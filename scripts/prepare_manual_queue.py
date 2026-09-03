"""Re-order the manual queue for a full coverage pass over the residue —
disposition-ordered (likely-reject -> ambiguous -> confirmed-ADC), not
stratified by score band/archetype. See labelling/queue_order.py for the
bucketing rule.

This is a re-ordering step only — it makes no model calls. Run the two
model-backed coverage passes first, separately, whenever
ANTHROPIC_API_KEY is available:

    python scripts/run_layer3_overflow.py --max-spend 5   # the 174 manual-overflow candidates
    python scripts/run_triage_pipeline.py --full --max-spend N   # residue with no triage opinion at all

Safe to run before those finish, or skip them entirely — it orders
whatever triage state exists on disk right now. Re-running it is
idempotent: it recomputes the full order from current state each time,
the same way serve_validation_sample.py re-prioritizes without
duplicating or dropping anything.

    python scripts/prepare_manual_queue.py [--dry-run]
"""
from __future__ import annotations

import argparse

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import queue_order as qo
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.labelling import triage_serve
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report bucket counts only, don't touch the session")
    args = ap.parse_args()

    programs = pp.load_materialized()
    gold_records = store.load_records()
    reviewed = store.reviewed_program_ids(gold_records)

    heme_holdout_ids = {item["program_id"] for item in ts.load_validation_sample()}
    triage_holdout_ids = {d["program_id"] for d in tval.load_validation_sample()}
    heme_auto_ok, heme_reason = triage_serve.heme_only_auto_exclude_allowed(programs, gold_records)
    model_gate_ok, model_reason = triage_serve.model_layer_gate_passed(gold_records)
    staged_by_program = staging.latest_by_program(staging.load_records())

    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)
        print("No existing session — created a fresh one.")

    triage_serve.ingest_reopens(session, {p["program_id"] for p in programs})
    remaining_ids = list(session.get("reopen_queue") or []) + list(session.get("order") or [])
    remaining_ids = [pid for pid in remaining_ids if pid not in reviewed]

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        ordered, counts = qo.build_disposition_order(
            programs, remaining_ids,
            gold_records=gold_records, heme_auto_ok=heme_auto_ok, model_gate_ok=model_gate_ok,
            heme_holdout_ids=heme_holdout_ids, triage_holdout_ids=triage_holdout_ids,
            staged_by_program=staged_by_program, con=con,
        )
    finally:
        con.close()

    print(f"heme_only auto-exclude: {'ON' if heme_auto_ok else 'off'} ({heme_reason})")
    print(f"model-layer (Layer 2/3) auto-skip: {'ON' if model_gate_ok else 'off'} ({model_reason})")
    print(f"\n{len(ordered)} candidate(s) will be served, disposition-ordered:")
    for bucket in qo.BUCKET_ORDER:
        print(f"  {bucket}: {counts[bucket]}")
    n_auto_excluded = len(remaining_ids) - len(ordered)
    print(f"  (auto-excluded from the queue entirely, already decided by triage: {n_auto_excluded})")

    if args.dry_run:
        print("\n--dry-run: session not modified.")
        return

    session["order"] = [pid for pid in ordered if pid not in set(session.get("reopen_queue") or [])]
    q.save_session(session)
    print(f"\nInstalled disposition order as the manual queue ({len(session['order'])} candidate(s)). "
          "Reopens (if any) still come first, ahead of this order.")


if __name__ == "__main__":
    main()
