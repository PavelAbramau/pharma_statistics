"""Serve only the part of the queue that can't be overturned by the
still-running background triage pipeline or the not-yet-judged
validation gate:

  Pool A — no real staged machine decision at all (never touched, or
           only manual_overflow-flagged with no verdict yet). Needs the
           normal full three-gate review regardless of anything else.
  Pool B — staged is_adc=yes, awaiting a human Gate-2/Gate-3 call. Layer
           2/3 never resolves in_scope, so this can only ever be settled
           by a human — nothing the background pipeline does changes it.

Explicitly EXCLUDED: any program with a staged is_adc=no ("reject") or
is_adc=unsure verdict — those are exactly what the validation gate might
overturn wholesale via triage/promote.py, and validation-sample program
ids themselves (reviewed blind, in the separate validation queue).

This is a snapshot, installed once into session["order"]: once serving
starts, nothing the background pipeline stages afterward can shrink or
reorder it out from under the reviewer — same guarantee as
prepare_manual_queue.py's disposition order, just a different filter.

    python scripts/serve_stable_pool.py
"""
from __future__ import annotations

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval


def stable_pool_ids(
    programs: list[dict], gold_records: list[dict], staged_records: list[dict],
    exclude_ids: set,
) -> tuple[list[str], dict]:
    reviewed = store.reviewed_program_ids(gold_records)
    staged_latest = staging.latest_by_program(staged_records)

    pool_a, pool_b = [], []
    for p in programs:
        pid = p["program_id"]
        if pid in reviewed or pid in exclude_ids:
            continue
        s = staged_latest.get(pid)
        is_adc = s.get("is_adc") if s else None
        if is_adc == "yes":
            pool_b.append(pid)
        elif is_adc in ("no", "unsure"):
            continue  # excluded: validation might overturn a "no"; "unsure" isn't stable either
        else:
            pool_a.append(pid)  # no record, or manual_overflow-flagged with no real verdict yet

    return pool_a + pool_b, {"pool_a_no_opinion": len(pool_a), "pool_b_awaiting_gate": len(pool_b)}


def main() -> None:
    programs = pp.load_materialized()
    gold_records = store.load_records()
    reviewed = store.reviewed_program_ids(gold_records)
    staged_records = staging.load_records()

    heme_holdout_ids = {item["program_id"] for item in ts.load_validation_sample()}
    triage_holdout_ids = {d["program_id"] for d in tval.load_validation_sample()}
    exclude_ids = heme_holdout_ids | triage_holdout_ids

    ids, counts = stable_pool_ids(programs, gold_records, staged_records, exclude_ids)

    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)
    session["order"] = ids
    q.save_session(session)

    print(f"Pool A (no staged opinion yet): {counts['pool_a_no_opinion']}")
    print(f"Pool B (is_adc=yes, awaiting Gate-2/3): {counts['pool_b_awaiting_gate']}")
    print(f"Installed {len(ids)} card(s) as the stable main queue.")
    print(f"Excluded: {len(exclude_ids)} validation-sample id(s) (blind, separate queue), "
          "plus every staged is_adc=no/unsure candidate.")
    print("\nStart the app with: python scripts/run_labelling_app.py")


if __name__ == "__main__":
    main()
