"""Commit Layer 1 auto-rejections into gold + the labelling session.

    python scripts/apply_triage_to_queue.py [--dry-run]

Does not commit Layer 2/3. Does not write is_adc=yes/in_scope=yes
(those stay in the queue at Gate 3). heme_only is withheld if MeSH
coverage is below threshold, if ambiguous classifications dominate,
or if the existing heme_only agreement gate is not open.
"""
from __future__ import annotations

import argparse

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import stats as label_stats
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.labelling import triage_serve
from pharma_stats.triage import apply as triage_apply
from pharma_stats.triage import validation as tval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    result = triage_apply.apply_layer1(programs, dry_run=args.dry_run)
    prefix = "[dry-run] " if result["dry_run"] else ""
    print(f"{prefix}Layer 1 auto-rejections: {result['n_commit']} "
          f"(is_adc=no {result['n_is_adc_no']}, in_scope=no {result['n_in_scope_no']})")
    print(f"heme_only auto-exclude: {'ON' if result['heme_auto_ok'] else 'OFF'} — {result['heme_reason']}")
    if result["dry_run"] and result.get("names"):
        for name, rule in result["names"][:30]:
            print(f"  - {name}  [{rule}]")
        extra = len(result["names"]) - 30
        if extra > 0:
            print(f"  … and {extra} more")

    gold_records = store.load_records()
    heme_auto_ok, _ = triage_serve.heme_only_auto_exclude_allowed(programs, gold_records)
    model_ok, model_reason = triage_serve.model_layer_gate_passed(gold_records)
    print(f"Layer 2/3 auto-commit gate: {'OPEN' if model_ok else 'CLOSED'} — {model_reason}")

    session = q.load_session()
    remaining = list(session["order"]) if session else [p["program_id"] for p in programs]
    reopen = session.get("reopen_queue", []) if session else []
    heme_holdout = {item["program_id"] for item in ts.load_validation_sample()}
    triage_holdout = {d["program_id"] for d in tval.load_validation_sample()}
    comp = triage_serve.queue_composition(
        programs, remaining,
        gold_records=gold_records,
        heme_auto_ok=heme_auto_ok,
        model_gate_ok=model_ok,
        heme_holdout_ids=heme_holdout,
        triage_holdout_ids=triage_holdout,
        reopen_ids=reopen,
    )
    median_s = comp["median_seconds_per_label"]
    hours = comp["hours_left_to_target"]
    print(f"\nManual queue: {comp['manual_queue']}")
    print(f"  enter at Gate 1: {comp['enter_gate1']}")
    print(f"  enter at Gate 2: {comp['enter_gate2']}")
    print(f"  enter at Gate 3: {comp['enter_gate3']}")
    if comp["skip_still_in_session_order"]:
        print(f"  (session.order still holds {comp['skip_still_in_session_order']} "
              "auto-skip ids — they will be dropped on serve)")
    print(f"Gate-3 labelled: {comp['gate3_labelled']} / target {comp['gate3_target']} "
          f"({comp['remaining_to_target']} remaining)")
    if median_s is not None and hours is not None:
        print(f"Median {median_s:.0f}s/gate-3 label → ~{hours:.1f} hours left to target "
              f"(Gate-3-entry cards skip Gates 1–2, so this is an upper bound on the "
              f"{comp['enter_gate3']} already-resolved ADC cards).")
    else:
        print("No gate-3 timing data on file — cannot estimate hours.")

    # Keep labelled_count in the report too, for the topbar number.
    print(f"Gold gate counts: {label_stats.gate_counts(gold_records)}")


if __name__ == "__main__":
    main()
