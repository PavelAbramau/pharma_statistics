"""The one place Layer 2/3 staged decisions can become gold — gated on
triage/validation.check_gate() passing against the drawn, judged sample.

    python scripts/promote_triage_decisions.py --check    # report gate status only, writes nothing
    python scripts/promote_triage_decisions.py --accept   # gate must pass; writes gold, marks staging accepted
    python scripts/promote_triage_decisions.py --reject    # marks staging rejected; writes nothing to gold

Fully reversible: gold gets NEW append-only records (decided_by=auto);
staging gets NEW status-tagged records for the same program_ids. Nothing
is overwritten or deleted either way — see triage/promote.py.
"""
from __future__ import annotations

import argparse

from pharma_stats.labelling import store as gold_store
from pharma_stats.triage import promote, staging, validation as val


def cmd_check() -> None:
    sample = val.load_validation_sample()
    if not sample:
        print(f"No validation sample at {val.VALIDATION_SAMPLE_PATH}.")
        return
    gold_records = gold_store.load_records()
    agreement = val.compute_agreement(sample, gold_records)
    passed, reason = val.check_gate(agreement)

    print(f"Sample: {len(sample)} drawn.")
    print(f"is_adc:    {agreement['is_adc']['agree']}/{agreement['is_adc']['compared']} "
          f"= {agreement['is_adc']['agreement_rate']:.1%}" if agreement['is_adc']['agreement_rate'] is not None
          else f"is_adc:    {agreement['is_adc']['compared']} judged so far")
    print(f"in_scope:  {agreement['in_scope']['agree']}/{agreement['in_scope']['compared']} "
          f"= {agreement['in_scope']['agreement_rate']:.1%}" if agreement['in_scope']['agreement_rate'] is not None
          else "in_scope:  not enough gate-2+ comparisons yet")
    print("\nPer stratum:")
    for k, v in agreement["by_stratum"].items():
        rate = f"{v['agreement_rate']:.1%}" if v["agreement_rate"] is not None else "n/a"
        print(f"  {k}: {v['agree']}/{v['compared']} = {rate}")
    if agreement["appendix"]["compared"]:
        a = agreement["appendix"]
        rate = f"{a['agreement_rate']:.1%}" if a["agreement_rate"] is not None else "n/a"
        print(f"  no_usable_evidence/appendix (not counted toward the gate): {a['agree']}/{a['compared']} = {rate}")

    gap = val.recall_vs_text_gap(agreement)
    if gap:
        print(f"\ntext {gap['text_agreement_rate']:.1%} (n={gap['text_compared']}) vs "
              f"recall/no-usable-evidence {gap['recall_agreement_rate']:.1%} (n={gap['recall_compared']}) "
              f"— gap {gap['gap']:+.1%}")

    print(f"\nGate: {'PASS' if passed else 'FAIL'} — {reason}")
    if passed:
        pending = promote.pending_committable_layer23(staging.load_records(), gold_records)
        print(f"Would accept {len(pending)} pending is_adc=no decision(s) into gold with --accept.")
    else:
        pending = [
            r for r in staging.latest_by_program(staging.load_records()).values()
            if r.get("layer") in (2, 3) and r.get("status") == "pending" and not r.get("manual_overflow")
        ]
        print(f"Would mark {len(pending)} pending Layer 2/3 decision(s) rejected with --reject.")


def cmd_accept() -> None:
    try:
        result = promote.accept_all_pending()
    except promote.PromotionError as e:
        print(f"REFUSED: {e}")
        return
    print(f"Accepted {result['n_accepted']} decision(s) into gold (run_id={result['run_id']}).")
    print("Reversible: these are decided_by=auto gold records — a human re-review is always a "
          "newer record and overrides it, same as any other gold entry.")


def cmd_reject() -> None:
    result = promote.reject_all_pending()
    print(f"Rejected {result['n_rejected']} pending Layer 2/3 decision(s) — reason: {result['reason']!r}. "
          f"Nothing written to gold; all stay in the manual queue (run_id={result['run_id']}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--reject", action="store_true")
    args = ap.parse_args()

    modes = [m for m in (args.check, args.accept, args.reject) if m]
    if len(modes) != 1:
        print("Pass exactly one of --check / --accept / --reject.")
        return

    if args.check:
        cmd_check()
    elif args.accept:
        cmd_accept()
    else:
        cmd_reject()


if __name__ == "__main__":
    main()
