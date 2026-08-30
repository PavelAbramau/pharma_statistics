"""Apply the MeSH-based heme_only auto-exclusion, with a held-out
validation sample and a self-enforced agreement kill-switch.

    python scripts/apply_auto_scope_exclusions.py [--dry-run]

Never overrides a human decision — only ever considers programs with no
existing label record at all — and never deletes anything: an excluded
asset stays in asset_candidates/provisional_programs, it's just no longer
served by the review app (store.reviewed_program_ids treats any label
record, auto or human, as terminal, same as a manual gate-1/2 rejection).

Safety, two independent gates:

1. MeSH coverage across in-universe trials must be >= trial_scope.
   MESH_COVERAGE_THRESHOLD (audit/universe.py's coverage gate checks the
   same number) — a heme_only count from near-zero coverage isn't a clean
   result, it's an absence of one. Run scripts/fetch_current_state.py
   first if this refuses.
2. The CURRENT agreement rate on the held-out validation sample (see
   labelling/trial_scope.py; audit/universe.py's "heme_only auto-exclusion
   agreement" check runs the same comparison) must be >=
   trial_scope.AGREEMENT_THRESHOLD.

Below either, this refuses to write any new auto-exclusions — the whole
mechanism reverts to manual until both clear — though the validation
sample itself still gets topped up so there's something to keep measuring
against once coverage exists.
"""
from __future__ import annotations

import argparse

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts

SESSION_ID = "auto:apply_auto_scope_exclusions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report what would happen, write nothing")
    args = parser.parse_args()

    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    coverage = ts.mesh_coverage(programs)
    print(f"MeSH coverage: {coverage['coverage_rate']:.1%} ({coverage['covered']} / {coverage['total']})")
    if coverage["coverage_rate"] < ts.MESH_COVERAGE_THRESHOLD:
        print(f"Below the {ts.MESH_COVERAGE_THRESHOLD:.0%} coverage gate — refusing to run. "
              "Run scripts/fetch_current_state.py to raise coverage first.")
        return

    records = store.load_records()
    reviewed = store.reviewed_program_ids(records)

    qualifying = [
        p for p in programs
        if p.get("scope_category") == "heme_only" and p["program_id"] not in reviewed
    ]
    by_id = {p["program_id"]: p for p in qualifying}
    candidate_ids = sorted(by_id)

    existing_sample = ts.load_validation_sample()
    already_reserved = {item["program_id"] for item in existing_sample}

    sample_ids = ts.draw_validation_sample(candidate_ids, already_reserved)
    new_sample = [
        {"program_id": pid, "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"}
        for pid in sample_ids
    ]
    newly_reserved = [pid for pid in sample_ids if pid not in already_reserved]

    agreement = ts.validation_agreement(new_sample, records)
    print(f"Qualifying heme_only, unreviewed assets: {len(qualifying)}")
    print(f"Validation sample: {len(new_sample)} held out ({len(newly_reserved)} newly reserved this run).")
    if agreement["agreement_rate"] is not None:
        print(f"Agreement so far: {agreement['agreements']}/{agreement['compared']} compared "
              f"({agreement['agreement_rate']:.0%})")
    else:
        print("Agreement so far: none of the sample has been reviewed yet.")

    sample_id_set = set(sample_ids)
    to_exclude = [p for p in qualifying if p["program_id"] not in sample_id_set]

    gate_open = (
        agreement["agreement_rate"] is not None
        and agreement["agreement_rate"] >= ts.AGREEMENT_THRESHOLD
    )
    # A still-empty gold set (nothing reviewed at all yet, no prior sample
    # on file) has no track record to distrust, but also none to trust —
    # that one case is the legitimate bootstrap: exclude, and let the
    # validation sample start accumulating comparisons from here on.
    bootstrapping = agreement["compared"] == 0 and not existing_sample

    ts.save_validation_sample(new_sample)

    if not gate_open and not bootstrapping:
        print(f"Agreement below {ts.AGREEMENT_THRESHOLD:.0%} (or the sample isn't reviewed enough yet) — "
              f"auto-exclusion is OFF. {len(to_exclude)} otherwise-qualifying asset(s) "
              "stay in the manual queue.")
        return

    if args.dry_run:
        print(f"[dry-run] would auto-exclude {len(to_exclude)} asset(s):")
        for p in to_exclude:
            print(f"  - {p['proposed_name']} ({p['program_id']})")
        return

    for p in to_exclude:
        decision = ts.auto_scope_decision(p["scope_category"])
        body = {
            "action": "label", "program_id": p["program_id"],
            "candidate_id": p["candidate_id"], "proposed_name": p["proposed_name"],
            "gate_reached": 2, "triage_layer": 1, "triage_rule": "layer1_mesh_heme_only",
            "discovery_strategy": p.get("discovery_strategy"),
            "match_strength": p.get("match_strength"), "matched_term": p.get("matched_term"),
            **decision,
        }
        store.validate_label_payload(body)
        record = store.build_record(body, session_id=SESSION_ID, served_stratum={})
        store.append_record(record)

    print(f"Auto-excluded {len(to_exclude)} heme_only asset(s), decided_by=auto, in_scope=no/heme_only.")


if __name__ == "__main__":
    main()
