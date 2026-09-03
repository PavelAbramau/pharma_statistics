"""Layer 1.5: ChEMBL molecule_type lookup for candidates Layer 1's local
rules and Layer 2/3's model calls all left unsure. Free, no auth,
deterministic — see triage/layer1_5.py for why this exists and what it
does and does not resolve.

    python scripts/run_layer1_5_chembl.py --dry-run
    python scripts/run_layer1_5_chembl.py

Stages resolved decisions (layer=1, decided_by=auto, rule=
layer1_5_chembl:<value>:<chembl_id>) to triage/staged_decisions.jsonl —
does NOT write to gold/labels.jsonl. Promote via the same review path any
other staged decision goes through.
"""
from __future__ import annotations

import argparse
import uuid

from pharma_stats.clients.chembl import ChemblClient
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.triage import layer1_5, pool as tpool, staging


def _load_unsure_candidates(staged_records: list[dict]) -> list[dict]:
    latest = staging.latest_by_program(staged_records)
    return [r for r in latest.values() if r.get("layer") == 3 and r.get("is_adc") == "unsure"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, stage nothing")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    staged_records = staging.load_records()
    candidates = _load_unsure_candidates(staged_records)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"{len(candidates)} unsure candidate(s) loaded from {staging.STAGING_PATH}")

    programs = pp.load_materialized()
    by_pid = {p["program_id"]: p for p in programs}

    client = ChemblClient()
    run_id = f"triage:layer1_5:{uuid.uuid4().hex[:8]}"

    n_resolved = n_yes = n_no = n_missing_program = 0
    hits: list[dict] = []

    for c in candidates:
        program = by_pid.get(c["program_id"])
        if program is None:
            n_missing_program += 1
            continue
        result, hit = layer1_5.evaluate_layer1_5(program, client=client)
        if result is None:
            continue
        n_resolved += 1
        if result.is_adc == "no":
            n_no += 1
        else:
            n_yes += 1
        hits.append({
            "program_id": c["program_id"], "name": program.get("proposed_name"),
            "is_adc": result.is_adc, "source": hit.source, "field": hit.field,
            "value": hit.value, "record_id": hit.record_id,
            "matched_name": hit.matched_name, "matched_via": hit.matched_via,
        })
        if not args.dry_run:
            try:
                tpool.assert_not_reviewed(c["program_id"])
            except tpool.PoolIntegrityError as e:
                print(f"  SKIPPED (pool integrity): {e}")
                continue
            record = staging.build_record({
                "program_id": c["program_id"], "proposed_name": program.get("proposed_name"),
                "is_adc": result.is_adc, "in_scope": result.in_scope, "layer": 1, "rule": result.rule,
            }, run_id=run_id)
            staging.append_record(record)

    print()
    print(f"Resolved: {n_resolved}/{len(candidates)} ({n_resolved / len(candidates):.1%} of the 232)"
          if candidates else "No candidates to process.")
    print(f"  is_adc=no:  {n_no}")
    print(f"  is_adc=yes: {n_yes} (still needs a scope call before committable)")
    if n_missing_program:
        print(f"  {n_missing_program} program_id(s) not found in materialized universe — skipped")
    print()
    for h in hits:
        print(f"  {h['name']} ({h['program_id']}): is_adc={h['is_adc']} via "
              f"{h['source']}.{h['field']}={h['value']!r} "
              f"(matched {h['matched_via']}={h['matched_name']!r}, id={h['record_id']})")

    if args.dry_run:
        print(f"\n[dry run] nothing staged. Re-run without --dry-run to write {n_resolved} decision(s) "
              f"to {staging.STAGING_PATH}.")
    else:
        print(f"\nStaged {n_resolved} decision(s) to {staging.STAGING_PATH} (run_id={run_id}).")


if __name__ == "__main__":
    main()
