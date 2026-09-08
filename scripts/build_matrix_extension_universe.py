"""Matrix-only universe extension (docs/decisions/0008): discovers ADC
trials from 2000-2012 and haematological-indication ADC trials via CT.gov,
applies the deterministic dead_historical rule (labelling/dead_historical
.py — no manual labelling for this cohort, ever), and writes the result
to data/matrix_extension_programs.json.

Deliberately NOT written to gold/labels.jsonl, provisional_programs, or
asset_candidates — this cohort stays out of the detector's scope and out
of the gold set by construction (docs/decisions/0008). It exists solely
so scripts/build_adc_indication_table.py's coarse (payload x tumour-
system) matrix view can show it as a separate, visually-distinct,
lower-confidence column.

    python scripts/build_matrix_extension_universe.py
"""
from __future__ import annotations

import dataclasses
import json
import time

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import DATA_DIR
from pharma_stats.discovery import matrix_extension as mx
from pharma_stats.discovery.candidates import build_candidate_table
from pharma_stats.labelling import provisional_programs as pp

OUT_PATH = DATA_DIR / "matrix_extension_programs.json"


def main() -> None:
    client = CtgovClient()

    t0 = time.time()
    print("Discovering ADC trials 2000-present, no heme exclusion (pattern-matching pass only)...")
    mentions, conditions_by_nct = mx.discover_extension_mentions(client)
    print(f"  {len(mentions)} mentions, {len({m.nct_id for m in mentions})} distinct trials "
          f"[{time.time() - t0:.0f}s]")

    candidates = build_candidate_table(mentions)
    print(f"Clustered into {len(candidates)} candidate assets.")

    main_programs = pp.load_materialized()
    existing_candidate_ids = frozenset(
        p["candidate_id"] for p in main_programs if p.get("candidate_id")
    )
    programs, build_stats = mx.build_extension_programs(
        candidates, conditions_by_nct, existing_candidate_ids=existing_candidate_ids,
    )
    print(f"\n{build_stats['n_candidates_discovered']} candidates discovered.")
    print(f"  excluded (not confirmed ADC by Layer-1 rules): {build_stats['n_excluded_not_adc']}")
    print(f"  excluded (not all-industry sponsors): {build_stats['n_excluded_non_industry']}")
    print(f"  excluded (same candidate_id already in the main asset_candidates/provisional_programs "
          f"pipeline): {build_stats['n_excluded_already_in_main_universe']}")
    print(f"  excluded (neither pre_2012 nor heme -- belongs to the main universe instead): "
          f"{build_stats['n_excluded_neither_reason']}")
    print(f"  in extension cohort: {build_stats['n_in_extension']}")
    n_pre_2012 = sum(1 for p in programs if "pre_2012" in p.extension_reasons)
    n_heme = sum(1 for p in programs if "heme" in p.extension_reasons)
    print(f"    of which pre_2012: {n_pre_2012}, heme: {n_heme} "
          f"(a program can be both).")

    print("\nBuilding sponsor x target population index (extension + main universe) "
          "for the successor check...")
    population_index = mx.build_population_index(programs, main_programs)

    mx.classify_extension_cohort(programs, population_index)
    n_dead_historical = sum(1 for p in programs if p.dead_historical == "dead_historical")
    n_not_classified = len(programs) - n_dead_historical
    print(f"\ndead_historical: {n_dead_historical} of {len(programs)} extension programs "
          f"({n_not_classified} not classified -- recent trial, known-approved, or a successor "
          "asset; see labelling/dead_historical.py for the exact rule).")
    print("STATE CLEARLY: every dead_historical status here is an INFERENCE from a deterministic "
          "rule, never a labelled/gold outcome (docs/decisions/0008).")

    records = [dataclasses.asdict(p) for p in programs]
    for r in records:
        r["trial_start_dates"] = [d.isoformat() for d in r["trial_start_dates"]]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nWrote {len(records)} extension program(s) to {OUT_PATH}")


if __name__ == "__main__":
    main()
