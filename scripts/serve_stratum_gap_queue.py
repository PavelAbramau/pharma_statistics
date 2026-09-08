"""Serve exactly the programs in the two (band, archetype) strata that
pharma_stats.stats.label_statistics currently reports as having zero
labels — the InsufficientStratumCoverageError that blocks every
weighted population estimate in the project (see
reports/label_statistics.md). Labelling these unlocks every IPW
population figure at once; there's no larger population-estimate task
waiting on more than this.

    python scripts/serve_stratum_gap_queue.py

Only reorders/populates session state (stratum_gap_order) — writes no
gold. Fully reversible, same as scripts/serve_auto_gate2_rejects.py.
"""
from __future__ import annotations

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store
from pharma_stats.stats import label_statistics as ls

# The exact two zero-label cells label_statistics.compute_stratum_weights
# refuses on today. Hand-pinned, not re-derived from the exception at
# script run time: the whole point is to serve a fixed, known-good batch
# and see the coverage gap close, not to silently chase a moving target
# if the population shifts underneath a re-fetch mid-session.
TARGET_STRATA = [
    (1, "registry_terminated_stated_reason"),
    (4, "registry_terminated_vague_reason"),
]


def stratum_gap_ids(programs: list[dict], gold_records: list[dict]) -> list[str]:
    labelled = set(ls.gate3_labels_by_program(gold_records))
    ids = [
        p["program_id"] for p in programs
        if (p["band"], p["primary_archetype"]) in TARGET_STRATA and p["program_id"] not in labelled
    ]
    return sorted(ids)


def main() -> None:
    programs = pp.load_materialized()
    gold_records = store.load_records()
    ids = stratum_gap_ids(programs, gold_records)

    print(f"{len(ids)} program(s) across the {len(TARGET_STRATA)} zero-label strata:")
    for band, archetype in TARGET_STRATA:
        n = sum(
            1 for p in programs
            if p["band"] == band and p["primary_archetype"] == archetype and p["program_id"] in ids
        )
        print(f"  (band={band}, {archetype}): {n}")

    reviewed = store.reviewed_program_ids(gold_records)
    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)

    session["stratum_gap_order"] = ids
    q.save_session(session)
    print(f"\nInstalled {len(ids)} candidate(s) into the separate stratum-gap queue.")
    print(f"active_queue is still {session.get('active_queue', 'main')!r} — switch to 'stratum_gap' "
          "in the app (topbar, 'coverage gap') whenever you're ready to work through it.")


if __name__ == "__main__":
    main()
