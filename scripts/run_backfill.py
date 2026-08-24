"""Resumable, priority-ordered history backfill.

  python scripts/run_backfill.py --signal-labels none --max-seconds 3600
      # index-only pass across the full trial universe (no body fetches);
      # use this first to get real request-count numbers.

  python scripts/run_backfill.py --signal-labels recommended --max-seconds 3600
      # index refresh + selective body fetch, bounded to ~1 hour.
      # Ctrl-C / SIGKILL at any point is safe — rerun the same command to resume.

--signal-labels: "requested" = exactly the user's literal module list
  (Study Design, Outcome Measures, Study Status, Sponsor/Collaborators —
  "Recruitment Status" folded into Study Status, see history/index.py);
  "recommended" = requested + Arms and Interventions (needed for
  arm_removed/cohort_dropped per the step-4 event spec); "none" = index
  only, fetch no version bodies.
"""
import argparse
import sys

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.history.index import (
    RECOMMENDED_SIGNAL_LABELS,
    USER_REQUESTED_SIGNAL_LABELS,
    module_filter_stats,
)
from pharma_stats.history.orchestrator import run_backfill

LABEL_SETS = {
    "requested": USER_REQUESTED_SIGNAL_LABELS,
    "recommended": RECOMMENDED_SIGNAL_LABELS,
    "none": frozenset(),
}


def trial_universe(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT nct_ids FROM asset_candidates").fetchall()
    ids: set[str] = set()
    for (nct_ids,) in rows:
        ids.update(nct_ids)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-labels", choices=list(LABEL_SETS), default="recommended")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--max-trials", type=int, default=None)
    args = ap.parse_args()

    signal_labels = LABEL_SETS[args.signal_labels]
    con = duckdb.connect(str(WAREHOUSE_DB))
    universe = trial_universe(con)
    print(f"Trial universe: {len(universe)} trials")
    print(f"Signal labels ({args.signal_labels}): {sorted(signal_labels) or '(none — index only)'}")

    client = CtgovClient()
    result = run_backfill(
        client, con, universe,
        signal_labels=signal_labels,
        max_seconds=args.max_seconds,
        max_trials=args.max_trials,
    )
    print("\n--- run result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n--- index coverage so far ---")
    for label_name, labels in [("requested", USER_REQUESTED_SIGNAL_LABELS),
                                ("recommended", RECOMMENDED_SIGNAL_LABELS)]:
        stats = module_filter_stats(con, labels)
        print(f"  [{label_name}] {stats}")

    con.close()


if __name__ == "__main__":
    sys.exit(main())
