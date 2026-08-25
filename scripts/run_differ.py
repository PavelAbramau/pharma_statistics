"""Extract EvidenceEvents from every fetched adjacent trial-version pair.

Pure local computation — reads raw/ + history_index only, no network.

    python scripts/run_differ.py

Writes data/warehouse.duckdb::evidence_events and
reports/differ_noise_floor.md (noise floor + firing frequency +
negative control — read this before trusting any event).
"""
from __future__ import annotations

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.differ import extract, report


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE_DB))
    try:
        neg_control = report.negative_control(con)
    finally:
        con.close()
    print(f"Negative control: {'PASS' if neg_control[0] else 'FAIL'} — {neg_control[1]}")

    n_events, stats = extract.materialize()
    print(f"Extracted {n_events} events from {stats.total_pairs} version pairs "
          f"({stats.total_trials_with_2plus_versions} trials)")
    print(f"Zero-event pairs: {stats.pairs_with_zero_events} ({stats.zero_event_fraction:.1%})")

    path = report.write_report(stats, neg_control)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
