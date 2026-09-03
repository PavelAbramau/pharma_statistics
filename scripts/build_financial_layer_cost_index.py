"""Item 1 of the financial layer: synthetic program cost index + the
conviction ratio, computed for the whole universe (this needs no filing
data, unlike items 2-5) and emitted as monthly financial_events rows.

    python scripts/build_financial_layer_cost_index.py [--limit N]

See docs/decisions/0003-synthetic-cost-benchmark.md for the benchmark
citation and formula construction, and audit/leakage.md for this
feature's knowability-date contract. Nothing here writes gold; this is a
derived, rebuildable view, same as provisional_programs.py.
"""
from __future__ import annotations

import argparse
from datetime import date

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.finance import conviction as cv
from pharma_stats.finance import cost_model as cm
from pharma_stats.finance import store as fstore
from pharma_stats.labelling import provisional_programs as pp

SOURCE_TAG = "sertkaya_2016_synthetic"


def _month_range(start: date, end: date) -> list[date]:
    months = []
    cursor = start.replace(day=1)
    while cursor <= end:
        months.append(cursor)
        year, month = cursor.year, cursor.month
        cursor = date(year + (month // 12), (month % 12) + 1, 1)
    return months


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap on programs processed (debugging)")
    ap.add_argument("--dry-run", action="store_true", help="compute and print summary, write nothing")
    args = ap.parse_args()

    programs = pp.load_materialized()
    if args.limit:
        programs = programs[: args.limit]
    print(f"{len(programs)} program(s) in the universe.")

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        today = date.today()

        # Present-day ranking snapshot (all 4 factors, including today's
        # site count) -- for the "relative index, ranking is what
        # matters" use case, not for the backtest-safe monthly series.
        snapshots = {}
        for p in programs:
            snap = cm.program_cost_index_snapshot(p, con, as_of=today)
            if snap["cost_index"] > 0:
                snapshots[p["program_id"]] = snap["cost_index"]
        print(f"{len(snapshots)}/{len(programs)} program(s) have a nonzero cost index today "
              "(the rest have no trial with any indexed history yet).")

        # Time-cut-safe monthly series, one point per program per month --
        # what actually gets registered in audit/leakage.md.
        by_program_series: dict[str, list[dict]] = {}
        earliest, latest = None, None
        for p in programs:
            series = cm.monthly_cost_index_series(p, con, end=today)
            if series:
                by_program_series[p["program_id"]] = series
                first_month = date.fromisoformat(series[0]["as_of"])
                earliest = first_month if earliest is None else min(earliest, first_month)
                latest = today if latest is None else latest

        n_months = len(_month_range(earliest, today)) if earliest else 0
        print(f"{len(by_program_series)}/{len(programs)} program(s) have a resolvable monthly series "
              f"({n_months} months, {earliest} to {today} if any).")

        if args.dry_run:
            print("[dry run] no financial_events written.")
            return

        events = []
        programs_by_id = {p["program_id"]: p for p in programs}

        # Per-month conviction ratios: at each month, compare each
        # program's SAME-MONTH cost index (not its current one) to its
        # SAME-MONTH peers -- genuinely time-cut safe, not just the
        # inputs to it.
        all_months = _month_range(earliest, today) if earliest else []
        series_by_month: dict[date, dict[str, float]] = {m: {} for m in all_months}
        for pid, series in by_program_series.items():
            for point in series:
                m = date.fromisoformat(point["as_of"])
                if point["cost_index"] > 0:
                    series_by_month[m][pid] = point["cost_index"]

        for pid, series in by_program_series.items():
            program = programs_by_id[pid]
            for point in series:
                m = date.fromisoformat(point["as_of"])
                events.append(fstore.FinancialEvent(
                    subject_type="program", subject_id=pid, event_date=m,
                    event_type="synthetic_cost_index_monthly",
                    detail="phase_weight x enrollment x elapsed_months, versioned-history only",
                    source=SOURCE_TAG, value=point["cost_index"],
                ))

        for m, spend_by_program in series_by_month.items():
            month_programs = [programs_by_id[pid] for pid in spend_by_program]
            ratios = cv.compute_conviction_ratios(month_programs, spend_by_program)
            for pid, r in ratios.items():
                if r["conviction_ratio"] is None:
                    continue
                events.append(fstore.FinancialEvent(
                    subject_type="program", subject_id=pid, event_date=m,
                    event_type="conviction_ratio_monthly",
                    detail=f"peer_group={r['peer_group']}, n_peers={r['n_peers']}",
                    source=SOURCE_TAG, value=r["conviction_ratio"],
                ))
    finally:
        con.close()

    n = fstore.append_records(events)
    print(f"\nWrote {n} financial_events row(s) "
          f"({sum(1 for e in events if e.event_type == 'synthetic_cost_index_monthly')} cost-index, "
          f"{sum(1 for e in events if e.event_type == 'conviction_ratio_monthly')} conviction-ratio).")


if __name__ == "__main__":
    main()
