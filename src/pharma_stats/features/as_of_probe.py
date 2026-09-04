"""Real-data leakage probe for the panel: pick trials with 2+ FETCHED
(body-on-disk) versions, resolve each at its earliest FETCHED version's
own date, and assert the result matches that version's own fields —
never a later version's (and never the current-state fetch's). Tests
the earliest FETCHED version deliberately, not the earliest INDEXED one
— version 0 and many interior versions are never fetched at all (see
trial_asof.py's module docstring: the backfill's selective body-fetch,
confirmed 56.6% of version>0 rows have no body), so a date before any
fetched version genuinely resolves to None, which is correct
carry-forward behaviour, not a leak, and not what this probe checks.

tests/test_features_panel.py covers the same as-of property on synthetic
fixtures; this runs it against whatever real data is actually on disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.features.trial_asof import _fetched_version_states, resolve_trial_summary_as_of


@dataclass
class ProbeResult:
    n_checked: int
    n_passed: int
    failures: list[str]

    @property
    def passed(self) -> bool:
        return self.n_checked > 0 and not self.failures


def run_asof_probe(con: duckdb.DuckDBPyConnection, *, sample_size: int = 50) -> ProbeResult:
    candidate_ids = [
        r[0] for r in con.execute(
            "SELECT nct_id FROM history_index GROUP BY nct_id HAVING count(*) >= 2 ORDER BY nct_id LIMIT ?",
            [sample_size * 4],  # oversample -- most won't have >=2 FETCHED versions, filtered below
        ).fetchall()
    ]

    failures = []
    n_checked = 0
    for nct_id in candidate_ids:
        if n_checked >= sample_size:
            break
        states = _fetched_version_states(nct_id, con)
        if len(states) < 2:
            continue  # needs at least 2 fetched versions to meaningfully probe "not the later one"
        n_checked += 1
        earliest = states[0]
        as_of = earliest["posted_date"]
        summary = resolve_trial_summary_as_of(nct_id, as_of, con)
        if summary is None:
            failures.append(f"{nct_id}: resolved to None at its own earliest FETCHED version's date ({as_of})")
            continue
        if summary.source_snapshot != f"versioned:v{earliest['version']}":
            failures.append(
                f"{nct_id}: resolving as-of its earliest fetched version's own posted_date ({as_of}) "
                f"returned {summary.source_snapshot}, not versioned:v{earliest['version']} — a later "
                "version leaked into an earlier month."
            )

    return ProbeResult(n_checked=n_checked, n_passed=n_checked - len(failures), failures=failures)


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        result = run_asof_probe(con)
    finally:
        con.close()
    print(f"as-of probe: {result.n_passed}/{result.n_checked} passed")
    for f in result.failures:
        print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
