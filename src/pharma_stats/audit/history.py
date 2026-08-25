"""History index stage: every universe trial indexed or a logged fetch
failure, no version-number holes, dates monotonic, module labels within
the known vocabulary (an unseen label may be endpoint schema drift, not
a new trial type — WARN, don't drop it silently)."""
from __future__ import annotations

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok, warn
from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.history.index import (
    COSMETIC_OR_OUT_OF_SCOPE_LABELS,
    RECOMMENDED_SIGNAL_LABELS,
)

STAGE = "history_index"

KNOWN_MODULE_LABELS = RECOMMENDED_SIGNAL_LABELS | COSMETIC_OR_OUT_OF_SCOPE_LABELS


def run() -> list[Check]:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        checks: list[Check] = []
        checks += _coverage_check(con)
        checks += _version_contiguity_and_monotonicity(con)
        checks += _module_vocab_check(con)
        return checks
    finally:
        con.close()


def _coverage_check(con) -> list[Check]:
    universe_trials = {
        r[0] for r in con.execute(
            "SELECT DISTINCT nct_id FROM (SELECT unnest(nct_ids) AS nct_id FROM asset_candidates)"
        ).fetchall()
    }
    indexed_trials = {r[0] for r in con.execute("SELECT DISTINCT nct_id FROM history_index").fetchall()}
    logged_failure_trials = {
        r[0] for r in con.execute(
            "SELECT nct_id FROM backfill_queue WHERE status = 'error'"
        ).fetchall()
    }
    tracked_trials = {r[0] for r in con.execute("SELECT nct_id FROM backfill_queue").fetchall()}

    gap = universe_trials - indexed_trials - logged_failure_trials
    untracked = universe_trials - tracked_trials

    checks = [
        (fail if gap else ok)(
            STAGE, "every universe trial has a history_index row or a logged fetch failure",
            expected="0 silent gaps", actual=f"{len(gap)} / {len(universe_trials)} silent gaps",
            detail=", ".join(sorted(gap)[:10]),
        ),
    ]
    if untracked:
        checks.append(warn(
            STAGE, "universe trials present in backfill_queue",
            expected=f"{len(universe_trials)} / {len(universe_trials)} tracked",
            actual=f"{len(universe_trials) - len(untracked)} / {len(universe_trials)} tracked",
            detail=", ".join(sorted(untracked)[:10]),
        ))
    return checks


def _version_contiguity_and_monotonicity(con) -> list[Check]:
    rows = con.execute(
        "SELECT nct_id, version, posted_date FROM history_index ORDER BY nct_id, version"
    ).fetchall()

    by_trial: dict[str, list[tuple[int, object]]] = {}
    for nct_id, version, posted_date in rows:
        by_trial.setdefault(nct_id, []).append((version, posted_date))

    holes, non_monotonic = [], []
    for nct_id, versions in by_trial.items():
        vs = [v for v, _ in versions]
        if vs != list(range(vs[0], vs[0] + len(vs))) or vs[0] != 0:
            holes.append(nct_id)
        dates = [d for _, d in versions if d is not None]
        if any(a > b for a, b in zip(dates, dates[1:])):
            non_monotonic.append(nct_id)

    return [
        (fail if holes else ok)(
            STAGE, "version numbers contiguous from 0 with no holes",
            expected="0 trials with gaps", actual=f"{len(holes)} / {len(by_trial)} trials",
            detail=", ".join(holes[:10]),
        ),
        (fail if non_monotonic else ok)(
            STAGE, "version posted_date is monotonically non-decreasing",
            expected="0 trials with an out-of-order version",
            actual=f"{len(non_monotonic)} / {len(by_trial)} trials",
            detail=", ".join(non_monotonic[:10]),
        ),
    ]


def _module_vocab_check(con) -> list[Check]:
    rows = con.execute(
        "SELECT DISTINCT unnest(changed_modules) AS label FROM history_index WHERE changed_modules IS NOT NULL"
    ).fetchall()
    seen = {r[0] for r in rows}
    unknown = sorted(seen - KNOWN_MODULE_LABELS)

    return [(warn if unknown else ok)(
        STAGE, "changed_modules labels are within the known CT.gov module vocabulary",
        expected="0 unrecognised labels", actual=f"{len(unknown)} unrecognised",
        detail=", ".join(unknown[:15]) + (
            " — could be endpoint schema drift; check schema_guard before assuming it's benign"
            if unknown else ""
        ),
    )]
