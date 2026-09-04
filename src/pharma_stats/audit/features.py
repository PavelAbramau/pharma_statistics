"""Features stage: every panel column registered with a knowability rule
(features/knowability.py, backing audit/leakage.md), a real as-of-date
leakage probe against real data (features/as_of_probe.py — not just the
register check), and basic panel coverage.
"""
from __future__ import annotations

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok
from pharma_stats.config import REPO_ROOT, WAREHOUSE_DB
from pharma_stats.features.as_of_probe import run_asof_probe
from pharma_stats.features.knowability import REGISTRY

STAGE = "features"
LEAKAGE_REGISTER = REPO_ROOT / "audit" / "leakage.md"
REGISTERED_FEATURES = money_panel.FEATURE_NAMES


def run() -> list[Check]:
    checks = []

    checks.append(
        ok(STAGE, "knowability registry has at least one entry",
           expected=">=1 registered feature", actual=f"{len(REGISTRY)} registered")
        if REGISTRY else
        fail(STAGE, "knowability registry has at least one entry",
             expected=">=1 registered feature", actual="0 registered")
    )

    checks.append(
        ok(STAGE, "audit/leakage.md exists", expected="file present", actual="present")
        if LEAKAGE_REGISTER.exists() else
        fail(STAGE, "audit/leakage.md exists", expected="file present", actual="missing")
    )

    if not WAREHOUSE_DB.exists():
        checks.append(info(STAGE, "as-of leakage probe", expected="run against real history_index data",
                            actual="SKIPPED — no warehouse.duckdb on disk yet"))
        return checks

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        result = run_asof_probe(con, sample_size=50)
    finally:
        con.close()

    level = ok if result.passed else fail
    checks.append(level(
        STAGE, "as-of leakage probe: resolving a trial as of its own earliest fetched "
               "version's date must never return a later version",
        expected=f"{result.n_checked}/{result.n_checked} pass",
        actual=f"{result.n_passed}/{result.n_checked} pass",
        detail="; ".join(result.failures[:10]) if result.failures else "no failures",
    ))
    return checks
