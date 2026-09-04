"""Features stage: two real feature sources, checked together.

1. The program x month panel (features/panel.py) — silence_score_asof,
   band_asof, cost_index, contacts_locations_amendment_cadence_asof
   (time-varying) plus target, payload_chemotype, indication_mesh_term
   (static). Checked via the knowability registry
   (features/knowability.py) and a real as-of leakage probe against
   ACTUAL history_index data (features/as_of_probe.py) — stronger than a
   check against the money panel alone, since it exercises the real
   carry-forward/version-resolution logic against whatever's really on
   disk, not just a derived feature table.

2. The money-layer feature panel (finance/panel.py) — conviction_ratio,
   estimated_cumulative_spend, built from financial_events. Checked via
   row count, a knowability_date<=as_of violation scan, and per-feature
   NaN rates (_money_layer_checks) — this is the only thing that
   exercises the financial_events slice at all; the program-panel checks
   above never touch it.

Both sources register in the SAME audit/leakage.md, so they share ONE
combined registration check over the union of
features.knowability.REGISTRY and finance.panel.FEATURE_NAMES — a
feature from either source missing from the register is a FAIL, not two
separate, each-incomplete checks that could both silently pass while a
feature from the OTHER source goes unregistered.

Still not built, reported as an honest INFO rather than omitted:
Differ-event-derived features beyond contacts_locations_amendment_cadence
(e.g. no trial_reopened/arm_removed counts yet) and anything requiring
the five-entity warehouse. The program x month panel itself is real,
not a stub — confirmed against features/panel.py and its test suite
before dropping the old "not built" placeholder for it.
"""
from __future__ import annotations

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok
from pharma_stats.config import REPO_ROOT, WAREHOUSE_DB
from pharma_stats.features.as_of_probe import run_asof_probe
from pharma_stats.features.knowability import REGISTRY as PANEL_REGISTRY
from pharma_stats.finance import panel as money_panel

STAGE = "features"
LEAKAGE_REGISTER = REPO_ROOT / "audit" / "leakage.md"


def _leakage_register_check() -> list[Check]:
    """Union of both feature sources — see module docstring on why this
    must be one check, not two."""
    panel_features = sorted(PANEL_REGISTRY.keys())
    money_features = sorted(money_panel.FEATURE_NAMES)
    all_features = sorted(set(panel_features) | set(money_features))

    if not LEAKAGE_REGISTER.exists():
        return [fail(
            STAGE, "audit/leakage.md exists",
            expected="present", actual="missing",
            detail="every time-varying feature (program panel or money layer) must carry a "
                   "knowability-date contract here before it's trusted (CLAUDE.md)",
        )]
    text = LEAKAGE_REGISTER.read_text(encoding="utf-8")
    missing = [f for f in all_features if f not in text]
    return [(fail if missing else ok)(
        STAGE, "every feature (program-panel + money-layer) is registered in audit/leakage.md",
        expected=f"{len(all_features)} registered ({len(panel_features)} program-panel + "
                 f"{len(money_features)} money-layer)",
        actual=f"{len(all_features) - len(missing)} registered",
        detail=f"missing: {', '.join(missing)}" if missing else "",
    )]


def _panel_asof_probe_checks() -> list[Check]:
    checks = [
        ok(STAGE, "program x month panel knowability registry has at least one entry",
           expected=">=1 registered feature", actual=f"{len(PANEL_REGISTRY)} registered")
        if PANEL_REGISTRY else
        fail(STAGE, "program x month panel knowability registry has at least one entry",
             expected=">=1 registered feature", actual="0 registered")
    ]

    if not WAREHOUSE_DB.exists():
        checks.append(info(
            STAGE, "as-of leakage probe (program panel, real history_index data)",
            expected="run against real history_index data",
            actual="SKIPPED — no warehouse.duckdb on disk yet",
        ))
        return checks

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        result = run_asof_probe(con, sample_size=50)
    finally:
        con.close()
    level = ok if result.passed else fail
    checks.append(level(
        STAGE, "as-of leakage probe (program panel): resolving a trial as of its own earliest "
               "fetched version's date must never return a later version",
        expected=f"{result.n_checked}/{result.n_checked} pass",
        actual=f"{result.n_passed}/{result.n_checked} pass",
        detail="; ".join(result.failures[:10]) if result.failures else "no failures",
    ))
    return checks


def _money_layer_checks(panel: list[dict]) -> list[Check]:
    checks = [info(
        STAGE, "money-layer feature panel row count",
        expected="n/a", actual=f"{len(panel)} (program, month) row(s)",
    )]

    # As-of-date leakage probe: every row's knowability_date must equal its
    # own as_of (panel.py's construction contract, audit/leakage.md's whole
    # basis for trusting these two features) — never later.
    violations = [r for r in panel if r["knowability_date"] > r["as_of"]]
    checks.append((fail if violations else ok)(
        STAGE, "no money-layer feature row's knowability_date is later than its as_of date",
        expected="0 violations", actual=f"{len(violations)} violations",
        detail=", ".join(f"{v['program_id']}@{v['as_of']}" for v in violations[:5]),
    ))

    for feature in money_panel.FEATURE_NAMES:
        n_present = sum(1 for r in panel if r.get(feature) is not None)
        nan_rate = 1 - (n_present / len(panel))
        checks.append(info(
            STAGE, f"{feature} NaN rate",
            expected="n/a", actual=f"{nan_rate:.1%} missing ({n_present}/{len(panel)} present)",
        ))
    return checks


def run() -> list[Check]:
    checks: list[Check] = []
    checks += _leakage_register_check()
    checks += _panel_asof_probe_checks()

    money_layer_panel = money_panel.build_money_layer_panel()
    if not money_layer_panel:
        checks.append(info(
            STAGE, "money-layer feature panel (conviction_ratio, estimated_cumulative_spend)",
            expected="computed from financial_events",
            actual="0 rows — financial_events is empty, WAREHOUSE_DB doesn't exist yet, or "
                   "the layer hasn't been materialized",
            detail="run scripts/build_financial_layer_cost_index.py first",
        ))
    else:
        checks += _money_layer_checks(money_layer_panel)

    checks.append(info(
        STAGE, "still not built",
        expected="n/a",
        actual="Differ-event-derived features beyond contacts_locations_amendment_cadence_asof "
               "(e.g. trial_reopened/arm_removed counts) and anything requiring the "
               "five-entity warehouse — the program x month panel itself (silence_score_asof, "
               "cost_index, target/payload/indication) is real, not a stub.",
    ))
    return checks
