"""Features stage: the money-layer slice of the feature panel
(conviction_ratio, estimated_cumulative_spend — pharma_stats.finance.panel)
is real and checked here in full: 100% registered in audit/leakage.md
(unregistered = FAIL), an as-of-date leakage probe against the panel
itself, and per-feature NaN rates. Everything else — structural silence
features, staleness monotonicity, Differ-derived features — still depends
on the five-entity warehouse and Differ output, neither of which are
built, and is reported as an honest "not built" INFO below."""
from __future__ import annotations

from pharma_stats.audit.stubs import not_built
from pharma_stats.audit.types import Check, fail, info, ok
from pharma_stats.config import REPO_ROOT
from pharma_stats.finance import panel as money_panel

STAGE = "features"
LEAKAGE_REGISTER = REPO_ROOT / "audit" / "leakage.md"
REGISTERED_FEATURES = money_panel.FEATURE_NAMES


def run() -> list[Check]:
    checks: list[Check] = []
    checks += _leakage_register_check()

    panel = money_panel.build_money_layer_panel()
    if not panel:
        checks.append(info(
            STAGE, "money-layer feature panel (conviction_ratio, estimated_cumulative_spend)",
            expected="computed from financial_events",
            actual="0 rows — financial_events is empty or not materialized yet",
            detail="run scripts/build_financial_layer_cost_index.py first",
        ))
    else:
        checks += _money_layer_checks(panel)

    checks += not_built(
        STAGE,
        "The full program x month feature panel (structural silence features, staleness "
        "monotonicity, Differ-derived features) still depends on the five-entity warehouse and "
        "Differ output, neither of which are built. Only the money-layer slice "
        f"({', '.join(REGISTERED_FEATURES)}) is real so far — see checks above.",
    )
    return checks


def _leakage_register_check() -> list[Check]:
    if not LEAKAGE_REGISTER.exists():
        return [fail(
            STAGE, "audit/leakage.md exists",
            expected="present", actual="missing",
            detail="every time-varying feature must carry a knowability-date contract here "
                   "before it's trusted (CLAUDE.md)",
        )]
    text = LEAKAGE_REGISTER.read_text(encoding="utf-8")
    missing = [f for f in REGISTERED_FEATURES if f not in text]
    return [(fail if missing else ok)(
        STAGE, "every money-layer feature is registered in audit/leakage.md",
        expected=f"{len(REGISTERED_FEATURES)} registered",
        actual=f"{len(REGISTERED_FEATURES) - len(missing)} registered",
        detail=f"missing: {', '.join(missing)}" if missing else "",
    )]


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
        STAGE, "no feature row's knowability_date is later than its as_of date",
        expected="0 violations", actual=f"{len(violations)} violations",
        detail=", ".join(f"{v['program_id']}@{v['as_of']}" for v in violations[:5]),
    ))

    for feature in REGISTERED_FEATURES:
        n_present = sum(1 for r in panel if r.get(feature) is not None)
        nan_rate = 1 - (n_present / len(panel))
        checks.append(info(
            STAGE, f"{feature} NaN rate",
            expected="n/a", actual=f"{nan_rate:.1%} missing ({n_present}/{len(panel)} present)",
        ))
    return checks
