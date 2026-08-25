"""Features stage: every feature registered with a knowability date in
audit/leakage.md, a real as-of-date leakage probe (not just the register
check), NaN rates, and staleness-feature monotonicity. None of this is
computable until a feature panel exists."""
from __future__ import annotations

from pharma_stats.audit.stubs import not_built
from pharma_stats.audit.types import Check
from pharma_stats.config import REPO_ROOT

STAGE = "features"
LEAKAGE_REGISTER = REPO_ROOT / "audit" / "leakage.md"


def run() -> list[Check]:
    checks = not_built(
        STAGE,
        "No feature panel exists yet — it depends on the five-entity warehouse and Differ output, "
        "neither of which are built. Once features exist, wire up: 100% carry a knowability date "
        "in audit/leakage.md (unregistered = FAIL), an as-of-date recompute that asserts no value "
        "depends on a snapshot posted after the cut date, per-feature NaN rates, and a "
        "monotonicity sanity check that staleness features increase with time for a program with "
        "no new activity.",
    )
    if not LEAKAGE_REGISTER.exists():
        checks[0].detail += " audit/leakage.md does not exist yet either."
    return checks
