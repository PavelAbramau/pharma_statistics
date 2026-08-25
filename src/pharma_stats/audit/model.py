"""Model stage: calibration reliability curve, concordance index, and
the check that matters most — the model must beat the silence-score
heuristic on lead time at matched precision, or this FAILs loudly rather
than reporting a respectable-looking AUC. None of this is computable
until a model exists."""
from __future__ import annotations

from pharma_stats.audit.stubs import not_built
from pharma_stats.audit.types import Check

STAGE = "model"


def run() -> list[Check]:
    return not_built(
        STAGE,
        "No model has been trained yet (README.md: 'Program-status detection / silent-kill "
        "backtest — not started'). Once one exists, wire up: a calibration reliability curve, "
        "concordance index, and — the check that matters more than either — a head-to-head "
        "comparison against the provisional_programs.py silence-score heuristic on lead time at "
        "matched precision. If the model does not beat ten lines of arithmetic, FAIL loudly.",
    )
