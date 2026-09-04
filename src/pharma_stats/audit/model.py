"""Model stage: reads scripts/run_model_backtest.py's published result
(data/model_backtest_result.json) and enforces the one check that
matters — the discrete-time competing-risks model must beat the
silence-score heuristic on lead time at matched precision, or this FAILs
loudly, per the user's own gate. Does not re-run the (expensive, ~30min)
backtest itself — see run_model_backtest.py for that.
"""
from __future__ import annotations

import json

from pharma_stats.audit.types import Check, fail, info, ok
from pharma_stats.config import DATA_DIR

STAGE = "model"
MODEL_RESULT_PATH = DATA_DIR / "model_backtest_result.json"


def run() -> list[Check]:
    if not MODEL_RESULT_PATH.exists():
        return [info(
            STAGE, "model beats the silence-score heuristic at matched precision (the gate)",
            expected="scripts/run_model_backtest.py has been run at least once",
            actual="NOT RUN — no data/model_backtest_result.json on disk yet",
            detail="run `python scripts/run_model_backtest.py` before this stage can check anything.",
        )]

    try:
        result = json.loads(MODEL_RESULT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [fail(
            STAGE, "model beats the silence-score heuristic at matched precision (the gate)",
            expected="a readable JSON result", actual=f"unreadable: {e}",
        )]

    counts = result.get("training_event_counts", {})
    checks = [info(
        STAGE, "training event counts (dead is the only outcome with a real ground-truth date "
               "and enough events — see docs/decisions/0005-0007)",
        expected="dead has enough events to be meaningful; approved/superseded reported honestly regardless",
        actual=f"dead={counts.get('dead', 0)}, approved={counts.get('approved', 0)}, "
               f"superseded={counts.get('superseded', 0)}",
    )]

    level = ok if result.get("gate_passed") else fail
    checks.append(level(
        STAGE, "model beats the silence-score heuristic at matched precision (the gate)",
        expected=f"model's best median lead time (dead only) beats the heuristic's, "
                 f"both at >= {result.get('min_precision', 0.5):.0%} precision",
        actual="PASS" if result.get("gate_passed") else "FAIL",
        detail=result.get("gate_reason", ""),
    ))
    return checks
