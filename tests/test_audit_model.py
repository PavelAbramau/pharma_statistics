"""Tests for audit/model.py — the stage that enforces the "model must
beat the silence-score heuristic or FAIL" gate, reading the published
backtest result rather than re-running the (expensive) backtest itself."""
from __future__ import annotations

import json

from pharma_stats.audit import model


def test_model_stage_reports_not_run_when_no_result_file(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "MODEL_RESULT_PATH", tmp_path / "model_backtest_result.json")
    checks = model.run()
    assert len(checks) == 1
    assert checks[0].level == "INFO"
    assert "NOT RUN" in checks[0].actual


def test_model_stage_passes_when_gate_passed(tmp_path, monkeypatch):
    result_path = tmp_path / "model_backtest_result.json"
    result_path.write_text(json.dumps({
        "gate_passed": True, "gate_reason": "model beats heuristic",
        "training_event_counts": {"dead": 63, "approved": 15, "superseded": 3},
        "min_precision": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(model, "MODEL_RESULT_PATH", result_path)
    checks = model.run()
    gate_check = checks[-1]
    assert gate_check.level == "PASS"
    assert "model beats heuristic" in gate_check.detail


def test_model_stage_fails_loudly_when_gate_failed(tmp_path, monkeypatch):
    result_path = tmp_path / "model_backtest_result.json"
    result_path.write_text(json.dumps({
        "gate_passed": False, "gate_reason": "heuristic beats model",
        "training_event_counts": {"dead": 63, "approved": 15, "superseded": 3},
        "min_precision": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(model, "MODEL_RESULT_PATH", result_path)
    checks = model.run()
    gate_check = checks[-1]
    assert gate_check.level == "FAIL"
    assert "heuristic beats model" in gate_check.detail


def test_model_stage_reports_event_counts_regardless_of_gate_outcome(tmp_path, monkeypatch):
    result_path = tmp_path / "model_backtest_result.json"
    result_path.write_text(json.dumps({
        "gate_passed": False, "gate_reason": "x",
        "training_event_counts": {"dead": 63, "approved": 15, "superseded": 3},
        "min_precision": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(model, "MODEL_RESULT_PATH", result_path)
    checks = model.run()
    counts_check = checks[0]
    assert "dead=63" in counts_check.actual
    assert "approved=15" in counts_check.actual
    assert "superseded=3" in counts_check.actual
