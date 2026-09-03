"""Tests for triage/promote.py — the one path Layer 2/3 verdicts can
reach gold through, and only when the validation gate passes."""
from __future__ import annotations

import pytest

from pharma_stats.triage import promote, staging, validation as val


def _staged(pid, is_adc, *, layer=2, status="pending", manual_overflow=False, rule=None):
    return {
        "event_id": f"e-{pid}", "timestamp": "2026-01-01T00:00:00Z", "run_id": "r1",
        "status": status, "program_id": pid, "proposed_name": pid, "is_adc": is_adc,
        "in_scope": None, "scope_reason": None, "layer": layer, "rule": rule,
        "model": "claude-sonnet-4-6", "prompt_version": "layer3-v2-terse",
        "from_recall": False, "quote": "x", "manual_overflow": manual_overflow,
        "manual_overflow_reason": None,
    }


def _passing_agreement():
    return {
        "is_adc": {"compared": 80, "agree": 78, "agreement_rate": 78 / 80},
        "in_scope": {"compared": 0, "agree": 0, "agreement_rate": None},
        "by_stratum": {}, "appendix": {"compared": 0, "agree": 0, "agreement_rate": None},
    }


def test_pending_committable_layer23_filters_correctly():
    staged = [
        _staged("p1", "no"),                                  # committable
        _staged("p2", "yes"),                                 # not committable (needs scope/gate3)
        _staged("p3", "no", layer=1),                          # Layer 1 — apply.py's job, not this
        _staged("p4", "no", status="accepted"),                # already accepted
        _staged("p5", "no", manual_overflow=True),              # overflow — never auto-decided
    ]
    result = promote.pending_committable_layer23(staged, gold_records=[])
    assert {r["program_id"] for r in result} == {"p1"}


def test_pending_committable_layer23_excludes_already_reviewed():
    staged = [_staged("p1", "no")]
    gold_records = [{"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
                      "timestamp": "2026-01-01", "is_repeat_probe": False}]
    result = promote.pending_committable_layer23(staged, gold_records)
    assert result == []


def test_accept_all_pending_refuses_when_gate_fails(monkeypatch):
    failing_agreement = {
        "is_adc": {"compared": 80, "agree": 60, "agreement_rate": 60 / 80},
        "in_scope": {"compared": 0, "agree": 0, "agreement_rate": None},
    }
    monkeypatch.setattr(val, "compute_agreement", lambda sample, gold_records: failing_agreement)
    with pytest.raises(promote.PromotionError):
        promote.accept_all_pending(sample=[{"program_id": "x"}], gold_records=[], staged_records=[])


def test_accept_all_pending_writes_gold_and_marks_staging_accepted(tmp_path, monkeypatch):
    from pharma_stats.labelling import store as gold_store
    from pharma_stats.triage import pool as tpool

    monkeypatch.setattr(val, "compute_agreement", lambda sample, gold_records: _passing_agreement())
    monkeypatch.setattr(gold_store, "LABELS_PATH", tmp_path / "labels.jsonl")
    monkeypatch.setattr(staging, "STAGING_PATH", tmp_path / "staged.jsonl")
    staged = [_staged("p1", "no", layer=3, rule="unsure_but_no")]
    result = promote.accept_all_pending(sample=[{"program_id": "x"}], gold_records=[], staged_records=staged)
    assert result["n_accepted"] == 1

    gold_records = gold_store.load_records()
    assert len(gold_records) == 1
    assert gold_records[0]["decided_by"] == "auto"
    assert gold_records[0]["is_adc"] == "no"
    assert gold_records[0]["gate_reached"] == 1
    assert gold_records[0]["triage_layer"] == 3

    latest = staging.latest_by_program(staging.load_records())
    assert latest["p1"]["status"] == "accepted"


def test_accept_all_pending_never_touches_is_adc_yes(tmp_path, monkeypatch):
    from pharma_stats.labelling import store as gold_store
    from pharma_stats.triage import pool as tpool

    monkeypatch.setattr(val, "compute_agreement", lambda sample, gold_records: _passing_agreement())
    monkeypatch.setattr(gold_store, "LABELS_PATH", tmp_path / "labels.jsonl")
    monkeypatch.setattr(staging, "STAGING_PATH", tmp_path / "staged.jsonl")
    staged = [_staged("p1", "yes", layer=2)]
    result = promote.accept_all_pending(sample=[], gold_records=[], staged_records=staged)
    assert result["n_accepted"] == 0
    assert not (tmp_path / "labels.jsonl").exists()


def test_reject_all_pending_writes_no_gold_and_marks_rejected(tmp_path, monkeypatch):
    from pharma_stats.triage import pool as tpool

    monkeypatch.setattr(staging, "STAGING_PATH", tmp_path / "staged.jsonl")
    staged = [_staged("p1", "no"), _staged("p2", "yes"), _staged("p3", "no", manual_overflow=True)]
    result = promote.reject_all_pending(staged_records=staged, reason="agreement too low")
    assert result["n_rejected"] == 2  # p3 (overflow) excluded

    staged_after = staging.load_records()
    latest = staging.latest_by_program(staged_after)
    assert latest["p1"]["status"] == "rejected"
    assert latest["p2"]["status"] == "rejected"
    assert "p3" not in {r["program_id"] for r in staged_after}
    assert not (tmp_path / "labels.jsonl").exists()
