"""Tests for scripts/serve_stable_pool.py's pool-selection logic — must
exclude anything a still-running background pipeline or an unjudged
validation gate could still overturn."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_stable_pool.py"


@pytest.fixture()
def script():
    spec = importlib.util.spec_from_file_location("serve_stable_pool", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["serve_stable_pool"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("serve_stable_pool", None)


def _program(pid):
    return {"program_id": pid}


def _staged(pid, is_adc=None, manual_overflow=False):
    return {"program_id": pid, "is_adc": is_adc, "manual_overflow": manual_overflow,
            "status": "pending", "layer": 3, "timestamp": "2026-01-01T00:00:00Z"}


def test_no_staged_record_is_pool_a(script):
    programs = [_program("p1")]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=[], exclude_ids=set())
    assert ids == ["p1"]
    assert counts == {"pool_a_no_opinion": 1, "pool_b_awaiting_gate": 0}


def test_manual_overflow_flag_with_no_verdict_is_pool_a(script):
    programs = [_program("p1")]
    staged = [_staged("p1", is_adc=None, manual_overflow=True)]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=staged, exclude_ids=set())
    assert ids == ["p1"]
    assert counts["pool_a_no_opinion"] == 1


def test_staged_is_adc_yes_is_pool_b(script):
    programs = [_program("p1")]
    staged = [_staged("p1", is_adc="yes")]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=staged, exclude_ids=set())
    assert ids == ["p1"]
    assert counts == {"pool_a_no_opinion": 0, "pool_b_awaiting_gate": 1}


def test_staged_is_adc_no_is_excluded(script):
    programs = [_program("p1")]
    staged = [_staged("p1", is_adc="no")]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=staged, exclude_ids=set())
    assert ids == []


def test_staged_is_adc_unsure_is_excluded(script):
    programs = [_program("p1")]
    staged = [_staged("p1", is_adc="unsure")]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=staged, exclude_ids=set())
    assert ids == []


def test_reviewed_program_excluded(script):
    from pharma_stats.labelling import store
    programs = [_program("p1")]
    gold = [store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no"},
        session_id="s1", served_stratum={},
    )]
    ids, counts = script.stable_pool_ids(programs, gold_records=gold, staged_records=[], exclude_ids=set())
    assert ids == []


def test_validation_holdout_excluded_even_with_no_opinion(script):
    programs = [_program("p1")]
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=[], exclude_ids={"p1"})
    assert ids == []


def test_pool_a_before_pool_b_in_order(script):
    programs = [_program("b1"), _program("a1")]
    staged = [_staged("b1", is_adc="yes")]  # a1 has no staged record at all
    ids, counts = script.stable_pool_ids(programs, gold_records=[], staged_records=staged, exclude_ids=set())
    assert ids == ["a1", "b1"]
