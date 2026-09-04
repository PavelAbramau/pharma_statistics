"""Tests for the third (auto_review) queue: generalized queue.py support,
and scripts/serve_auto_gate2_rejects.py's selection logic."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pharma_stats.labelling import queue as q

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_auto_gate2_rejects.py"


@pytest.fixture()
def script():
    spec = importlib.util.spec_from_file_location("serve_auto_gate2_rejects", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["serve_auto_gate2_rejects"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("serve_auto_gate2_rejects", None)


def test_new_session_has_auto_review_order():
    session = q.new_session([], exclude_ids=set())
    assert session["auto_review_order"] == []


def test_switch_queue_supports_auto_review():
    session = q.new_session([], exclude_ids=set())
    q.switch_queue(session, "auto_review")
    assert session["active_queue"] == "auto_review"


def test_switch_queue_still_rejects_unknown_name():
    session = q.new_session([], exclude_ids=set())
    with pytest.raises(ValueError):
        q.switch_queue(session, "bogus")


def test_pop_next_uses_auto_review_order_when_active():
    session = q.new_session([], exclude_ids=set())
    session["auto_review_order"] = ["a1", "a2"]
    q.switch_queue(session, "auto_review")
    pid, is_repeat = q.pop_next(session, labelled_ids=set())
    assert pid == "a1"
    assert is_repeat is False
    assert session["auto_review_order"] == ["a2"]


def test_repeat_probe_never_fires_on_auto_review_queue():
    session = q.new_session([], exclude_ids=set())
    session["auto_review_order"] = [f"a{i}" for i in range(30)]
    q.switch_queue(session, "auto_review")
    session["total_served"] = 9  # next call would be a repeat probe on main (multiple of 10)
    pid, is_repeat = q.pop_next(session, labelled_ids={"x", "y"})
    assert is_repeat is False
    assert pid == "a0"


def test_bypass_reviewed_queues_contains_auto_review():
    assert "auto_review" in q.BYPASS_REVIEWED_QUEUES
    assert "main" not in q.BYPASS_REVIEWED_QUEUES
    assert "validation" not in q.BYPASS_REVIEWED_QUEUES


def _gold(pid, decided_by="auto", gate_reached=2, is_adc="yes", in_scope="no", ts="2026-01-01T00:00:00Z"):
    return {"action": "label", "program_id": pid, "decided_by": decided_by, "gate_reached": gate_reached,
            "is_adc": is_adc, "in_scope": in_scope, "timestamp": ts, "is_repeat_probe": False}


def test_auto_gate2_reject_ids_selects_correctly(script):
    records = [
        _gold("p1"),  # auto, gate 2 -- included
        _gold("p2", decided_by="human"),  # human -- excluded
        _gold("p3", gate_reached=1),  # auto, but gate 1 -- excluded
        _gold("p4", gate_reached=3, is_adc="yes", in_scope="yes"),  # auto gate 3 shouldn't exist, but excluded anyway
    ]
    ids = script.auto_gate2_reject_ids(records)
    assert ids == ["p1"]


def test_auto_gate2_reject_ids_uses_latest_only():
    from pharma_stats.labelling import store
    records = [
        _gold("p1", ts="2026-01-01T00:00:00Z"),  # original auto gate-2 reject
        _gold("p1", decided_by="human", gate_reached=3, is_adc="yes", in_scope="yes", ts="2026-02-01T00:00:00Z"),
    ]
    spec = importlib.util.spec_from_file_location("serve_auto_gate2_rejects", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = module.auto_gate2_reject_ids(records)
    assert ids == []  # already reopened and re-decided by a human -- drops out
