"""Tests for the fourth (stratum_gap) queue: generalized queue.py
support, and scripts/serve_stratum_gap_queue.py's selection logic."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pharma_stats.labelling import queue as q

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_stratum_gap_queue.py"


@pytest.fixture()
def script():
    spec = importlib.util.spec_from_file_location("serve_stratum_gap_queue", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["serve_stratum_gap_queue"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("serve_stratum_gap_queue", None)


def test_new_session_has_stratum_gap_order():
    session = q.new_session([], exclude_ids=set())
    assert session["stratum_gap_order"] == []


def test_switch_queue_supports_stratum_gap():
    session = q.new_session([], exclude_ids=set())
    q.switch_queue(session, "stratum_gap")
    assert session["active_queue"] == "stratum_gap"


def test_pop_next_uses_stratum_gap_order_when_active():
    session = q.new_session([], exclude_ids=set())
    session["stratum_gap_order"] = ["a1", "a2"]
    q.switch_queue(session, "stratum_gap")
    pid, is_repeat = q.pop_next(session, labelled_ids=set())
    assert pid == "a1"
    assert is_repeat is False


def test_stratum_gap_not_in_bypass_reviewed_queues():
    # unlike auto_review, these are genuinely never-labelled programs --
    # the normal gate 1/2/3 flow applies, no already-reviewed bypass needed.
    assert "stratum_gap" not in q.BYPASS_REVIEWED_QUEUES


def _program(pid, band, archetype):
    return {"program_id": pid, "band": band, "primary_archetype": archetype}


def _gold(pid, gate_reached=3, status="active", ts="2026-01-01T00:00:00Z"):
    return {"action": "label", "program_id": pid, "gate_reached": gate_reached,
            "is_adc": "yes", "in_scope": "yes", "status": status,
            "timestamp": ts, "is_repeat_probe": False}


def test_stratum_gap_ids_selects_only_target_strata(script):
    programs = [
        _program("p1", 1, "registry_terminated_stated_reason"),  # target cell A
        _program("p2", 4, "registry_terminated_vague_reason"),  # target cell B
        _program("p3", 1, "other"),  # wrong archetype
        _program("p4", 2, "registry_terminated_stated_reason"),  # wrong band
    ]
    ids = script.stratum_gap_ids(programs, gold_records=[])
    assert ids == ["p1", "p2"]


def test_stratum_gap_ids_excludes_already_labelled(script):
    programs = [
        _program("p1", 1, "registry_terminated_stated_reason"),
        _program("p2", 1, "registry_terminated_stated_reason"),
    ]
    records = [_gold("p1")]  # already gate-3 labelled
    ids = script.stratum_gap_ids(programs, gold_records=records)
    assert ids == ["p2"]


def test_stratum_gap_ids_matches_reported_counts(script):
    """Regression pin: as of 2026-09-04, label_statistics reports exactly
    19 + 2 = 21 unlabelled programs across these two cells."""
    from pharma_stats.labelling import provisional_programs as pp
    from pharma_stats.labelling import store

    programs = pp.load_materialized()
    if not programs:
        pytest.skip("no materialized provisional_programs table on this machine")
    ids = script.stratum_gap_ids(programs, store.load_records())
    assert len(ids) == 21
