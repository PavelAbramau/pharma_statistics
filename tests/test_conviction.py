"""Tests for finance/conviction.py — the conviction ratio compares a
program's spend to its phase/indication peers, and must refuse to
produce a ratio when there's no honest peer denominator."""
from __future__ import annotations

from pharma_stats.finance import conviction as cv


def _program(pid, phase, scope_category="solid"):
    return {"program_id": pid, "scope_category": scope_category, "trials": [{"phases": [phase]}]}


def test_conviction_ratio_basic():
    assert cv.conviction_ratio(200.0, [100.0, 100.0, 100.0]) == 2.0


def test_conviction_ratio_none_with_fewer_than_two_peers():
    assert cv.conviction_ratio(200.0, [100.0]) is None
    assert cv.conviction_ratio(200.0, []) is None


def test_conviction_ratio_none_when_peer_median_is_zero():
    assert cv.conviction_ratio(200.0, [0.0, 0.0]) is None


def test_conviction_ratio_ignores_zero_peers_in_median():
    # two real peers at 100, one zero (no spend data) -- median of the
    # usable peers should still be 100, not pulled toward 0.
    assert cv.conviction_ratio(150.0, [100.0, 100.0, 0.0]) == 1.5


def test_peer_group_key_uses_highest_phase_and_scope_category():
    p = {"scope_category": "solid", "trials": [{"phases": ["PHASE1"]}, {"phases": ["PHASE2"]}]}
    assert cv.peer_group_key(p) == ("PHASE2", "solid")


def test_peer_group_key_defaults_scope_category_to_unknown():
    p = {"trials": [{"phases": ["PHASE1"]}]}
    assert cv.peer_group_key(p) == ("PHASE1", "unknown")


def test_compute_conviction_ratios_groups_by_peer_key():
    programs = [
        _program("a", "PHASE2"), _program("b", "PHASE2"), _program("c", "PHASE2"),
        _program("d", "PHASE1"),  # different peer group -- must not mix with the PHASE2 group
    ]
    spend = {"a": 300.0, "b": 100.0, "c": 100.0, "d": 50.0}
    result = cv.compute_conviction_ratios(programs, spend)

    assert result["a"]["peer_group"] == ("PHASE2", "solid")
    assert result["a"]["n_peers"] == 2  # b, c
    assert result["a"]["peer_median"] == 100.0
    assert result["a"]["conviction_ratio"] == 3.0

    assert result["d"]["n_peers"] == 0
    assert result["d"]["conviction_ratio"] is None  # no peers at all in its group


def test_compute_conviction_ratios_skips_programs_without_spend_data():
    programs = [_program("a", "PHASE2"), _program("b", "PHASE2")]
    spend = {"a": 100.0}  # b has no spend estimate at all
    result = cv.compute_conviction_ratios(programs, spend)
    assert "b" not in result
    assert result["a"]["n_peers"] == 0
