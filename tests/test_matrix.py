"""Tests for attributes/matrix.py — B5 crowding/failure-density
classification. The quadrant logic and population rule are the load-
bearing parts; both get dedicated coverage since a mistake here directly
misreports the graveyard quadrant, the actual deliverable."""
from __future__ import annotations

from pharma_stats.attributes import matrix as mx


def _gold(pid, is_adc="yes", in_scope="yes", status=None, kill_reason=None):
    return {"action": "label", "program_id": pid, "gate_reached": 3 if status else 2,
            "is_adc": is_adc, "in_scope": in_scope, "status": status, "kill_reason": kill_reason,
            "timestamp": "2026-01-01T00:00:00Z", "is_repeat_probe": False}


def test_program_live_dead_status_none_when_not_in_gold():
    assert mx.program_live_dead_status({"program_id": "p1"}, gold_latest={}) is None


def test_program_live_dead_status_none_when_gold_says_out_of_scope():
    gold_latest = {"p1": _gold("p1", in_scope="no")}
    assert mx.program_live_dead_status({"program_id": "p1"}, gold_latest) is None


def test_program_live_dead_status_dead_confirmed_from_gold():
    gold_latest = {"p1": _gold("p1", status="dead_confirmed", kill_reason="futility_efficacy")}
    result = mx.program_live_dead_status({"program_id": "p1"}, gold_latest)
    assert result.is_dead is True
    assert result.basis == "gold"
    assert result.kill_reason == "futility_efficacy"


def test_program_live_dead_status_superseded_counts_as_dead():
    gold_latest = {"p1": _gold("p1", status="superseded")}
    result = mx.program_live_dead_status({"program_id": "p1"}, gold_latest)
    assert result.is_dead is True


def test_program_live_dead_status_active_from_gold_is_live():
    gold_latest = {"p1": _gold("p1", status="active")}
    result = mx.program_live_dead_status({"program_id": "p1"}, gold_latest)
    assert result.is_dead is False
    assert result.basis == "gold"


def test_program_live_dead_status_unknown_status_falls_to_silence_proxy():
    gold_latest = {"p1": _gold("p1", status="unknown")}
    program = {"program_id": "p1", "history_coverage": "full", "band": 4}
    result = mx.program_live_dead_status(program, gold_latest)
    assert result.is_dead is True
    assert result.basis == "silence_proxy"


def test_program_live_dead_status_incomplete_coverage_never_dead_by_proxy():
    gold_latest = {"p1": _gold("p1", status="unknown")}
    program = {"program_id": "p1", "history_coverage": "partial", "band": 4}
    result = mx.program_live_dead_status(program, gold_latest)
    assert result.is_dead is False
    assert result.basis == "assumed_live"


def test_program_live_dead_status_low_band_is_assumed_live():
    gold_latest = {"p1": _gold("p1", status="unknown")}
    program = {"program_id": "p1", "history_coverage": "full", "band": 1}
    result = mx.program_live_dead_status(program, gold_latest)
    assert result.is_dead is False


def test_classify_quadrant_untested_white_space():
    assert mx.classify_quadrant(0, 0, min_n=5) == "untested_white_space"


def test_classify_quadrant_insufficient_evidence_below_min_n():
    assert mx.classify_quadrant(2, 1, min_n=5) == "insufficient_evidence"
    assert mx.classify_quadrant(0, 3, min_n=5) == "insufficient_evidence"


def test_classify_quadrant_red_ocean():
    assert mx.classify_quadrant(6, 0, min_n=5) == "red_ocean"


def test_classify_quadrant_graveyard():
    assert mx.classify_quadrant(0, 6, min_n=5) == "graveyard"
    assert mx.classify_quadrant(1, 6, min_n=5) == "graveyard"


def test_classify_quadrant_contested_and_hard_both_many():
    assert mx.classify_quadrant(6, 6, min_n=5) == "contested_and_hard"


def test_classify_quadrant_contested_and_hard_mixed_neither_dominant():
    # total (7) clears min_n but neither side alone does -- falls to
    # contested_and_hard as the closest-fitting of the four quadrants
    assert mx.classify_quadrant(4, 3, min_n=5) == "contested_and_hard"


def test_build_matrix_excludes_out_of_scope_and_groups_by_target_indication():
    programs = [
        {"program_id": "p1", "proposed_name": "Trastuzumab deruxtecan", "synonyms": [],
         "trials": [], "history_coverage": "full", "band": 0},
        {"program_id": "p2", "proposed_name": "Sacituzumab govitecan", "synonyms": [],
         "trials": [], "history_coverage": "full", "band": 0},
        {"program_id": "p3", "proposed_name": "Excluded Co", "synonyms": [],
         "trials": [], "history_coverage": "full", "band": 0},
    ]
    import pharma_stats.attributes.matrix as mx_mod
    from pharma_stats.labelling import store as gold_store

    records = [
        _gold("p1", status="active"),
        _gold("p2", status="dead_confirmed", kill_reason="toxicity_safety"),
        _gold("p3", in_scope="no"),  # excluded from population
    ]
    orig_load = gold_store.load_records
    gold_store.load_records = lambda: records
    try:
        cells, attrs = mx_mod.build_matrix(programs, min_n=1)
    finally:
        gold_store.load_records = orig_load

    assert "p3" not in attrs
    assert "p1" in attrs and "p2" in attrs
    # both cluster under ERBB2/TACSTD2 respectively (antibody-stem dictionary), unknown indication (no MeSH data)
    key1 = ("ERBB2", "unknown")
    key2 = ("TACSTD2", "unknown")
    assert cells[key1].n_live == 1
    assert cells[key2].n_dead == 1
