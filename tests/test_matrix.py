"""Tests for attributes/matrix.py — B5 crowding/failure-density
classification. The quadrant logic, population rule, and the two
exclude-entirely (never bucket) rules on the payload/tumour-system axes
are the load-bearing parts; all get dedicated coverage since a mistake
here directly misreports the graveyard cells, the actual deliverable."""
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


def _program(pid, name, meshes_by_nct):
    return {
        "program_id": pid, "proposed_name": name, "synonyms": [],
        "trials": [{"nct_id": nct} for nct in meshes_by_nct],
        "history_coverage": "full", "band": 0,
    }


def _with_gold_and_mesh(monkeypatch, records, mesh_by_nct):
    from pharma_stats.labelling import store as gold_store
    from pharma_stats.attributes import tumour_system as tsys_mod

    monkeypatch.setattr(gold_store, "load_records", lambda: records)
    monkeypatch.setattr(
        tsys_mod, "_condition_browse_data",
        lambda nct_id: (mesh_by_nct.get(nct_id, []), []),
    )


def test_build_matrix_excludes_out_of_scope_and_groups_by_payload_and_system(monkeypatch):
    # p1: trastuzumab deruxtecan -> camptothecin_topo1, breast (D001943)
    # p2: sacituzumab govitecan -> camptothecin_topo1, gi_hepatobiliary (D015179, colorectal)
    # p3: excluded via in_scope=no
    programs = [
        _program("p1", "Trastuzumab deruxtecan", ["NCT001"]),
        _program("p2", "Sacituzumab govitecan", ["NCT002"]),
        _program("p3", "Excluded Co", ["NCT003"]),
    ]
    records = [
        _gold("p1", status="active"),
        _gold("p2", status="dead_confirmed", kill_reason="toxicity_safety"),
        _gold("p3", in_scope="no"),
    ]
    mesh_by_nct = {
        "NCT001": [{"id": "D001943", "term": "Breast Neoplasms"}],
        "NCT002": [{"id": "D015179", "term": "Colorectal Neoplasms"}],
        "NCT003": [{"id": "D001943", "term": "Breast Neoplasms"}],
    }
    _with_gold_and_mesh(monkeypatch, records, mesh_by_nct)

    cells, attrs, stats = mx.build_matrix(programs, min_n=1)

    assert "p3" not in attrs
    assert "p1" in attrs and "p2" in attrs
    key1 = ("camptothecin_topo1", "breast")
    key2 = ("camptothecin_topo1", "gi_hepatobiliary")
    assert cells[key1].n_live == 1
    assert cells[key2].n_dead == 1
    assert stats["n_in_population"] == 2


def test_build_matrix_excludes_undisclosed_payload_entirely(monkeypatch):
    # bare dev code -> no INN suffix -> "undisclosed" -> excluded, not bucketed
    programs = [_program("p1", "XMT-1592", ["NCT001"])]
    records = [_gold("p1", status="active")]
    mesh_by_nct = {"NCT001": [{"id": "D001943", "term": "Breast Neoplasms"}]}
    _with_gold_and_mesh(monkeypatch, records, mesh_by_nct)

    cells, attrs, stats = mx.build_matrix(programs, min_n=1)

    assert attrs == {}
    assert cells == {}
    assert stats["n_scope_confirmed"] == 1
    assert stats["n_excluded_payload_undisclosed"] == 1
    assert stats["n_excluded_system_unresolved"] == 0


def test_build_matrix_excludes_unresolved_tumour_system_entirely(monkeypatch):
    # only a generic root MeSH term on file -> no resolvable system -> excluded
    programs = [_program("p1", "Trastuzumab deruxtecan", ["NCT001"])]
    records = [_gold("p1", status="active")]
    mesh_by_nct = {"NCT001": [{"id": "D009369", "term": "Neoplasms"}]}
    _with_gold_and_mesh(monkeypatch, records, mesh_by_nct)

    cells, attrs, stats = mx.build_matrix(programs, min_n=1)

    assert attrs == {}
    assert cells == {}
    assert stats["n_excluded_payload_undisclosed"] == 0
    assert stats["n_excluded_system_unresolved"] == 1


def test_build_matrix_excludes_site_agnostic_histology_term(monkeypatch):
    # "Carcinoma" is real "solid" data but names no body site -- must not
    # become its own cell or fall into any of the 8 systems
    programs = [_program("p1", "Trastuzumab deruxtecan", ["NCT001"])]
    records = [_gold("p1", status="active")]
    mesh_by_nct = {"NCT001": [{"id": "D002277", "term": "Carcinoma"}]}
    _with_gold_and_mesh(monkeypatch, records, mesh_by_nct)

    cells, attrs, stats = mx.build_matrix(programs, min_n=1)

    assert attrs == {}
    assert stats["n_excluded_system_unresolved"] == 1
