"""Tests for attributes/tumour_system.py — the opportunity matrix's
coarse tumour-system axis. system_for's None-vs-group distinction is the
load-bearing behaviour: None must mean "exclude," and only that, never a
guessed or catch-all group (see attributes/matrix.py, docs/decisions/0005)."""
from __future__ import annotations

from pharma_stats.attributes import tumour_system as tsys


def test_system_for_known_solid_id_resolves():
    assert tsys.system_for("D001943") == "breast"  # Breast Neoplasms


def test_system_for_generic_basket_id_is_none():
    assert tsys.system_for("D009369") is None  # Neoplasms (root umbrella term)


def test_system_for_site_agnostic_histology_is_none():
    # real "solid" MeSH data, but names no body site -- must not resolve
    assert tsys.system_for("D002277") is None  # Carcinoma
    assert tsys.system_for("D000230") is None  # Adenocarcinoma


def test_system_for_heme_id_is_none():
    assert tsys.system_for("D007938") is None  # Leukemia


def test_system_for_unknown_id_is_none():
    assert tsys.system_for("D999999") is None


def test_system_for_none_id_is_none():
    assert tsys.system_for(None) is None


def test_program_tumour_system_none_with_no_trials():
    assert tsys.program_tumour_system({"trials": []}) is None


def test_program_tumour_system_most_common_wins(monkeypatch):
    mesh_by_nct = {
        "NCT1": [{"id": "D001943"}],  # breast
        "NCT2": [{"id": "D001943"}],  # breast
        "NCT3": [{"id": "D015179"}],  # gi_hepatobiliary
    }
    monkeypatch.setattr(tsys, "_condition_browse_data", lambda nct_id: (mesh_by_nct.get(nct_id, []), []))
    program = {"trials": [{"nct_id": n} for n in mesh_by_nct]}
    assert tsys.program_tumour_system(program) == "breast"


def test_program_tumour_system_none_when_only_generic_terms(monkeypatch):
    mesh_by_nct = {"NCT1": [{"id": "D009369"}]}  # Neoplasms
    monkeypatch.setattr(tsys, "_condition_browse_data", lambda nct_id: (mesh_by_nct.get(nct_id, []), []))
    program = {"trials": [{"nct_id": "NCT1"}]}
    assert tsys.program_tumour_system(program) is None
