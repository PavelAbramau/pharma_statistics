"""Tests for triage/layer1_5.py and clients/chembl.py — the ChEMBL HTTP
session is mocked; no test here makes a real network call."""
from __future__ import annotations

import pytest

from pharma_stats.clients.chembl import ChemblClient, ChemblError
from pharma_stats.triage import layer1_5


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = responses  # {query: _FakeResponse}
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        query = params.get("q")
        self.calls.append(query)
        return self.responses.get(query, _FakeResponse(200, {"molecules": []}))


def _molecule(molecule_type, pref_name=None, synonyms=None, chembl_id="CHEMBL1"):
    return {
        "molecule_type": molecule_type, "pref_name": pref_name, "molecule_chembl_id": chembl_id,
        "molecule_synonyms": [{"molecule_synonym": s, "syn_type": "RESEARCH_CODE"} for s in (synonyms or [])],
    }


def test_search_molecule_raises_on_non_200():
    session = _FakeSession({"x": _FakeResponse(500, text="server error")})
    client = ChemblClient(session=session, min_interval=0.0)
    with pytest.raises(ChemblError):
        client.search_molecule("x")


def test_search_molecule_returns_molecules_list():
    session = _FakeSession({"x": _FakeResponse(200, {"molecules": [_molecule("Antibody drug conjugate")]})})
    client = ChemblClient(session=session, min_interval=0.0)
    molecules = client.search_molecule("x")
    assert len(molecules) == 1
    assert molecules[0]["molecule_type"] == "Antibody drug conjugate"


def test_chembl_lookup_requires_exact_match_not_just_top_result():
    # ChEMBL's search is fuzzy — a returned molecule that does NOT
    # actually match our name/synonym must never be trusted
    session = _FakeSession({
        "Foo-123": _FakeResponse(200, {"molecules": [_molecule("Antibody drug conjugate", pref_name="Something Else")]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    hit = layer1_5.chembl_lookup("Foo-123", [], client=client)
    assert hit is None


def test_chembl_lookup_matches_on_pref_name():
    session = _FakeSession({
        "Trastuzumab Deruxtecan": _FakeResponse(200, {"molecules": [
            _molecule("Antibody drug conjugate", pref_name="TRASTUZUMAB DERUXTECAN", chembl_id="CHEMBL123"),
        ]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    hit = layer1_5.chembl_lookup("Trastuzumab Deruxtecan", [], client=client)
    assert hit is not None
    assert hit.value == "Antibody drug conjugate"
    assert hit.matched_via == "pref_name"
    assert hit.record_id == "CHEMBL123"


def test_chembl_lookup_matches_on_synonym_and_tries_multiple_queries():
    session = _FakeSession({
        "XL114": _FakeResponse(200, {"molecules": []}),  # first query: no hit
        "Some Other Name": _FakeResponse(200, {"molecules": [
            _molecule("Small molecule", pref_name="SOME OTHER NAME", synonyms=["XL114"]),
        ]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    hit = layer1_5.chembl_lookup("XL114", ["Some Other Name"], client=client)
    assert hit is not None
    assert hit.value == "Small molecule"
    assert hit.matched_via == "pref_name"


def test_chembl_lookup_ignores_unrecognized_molecule_type():
    session = _FakeSession({
        "Foo": _FakeResponse(200, {"molecules": [_molecule("Oligonucleotide", pref_name="FOO")]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    hit = layer1_5.chembl_lookup("Foo", [], client=client)
    assert hit is None  # not in CONFIDENT_MOLECULE_TYPE_MAP — never guessed


def test_chembl_lookup_treats_network_error_as_no_hit():
    class _RaisingSession(_FakeSession):
        def get(self, url, params=None, timeout=None):
            raise ChemblError("boom")

    client = ChemblClient(session=_RaisingSession({}), min_interval=0.0)
    hit = layer1_5.chembl_lookup("anything", [], client=client)
    assert hit is None


def test_evaluate_layer1_5_no_hit_is_committable():
    session = _FakeSession({
        "Foo": _FakeResponse(200, {"molecules": [_molecule("Small molecule", pref_name="FOO", chembl_id="CHEMBL9")]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    program = {"proposed_name": "Foo", "synonyms": []}
    result, hit = layer1_5.evaluate_layer1_5(program, client=client)
    assert result.is_adc == "no"
    assert result.committable is True
    assert hit.source == "chembl"
    assert "layer1_5_chembl:Small molecule:CHEMBL9" == result.rule


def test_evaluate_layer1_5_yes_hit_is_not_committable_alone():
    session = _FakeSession({
        "Bar": _FakeResponse(200, {"molecules": [_molecule("Antibody drug conjugate", pref_name="BAR")]}),
    })
    client = ChemblClient(session=session, min_interval=0.0)
    program = {"proposed_name": "Bar", "synonyms": []}
    result, hit = layer1_5.evaluate_layer1_5(program, client=client)
    assert result.is_adc == "yes"
    assert result.committable is False  # still needs a scope call, same as any is_adc=yes rule


def test_evaluate_layer1_5_returns_none_when_nothing_matches():
    session = _FakeSession({})
    client = ChemblClient(session=session, min_interval=0.0)
    program = {"proposed_name": "Totally Unknown Compound Zzz", "synonyms": []}
    result, hit = layer1_5.evaluate_layer1_5(program, client=client)
    assert result is None and hit is None


def test_drugbank_lookup_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        layer1_5.drugbank_lookup("anything", [])
