"""Tests for attributes/target_external.py -- the ChEMBL + Open Targets
residual target-antigen resolution. No network calls: fake clients stand
in, since the whole point of this module is the VERIFICATION discipline
around fuzzy/networked lookups, which is testable without a real API."""
from __future__ import annotations

from pharma_stats.attributes import target_external as te


class FakeChembl:
    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = responses
        self.queries: list[str] = []

    def search_molecule(self, query: str) -> list[dict]:
        self.queries.append(query)
        return self.responses.get(query, [])


class FakeOpenTargets:
    def __init__(self, mechanisms_by_id: dict[str, list[dict]]):
        self.mechanisms_by_id = mechanisms_by_id

    def drug_mechanisms(self, chembl_id: str) -> list[dict]:
        return self.mechanisms_by_id.get(chembl_id, [])


def _binding_row(symbol: str, moa: str = "binding agent") -> dict:
    return {"mechanismOfAction": moa, "actionType": "BINDING AGENT", "targets": [{"approvedSymbol": symbol}]}


def _inhibitor_row(symbols: list[str]) -> dict:
    return {"mechanismOfAction": "tubulin inhibitor", "actionType": "INHIBITOR",
            "targets": [{"approvedSymbol": s} for s in symbols]}


def test_resolves_via_exact_pref_name_match_and_binding_agent_mechanism():
    chembl = FakeChembl({"XYZ-001": [{"molecule_chembl_id": "CHEMBL1", "pref_name": "XYZ-001",
                                        "molecule_synonyms": []}]})
    ot = FakeOpenTargets({"CHEMBL1": [_binding_row("CEACAM5"), _inhibitor_row(["TUBB", "TUBA1A"])]})
    hit = te.resolve_target_external("XYZ-001", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is not None
    assert hit.target == "CEACAM5"
    assert hit.chembl_id == "CHEMBL1"
    assert hit.matched_via == "pref_name"


def test_resolves_via_synonym_when_pref_name_differs():
    chembl = FakeChembl({"MyDevCode": [{"molecule_chembl_id": "CHEMBL2", "pref_name": "REAL INN NAME",
                                          "molecule_synonyms": [{"molecule_synonym": "MyDevCode"}]}]})
    ot = FakeOpenTargets({"CHEMBL2": [_binding_row("FOLR1")]})
    hit = te.resolve_target_external("MyDevCode", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is not None
    assert hit.target == "FOLR1"
    assert hit.matched_via == "molecule_synonyms"


def test_never_trusts_fuzzy_rank_one_without_exact_match():
    """Real bug this guards against: ChEMBL's search for "TAK-188" (a
    genuine unrelated ADC dev code) returns 20 fuzzy hits whose first
    result, verified live 2026-09-07, is TOVORAFENIB -- a completely
    different, unrelated compound. Rank alone must never be trusted."""
    chembl = FakeChembl({"TAK-188": [
        {"molecule_chembl_id": "CHEMBL_WRONG", "pref_name": "TOVORAFENIB", "molecule_synonyms": []},
        {"molecule_chembl_id": "CHEMBL_RIGHT", "pref_name": "TAK-188", "molecule_synonyms": []},
    ]})
    ot = FakeOpenTargets({"CHEMBL_RIGHT": [_binding_row("MET")]})
    hit = te.resolve_target_external("TAK-188", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is not None
    assert hit.chembl_id == "CHEMBL_RIGHT"
    assert hit.target == "MET"


def test_no_chembl_hit_at_all_returns_none():
    chembl = FakeChembl({})
    ot = FakeOpenTargets({})
    hit = te.resolve_target_external("SomeObscureDevCode", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is None


def test_chembl_hit_but_no_binding_agent_mechanism_on_open_targets_returns_none():
    chembl = FakeChembl({"XYZ-002": [{"molecule_chembl_id": "CHEMBL3", "pref_name": "XYZ-002",
                                        "molecule_synonyms": []}]})
    ot = FakeOpenTargets({"CHEMBL3": [_inhibitor_row(["TUBB"])]})  # only the payload mechanism on file
    hit = te.resolve_target_external("XYZ-002", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is None


def test_multiple_binding_agent_rows_is_ambiguous_not_guessed():
    """A real bispecific ADC (two distinct binding-agent mechanisms) must
    stay unresolved -- same "don't guess on ambiguity" policy as
    target.py's own trial-text extraction."""
    chembl = FakeChembl({"BISP-001": [{"molecule_chembl_id": "CHEMBL4", "pref_name": "BISP-001",
                                         "molecule_synonyms": []}]})
    ot = FakeOpenTargets({"CHEMBL4": [_binding_row("ERBB3"), _binding_row("TACSTD2")]})
    hit = te.resolve_target_external("BISP-001", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is None


def test_multiple_targets_on_one_binding_agent_row_is_ambiguous():
    chembl = FakeChembl({"XYZ-003": [{"molecule_chembl_id": "CHEMBL5", "pref_name": "XYZ-003",
                                        "molecule_synonyms": []}]})
    ot = FakeOpenTargets({"CHEMBL5": [_binding_row("ERBB2")]})
    ot.mechanisms_by_id["CHEMBL5"][0]["targets"].append({"approvedSymbol": "ERBB3"})
    hit = te.resolve_target_external("XYZ-003", [], chembl_client=chembl, opentargets_client=ot)
    assert hit is None
