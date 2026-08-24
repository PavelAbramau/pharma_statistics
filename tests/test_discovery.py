"""Unit tests for ADC candidate discovery: patterns and clustering.

No network. Mentions and studies are constructed in-memory.
"""
from datetime import date

from pharma_stats.discovery.candidates import (
    Mention,
    build_candidate_table,
    iter_pattern_matches,
)
from pharma_stats.discovery.patterns import (
    is_denylisted,
    looks_like_dev_code,
    matches_pattern,
)


def test_suffix_match_beats_literal():
    assert matches_pattern("trastuzumab deruxtecan") == ("suffix", "deruxtecan")
    assert matches_pattern("an antibody-drug conjugate of HER2") == (
        "literal",
        "antibody-drug conjugate",
    )
    assert matches_pattern("paclitaxel") is None


def test_dev_code_heuristic():
    assert looks_like_dev_code("ABBV-011")
    assert looks_like_dev_code("SKB264")
    assert not looks_like_dev_code("trastuzumab deruxtecan")
    assert not looks_like_dev_code("pembrolizumab")


def test_denylist_is_case_insensitive():
    assert is_denylisted("Pembrolizumab")
    assert not is_denylisted("trastuzumab deruxtecan")


def _mention(**overrides) -> Mention:
    base = dict(
        nct_id="NCT00000001",
        intervention_name="trastuzumab deruxtecan",
        other_names=["T-DXd", "Enhertu"],
        intervention_type="DRUG",
        strategy="pattern_match",
        lead_sponsor="Daiichi Sankyo",
        lead_sponsor_class="INDUSTRY",
        study_start_date=date(2018, 1, 1),
        overall_status="RECRUITING",
        brief_title="A mock ADC trial",
        match_strength="suffix",
    )
    base.update(overrides)
    return Mention(**base)


def test_synonym_mentions_cluster_into_one_candidate():
    mentions = [
        _mention(nct_id="NCT00000001", intervention_name="trastuzumab deruxtecan", other_names=["T-DXd"]),
        _mention(
            nct_id="NCT00000002",
            intervention_name="Enhertu",
            other_names=["trastuzumab deruxtecan"],
            strategy="seed_expansion",
            match_strength="seed",
        ),
    ]
    candidates = build_candidate_table(mentions)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.trial_count == 2
    assert set(c.nct_ids) == {"NCT00000001", "NCT00000002"}
    assert set(c.strategies) == {"pattern_match", "seed_expansion"}
    assert not c.ambiguous
    assert not c.dev_code_only
    names = {c.proposed_name, *c.synonyms}
    assert "trastuzumab deruxtecan" in names
    assert "Enhertu" in names


def test_unrelated_assets_stay_separate():
    mentions = [
        _mention(intervention_name="trastuzumab deruxtecan", other_names=[]),
        _mention(
            nct_id="NCT00000099",
            intervention_name="enfortumab vedotin",
            other_names=["Padcev"],
            match_strength="suffix",
        ),
    ]
    candidates = build_candidate_table(mentions)
    assert len(candidates) == 2
    names = {c.proposed_name.lower() for c in candidates}
    assert any("deruxtecan" in n for n in names)
    assert any("vedotin" in n for n in names)


def test_literal_only_cluster_is_flagged_ambiguous():
    mentions = [
        _mention(
            intervention_name="experimental ADC",
            other_names=[],
            match_strength="literal",
        )
    ]
    candidates = build_candidate_table(mentions)
    assert len(candidates) == 1
    assert candidates[0].ambiguous
    assert not candidates[0].dev_code_only


def test_dev_code_only_cluster_is_flagged():
    mentions = [
        _mention(
            intervention_name="ABBV-011",
            other_names=[],
            strategy="sponsor_expansion",
            match_strength="dev_code",
        )
    ]
    candidates = build_candidate_table(mentions)
    assert len(candidates) == 1
    assert candidates[0].dev_code_only
    assert not candidates[0].ambiguous


def test_substring_pass_merges_combo_arm_into_anchor_name():
    mentions = [
        _mention(intervention_name="belantamab mafodotin", other_names=[], nct_id="NCT00000010"),
        _mention(
            nct_id="NCT00000011",
            intervention_name="Arm A: Belantamab mafodotin, dexamethasone",
            other_names=[],
        ),
    ]
    candidates = build_candidate_table(mentions)
    assert len(candidates) == 1
    assert candidates[0].trial_count == 2


class _FakeSearchClient:
    def __init__(self, studies):
        self.studies = studies

    def search_studies(self, **kwargs):
        yield from self.studies


def _study(nct_id, intervention_name, other_names=None, start="2019-03-01"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": start},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Mock Pharma", "class": "INDUSTRY"},
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "DRUG",
                        "name": intervention_name,
                        "otherNames": other_names or [],
                    }
                ]
            },
        }
    }


def test_pattern_match_iterator_emits_suffix_hits_and_skips_pre_2012():
    client = _FakeSearchClient(
        [
            _study("NCT11111111", "sacituzumab govitecan", ["Trodelvy"]),
            _study("NCT22222222", "paclitaxel"),
            _study("NCT33333333", "trastuzumab emtansine", start="2010-01-01"),
        ]
    )
    mentions = list(iter_pattern_matches(client))
    assert [m.nct_id for m in mentions] == ["NCT11111111"]
    assert mentions[0].match_strength == "suffix"
    assert mentions[0].intervention_name == "sacituzumab govitecan"
