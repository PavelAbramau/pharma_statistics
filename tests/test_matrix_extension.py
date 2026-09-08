"""Tests for discovery/matrix_extension.py's non-network logic: which
candidates make it into the extension cohort and why, and the population
index dead_historical's successor check reads from. Discovery itself
(discover_extension_mentions) hits CT.gov and isn't exercised here — see
scripts/build_matrix_extension_universe.py for that."""
from __future__ import annotations

from datetime import date

from pharma_stats.discovery import matrix_extension as mx
from pharma_stats.discovery.candidates import CandidateAsset


def _candidate(
    candidate_id, name, *, synonyms=None, sponsor="Acme Oncology", sponsor_class="INDUSTRY",
    first="2008-01-01", last="2009-01-01", nct_ids=None,
):
    return CandidateAsset(
        candidate_id=candidate_id, proposed_name=name, synonyms=synonyms or [],
        sponsors_over_time=[{"sponsor": sponsor, "class": sponsor_class,
                              "first_seen": first, "last_seen": last}],
        trial_count=1, nct_ids=nct_ids or [f"NCT{candidate_id}"],
        first_trial_start_date=first, last_trial_start_date=last,
        strategies=["pattern_match_extension"], ambiguous=False, dev_code_only=False,
    )


def test_build_extension_programs_includes_pre_2012_solid_candidate():
    candidates = [_candidate("c1", "Old Vedotin Compound", first="2008-01-01", last="2009-06-01")]
    conditions = {"NCTc1": ["Breast Cancer"]}
    programs, stats = mx.build_extension_programs(candidates, conditions)
    assert len(programs) == 1
    assert programs[0].extension_reasons == ["pre_2012"]
    assert programs[0].tumour_bucket == "breast"
    assert stats["n_in_extension"] == 1


def test_build_extension_programs_includes_heme_candidate_even_if_recent():
    candidates = [_candidate("c2", "New Vedotin Compound", first="2020-01-01", last="2022-01-01")]
    conditions = {"NCTc2": ["Diffuse Large B-Cell Lymphoma"]}
    programs, stats = mx.build_extension_programs(candidates, conditions)
    assert len(programs) == 1
    assert programs[0].extension_reasons == ["heme"]
    assert programs[0].tumour_bucket == "heme"


def test_build_extension_programs_excludes_candidate_fully_inside_main_universe():
    # 2012+, solid -- this belongs to the real universe, not the extension
    candidates = [_candidate("c3", "New Vedotin Compound", first="2020-01-01", last="2022-01-01")]
    conditions = {"NCTc3": ["Breast Cancer"]}
    programs, stats = mx.build_extension_programs(candidates, conditions)
    assert programs == []
    assert stats["n_excluded_neither_reason"] == 1


def test_build_extension_programs_excludes_non_adc():
    candidates = [_candidate("c4", "Some Random Monoclonal Antibody", first="2008-01-01")]
    conditions = {"NCTc4": ["Breast Cancer"]}
    programs, stats = mx.build_extension_programs(candidates, conditions)
    assert programs == []
    assert stats["n_excluded_not_adc"] == 1


def test_build_extension_programs_excludes_non_industry_sponsor():
    candidates = [_candidate("c5", "Old Vedotin Compound", sponsor="National Cancer Institute",
                              sponsor_class="OTHER", first="2008-01-01")]
    conditions = {"NCTc5": ["Breast Cancer"]}
    programs, stats = mx.build_extension_programs(candidates, conditions)
    assert programs == []
    assert stats["n_excluded_non_industry"] == 1


def test_build_extension_programs_excludes_candidate_already_in_main_universe():
    """Real bug found 2026-09-08: this module's own clustering can
    rediscover the SAME asset the main pipeline already has (a heme-only
    compound with some trials in the 2012+ window gets caught by both
    passes), generating the same candidate_id hash. Left unexcluded, the
    successor check in dead_historical could read the SAME asset's two
    independently-discovered copies as two different programs and treat
    one as the other's successor."""
    candidates = [_candidate("c6", "Old Vedotin Compound", first="2008-01-01")]
    conditions = {"NCTc6": ["Diffuse Large B-Cell Lymphoma"]}
    programs, stats = mx.build_extension_programs(
        candidates, conditions, existing_candidate_ids=frozenset({"c6"}),
    )
    assert programs == []
    assert stats["n_excluded_already_in_main_universe"] == 1


def test_build_population_index_combines_extension_and_main_programs():
    ext = [mx.ExtensionProgram(
        program_id="mx-ext-1", proposed_name="Old Compound", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT1"], trial_start_dates=[date(2008, 1, 1)], target="MSLN",
        payload_chemotype="auristatin", tumour_bucket="breast", extension_reasons=["pre_2012"],
    )]
    main = [{
        "program_id": "p-main-1", "proposed_name": "New Compound Vedotin", "synonyms": [],
        "sponsors_over_time": [{"sponsor": "Acme"}],
        "trials": [{"start_date": "2015-01-01"}],
    }]
    index = mx.build_population_index(ext, main)
    assert ("Acme", "MSLN") in index
    ids = {pid for pid, _start in index[("Acme", "MSLN")]}
    assert "mx-ext-1" in ids


def test_classify_extension_cohort_marks_dead_historical(monkeypatch):
    p = mx.ExtensionProgram(
        program_id="mx-ext-1", proposed_name="Old Compound", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT1"], trial_start_dates=[date(2008, 1, 1)], target="MSLN",
        payload_chemotype="auristatin", tumour_bucket="breast", extension_reasons=["pre_2012"],
    )
    mx.classify_extension_cohort([p], population_index={}, today=date(2026, 9, 8))
    assert p.dead_historical == "dead_historical"


def test_fold_into_coarse_matrix_adds_dead_historical_only_never_dead():
    from pharma_stats.attributes.matrix import Cell

    cells = {("auristatin", "breast"): Cell(payload="auristatin", tumour_system="breast")}
    p_heme = mx.ExtensionProgram(
        program_id="mx-ext-1", proposed_name="Old Compound", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT1"], trial_start_dates=[date(2008, 1, 1)], target="MSLN",
        payload_chemotype="auristatin", tumour_bucket="heme", extension_reasons=["heme"],
        dead_historical="dead_historical",
    )
    p_not_classified = mx.ExtensionProgram(
        program_id="mx-ext-2", proposed_name="Recent Compound", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT2"], trial_start_dates=[date(2022, 1, 1)], target="MSLN",
        payload_chemotype="auristatin", tumour_bucket="heme", extension_reasons=["heme"],
        dead_historical=None,
    )
    stats = mx.fold_into_coarse_matrix(cells, [p_heme, p_not_classified])

    assert stats["n_dead_historical_folded_in"] == 1
    key = ("auristatin", "heme")
    assert key in cells  # a new cell, never present in the gold-only matrix (heme has no tumour_system)
    assert cells[key].n_dead_historical == 1
    assert cells[key].n_dead == 0  # never counted as a gold/proxy dead
    # the pre-existing gold-quality cell is untouched
    assert cells[("auristatin", "breast")].n_dead_historical == 0


def test_fold_into_coarse_matrix_excludes_undisclosed_payload():
    cells = {}
    p = mx.ExtensionProgram(
        program_id="mx-ext-1", proposed_name="Dev Code 123", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT1"], trial_start_dates=[date(2008, 1, 1)], target="MSLN",
        payload_chemotype="undisclosed", tumour_bucket="heme", extension_reasons=["heme"],
        dead_historical="dead_historical",
    )
    stats = mx.fold_into_coarse_matrix(cells, [p])
    assert stats["n_excluded_payload_undisclosed"] == 1
    assert cells == {}


def test_classify_extension_cohort_withholds_when_successor_exists():
    p = mx.ExtensionProgram(
        program_id="mx-ext-1", proposed_name="Old Compound", synonyms=[], sponsors=["Acme"],
        nct_ids=["NCT1"], trial_start_dates=[date(2008, 1, 1)], target="MSLN",
        payload_chemotype="auristatin", tumour_bucket="breast", extension_reasons=["pre_2012"],
    )
    index = {("Acme", "MSLN"): [("mx-ext-1", date(2008, 1, 1)), ("p-main-1", date(2015, 1, 1))]}
    mx.classify_extension_cohort([p], population_index=index, today=date(2026, 9, 8))
    assert p.dead_historical is None
