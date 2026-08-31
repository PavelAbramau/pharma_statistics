"""Tests for pharma_stats.triage — Layer 1 (deterministic, no API) fully
exercised for real; Layer 2 (batched model calls) with
silver.model_client's batch functions mocked — no test here makes a real
API call."""
from __future__ import annotations

import json

import pytest

from pharma_stats.labelling import trial_scope as ts
from pharma_stats.silver import model_client
from pharma_stats.triage import deterministic as det
from pharma_stats.triage import layer2


# --------------------------------------------------------- deterministic --

def _program(**overrides) -> dict:
    base = {
        "program_id": "cand_x", "proposed_name": "Some Drug", "synonyms": [],
        "sponsors_over_time": [], "trial_scope": {}, "trials": [],
    }
    base.update(overrides)
    return base


def test_evaluate_is_adc_inn_suffix():
    # not a real known compound — isolates the suffix rule from rule 0 (known-list exact match)
    p = _program(proposed_name="Zorbamab Deruxtecan")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc == "yes"
    assert "deruxtecan" in rule


def test_evaluate_is_adc_known_list_beats_nothing_else_needed():
    # a real known compound resolves via rule 0 even without checking suffix separately
    p = _program(proposed_name="Trastuzumab Deruxtecan")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc == "yes"
    assert rule.startswith("layer1_known_list:")


def test_evaluate_is_adc_denylist():
    p = _program(proposed_name="Pembrolizumab")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc == "no"
    assert rule == "layer1_denylist"


def test_evaluate_is_adc_generic_class_label():
    p = _program(proposed_name="Cohort 18 (Nectin-4 ADC)")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc == "yes"
    assert rule == "layer1_generic_class_label"


def test_evaluate_is_adc_generic_class_label_rejects_bare_conjugate():
    # "conjugate" alone must NOT fire — see module docstring's false-positive note
    p = _program(proposed_name="Etirinotecan Pegol (Topoisomerase I Inhibitor Polymer Conjugate)")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc is None


def test_evaluate_is_adc_known_list_exact_match():
    p = _program(proposed_name="Some Dev Code", synonyms=["Kadcyla"])
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc == "yes"
    assert rule.startswith("layer1_known_list:")


def test_evaluate_is_adc_unresolved_by_default():
    p = _program(proposed_name="XYZ-1234")
    is_adc, rule = det.evaluate_is_adc(p)
    assert is_adc is None
    assert rule is None


def test_evaluate_in_scope_rejection_non_industry_requires_all():
    sponsors = [{"sponsor": "Acme Pharma", "class": "INDUSTRY"}, {"sponsor": "State U", "class": "OTHER"}]
    p = _program(sponsors_over_time=sponsors)
    in_scope, reason, rule = det.evaluate_in_scope_rejection(p)
    assert in_scope is None  # NOT all non-industry — must not reject a real industry sponsor's asset

    sponsors_all_other = [{"sponsor": "State U", "class": "OTHER"}]
    p2 = _program(sponsors_over_time=sponsors_all_other)
    in_scope2, reason2, rule2 = det.evaluate_in_scope_rejection(p2)
    assert in_scope2 == "no"
    assert reason2 == "non_industry"


def test_evaluate_in_scope_rejection_heme_only():
    p = _program(trial_scope={"NCT1": "heme", "NCT2": "heme"})
    in_scope, reason, rule = det.evaluate_in_scope_rejection(p)
    assert in_scope == "no"
    assert reason == "heme_only"


def test_evaluate_in_scope_positive_requires_all_three_conditions():
    good = _program(
        sponsors_over_time=[{"sponsor": "Acme Pharma", "class": "INDUSTRY"}],
        trial_scope={"NCT1": "solid", "NCT2": "solid"},
        trials=[{"nct_id": "NCT1", "start_date": "2015-01-01"}, {"nct_id": "NCT2", "start_date": "2018-01-01"}],
    )
    assert det.evaluate_in_scope_positive(good) == "layer1_positive_in_scope"

    # fails on start date before 2012
    too_old = _program(
        sponsors_over_time=[{"sponsor": "Acme Pharma", "class": "INDUSTRY"}],
        trial_scope={"NCT1": "solid"},
        trials=[{"nct_id": "NCT1", "start_date": "2005-01-01"}],
    )
    assert det.evaluate_in_scope_positive(too_old) is None

    # fails on mixed scope (not all solid)
    mixed = _program(
        sponsors_over_time=[{"sponsor": "Acme Pharma", "class": "INDUSTRY"}],
        trial_scope={"NCT1": "solid", "NCT2": "ambiguous"},
        trials=[{"nct_id": "NCT1", "start_date": "2015-01-01"}],
    )
    assert det.evaluate_in_scope_positive(mixed) is None

    # fails when not ALL sponsors are industry
    mixed_sponsor = _program(
        sponsors_over_time=[{"sponsor": "Acme Pharma", "class": "INDUSTRY"}, {"sponsor": "Hospital", "class": "OTHER"}],
        trial_scope={"NCT1": "solid"},
        trials=[{"nct_id": "NCT1", "start_date": "2015-01-01"}],
    )
    assert det.evaluate_in_scope_positive(mixed_sponsor) is None


def test_evaluate_full_is_adc_yes_in_scope_no_committable():
    p = _program(
        proposed_name="Foo Vedotin",
        sponsors_over_time=[{"sponsor": "State U", "class": "OTHER"}],
    )
    result = det.evaluate(p)
    assert result.is_adc == "yes"
    assert result.in_scope == "no"
    assert result.committable is True


def test_evaluate_positive_rule_fires_even_with_is_adc_pending():
    p = _program(
        proposed_name="XYZ-1234",  # no is_adc rule fires
        sponsors_over_time=[{"sponsor": "Acme Pharma", "class": "INDUSTRY"}],
        trial_scope={"NCT1": "solid"},
        trials=[{"nct_id": "NCT1", "start_date": "2015-01-01"}],
    )
    result = det.evaluate(p)
    assert result is not None
    assert result.is_adc is None
    assert result.in_scope == "yes"
    assert result.rule == "layer1_positive_in_scope"
    assert result.committable is False


def test_evaluate_returns_none_when_nothing_resolves():
    p = _program(proposed_name="XYZ-1234")
    assert det.evaluate(p) is None


def test_known_adc_names_includes_seed_and_fixture_entries():
    names = det.known_adc_names()
    assert "kadcyla" in names          # seed_assets.json synonym
    assert "brentuximab vedotin" in names  # known_adcs.txt entry


# ----------------------------------------------------------------- layer2 --

def _evidence(pid, name):
    return {"program_id": pid, "name": name, "synonyms": [], "lead_sponsor": None,
            "conditions": [], "text_snippets": []}


def _fake_response(answers: list[dict]) -> tuple:
    return (json.dumps(answers), model_client.Usage(input_tokens=500, output_tokens=100, model="claude-sonnet-4-6"))


def test_run_layer2_stays_at_initial_k_on_unanimous_agreement(monkeypatch):
    evidences = [_evidence("p1", "Drug A"), _evidence("p2", "Drug B")]
    unanimous_answer = [
        {"name": "Drug A", "is_adc": "yes", "from_recall": False, "quote": "an antibody-drug conjugate"},
        {"name": "Drug B", "is_adc": "no", "from_recall": False, "quote": "a small molecule inhibitor"},
    ]

    submitted = []
    monkeypatch.setattr(layer2.model_client, "submit_batch", lambda requests, model: submitted.append(requests) or "batch1")
    monkeypatch.setattr(layer2.model_client, "poll_batch_until_done", lambda batch_id, **kw: {"processing_status": "ended"})

    def fake_collect(batch_id, **kw):
        # every custom_id in round 1 gets the same unanimous answer
        reqs = submitted[-1]
        return {r["custom_id"]: _fake_response(unanimous_answer) for r in reqs}

    monkeypatch.setattr(layer2.model_client, "collect_batch_results", fake_collect)

    results, log = layer2.run_layer2(evidences)
    assert results["p1"].is_adc == "yes"
    assert results["p1"].k == layer2.INITIAL_K
    assert results["p1"].disagreement is False
    assert results["p2"].is_adc == "no"
    assert log["n_escalated_groups"] == 0
    assert len(submitted) == 1  # no second (escalation) round submitted


def test_run_layer2_escalates_and_resolves_by_majority(monkeypatch):
    evidences = [_evidence("p1", "Drug A")]
    call_count = {"n": 0}

    def fake_collect(batch_id, **kw):
        call_count["n"] += 1
        reqs = submitted[-1]
        # round 1 disagrees (yes, no, no); round 2's 2 samples both say
        # "no" -> final tally over all 5 is 4-1 "no", a clear majority
        votes = ["yes", "no", "no"] if call_count["n"] == 1 else ["no", "no"]
        out = {}
        for r, v in zip(reqs, votes):
            out[r["custom_id"]] = _fake_response([{"name": "Drug A", "is_adc": v, "from_recall": False, "quote": "x"}])
        return out

    submitted = []
    monkeypatch.setattr(layer2.model_client, "submit_batch", lambda requests, model: submitted.append(requests) or f"batch{len(submitted)}")
    monkeypatch.setattr(layer2.model_client, "poll_batch_until_done", lambda batch_id, **kw: {"processing_status": "ended"})
    monkeypatch.setattr(layer2.model_client, "collect_batch_results", fake_collect)

    results, log = layer2.run_layer2(evidences)
    assert len(submitted) == 2  # escalation round did fire
    assert results["p1"].k == layer2.ESCALATED_K
    # majority vote per candidate (per spec), not silver's strict-unanimity
    # abstention rule — 4/5 "no" resolves to "no"
    assert results["p1"].is_adc == "no"
    # but disagreement stays True — round 1 needed a second look, so this
    # candidate is still routed to Layer 3 preferentially regardless of
    # how confident the final majority looks (see route_to_layer3)
    assert results["p1"].disagreement is True
    assert log["n_escalated_groups"] == 1


def test_run_layer2_disagreement_flag_reflects_initial_round_only():
    # a single round-1 dissenting vote can never be erased by escalation
    # samples (majority of 5 that started 2-1 is at best 4-1, never
    # unanimous) — disagreement must come from round 1 alone, not be
    # re-derived from the final tally, or every escalated candidate would
    # always read as "still disagreeing" even after a clean 2-0 round 2
    group = [{"program_id": "p1", "name": "Drug A"}]
    round1_indexed = [
        {"p1": {"name": "Drug A", "is_adc": "yes"}},
        {"p1": {"name": "Drug A", "is_adc": "no"}},
        {"p1": {"name": "Drug A", "is_adc": "no"}},
    ]
    initial = layer2._initial_disagreement(group, round1_indexed)
    assert initial["p1"] is True

    full_indexed = round1_indexed + [
        {"p1": {"name": "Drug A", "is_adc": "no"}},
        {"p1": {"name": "Drug A", "is_adc": "no"}},
    ]
    resolved = layer2._resolve_group(group, full_indexed, 5, initial)
    assert resolved["p1"].is_adc == "no"
    assert resolved["p1"].disagreement is True  # still flagged, from round 1


def test_route_to_layer3_on_unsure_disagreement_or_recall():
    unsure = layer2.CandidateAnswer("p1", "x", "unsure", False, None, 3, False, [])
    disagreeing = layer2.CandidateAnswer("p2", "x", "no", False, None, 5, True, [])
    recall = layer2.CandidateAnswer("p3", "x", "yes", True, None, 3, False, [])
    confident = layer2.CandidateAnswer("p4", "x", "yes", False, "quoted text", 3, False, [])
    assert layer2.route_to_layer3(unsure) is True
    assert layer2.route_to_layer3(disagreeing) is True
    assert layer2.route_to_layer3(recall) is True
    assert layer2.route_to_layer3(confident) is False


def test_group_into_batches_respects_batch_size():
    evidences = [_evidence(f"p{i}", f"Drug {i}") for i in range(45)]
    groups = layer2.group_into_batches(evidences, batch_size=20)
    assert [len(g) for g in groups] == [20, 20, 5]


# ----------------------------------------------------------------- layer3 --

def test_run_layer3_parses_answers(monkeypatch):
    from pharma_stats.triage import layer3

    candidates = [{"program_id": "p1", "name": "Drug A"}, {"program_id": "p2", "name": "Drug B"}]
    monkeypatch.setattr(layer3.model_client, "submit_batch", lambda requests, model: "batch1")
    monkeypatch.setattr(layer3.model_client, "poll_batch_until_done", lambda batch_id, **kw: {"processing_status": "ended"})
    monkeypatch.setattr(layer3.model_client, "collect_batch_results", lambda batch_id, **kw: {
        "l3:p1": (json.dumps({"is_adc": "yes", "quote": "an ADC", "source_url": "https://x"}),
                  model_client.Usage(100, 50, "claude-sonnet-4-6")),
        # p2 missing entirely — simulates an errored/expired batch request
    })

    answers, log = layer3.run_layer3(candidates)
    assert answers["p1"].is_adc == "yes"
    assert answers["p1"].quote == "an ADC"
    assert answers["p2"].is_adc == "unsure"  # missing result never guessed
    assert log["n_candidates"] == 2
    assert log["n_unsure"] == 1


def test_run_layer3_refuses_to_exceed_cap(monkeypatch):
    from pharma_stats.triage import layer3

    candidates = [{"program_id": f"p{i}", "name": f"Drug {i}"} for i in range(layer3.MAX_LAYER3_CANDIDATES + 1)]
    with pytest.raises(model_client.ModelClientError):
        layer3.run_layer3(candidates)


# -------------------------------------------------------------- validation --

def _decision(pid, is_adc, from_recall):
    return {"program_id": pid, "is_adc": is_adc, "in_scope": "no" if is_adc == "no" else None,
            "from_recall": from_recall}


def test_draw_stratified_sample_balances_four_strata():
    from pharma_stats.triage import validation as val

    decisions = (
        [_decision(f"ta{i}", "yes", False) for i in range(10)]   # text/accept
        + [_decision(f"tr{i}", "no", False) for i in range(10)]  # text/reject
        + [_decision(f"ra{i}", "yes", True) for i in range(10)]  # recall/accept
        + [_decision(f"rr{i}", "no", True) for i in range(10)]   # recall/reject
    )
    sample = val.draw_stratified_sample(decisions, sample_size=8, seed=0)
    from collections import Counter
    counts = Counter(val._stratum(d) for d in sample)
    assert len(sample) == 8
    assert all(n == 2 for n in counts.values())  # 8 // 4 = 2 per stratum


def test_draw_stratified_sample_keeps_already_reserved():
    from pharma_stats.triage import validation as val

    decisions = [_decision(f"p{i}", "yes", False) for i in range(20)]
    sample = val.draw_stratified_sample(decisions, sample_size=4, seed=0, already_reserved={"p5"})
    assert "p5" in {d["program_id"] for d in sample}


def test_compute_agreement_excludes_unreviewed_programs(monkeypatch):
    from pharma_stats.triage import validation as val

    sample = [_decision("p1", "yes", False), _decision("p2", "no", False)]
    gold_records = [{
        "action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "yes",
        "timestamp": "2026-01-01", "is_repeat_probe": False,
    }]
    agreement = val.compute_agreement(sample, gold_records)
    assert agreement["is_adc"]["compared"] == 1  # p2 excluded, never reached by a human
    assert agreement["is_adc"]["agree"] == 1


def test_check_gate_rejects_below_threshold():
    from pharma_stats.triage import validation as val

    agreement = {
        "is_adc": {"compared": 80, "agree": 70, "agreement_rate": 70 / 80},  # 87.5% < 95%
        "in_scope": {"compared": 10, "agree": 10, "agreement_rate": 1.0},
    }
    passed, reason = val.check_gate(agreement)
    assert passed is False
    assert "is_adc" in reason


def test_check_gate_passes_above_threshold():
    from pharma_stats.triage import validation as val

    agreement = {
        "is_adc": {"compared": 80, "agree": 78, "agreement_rate": 78 / 80},
        "in_scope": {"compared": 20, "agree": 19, "agreement_rate": 19 / 20},
    }
    passed, reason = val.check_gate(agreement)
    assert passed is True


def test_check_gate_refuses_when_not_enough_reviewed_yet():
    from pharma_stats.triage import validation as val

    agreement = {
        "is_adc": {"compared": 10, "agree": 10, "agreement_rate": 1.0},
        "in_scope": {"compared": 0, "agree": 0, "agreement_rate": None},
    }
    passed, reason = val.check_gate(agreement)
    assert passed is False
    assert "10/80" in reason


# -------------------------------------------------------------------- pool --

def test_select_candidate_pool_excludes_any_reviewed_gate(tmp_path, monkeypatch):
    from pharma_stats.triage import pool as tpool

    programs = [
        {"program_id": "p1", "proposed_name": "A"},
        {"program_id": "p2", "proposed_name": "B"},
        {"program_id": "p3", "proposed_name": "C"},
    ]
    gold_records = [
        {"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
         "timestamp": "2026-01-01", "is_repeat_probe": False},
        {"action": "label", "program_id": "p2", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
         "status": "active", "timestamp": "2026-01-02", "is_repeat_probe": False},
    ]
    pool, stats = tpool.select_candidate_pool(programs, gold_records)
    assert {p["program_id"] for p in pool} == {"p3"}
    assert stats["overlap_count"] == 0
    assert stats["total_reviewed_programs"] == 2
    assert stats["gate1_rejected_count"] == 1
    assert stats["gate3_labelled_count"] == 1


def test_assert_not_reviewed_raises_for_a_reviewed_program():
    from pharma_stats.triage import pool as tpool

    gold_records = [{"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
                      "timestamp": "2026-01-01", "is_repeat_probe": False}]
    with pytest.raises(tpool.PoolIntegrityError):
        tpool.assert_not_reviewed("p1", gold_records)
    tpool.assert_not_reviewed("p2", gold_records)  # not reviewed — fine


# ---------------------------------------------------------------- staging --

def test_staging_append_refuses_reviewed_program(tmp_path, monkeypatch):
    from pharma_stats.triage import staging

    gold_records = [{"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
                      "timestamp": "2026-01-01", "is_repeat_probe": False}]
    from pharma_stats.triage import pool as tpool
    monkeypatch.setattr(tpool.gold_store, "load_records", lambda: gold_records)

    record = staging.build_record({"program_id": "p1", "is_adc": "no"}, run_id="r1")
    with pytest.raises(Exception):
        staging.append_record(record, path=tmp_path / "staged.jsonl")


def test_staging_roundtrip(tmp_path, monkeypatch):
    from pharma_stats.triage import staging

    from pharma_stats.triage import pool as tpool
    monkeypatch.setattr(tpool.gold_store, "load_records", lambda: [])
    path = tmp_path / "staged.jsonl"
    record = staging.build_record(
        {"program_id": "p1", "proposed_name": "X", "is_adc": "yes", "layer": 2}, run_id="r1",
    )
    staging.append_record(record, path=path)
    loaded = staging.load_records(path=path)
    assert len(loaded) == 1
    assert loaded[0]["status"] == "pending"
    assert loaded[0]["layer"] == 2


# --------------------------------------------------------------- pipeline --

def test_partition_by_text_evidence():
    from pharma_stats.triage import pipeline as tpl

    evidences = [
        {"program_id": "p1", "text_snippets": ["some text"]},
        {"program_id": "p2", "text_snippets": []},
        {"program_id": "p3", "text_snippets": None},
    ]
    with_text, no_text = tpl.partition_by_text_evidence(evidences)
    assert [e["program_id"] for e in with_text] == ["p1"]
    assert [e["program_id"] for e in no_text] == ["p2", "p3"]


def test_cap_layer3_queue_flags_overflow_never_drops():
    from pharma_stats.triage import pipeline as tpl

    candidates = [{"program_id": f"p{i}"} for i in range(200)]
    within, overflow = tpl.cap_layer3_queue(candidates, cap=150)
    assert len(within) == 150
    assert len(overflow) == 50
    assert len(within) + len(overflow) == len(candidates)  # nothing dropped


def test_stage_manual_overflow_flags_every_candidate(tmp_path, monkeypatch):
    from pharma_stats.triage import pipeline as tpl
    from pharma_stats.triage import staging

    from pharma_stats.triage import pool as tpool
    monkeypatch.setattr(staging, "STAGING_PATH", tmp_path / "staged.jsonl")
    monkeypatch.setattr(tpool.gold_store, "load_records", lambda: [])

    overflow = [{"program_id": "p1", "name": "Drug A"}, {"program_id": "p2", "name": "Drug B"}]
    n = tpl.stage_manual_overflow(overflow, run_id="r1", reason="layer3 cap exceeded")
    assert n == 2
    records = staging.load_records(path=tmp_path / "staged.jsonl")
    assert all(r["manual_overflow"] for r in records)
    assert all(r["manual_overflow_reason"] == "layer3 cap exceeded" for r in records)


def test_recall_vs_text_gap_reports_both_rates():
    from pharma_stats.triage import validation as val

    agreement = {
        "by_stratum": {
            "text/accept": {"compared": 10, "agree": 10, "agreement_rate": 1.0},
            "text/reject": {"compared": 10, "agree": 9, "agreement_rate": 0.9},
            "recall/accept": {"compared": 10, "agree": 6, "agreement_rate": 0.6},
            "recall/reject": {"compared": 10, "agree": 7, "agreement_rate": 0.7},
        },
    }
    gap = val.recall_vs_text_gap(agreement)
    assert gap["text_agreement_rate"] == pytest.approx(19 / 20)
    assert gap["recall_agreement_rate"] == pytest.approx(13 / 20)
    assert gap["gap"] > 0.2  # recall is materially worse in this example
