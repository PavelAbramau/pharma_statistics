"""Tests for the silver (auto-labelled) track: the gold/silver isolation
guarantee, the decomposed-question deterministic rules, the citation
gate, and the strict eval-split/scoring math. Model-invocation stubs
(sampling.sample_answers, red_team.generate_objection) are checked only
for raising NotImplementedError — there's nothing else to test until a
model-invocation mechanism is chosen."""
from __future__ import annotations

import pytest

from pharma_stats.audit import gold_set
from pharma_stats.labelling import store
from pharma_stats.silver import citations, eval as silver_eval, questions, red_team, retrieval_agent, sampling
from pharma_stats.silver import store as silver_store
from pharma_stats.silver.questions import (
    NOT_DETERMINABLE,
    Citation,
    DecomposedAnswers,
    DiscontinuationStatementAnswer,
    StopReasonAnswer,
    SuccessorAssetAnswer,
    TrialInitiatedSinceAnswer,
)


# ---------------------------------------------------------- gold/silver isolation --

def test_silver_and_gold_paths_are_structurally_different():
    assert silver_store.SILVER_LABELS_PATH != store.LABELS_PATH
    assert "silver" in str(silver_store.SILVER_LABELS_PATH)
    assert "gold" in str(store.LABELS_PATH)


def test_silver_build_record_always_stamps_labeller_auto():
    record = silver_store.build_record({"program_id": "p1", "status": "active"}, session_id="s1")
    assert record["labeller"] == "auto"


def test_silver_append_record_refuses_non_auto_labeller(tmp_path):
    path = tmp_path / "labels.jsonl"
    bad_record = silver_store.build_record({"program_id": "p1"}, session_id="s1")
    bad_record["labeller"] = "human"  # tampered after build_record
    with pytest.raises(ValueError):
        silver_store.append_record(bad_record, path=path)
    assert not path.exists()  # the refusal must happen before any write


def test_silver_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "labels.jsonl"
    record = silver_store.build_record({"program_id": "p1", "status": "dormant_suspected"}, session_id="s1")
    silver_store.append_record(record, path=path)
    loaded = silver_store.load_records(path=path)
    assert len(loaded) == 1
    assert loaded[0]["labeller"] == "auto"


def test_gold_set_audit_catches_auto_sourced_record_in_gold(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    contaminated = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "gate_reached": 3, "is_adc": "yes", "in_scope": "yes", "labeller": "auto",
        "program_id": "p1", "status": "active", "kill_reason": None,
        "confidence": "high", "evidence_note": "", "label_evidence_date": None,
        "public_confirmation_date": None, "never_publicly_confirmed": False,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
        "history_coverage_at_serve_time": "full",
    }
    store.append_record(contaminated, path=gold_path)

    checks = gold_set.run()
    isolation = next(c for c in checks if "auto-sourced" in c.name)
    assert isolation.level == "FAIL"
    assert "e1" in isolation.detail


def test_gold_set_audit_passes_isolation_on_clean_gold(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    checks = gold_set.run()  # empty gold set — isolation check must still run
    isolation = next(c for c in checks if "auto-sourced" in c.name)
    assert isolation.level == "PASS"


# --------------------------------------------------- decomposed-question rules --

def _answers(
    trial_since=False, statement_exists=False, statement_date=None,
    reason=NOT_DETERMINABLE, successor_exists=False,
) -> DecomposedAnswers:
    return DecomposedAnswers(
        trial_initiated_since=TrialInitiatedSinceAnswer(value=trial_since, since_date="2024-01-01"),
        discontinuation_statement=DiscontinuationStatementAnswer(exists=statement_exists, statement_date=statement_date),
        stop_reason=StopReasonAnswer(category=reason),
        successor_asset=SuccessorAssetAnswer(exists=successor_exists),
    )


def test_deterministic_rules_dead_confirmed_with_reason():
    answers = _answers(statement_exists=True, statement_date="2025-01-01", reason="efficacy")
    result = questions.apply_deterministic_rules(answers)
    assert result == {
        "status": "dead_confirmed", "kill_reason": "futility_efficacy",
        "public_confirmation_date": "2025-01-01", "never_publicly_confirmed": False, "abstain": False,
    }


def test_deterministic_rules_abstains_on_confirmed_but_reason_not_determinable():
    answers = _answers(statement_exists=True, statement_date="2025-01-01", reason=NOT_DETERMINABLE)
    result = questions.apply_deterministic_rules(answers)
    assert result["abstain"] is True


def test_deterministic_rules_abstains_when_core_evidence_not_determinable():
    answers = _answers(trial_since=NOT_DETERMINABLE)
    assert questions.apply_deterministic_rules(answers)["abstain"] is True

    answers2 = _answers(statement_exists=NOT_DETERMINABLE)
    assert questions.apply_deterministic_rules(answers2)["abstain"] is True


def test_deterministic_rules_superseded_when_successor_exists_and_no_new_trial():
    answers = _answers(trial_since=False, statement_exists=False, successor_exists=True)
    result = questions.apply_deterministic_rules(answers)
    assert result["status"] == "superseded"
    assert result["abstain"] is False


def test_deterministic_rules_dormant_when_no_new_trial_no_statement_no_successor():
    answers = _answers(trial_since=False, statement_exists=False, successor_exists=False)
    result = questions.apply_deterministic_rules(answers)
    assert result["status"] == "dormant_suspected"


def test_deterministic_rules_active_when_trial_initiated_since():
    answers = _answers(trial_since=True, statement_exists=False)
    result = questions.apply_deterministic_rules(answers)
    assert result["status"] == "active"


def test_deterministic_rules_abstains_when_successor_not_determinable():
    answers = _answers(trial_since=False, statement_exists=False, successor_exists=NOT_DETERMINABLE)
    assert questions.apply_deterministic_rules(answers)["abstain"] is True


# ----------------------------------------------------------------- citations --

def test_citation_verify_matches_with_whitespace_normalisation():
    c = Citation(source_type="fetched_url", locator="https://example.com", quote="the trial was  discontinued")
    assert citations.verify(c, "...the trial was\ndiscontinued due to...") is True
    assert citations.verify(c, "no relevant text here") is False


def test_citation_resolve_raises_for_missing_snapshot():
    c = Citation(source_type="raw_snapshot", locator="ctgov:NCT00000000_DOES_NOT_EXIST", quote="x")
    with pytest.raises(citations.CitationError):
        citations.resolve(c)


def test_citation_resolve_raises_for_unknown_source_type():
    c = Citation(source_type="telepathy", locator="x", quote="y")
    with pytest.raises(citations.CitationError):
        citations.resolve(c)


# ------------------------------------------------------------------- eval --

def _gold_records(entries):
    records = []
    for pid, candidate_id, gate, status, kill_reason, confirm_date in entries:
        r = store.build_record(
            {
                "action": "label", "program_id": pid, "candidate_id": candidate_id,
                "gate_reached": gate, "is_adc": "yes", "in_scope": "yes" if gate == 3 else "no",
                "scope_reason": None if gate == 3 else "heme_only",
                "status": status, "kill_reason": kill_reason,
                "public_confirmation_date": confirm_date, "confidence": "high",
            },
            session_id="s1", served_stratum={},
        )
        records.append(r)
    return records


def test_split_by_asset_keeps_all_programs_of_one_asset_together():
    records = _gold_records([
        ("p1", "assetA", 3, "active", None, None),
        ("p2", "assetA", 3, "dead_confirmed", "futility_efficacy", "2024-01-01"),
        ("p3", "assetB", 3, "active", None, None),
        ("p4", "assetC", 1, None, None, None),  # gate 1 — excluded entirely
    ])
    few_shot, held_out = silver_eval.split_by_asset(records, eval_fraction=0.5, seed=0)
    assert set(few_shot) | set(held_out) == {"p1", "p2", "p3"}
    assert not (set(few_shot) & set(held_out))
    # assetA's two programs must land in the same half
    assert ("p1" in few_shot) == ("p2" in few_shot)


def test_per_field_accuracy_scores_fields_separately_and_excludes_abstentions():
    records = _gold_records([
        ("p1", "assetA", 3, "dead_confirmed", "futility_efficacy", "2024-01-01"),
        ("p2", "assetB", 3, "active", None, None),
    ])
    predictions = {
        "p1": {"status": "dead_confirmed", "kill_reason": "futility_efficacy", "public_confirmation_date": "2024-06-01"},
        "p2": {"status": "active", "kill_reason": "not_determinable", "public_confirmation_date": None},
    }
    result = silver_eval.per_field_accuracy(predictions, records, ["p1", "p2"])
    assert result["status"]["accuracy"] == 1.0
    assert result["public_confirmation_date"]["n_compared"] == 1  # p2 abstained (None)
    assert result["public_confirmation_date"]["accuracy"] == 0.0  # p1's date was wrong
    assert result["kill_reason"]["n_abstained"] == 1


def test_accuracy_vs_self_consistency_ceiling():
    assert silver_eval.accuracy_vs_self_consistency_ceiling(0.9, 0.88) == {
        "comparable": True, "at_or_above_ceiling": True, "gap": pytest.approx(0.02),
    }
    assert silver_eval.accuracy_vs_self_consistency_ceiling(0.5, 0.88)["at_or_above_ceiling"] is False
    assert silver_eval.accuracy_vs_self_consistency_ceiling(None, 0.88)["comparable"] is False


# --------------------------------------------------------------- sampling --

def test_majority_vote_and_should_abstain():
    assert sampling.majority_vote(["a", "a", "a"]) == ("a", 3)
    assert sampling.should_abstain(["a", "a", "a"]) is False
    assert sampling.should_abstain(["a", "a", "b"]) is True
    assert sampling.should_abstain([]) is True


def test_sample_answers_not_implemented():
    with pytest.raises(NotImplementedError):
        sampling.sample_answers(lambda: "x")


# --------------------------------------------------------------- red team --

def test_forces_abstention_requires_strong_and_evidenced():
    from pharma_stats.silver.red_team import Objection
    strong_evidenced = Objection(strength="strong", argument="...", citations=[
        Citation(source_type="fetched_url", locator="https://x", quote="y"),
    ])
    strong_unevidenced = Objection(strength="strong", argument="...", citations=[])
    weak_evidenced = Objection(strength="weak", argument="...", citations=[
        Citation(source_type="fetched_url", locator="https://x", quote="y"),
    ])
    assert red_team.forces_abstention(strong_evidenced) is True
    assert red_team.forces_abstention(strong_unevidenced) is False
    assert red_team.forces_abstention(weak_evidenced) is False


def test_generate_objection_not_implemented():
    with pytest.raises(NotImplementedError):
        red_team.generate_objection({}, {})


# ---------------------------------------------------------- retrieval agent --

def test_search_sec_edgar_requires_explicit_user_agent():
    with pytest.raises(ValueError):
        retrieval_agent.search_sec_edgar_fulltext("some asset name")


def test_filing_url_reconstruction_from_hit():
    hit = {
        "_id": "0001234567-24-001234:filing-main.htm",
        "_source": {"ciks": ["0001234567"]},
    }
    url = retrieval_agent._filing_url(hit)
    assert url == "https://www.sec.gov/Archives/edgar/data/1234567/000123456724001234/filing-main.htm"


def test_filing_url_returns_none_for_malformed_hit():
    assert retrieval_agent._filing_url({"_id": "", "_source": {}}) is None


def test_press_archive_and_conference_search_not_implemented():
    with pytest.raises(NotImplementedError):
        retrieval_agent.search_sponsor_press_archive()
    with pytest.raises(NotImplementedError):
        retrieval_agent.search_conference_abstracts()
