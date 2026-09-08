"""Tests for labelling/dead_historical.py — the deterministic rule the
matrix-only universe extension uses instead of manual labelling
(docs/decisions/0008). No trial in >=10y, not a known-approved compound,
and no successor asset from the same sponsor on the same target."""
from __future__ import annotations

from datetime import date

from pharma_stats.labelling import dead_historical as dh

TODAY = date(2026, 9, 8)
APPROVED = frozenset({"brentuximab vedotin", "adcetris"})


def test_has_trial_in_last_10_years_true_when_recent():
    assert dh.has_trial_in_last_10_years(date(2020, 1, 1), today=TODAY) is True


def test_has_trial_in_last_10_years_false_when_old():
    assert dh.has_trial_in_last_10_years(date(2010, 1, 1), today=TODAY) is False


def test_has_trial_in_last_10_years_true_when_unknown():
    # no date to reason from -- never guessed as historical
    assert dh.has_trial_in_last_10_years(None, today=TODAY) is True


def test_is_known_approved_matches_case_insensitive_synonym():
    assert dh.is_known_approved(["Adcetris"], APPROVED) is True
    assert dh.is_known_approved(["SGN-99"], APPROVED) is False


def test_has_successor_true_when_later_program_same_sponsor_target():
    index = {("Acme", "MSLN"): [("p1", date(2005, 1, 1)), ("p2", date(2015, 1, 1))]}
    assert dh.has_successor("p1", "Acme", "MSLN", date(2005, 1, 1), index) is True


def test_has_successor_false_when_no_later_program():
    index = {("Acme", "MSLN"): [("p1", date(2005, 1, 1))]}
    assert dh.has_successor("p1", "Acme", "MSLN", date(2005, 1, 1), index) is False


def test_has_successor_ignores_self():
    index = {("Acme", "MSLN"): [("p1", date(2005, 1, 1))]}
    assert dh.has_successor("p1", "Acme", "MSLN", date(2005, 1, 1), index) is False


def test_classify_dead_historical_positive_case():
    result = dh.classify_dead_historical(
        "p1", ["Old Compound"], ["Acme"], "MSLN", [date(2008, 1, 1), date(2009, 1, 1)],
        today=TODAY, population_index={}, approved_names=APPROVED,
    )
    assert result == "dead_historical"


def test_classify_dead_historical_none_when_no_trial_dates():
    result = dh.classify_dead_historical(
        "p1", ["Old Compound"], ["Acme"], "MSLN", [],
        today=TODAY, population_index={}, approved_names=APPROVED,
    )
    assert result is None


def test_classify_dead_historical_none_when_recent_trial():
    result = dh.classify_dead_historical(
        "p1", ["Old Compound"], ["Acme"], "MSLN", [date(2008, 1, 1), date(2024, 1, 1)],
        today=TODAY, population_index={}, approved_names=APPROVED,
    )
    assert result is None


def test_classify_dead_historical_none_when_known_approved():
    result = dh.classify_dead_historical(
        "p1", ["Adcetris"], ["Acme"], "TNFRSF8", [date(2008, 1, 1)],
        today=TODAY, population_index={}, approved_names=APPROVED,
    )
    assert result is None


def test_classify_dead_historical_none_when_successor_exists():
    index = {("Acme", "MSLN"): [("p1", date(2008, 1, 1)), ("p2", date(2018, 1, 1))]}
    result = dh.classify_dead_historical(
        "p1", ["Old Compound"], ["Acme"], "MSLN", [date(2008, 1, 1)],
        today=TODAY, population_index=index, approved_names=APPROVED,
    )
    assert result is None


def test_classify_dead_historical_vacuous_when_target_unresolved():
    # can't check lineage without a resolved target -- never blocks on that alone
    result = dh.classify_dead_historical(
        "p1", ["Old Compound"], ["Acme"], None, [date(2008, 1, 1)],
        today=TODAY, population_index={}, approved_names=APPROVED,
    )
    assert result == "dead_historical"
