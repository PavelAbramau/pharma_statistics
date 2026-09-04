"""Tests for models/discrete_time_survival.py — outcome/event-date
determination (the part with real, checkable data-quality rules) and
the cause-specific hazard fit's fallback to intercept-only under data
poverty."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pharma_stats.models import discrete_time_survival as dts


def _program(pid, trials=None):
    return {"program_id": pid, "trials": trials or []}


def test_determine_outcome_dead_confirmed_uses_public_confirmation_date():
    g = {"status": "dead_confirmed", "public_confirmation_date": "2021-03-15"}
    outcome = dts.determine_program_outcome(_program("p1"), g, panel_end=date(2026, 1, 1))
    assert outcome.outcome == "dead"
    assert outcome.event_month == date(2021, 3, 15)
    assert outcome.basis == "public_confirmation_date"


def test_determine_outcome_dead_confirmed_never_publicly_confirmed_is_censored_not_guessed():
    g = {"status": "dead_confirmed", "public_confirmation_date": None, "never_publicly_confirmed": True}
    outcome = dts.determine_program_outcome(_program("p1"), g, panel_end=date(2026, 1, 1))
    assert outcome.outcome == "censored"
    assert "excluded_from_hazard" in outcome.basis
    # label_evidence_date must never be read for this, even if present
    g["label_evidence_date"] = "2015-01-01"
    outcome2 = dts.determine_program_outcome(_program("p1"), g, panel_end=date(2026, 1, 1))
    assert outcome2.event_month == outcome.event_month  # unchanged by label_evidence_date's presence


def test_determine_outcome_approved_uses_earliest_actual_primary_completion():
    trials = [
        {"primary_completion_type": "ESTIMATED", "primary_completion_date": "2038-01-01"},
        {"primary_completion_type": "ACTUAL", "primary_completion_date": "2019-06-01"},
        {"primary_completion_type": "ACTUAL", "primary_completion_date": "2017-03-10"},
    ]
    g = {"status": "approved"}
    outcome = dts.determine_program_outcome(_program("p1", trials), g, panel_end=date(2026, 1, 1))
    assert outcome.outcome == "approved"
    assert outcome.event_month == date(2017, 3, 10)  # earliest ACTUAL, never the ESTIMATED 2038 one
    assert outcome.basis == "earliest_actual_primary_completion_proxy"


def test_determine_outcome_approved_never_uses_gold_timestamp():
    """The rejected proxy (docs/decisions/0007): gold timestamp reflects
    labelling-queue order, not when the program was actually approved.
    determine_program_outcome must not accept or read a timestamp field
    at all for this purpose."""
    g = {"status": "approved", "timestamp": "2026-09-03T17:09:05+00:00"}
    outcome = dts.determine_program_outcome(_program("p1", trials=[]), g, panel_end=date(2026, 1, 1))
    assert outcome.outcome == "censored"  # no ACTUAL completion date -> censored, not the timestamp
    assert "no_completion_date_available" in outcome.basis


def test_determine_outcome_active_is_censored_at_panel_end():
    g = {"status": "active"}
    outcome = dts.determine_program_outcome(_program("p1"), g, panel_end=date(2026, 3, 1))
    assert outcome.outcome == "censored"
    assert outcome.event_month == date(2026, 3, 1)


def test_fit_cause_specific_hazard_falls_back_to_intercept_only_under_min_events():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "silence_score_asof": rng.uniform(0, 100, n),
        "log_cost_index": rng.uniform(0, 5, n),
        "sponsor": rng.integers(0, 10, n),
        "event_superseded": [1, 1, 1] + [0] * (n - 3),  # only 3 events -- below MIN_EVENTS_FOR_COVARIATES
    })
    hazard = dts.fit_cause_specific_hazard(df, "event_superseded")
    assert hazard.covariates == []
    assert hazard.n_events == 3


def test_fit_cause_specific_hazard_uses_covariates_above_min_events():
    rng = np.random.default_rng(0)
    n = 300
    silence = rng.uniform(0, 100, n)
    p_event = 1 / (1 + np.exp(-(0.03 * silence - 3)))
    events = (rng.random(n) < p_event).astype(int)
    df = pd.DataFrame({
        "silence_score_asof": silence,
        "log_cost_index": rng.uniform(0, 5, n),
        "sponsor": rng.integers(0, 15, n),
        "event_dead": events,
    })
    assert df["event_dead"].sum() >= dts.MIN_EVENTS_FOR_COVARIATES
    hazard = dts.fit_cause_specific_hazard(df, "event_dead")
    assert hazard.covariates == dts.COVARIATES
    preds = hazard.predict(df)
    assert len(preds) == n
    assert (preds >= 0).all() and (preds <= 1).all()


def test_build_training_table_marks_only_terminal_row_as_event(monkeypatch):
    from pharma_stats.labelling import store

    program = {"program_id": "p1", "proposed_name": "Drug X", "synonyms": [],
               "trials": [{"nct_id": "NCT001"}], "sponsors_over_time": []}
    gold = {"p1": {"status": "dead_confirmed", "public_confirmation_date": "2021-06-01",
                   "gate_reached": 3, "action": "label", "is_repeat_probe": False,
                   "program_id": "p1", "timestamp": "2021-06-01T00:00:00Z"}}
    monkeypatch.setattr(store, "load_records", lambda: [gold["p1"]])
    monkeypatch.setattr(store, "latest_by_program", lambda records: gold)

    fake_rows = [
        {"program_id": "p1", "as_of": "2020-01-01", "silence_score_asof": 10, "band_asof": 0,
         "cost_index": 100.0, "contacts_locations_amendment_cadence_asof": 0.0,
         "target": "ERBB2", "payload_chemotype": "camptothecin_topo1", "indication_mesh_term": "unknown"},
        {"program_id": "p1", "as_of": "2021-06-01", "silence_score_asof": 80, "band_asof": 3,
         "cost_index": 500.0, "contacts_locations_amendment_cadence_asof": 0.0,
         "target": "ERBB2", "payload_chemotype": "camptothecin_topo1", "indication_mesh_term": "unknown"},
    ]
    monkeypatch.setattr(dts, "build_program_month_panel", lambda program, con, end=None: fake_rows)

    df = dts.build_training_table([program], con=None, panel_end=date(2026, 1, 1))
    assert len(df) == 2
    assert df.iloc[0]["event_dead"] == 0
    assert df.iloc[1]["event_dead"] == 1
    assert df.iloc[1]["event_approved"] == 0
