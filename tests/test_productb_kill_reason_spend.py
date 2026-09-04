"""Tests for productb/kill_reason_spend.py (B1) — kill reason must be
resolved against spend AS OF the death date, never a later or final
value, and small-N summaries must be reported honestly, not smoothed."""
from __future__ import annotations

from datetime import date

from pharma_stats.finance import panel as fpanel
from pharma_stats.finance import store as fstore
from pharma_stats.productb import kill_reason_spend as krs


def _label(pid, *, status="dead_confirmed", kill_reason="futility_efficacy",
           label_evidence_date="2021-06-01", timestamp="2026-01-01T00:00:00+00:00",
           proposed_name=None, is_repeat_probe=False, gate_reached=3):
    return {
        "action": "label", "timestamp": timestamp, "program_id": pid,
        "proposed_name": proposed_name, "status": status, "kill_reason": kill_reason,
        "label_evidence_date": label_evidence_date, "is_repeat_probe": is_repeat_probe,
        "gate_reached": gate_reached,
    }


def test_dead_confirmed_records_filters_status_and_kill_reason():
    records = [
        _label("p1", status="dead_confirmed", kill_reason="futility_efficacy"),
        _label("p2", status="active", kill_reason=None),
        _label("p3", status="dead_confirmed", kill_reason=None),  # shouldn't happen, but filter it anyway
    ]
    dead = krs.dead_confirmed_records(records)
    assert {r["program_id"] for r in dead} == {"p1"}


def test_dead_confirmed_records_uses_latest_label_per_program():
    records = [
        _label("p1", status="dead_confirmed", kill_reason="futility_efficacy", timestamp="2026-01-01T00:00:00+00:00"),
        _label("p1", status="active", kill_reason=None, timestamp="2026-02-01T00:00:00+00:00"),
    ]
    dead = krs.dead_confirmed_records(records)
    assert dead == []  # the later, non-dead label supersedes the first


def test_dead_confirmed_records_ignores_repeat_probes():
    records = [_label("p1", is_repeat_probe=True)]
    assert krs.dead_confirmed_records(records) == []


def test_spend_at_death_rows_resolves_value_as_of_label_evidence_date(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        fstore.FinancialEvent(subject_type="program", subject_id="p1", event_date=date(2021, 1, 1),
                               event_type=fpanel.COST_EVENT_TYPE, detail="", source="s", value=10.0),
        # this later point must NEVER leak into a row resolved as of an earlier death date
        fstore.FinancialEvent(subject_type="program", subject_id="p1", event_date=date(2021, 12, 1),
                               event_type=fpanel.COST_EVENT_TYPE, detail="", source="s", value=1000.0),
    ], warehouse_db=db)
    panel = fpanel.build_money_layer_panel(db)

    dead = [_label("p1", label_evidence_date="2021-06-01", proposed_name="DrugX")]
    rows = krs.spend_at_death_rows(dead, panel)

    assert len(rows) == 1
    assert rows[0].estimated_cumulative_spend == 10.0
    assert rows[0].kill_reason == "futility_efficacy"
    assert rows[0].proposed_name == "DrugX"


def test_spend_at_death_rows_none_when_no_spend_data_yet(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        fstore.FinancialEvent(subject_type="program", subject_id="p1", event_date=date(2021, 6, 1),
                               event_type=fpanel.COST_EVENT_TYPE, detail="", source="s", value=10.0),
    ], warehouse_db=db)
    panel = fpanel.build_money_layer_panel(db)

    dead = [_label("p1", label_evidence_date="2021-01-01")]  # death predates the only cost point
    rows = krs.spend_at_death_rows(dead, panel)
    assert rows[0].estimated_cumulative_spend is None


def test_spend_at_death_rows_skips_records_without_label_evidence_date():
    dead = [_label("p1", label_evidence_date=None)]
    assert krs.spend_at_death_rows(dead, []) == []


def test_summarize_by_kill_reason_computes_median_and_counts():
    rows = [
        krs.KillReasonSpendRow("p1", None, "futility_efficacy", "2021-01-01", 100.0, None),
        krs.KillReasonSpendRow("p2", None, "futility_efficacy", "2021-01-01", 300.0, None),
        krs.KillReasonSpendRow("p3", None, "strategic_portfolio", "2021-01-01", 10.0, None),
        krs.KillReasonSpendRow("p4", None, "strategic_portfolio", "2021-01-01", None, None),  # no spend data
    ]
    summary = krs.summarize_by_kill_reason(rows)

    assert summary["futility_efficacy"]["n"] == 2
    assert summary["futility_efficacy"]["n_with_spend_data"] == 2
    assert summary["futility_efficacy"]["median_spend"] == 200.0

    assert summary["strategic_portfolio"]["n"] == 2
    assert summary["strategic_portfolio"]["n_with_spend_data"] == 1
    assert summary["strategic_portfolio"]["median_spend"] == 10.0


def test_summarize_by_kill_reason_handles_reason_with_zero_spend_data():
    rows = [krs.KillReasonSpendRow("p1", None, "unknown_silent", "2021-01-01", None, None)]
    summary = krs.summarize_by_kill_reason(rows)
    assert summary["unknown_silent"]["n"] == 1
    assert summary["unknown_silent"]["n_with_spend_data"] == 0
    assert summary["unknown_silent"]["median_spend"] is None
