"""Tests for finance/panel.py — the money-layer feature panel is the
first real consumer of financial_events, and its whole value rests on the
leakage contract in audit/leakage.md: a feature row's knowability_date
must never exceed its own as_of."""
from __future__ import annotations

from datetime import date

from pharma_stats.finance import panel as fpanel
from pharma_stats.finance import store as fstore


def _event(pid, event_type, month, value):
    return fstore.FinancialEvent(
        subject_type="program", subject_id=pid, event_date=month,
        event_type=event_type, detail="test", source="test", value=value,
    )


def test_panel_empty_when_no_financial_events(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    assert fpanel.build_money_layer_panel(db) == []


def test_cumulative_spend_accumulates_across_months(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 2, 1), 15.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 3, 1), 5.0),
    ], warehouse_db=db)

    panel = fpanel.build_money_layer_panel(db)
    by_month = {r["as_of"]: r["estimated_cumulative_spend"] for r in panel}
    assert by_month["2021-01-01"] == 10.0
    assert by_month["2021-02-01"] == 25.0
    assert by_month["2021-03-01"] == 30.0


def test_conviction_ratio_only_set_for_months_with_its_own_event(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 2, 1), 10.0),
        _event("p1", fpanel.CONVICTION_EVENT_TYPE, date(2021, 2, 1), 2.5),
    ], warehouse_db=db)

    panel = fpanel.build_money_layer_panel(db)
    by_month = {r["as_of"]: r["conviction_ratio"] for r in panel}
    assert by_month["2021-01-01"] is None
    assert by_month["2021-02-01"] == 2.5


def test_knowability_date_never_exceeds_as_of(tmp_path):
    """The leakage invariant audit/leakage.md and audit/features.py both
    depend on: every row's knowability_date equals its own as_of."""
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 2, 1), 20.0),
        _event("p2", fpanel.CONVICTION_EVENT_TYPE, date(2021, 5, 1), 1.1),
    ], warehouse_db=db)

    panel = fpanel.build_money_layer_panel(db)
    assert panel
    for row in panel:
        assert row["knowability_date"] == row["as_of"]


def test_month_with_no_event_produces_no_row(tmp_path):
    """A gap month (no cost or conviction event at all) must never be
    synthesized as a row — that would conflate "unknown" with "zero"."""
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 3, 1), 10.0),
    ], warehouse_db=db)

    panel = fpanel.build_money_layer_panel(db)
    months = {r["as_of"] for r in panel if r["program_id"] == "p1"}
    assert months == {"2021-01-01", "2021-03-01"}


def test_value_as_of_forward_fills_from_latest_point_at_or_before(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 2, 1), 5.0),
    ], warehouse_db=db)
    panel = fpanel.build_money_layer_panel(db)
    by_program = fpanel.index_by_program(panel)
    rows = by_program["p1"]

    # a date between the two points sees only the first
    assert fpanel.value_as_of(rows, date(2021, 1, 15), "estimated_cumulative_spend") == 10.0
    # a date well after both points sees the full cumulative sum
    assert fpanel.value_as_of(rows, date(2021, 6, 1), "estimated_cumulative_spend") == 15.0
    # a date before any point resolves to None, never 0
    assert fpanel.value_as_of(rows, date(2020, 1, 1), "estimated_cumulative_spend") is None


def test_value_as_of_never_uses_a_point_after_the_given_date(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 10.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 12, 1), 1000.0),
    ], warehouse_db=db)
    panel = fpanel.build_money_layer_panel(db)
    rows = fpanel.index_by_program(panel)["p1"]

    # resolving mid-year must never see the December point
    assert fpanel.value_as_of(rows, date(2021, 6, 1), "estimated_cumulative_spend") == 10.0


def test_index_by_program_groups_and_preserves_order(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    fstore.append_records([
        _event("p2", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 1.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 2, 1), 2.0),
        _event("p1", fpanel.COST_EVENT_TYPE, date(2021, 1, 1), 1.0),
    ], warehouse_db=db)
    panel = fpanel.build_money_layer_panel(db)
    idx = fpanel.index_by_program(panel)
    assert set(idx.keys()) == {"p1", "p2"}
    assert [r["as_of"] for r in idx["p1"]] == ["2021-01-01", "2021-02-01"]
