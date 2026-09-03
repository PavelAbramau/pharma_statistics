"""Tests for finance/store.py — financial_events, a separate table from
differ's evidence_events (different grain: subject_type/subject_id, not
nct_id/version)."""
from __future__ import annotations

from datetime import date

from pharma_stats.finance import store as fstore


def test_load_records_empty_when_table_never_created(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    assert fstore.load_records(db) == []


def test_append_and_load_round_trip(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    records = [
        fstore.FinancialEvent(
            subject_type="program", subject_id="p1", event_date=date(2021, 3, 1),
            event_type="synthetic_cost_index_monthly", detail="test",
            source="sertkaya_2016_synthetic", value=123.4,
        ),
    ]
    n = fstore.append_records(records, warehouse_db=db)
    assert n == 1

    loaded = fstore.load_records(db)
    assert len(loaded) == 1
    assert loaded[0]["subject_id"] == "p1"
    assert loaded[0]["value"] == 123.4
    assert loaded[0]["event_date"] == date(2021, 3, 1)


def test_load_records_filters_by_event_type(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    records = [
        fstore.FinancialEvent(subject_type="program", subject_id="p1", event_date=date(2021, 1, 1),
                               event_type="synthetic_cost_index_monthly", detail="a", source="s"),
        fstore.FinancialEvent(subject_type="program", subject_id="p1", event_date=date(2021, 1, 1),
                               event_type="conviction_ratio_monthly", detail="b", source="s"),
    ]
    fstore.append_records(records, warehouse_db=db)
    only_cost = fstore.load_records(db, event_type="synthetic_cost_index_monthly")
    assert len(only_cost) == 1
    assert only_cost[0]["detail"] == "a"


def test_append_records_empty_list_is_noop(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    assert fstore.append_records([], warehouse_db=db) == 0
    assert fstore.load_records(db) == []
