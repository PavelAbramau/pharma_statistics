"""Tests for finance/cost_model.py — the synthetic cost index must only
ever resolve trial state through the time-cut versioned-history path
(never a current-state read) for the monthly series, and site count must
stay a static, separately-exposed factor, never inside that series."""
from __future__ import annotations

import functools
import json
from datetime import date, datetime, timezone

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.finance import cost_model as cm


def _study(phase="PHASE2", enrollment=100, start="2020-01-01", locations=None):
    ps = {
        "statusModule": {"startDateStruct": {"date": start}},
        "designModule": {"phases": [phase], "enrollmentInfo": {"count": enrollment}},
    }
    if locations is not None:
        ps["contactsLocationsModule"] = {"locations": locations}
    return {"protocolSection": ps}


@pytest.fixture()
def con_and_env(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"
    db_path = tmp_path / "warehouse.duckdb"

    def save(snap_id, study, fetched_at):
        snap.save_snapshot("ctgov", snap_id, url="https://x", body=json.dumps(study),
                            fetched_at=fetched_at, raw_dir=raw_dir, manifest_db=manifest_db)

    # v1: posted 2020-02-01, enrollment 50, PHASE1
    save("NCT001:v1", _study(phase="PHASE1", enrollment=50, start="2020-01-01"),
         datetime(2020, 2, 1, tzinfo=timezone.utc))
    # v2: posted 2020-08-01, enrollment 100, PHASE2 (escalated)
    save("NCT001:v2", _study(phase="PHASE2", enrollment=100, start="2020-01-01"),
         datetime(2020, 8, 1, tzinfo=timezone.utc))
    # current-state fetch: has locations, but must NEVER be read for enrollment/phase
    save("NCT001", _study(phase="PHASE2", enrollment=999, start="2020-01-01", locations=[{"city": "Boston"}] * 12),
         datetime(2026, 9, 1, tzinfo=timezone.utc))

    monkeypatch.setattr(snap, "latest", functools.partial(snap.latest, manifest_db=manifest_db))

    con = duckdb.connect(str(db_path))
    con.execute(HISTORY_INDEX_SCHEMA)
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
            study_type, changed_modules, schema_hash, indexed_at)
           VALUES ('NCT001', 1, ?, ?, 'RECRUITING', 'INTERVENTIONAL', [], 'h', ?)""",
        [date(2020, 2, 1), date(2020, 1, 25), datetime(2020, 2, 1, tzinfo=timezone.utc)],
    )
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
            study_type, changed_modules, schema_hash, indexed_at)
           VALUES ('NCT001', 2, ?, ?, 'RECRUITING', 'INTERVENTIONAL', [], 'h', ?)""",
        [date(2020, 8, 1), date(2020, 7, 25), datetime(2020, 8, 1, tzinfo=timezone.utc)],
    )
    yield con
    con.close()


def test_resolve_trial_state_as_of_uses_the_right_version(con_and_env):
    con = con_and_env
    state = cm.resolve_trial_state_as_of("NCT001", date(2020, 5, 1), con)
    assert state.version == 1
    assert state.phase == "PHASE1"
    assert state.enrollment_count == 50


def test_resolve_trial_state_as_of_picks_up_later_version_once_posted(con_and_env):
    con = con_and_env
    state = cm.resolve_trial_state_as_of("NCT001", date(2020, 9, 1), con)
    assert state.version == 2
    assert state.phase == "PHASE2"
    assert state.enrollment_count == 100


def test_resolve_trial_state_as_of_none_before_any_version_posted(con_and_env):
    con = con_and_env
    state = cm.resolve_trial_state_as_of("NCT001", date(2020, 1, 1), con)
    assert state is None


def test_resolve_trial_state_as_of_never_reads_current_state_fetch(con_and_env):
    """The current-state snapshot claims enrollment=999 and PHASE2 as of
    'today' -- resolving at a date AFTER v2 must still return v2's real
    values (100), never the current-state fetch's inflated 999."""
    con = con_and_env
    state = cm.resolve_trial_state_as_of("NCT001", date(2026, 1, 1), con)
    assert state.enrollment_count == 100
    assert state.source == "versioned:v2"


def test_elapsed_months_zero_before_start():
    assert cm.elapsed_months(date(2021, 1, 1), date(2020, 1, 1)) == 0.0


def test_elapsed_months_positive_after_start():
    months = cm.elapsed_months(date(2020, 1, 1), date(2020, 7, 1))
    assert 5.5 < months < 6.5


def test_trial_cost_index_as_of_uses_phase_weight_and_elapsed_time(con_and_env):
    con = con_and_env
    idx = cm.trial_cost_index_as_of("NCT001", date(2020, 9, 1), con)
    expected = cm.PHASE_COST_WEIGHT["PHASE2"] * 100 * cm.elapsed_months(date(2020, 1, 1), date(2020, 9, 1))
    assert idx == pytest.approx(expected)


def test_trial_cost_index_as_of_zero_when_unresolvable(con_and_env):
    con = con_and_env
    assert cm.trial_cost_index_as_of("NCT001", date(2019, 1, 1), con) == 0.0


def test_current_site_count_reads_current_state_only(con_and_env):
    con = con_and_env
    program = {"program_id": "p1", "trials": [{"nct_id": "NCT001"}]}
    assert cm.current_site_count(program) == 12


def test_monthly_series_never_includes_site_count(con_and_env):
    con = con_and_env
    program = {"program_id": "p1", "trials": [{"nct_id": "NCT001"}]}
    series = cm.monthly_cost_index_series(program, con, end=date(2020, 9, 1))
    assert series
    for point in series:
        assert "site_count" not in point
        assert point["knowability_date"] == point["as_of"]


def test_program_cost_index_snapshot_includes_site_count(con_and_env):
    con = con_and_env
    program = {"program_id": "p1", "trials": [{"nct_id": "NCT001"}]}
    snapshot = cm.program_cost_index_snapshot(program, con, as_of=date(2020, 9, 1))
    assert snapshot["site_count"] == 12
    assert snapshot["cost_index"] == pytest.approx(snapshot["base_index"] * cm.site_count_factor(12))


def test_site_count_factor_neutral_when_unknown():
    assert cm.site_count_factor(None) == 1.0
