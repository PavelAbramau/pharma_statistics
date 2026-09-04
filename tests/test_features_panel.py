"""Tests for features/panel.py and features/trial_asof.py — the program
x month panel must never leak a later or current-state value into an
earlier month's row."""
from __future__ import annotations

import functools
import json
from datetime import date, datetime, timezone

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.features import panel as fp
from pharma_stats.features import trial_asof


def _study(status="RECRUITING", phase="PHASE2", enrollment=100, start="2020-01-01", locations=None):
    ps = {
        "statusModule": {
            "overallStatus": status,
            "startDateStruct": {"date": start},
            "lastUpdatePostDateStruct": {"date": start, "type": "ACTUAL"},
            "statusVerifiedDate": start[:7],
        },
        "designModule": {"phases": [phase], "enrollmentInfo": {"count": enrollment, "type": "ACTUAL"}},
        "conditionsModule": {"conditions": ["Breast Cancer"]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Oncology"}},
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

    save("NCT001:v1", _study(status="RECRUITING", enrollment=50, start="2020-01-01"),
         datetime(2020, 2, 1, tzinfo=timezone.utc))
    save("NCT001:v2", _study(status="ACTIVE_NOT_RECRUITING", enrollment=100, start="2020-01-01"),
         datetime(2020, 8, 1, tzinfo=timezone.utc))
    # current-state fetch: a much later, inflated view -- must never leak into an early month
    save("NCT001", _study(status="TERMINATED", enrollment=999, start="2020-01-01", locations=[{"c": 1}] * 5),
         datetime(2026, 9, 1, tzinfo=timezone.utc))

    monkeypatch.setattr(snap, "latest", functools.partial(snap.latest, manifest_db=manifest_db))

    con = duckdb.connect(str(db_path))
    con.execute(HISTORY_INDEX_SCHEMA)
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
            study_type, changed_modules, schema_hash, indexed_at)
           VALUES ('NCT001', 1, ?, ?, 'RECRUITING', 'INTERVENTIONAL', ['Contacts/Locations'], 'h', ?)""",
        [date(2020, 2, 1), date(2020, 1, 25), datetime(2020, 2, 1, tzinfo=timezone.utc)],
    )
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
            study_type, changed_modules, schema_hash, indexed_at)
           VALUES ('NCT001', 2, ?, ?, 'ACTIVE_NOT_RECRUITING', 'INTERVENTIONAL', ['Study Status'], 'h', ?)""",
        [date(2020, 8, 1), date(2020, 7, 25), datetime(2020, 8, 1, tzinfo=timezone.utc)],
    )
    yield con
    con.close()


def test_resolve_trial_summary_as_of_never_leaks_current_state(con_and_env):
    con = con_and_env
    summary = trial_asof.resolve_trial_summary_as_of("NCT001", date(2020, 9, 1), con)
    assert summary.status == "ACTIVE_NOT_RECRUITING"  # v2's real status, never TERMINATED (current-state)
    assert summary.enrollment_count == 100  # never 999
    assert summary.source_snapshot == "versioned:v2"


def test_resolve_trial_summary_as_of_carries_forward_when_target_version_body_missing(con_and_env):
    """Real bug found 2026-09-04: the backfill's selective body-fetch
    (history/orchestrator.py) skips any version whose changed_modules
    doesn't intersect the signal labels -- 56.6% of version>0 rows on
    real data have no fetched body at all. A naive lookup for "the exact
    version implied by posted_date <= as_of" would return None here even
    though v2's body (still valid -- v3 changed nothing signal-relevant)
    is on disk."""
    con = con_and_env
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
            study_type, changed_modules, schema_hash, indexed_at)
           VALUES ('NCT001', 3, ?, ?, 'ACTIVE_NOT_RECRUITING', 'INTERVENTIONAL', ['Contacts/Locations'], 'h', ?)""",
        [date(2021, 1, 1), date(2020, 12, 25), datetime(2021, 1, 1, tzinfo=timezone.utc)],
    )
    # v3's body is deliberately never saved -- exactly the "not fetched" case.
    summary = trial_asof.resolve_trial_summary_as_of("NCT001", date(2021, 3, 1), con)
    assert summary is not None
    assert summary.source_snapshot == "versioned:v2"  # carried forward, not None
    assert summary.status == "ACTIVE_NOT_RECRUITING"  # v2's real value, not a leak from current-state either


def test_resolve_trial_summary_as_of_truncates_history_to_asof(con_and_env):
    con = con_and_env
    summary = trial_asof.resolve_trial_summary_as_of("NCT001", date(2020, 3, 1), con)
    assert len(summary.history) == 1  # only v1's row, v2 posted later
    assert summary.history[0]["version"] == 1


def test_build_program_month_panel_columns_are_all_registered(con_and_env):
    con = con_and_env
    program = {"program_id": "p1", "proposed_name": "Trastuzumab deruxtecan", "synonyms": [],
               "trials": [{"nct_id": "NCT001"}]}
    rows = fp.build_program_month_panel(program, con, end=date(2020, 9, 1))
    assert rows  # non-empty
    assert rows[0]["as_of"] == "2020-02-01"  # starts at the first indexed version's month
    assert rows[-1]["as_of"] == "2020-09-01"
    for row in rows:
        assert row["target"] == "ERBB2"  # antibody-stem dictionary, static across all rows
        assert row["payload_chemotype"] == "camptothecin_topo1"


def test_build_program_month_panel_silence_score_uses_asof_state_only(con_and_env):
    con = con_and_env
    program = {"program_id": "p1", "proposed_name": "XL114", "synonyms": [], "trials": [{"nct_id": "NCT001"}]}
    rows = fp.build_program_month_panel(program, con, end=date(2020, 9, 1))
    by_month = {r["as_of"]: r for r in rows}
    # a RECRUITING trial resolved as of an early month should not carry the
    # TERMINATED status the current-state fetch injects
    early = by_month["2020-03-01"]
    assert early["silence_score_asof"] is not None


def test_build_program_month_panel_empty_when_no_history(con_and_env):
    con = con_and_env
    program = {"program_id": "p2", "proposed_name": "Nothing", "synonyms": [], "trials": [{"nct_id": "NCT999"}]}
    assert fp.build_program_month_panel(program, con) == []
