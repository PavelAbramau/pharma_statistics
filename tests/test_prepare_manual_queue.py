"""Tests for scripts/prepare_manual_queue.py — installs the disposition
order (likely_reject -> ambiguous -> confirmed_adc) as the session's
manual queue. Reuses test_labelling.py's warehouse fixture pattern."""
from __future__ import annotations

import functools
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.history.orchestrator import BACKFILL_QUEUE_SCHEMA
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_manual_queue.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_manual_queue", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_manual_queue"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script():
    mod = _load_script()
    yield mod
    sys.modules.pop("prepare_manual_queue", None)


def _study(overall_status="TERMINATED"):
    return {
        "protocolSection": {
            "statusModule": {
                "overallStatus": overall_status,
                "completionDateStruct": {"date": "2020-01-01", "type": "ACTUAL"},
                "primaryCompletionDateStruct": {"date": "2020-01-01", "type": "ACTUAL"},
                "lastUpdatePostDateStruct": {"date": "2020-06-01", "type": "ACTUAL"},
                "statusVerifiedDate": "2020-06",
                "startDateStruct": {"date": "2018-01-01"},
            },
            "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 100, "type": "ACTUAL"}},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Oncology"}},
            "conditionsModule": {"conditions": ["Breast Cancer"]},
        },
        "hasResults": False,
    }


@pytest.fixture()
def warehouse(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"
    db_path = tmp_path / "warehouse.duckdb"

    def save(nct_id, study):
        snap.save_snapshot("ctgov", nct_id, url="https://x", body=json.dumps(study),
                            fetched_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
                            raw_dir=raw_dir, manifest_db=manifest_db)

    save("NCT00000001", _study())
    save("NCT00000002", _study())
    monkeypatch.setattr(snap, "latest", functools.partial(snap.latest, manifest_db=manifest_db))

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE asset_candidates (
            candidate_id VARCHAR, proposed_name VARCHAR, synonyms VARCHAR[],
            sponsors_over_time JSON, trial_count INTEGER, nct_ids VARCHAR[],
            first_trial_start_date VARCHAR, last_trial_start_date VARCHAR,
            strategies VARCHAR[], ambiguous BOOLEAN, dev_code_only BOOLEAN,
            discovery_strategy VARCHAR, match_strength VARCHAR, matched_term VARCHAR,
            review_status VARCHAR
        )
    """)
    con.execute(HISTORY_INDEX_SCHEMA)
    con.execute(BACKFILL_QUEUE_SCHEMA)
    rows = [
        ("cand_a", "DrugA", [], "[]", 1, ["NCT00000001"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
        ("cand_b", "DrugB", [], "[]", 1, ["NCT00000002"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
    ]
    con.executemany("INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    now = datetime(2026, 8, 19, tzinfo=timezone.utc)
    for nct_id in ["NCT00000001", "NCT00000002"]:
        con.execute(
            """INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
                study_type, changed_modules, schema_hash, indexed_at)
               VALUES (?, 0, ?, ?, 'RECRUITING', 'INTERVENTIONAL', [], 'testhash', ?)""",
            [nct_id, date(2019, 1, 1), date(2019, 1, 1), now],
        )
        con.execute(
            """INSERT INTO backfill_queue (nct_id, priority_tier, priority_key, status,
                latest_version_indexed, bodies_fetched_through_version, updated_at)
               VALUES (?, 1, 0.0, 'done', 0, -1, ?)""",
            [nct_id, now],
        )
    con.close()
    return db_path


def test_prepare_manual_queue_installs_disposition_order(warehouse, monkeypatch, tmp_path, script):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    session_path = tmp_path / "session.json"
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(script, "WAREHOUSE_DB", warehouse)

    from pharma_stats.labelling import triage_serve
    monkeypatch.setattr(triage_serve, "REOPEN_PATH", tmp_path / "reopen.json")

    from pharma_stats.triage import evidence as tev
    monkeypatch.setattr(
        tev, "build_layer2_evidence",
        lambda program, con: {"text_snippets": ["administered orally as a tablet"]},
    )

    sys_argv_backup = sys.argv
    sys.argv = ["prepare_manual_queue.py"]
    try:
        script.main()
    finally:
        sys.argv = sys_argv_backup

    session = q.load_session(session_path)
    assert set(session["order"]) == {"cand_a", "cand_b"}


def test_prepare_manual_queue_dry_run_does_not_touch_session(warehouse, monkeypatch, tmp_path, script, capsys):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    session_path = tmp_path / "session.json"
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(script, "WAREHOUSE_DB", warehouse)

    from pharma_stats.labelling import triage_serve
    monkeypatch.setattr(triage_serve, "REOPEN_PATH", tmp_path / "reopen.json")

    from pharma_stats.triage import evidence as tev
    monkeypatch.setattr(tev, "build_layer2_evidence", lambda program, con: {"text_snippets": []})

    assert not session_path.exists()

    sys_argv_backup = sys.argv
    sys.argv = ["prepare_manual_queue.py", "--dry-run"]
    try:
        script.main()
    finally:
        sys.argv = sys_argv_backup

    out = capsys.readouterr().out
    assert "--dry-run: session not modified" in out
    assert not session_path.exists()
