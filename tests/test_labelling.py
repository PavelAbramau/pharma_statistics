"""Tests for the labelling app: provisional program scoring, the
stratified queue, gold JSONL validation, and the FastAPI routes end to
end (against real fixture snapshots, no mocking of the scoring logic
itself)."""
from __future__ import annotations

import functools
import json
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store


def _study(overall_status, why_stopped=None, enrollment_type="ACTUAL",
           enrollment_count=100, has_results=False, completion_date="2020-01-01",
           last_update="2020-06-01", status_verified="2020-06"):
    return {
        "protocolSection": {
            "statusModule": {
                "overallStatus": overall_status,
                "whyStopped": why_stopped,
                "completionDateStruct": {"date": completion_date, "type": "ACTUAL"},
                "primaryCompletionDateStruct": {"date": completion_date, "type": "ACTUAL"},
                "lastUpdatePostDateStruct": {"date": last_update, "type": "ACTUAL"},
                "statusVerifiedDate": status_verified,
                "startDateStruct": {"date": "2018-01-01"},
            },
            "designModule": {
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": enrollment_count, "type": enrollment_type},
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Oncology"}},
            "conditionsModule": {"conditions": ["Breast Cancer"]},
        },
        "hasResults": has_results,
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

    save("NCT00000001", _study("TERMINATED", why_stopped="Business decision."))
    save("NCT00000002", _study(
        "TERMINATED",
        why_stopped="The independent data monitoring committee recommended stopping "
                    "the trial after an interim analysis showed the experimental arm "
                    "did not meet the pre-specified efficacy threshold.",
    ))
    save("NCT00000003", _study("UNKNOWN"))
    save("NCT00000004", _study("RECRUITING", last_update="2020-01-01", status_verified="2019-01"))
    save("NCT00000005", _study("COMPLETED", has_results=False, completion_date="2018-01-01"))

    # provisional_programs.py resolves raw bodies via snap.latest(...) with
    # its default manifest_db; pin that default to this test's manifest so
    # nothing touches the real project's data/manifest.duckdb.
    monkeypatch.setattr(snap, "latest", functools.partial(snap.latest, manifest_db=manifest_db))

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE asset_candidates (
            candidate_id VARCHAR, proposed_name VARCHAR, synonyms VARCHAR[],
            sponsors_over_time JSON, trial_count INTEGER, nct_ids VARCHAR[],
            first_trial_start_date VARCHAR, last_trial_start_date VARCHAR,
            strategies VARCHAR[], ambiguous BOOLEAN, dev_code_only BOOLEAN, review_status VARCHAR
        )
    """)
    con.execute(HISTORY_INDEX_SCHEMA)  # unused by these fixtures but queried by summarize_trial
    rows = [
        ("cand_vague", "VagueMab", [], "[]", 1, ["NCT00000001"], None, None, [], False, False, "unreviewed"),
        ("cand_stated", "StatedMab", [], "[]", 1, ["NCT00000002"], None, None, [], False, False, "unreviewed"),
        ("cand_unknown", "UnknownMab", [], "[]", 1, ["NCT00000003"], None, None, [], False, False, "unreviewed"),
        ("cand_stale", "StaleMab", [], "[]", 1, ["NCT00000004"], None, None, [], False, False, "unreviewed"),
        ("cand_noresults", "NoResultsMab", [], "[]", 1, ["NCT00000005"], None, None, [], False, False, "unreviewed"),
    ]
    con.executemany(
        "INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.close()

    return db_path


def test_scoring_and_archetypes(warehouse):
    con = duckdb.connect(str(warehouse), read_only=True)
    programs = pp.build_all_programs(con, as_of=date(2026, 8, 19))
    con.close()
    by_id = {p["candidate_id"]: p for p in programs}

    assert by_id["cand_vague"]["primary_archetype"] == "registry_terminated_vague_reason"
    assert by_id["cand_stated"]["primary_archetype"] == "registry_terminated_stated_reason"
    assert by_id["cand_unknown"]["primary_archetype"] == "unknown_status"
    assert by_id["cand_noresults"]["primary_archetype"] == "completed_no_results"

    # a still-"RECRUITING" trial silently stale for years should score higher
    # than an openly, concretely terminated one
    assert by_id["cand_stale"]["silence_score"] > by_id["cand_stated"]["silence_score"]
    assert 0 <= by_id["cand_vague"]["silence_score"] <= 100


def test_materialize_and_load_roundtrip(warehouse):
    n = pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    assert n == 5
    loaded = pp.load_materialized(warehouse_db=warehouse)
    assert len(loaded) == 5
    assert all(p["provisional"] is True for p in loaded)
    assert all(p["indication_code"] == "UNSPECIFIED" for p in loaded)


def test_stratified_order_interleaves_bands(warehouse):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    programs = pp.load_materialized(warehouse_db=warehouse)
    order = q.build_stratified_order(programs, exclude_ids=set())
    assert set(order) == {p["program_id"] for p in programs}
    # not sorted by score (that's the whole point)
    scores = {p["program_id"]: p["silence_score"] for p in programs}
    assert [scores[pid] for pid in order] != sorted(scores[pid] for pid in order)[::-1] or len(order) < 2


def test_validate_label_payload_requires_dead_confirmed_fields():
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({"action": "label", "status": "dead_confirmed", "confidence": "high"})

    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            "action": "label", "status": "dead_confirmed", "confidence": "high",
            "kill_reason": "futility_efficacy", "label_evidence_date": "2024-01-01",
        })  # missing confirmation date / never_publicly_confirmed

    store.validate_label_payload({
        "action": "label", "status": "dead_confirmed", "confidence": "high",
        "kill_reason": "futility_efficacy", "label_evidence_date": "2024-01-01",
        "never_publicly_confirmed": True,
    })  # should not raise

    store.validate_label_payload({"action": "skip"})  # should not raise
    store.validate_label_payload({"action": "flag_invalid"})  # should not raise


def test_append_only_gold_store(tmp_path):
    path = tmp_path / "labels.jsonl"
    r1 = store.build_record(
        {"action": "label", "program_id": "p1", "status": "active", "confidence": "high"},
        session_id="s1", served_stratum={"band": 0, "archetype": "other", "silence_score": 5},
    )
    store.append_record(r1, path=path)
    r2 = store.build_record(
        {"action": "label", "program_id": "p1", "status": "dormant_suspected", "confidence": "medium"},
        session_id="s1", served_stratum={"band": 0, "archetype": "other", "silence_score": 5},
    )
    store.append_record(r2, path=path)

    records = store.load_records(path)
    assert len(records) == 2  # revision is a new line, not an edit
    latest = store.latest_by_program(records)
    assert latest["p1"]["status"] == "dormant_suspected"


def test_repeat_probe_fires_around_ten_percent():
    session = {"order": [f"p{i}" for i in range(50)], "total_served": 0,
               "pending_serve": {}, "served_log": []}
    labelled = {"p_already_labelled"}
    repeats = 0
    for _ in range(30):
        pid, is_repeat = q.pop_next(session, labelled)
        if is_repeat:
            repeats += 1
    assert repeats == 3  # every 10th serve, deterministic


def test_app_end_to_end(warehouse, monkeypatch, tmp_path):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))

    gold_path = tmp_path / "labels.jsonl"
    session_path = tmp_path / "session.json"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)

    import pharma_stats.labelling.app as appmod
    appmod._state.clear()
    appmod._init_state()

    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)

    r = client.get("/api/next?blind=true")
    body = r.json()
    assert body["done"] is False
    assert "silence_score" not in body["program"]  # blind: hidden pre-judgement
    token = body["serve_token"]

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "status": "dead_confirmed", "confidence": "high",
    })
    assert r.status_code == 422  # missing kill_reason / dates

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "status": "active", "confidence": "low",
        "blind": True, "seconds_spent": 12.5,
    })
    assert r.status_code == 200
    assert r.json()["reveal"]["silence_score"] is not None  # revealed after save

    records = store.load_records(gold_path)
    assert len(records) == 1
    assert records[0]["blind"] is True
    assert records[0]["app_version"]

    r = client.get("/api/session")
    stats = r.json()
    assert stats["labelled_count"] == 1
