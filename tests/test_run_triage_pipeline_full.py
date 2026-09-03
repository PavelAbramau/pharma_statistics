"""Regression test for the $37.59-vs-$5-cap incident: run_triage_pipeline.py
--full's --max-spend must abort mid-run against REAL accumulated cost
across BOTH Layer 2 and Layer 3, not just Layer 2 — the previous
Layer 3 code ran the whole queue as one unchunked, uncapped call."""
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
from pharma_stats.labelling import store
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_triage_pipeline.py"

N_CANDIDATES = 25  # > LAYER3_CHUNK_SIZE (20), so this forces 2 Layer 3 chunks


def _load_script():
    spec = importlib.util.spec_from_file_location("run_triage_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_triage_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script():
    mod = _load_script()
    yield mod
    sys.modules.pop("run_triage_pipeline", None)


def _study(nct_id):
    # No descriptionModule / interventions text at all -> zero-text bypass
    # (pipeline.partition_by_text_evidence) -> every candidate routes
    # straight into the Layer 3 queue, letting this test isolate Layer 3's
    # own chunking/cost-check without needing to mock Layer 2 too.
    return {
        "protocolSection": {
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2020-01-01"},
            },
            "designModule": {"phases": ["PHASE1"], "enrollmentInfo": {"count": 10, "type": "ESTIMATED"}},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Oncology"}},
            "conditionsModule": {"conditions": ["Solid Tumor"]},
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

    nct_ids = [f"NCT{i:08d}" for i in range(N_CANDIDATES)]
    for nct_id in nct_ids:
        save(nct_id, _study(nct_id))
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
    now = datetime(2026, 8, 19, tzinfo=timezone.utc)
    rows = []
    for i, nct_id in enumerate(nct_ids):
        rows.append((
            f"cand_{i}", f"XL{1000 + i}", [], "[]", 1, [nct_id], None, None, [], False, False,
            "pattern_match", "dev_code", "XL", "unreviewed",
        ))
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
    con.executemany("INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.close()
    return db_path


def test_full_run_max_spend_aborts_mid_layer3_not_after_everything(warehouse, monkeypatch, tmp_path, script, capsys):
    from pharma_stats.triage import layer3

    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    gold_path = tmp_path / "labels.jsonl"
    staging_path = tmp_path / "staged.jsonl"
    sample_path = tmp_path / "validation_sample.json"
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(staging, "STAGING_PATH", staging_path)
    monkeypatch.setattr(tval, "VALIDATION_SAMPLE_PATH", sample_path)
    monkeypatch.setattr(script, "WAREHOUSE_DB", warehouse)
    # A validation sample already existing is orthogonal to what this test
    # checks (Layer 3 chunking/cost-abort) — stub it out rather than
    # constructing synthetic staged decisions that satisfy the real
    # stratification feasibility checks too.
    monkeypatch.setattr(script.tval, "load_validation_sample", lambda: [{"program_id": "existing"}])

    calls = []

    def fake_run_layer3(chunk, *, model):
        calls.append(len(chunk))
        answers = {c["program_id"]: layer3.Layer3Answer(c["program_id"], c["name"], "no", None, None)
                   for c in chunk}
        # $6/chunk: after chunk 1, total is $6 (>5) -- the check fires
        # BEFORE starting the next chunk, so chunk 2 must never run.
        log = {"usage": {"cost_usd": 6.0}}
        return answers, log

    monkeypatch.setattr(script.layer3, "run_layer3", fake_run_layer3)
    monkeypatch.setattr(sys, "argv", ["run_triage_pipeline.py", "--full", "--max-spend", "5"])
    script.main()

    out = capsys.readouterr().out
    assert len(calls) == 1  # only the first 20-candidate chunk ran
    assert calls[0] == 20
    assert "stopping Layer 3 before chunk 2/2" in out
    assert "5 candidate(s) not yet run" in out

    staged = staging.load_records(staging_path)
    layer3_staged = [r for r in staged if r.get("layer") == 3]
    assert len(layer3_staged) == 20  # only the first chunk's candidates got staged
