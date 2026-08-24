"""End-to-end run of the implemented pipeline against in-memory mock data.

Covers the layers that exist today: snapshot store → candidate clustering →
warehouse load → history-index backfill. No ClinicalTrials.gov calls.
"""
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.discovery.candidates import Mention, build_candidate_table
from pharma_stats.history.index import module_filter_stats
from pharma_stats.history.orchestrator import run_backfill


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"
    return raw_dir, manifest_db


class FakeCtgovClient:
    def __init__(self, histories: dict[str, list[dict]]):
        self._histories = histories
        self.body_fetch_calls: list[tuple[str, int]] = []

    def get_history(self, nct_id: str) -> list[dict]:
        return self._histories[nct_id]

    def get_study_version(self, nct_id: str, version: int) -> dict:
        self.body_fetch_calls.append((nct_id, version))
        return {"protocolSection": {"identificationModule": {"nctId": nct_id}}}


def _history_entry(version, modules, d="2020-01-01", status="RECRUITING"):
    return {
        "version": version,
        "date": d,
        "status": status,
        "studyType": "INTERVENTIONAL",
        "moduleLabels": modules,
        "lastUpdateSubmitQcDate": d,
    }


def _mention(nct_id, name, strategy="pattern_match", strength="suffix", **extra) -> Mention:
    defaults = dict(
        nct_id=nct_id,
        intervention_name=name,
        other_names=[],
        intervention_type="DRUG",
        strategy=strategy,
        lead_sponsor="Mock Pharma",
        lead_sponsor_class="INDUSTRY",
        study_start_date=date(2019, 6, 1),
        overall_status="RECRUITING",
        brief_title=f"Mock trial {nct_id}",
        match_strength=strength,
    )
    defaults.update(extra)
    return Mention(**defaults)


def test_mock_pipeline_snapshot_candidates_backfill(store, tmp_path: Path):
    raw_dir, manifest_db = store

    # --- 1. immutable snapshots of two mock study records -----------------
    body_a = '{"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}'
    body_b = '{"protocolSection": {"identificationModule": {"nctId": "NCT00000002"}}}'
    snap.save_snapshot(
        "ctgov", "NCT00000001", "https://example.test/NCT00000001", body_a,
        fetched_at=_dt("2024-01-15T00:00:00"), raw_dir=raw_dir, manifest_db=manifest_db,
    )
    snap.save_snapshot(
        "ctgov", "NCT00000002", "https://example.test/NCT00000002", body_b,
        fetched_at=_dt("2024-01-15T00:00:00"), raw_dir=raw_dir, manifest_db=manifest_db,
    )

    as_of = snap.get_as_of("ctgov", "NCT00000001", date(2024, 6, 1), manifest_db=manifest_db)
    assert as_of is not None
    assert as_of.body_json()["protocolSection"]["identificationModule"]["nctId"] == "NCT00000001"
    assert snap.get_as_of("ctgov", "NCT00000001", date(2023, 1, 1), manifest_db=manifest_db) is None

    manifest_db.unlink()
    assert snap.rebuild_manifest(raw_dir=raw_dir, manifest_db=manifest_db) == 2

    # --- 2. cluster mock mentions into candidate assets -------------------
    mentions = [
        _mention("NCT00000001", "trastuzumab deruxtecan", other_names=["T-DXd"]),
        _mention(
            "NCT00000001", "Enhertu", other_names=["trastuzumab deruxtecan"],
            strategy="seed_expansion", strength="seed",
        ),
        _mention("NCT00000002", "enfortumab vedotin", other_names=["Padcev"]),
        _mention(
            "NCT00000003", "ABBV-011", strategy="sponsor_expansion", strength="dev_code",
        ),
    ]
    candidates = build_candidate_table(mentions)
    by_name = {c.proposed_name.lower(): c for c in candidates}
    assert len(candidates) == 3
    tdxd = next(c for c in candidates if "deruxtecan" in c.proposed_name.lower() or "enhertu" in c.proposed_name.lower())
    assert tdxd.trial_count == 1
    assert tdxd.nct_ids == ["NCT00000001"]
    assert not tdxd.dev_code_only
    abb = next(c for c in candidates if c.proposed_name == "ABBV-011")
    assert abb.dev_code_only

    # --- 3. load warehouse the same way the live script does --------------
    warehouse = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(warehouse))
    con.execute(
        """
        CREATE TABLE asset_candidates (
            candidate_id VARCHAR PRIMARY KEY,
            proposed_name VARCHAR,
            synonyms VARCHAR[],
            trial_count INTEGER,
            nct_ids VARCHAR[],
            strategies VARCHAR[],
            ambiguous BOOLEAN,
            dev_code_only BOOLEAN
        )
        """
    )
    for c in candidates:
        con.execute(
            "INSERT INTO asset_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                c.candidate_id, c.proposed_name, c.synonyms, c.trial_count,
                c.nct_ids, c.strategies, c.ambiguous, c.dev_code_only,
            ],
        )

    universe = {nct for c in candidates for nct in c.nct_ids}
    assert universe == {"NCT00000001", "NCT00000002", "NCT00000003"}

    # --- 4. history backfill with a fake CT.gov client --------------------
    histories = {
        "NCT00000001": [
            _history_entry(0, []),
            _history_entry(1, ["Study Design"], d="2021-03-01"),
            _history_entry(2, ["Contacts/Locations"], d="2021-06-01"),
            _history_entry(3, ["Study Status", "Outcome Measures"], d="2022-01-01"),
        ],
        "NCT00000002": [
            _history_entry(0, []),
            _history_entry(1, ["Arms and Interventions"], d="2020-08-01"),
        ],
        "NCT00000003": [
            _history_entry(0, []),
        ],
    }
    client = FakeCtgovClient(histories)
    signal = frozenset({"Study Design", "Study Status", "Outcome Measures", "Arms and Interventions"})
    result = run_backfill(client, con, universe, signal_labels=signal)

    assert result["errors"] == 0
    assert result["trials_done"] == 3
    assert result["remaining_in_queue"] == 0
    # v1 Study Design + v3 Study Status/Outcomes on NCT00000001, v1 arms on NCT00000002
    assert result["bodies_fetched"] == 3
    assert set(client.body_fetch_calls) == {
        ("NCT00000001", 1),
        ("NCT00000001", 3),
        ("NCT00000002", 1),
    }

    stats = module_filter_stats(con, signal)
    assert stats["total_trials"] == 3
    assert stats["signal_versions"] == 3

    gold_tier = con.execute(
        "SELECT nct_id FROM backfill_queue WHERE priority_tier = 0 ORDER BY nct_id"
    ).fetchall()
    # NCT00000001 and NCT00000002 belong to evidenced (not dev-code-only) candidates
    assert [r[0] for r in gold_tier] == ["NCT00000001", "NCT00000002"]

    assert any("deruxtecan" in n or "enhertu" in n for n in by_name)
    con.close()
