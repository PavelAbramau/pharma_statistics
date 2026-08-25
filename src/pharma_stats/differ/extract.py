"""Orchestrates the differ over the full corpus: every trial with 2+
fetched version bodies, adjacent-pair diffed, written to
warehouse.duckdb::evidence_events. Pure local computation — reads only
raw/ + history_index, no network.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import duckdb

from pharma_stats.config import MANIFEST_DB, WAREHOUSE_DB
from pharma_stats.differ.diff import diff_versions
from pharma_stats.differ.events import EVENT_TYPES, EvidenceEvent

EVIDENCE_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_events (
    event_id BIGINT,
    nct_id VARCHAR,
    from_version INTEGER,
    to_version INTEGER,
    event_date DATE,
    event_type VARCHAR,
    field VARCHAR,
    direction VARCHAR,
    from_value VARCHAR,
    to_value VARCHAR,
    detail VARCHAR,
    differ_version VARCHAR,
    extracted_at TIMESTAMP
)
"""


def _study_from_body(body: Any) -> dict:
    if isinstance(body, dict) and "study" in body and "protocolSection" in body["study"]:
        return body["study"]
    return body


@dataclass
class NoiseFloorStats:
    total_trials_with_2plus_versions: int
    total_pairs: int
    pairs_with_zero_events: int
    event_type_pair_counts: Counter  # how many pairs fired >=1 event of this type
    total_events: int
    events_by_type: Counter

    @property
    def zero_event_fraction(self) -> float:
        return self.pairs_with_zero_events / self.total_pairs if self.total_pairs else 0.0

    def firing_frequency(self, event_type: str) -> float:
        return self.event_type_pair_counts.get(event_type, 0) / self.total_pairs if self.total_pairs else 0.0


def _load_snapshot_path_index(manifest_db=None) -> dict:
    """One bulk read of the manifest instead of one duckdb connection per
    version (snap.latest() opens a fresh connection to manifest.duckdb
    on every call — fine for occasional lookups, ruinous over 50,000+
    versioned bodies). Returns {"NCT.....:vN": path}."""
    manifest_db = manifest_db or MANIFEST_DB
    con = duckdb.connect(str(manifest_db), read_only=True)
    try:
        rows = con.execute(
            "SELECT id, path FROM snapshots WHERE source = 'ctgov' AND id LIKE '%:v%'"
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


def _read_body_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        envelope = json.load(f)
    return json.loads(envelope["body"])


def _versions_with_bodies(nct_id: str, rows: list, path_index: dict) -> list[tuple]:
    out = []
    for version, posted_date in rows:
        path = path_index.get(f"{nct_id}:v{version}")
        if path is not None:
            out.append((version, posted_date, path))
    return out


def iter_all_pairs(con: duckdb.DuckDBPyConnection, path_index: Optional[dict] = None) -> Iterator[tuple]:
    """Yields (nct_id, from_version, to_version, prev_study, curr_study,
    event_date) for every adjacent pair of fetched-body versions, for
    every trial in history_index."""
    path_index = path_index if path_index is not None else _load_snapshot_path_index()
    rows_by_trial: dict[str, list] = {}
    for nct_id, version, posted_date in con.execute(
        "SELECT nct_id, version, posted_date FROM history_index ORDER BY nct_id, version"
    ).fetchall():
        rows_by_trial.setdefault(nct_id, []).append((version, posted_date))

    for nct_id, rows in rows_by_trial.items():
        versions = _versions_with_bodies(nct_id, rows, path_index)
        if len(versions) < 2:
            continue
        for (v1, _, p1), (v2, posted2, p2) in zip(versions, versions[1:]):
            if posted2 is None:
                continue  # can't assign a knowability date — skip rather than guess
            prev_study = _study_from_body(_read_body_json(p1))
            curr_study = _study_from_body(_read_body_json(p2))
            yield nct_id, v1, v2, prev_study, curr_study, posted2


def extract_all(con: duckdb.DuckDBPyConnection) -> tuple[list[EvidenceEvent], NoiseFloorStats]:
    events: list[EvidenceEvent] = []
    total_pairs = 0
    pairs_with_zero = 0
    event_type_pair_counts: Counter = Counter()
    events_by_type: Counter = Counter()
    trials_with_pairs = set()

    for nct_id, v1, v2, prev_study, curr_study, event_date in iter_all_pairs(con):
        total_pairs += 1
        trials_with_pairs.add(nct_id)
        pair_events = diff_versions(nct_id, v1, v2, prev_study, curr_study, event_date)
        if not pair_events:
            pairs_with_zero += 1
        else:
            fired_types = {e.event_type for e in pair_events}
            for t in fired_types:
                event_type_pair_counts[t] += 1
            for e in pair_events:
                events_by_type[e.event_type] += 1
            events.extend(pair_events)

    stats = NoiseFloorStats(
        total_trials_with_2plus_versions=len(trials_with_pairs),
        total_pairs=total_pairs,
        pairs_with_zero_events=pairs_with_zero,
        event_type_pair_counts=event_type_pair_counts,
        total_events=len(events),
        events_by_type=events_by_type,
    )
    return events, stats


def materialize(warehouse_db=None) -> tuple[int, NoiseFloorStats]:
    warehouse_db = warehouse_db or WAREHOUSE_DB
    con = duckdb.connect(str(warehouse_db))
    try:
        con.execute(EVIDENCE_EVENTS_SCHEMA)
        events, stats = extract_all(con)
        con.execute("DELETE FROM evidence_events")
        con.executemany(
            """
            INSERT INTO evidence_events
                (event_id, nct_id, from_version, to_version, event_date, event_type, field,
                 direction, from_value, to_value, detail, differ_version, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    i, e.nct_id, e.from_version, e.to_version, e.event_date, e.event_type, e.field,
                    e.direction, str(e.from_value) if e.from_value is not None else None,
                    str(e.to_value) if e.to_value is not None else None,
                    e.detail, e.differ_version, e.extracted_at,
                )
                for i, e in enumerate(events)
            ],
        )
        return len(events), stats
    finally:
        con.close()
