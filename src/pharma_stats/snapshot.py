"""Provenance-aware snapshot store.

Every fetch from an external source is written once, verbatim, to
``raw/{source}/{YYYY-MM-DD}/{id}.json`` and never mutated again. A DuckDB
manifest indexes every snapshot on disk so that callers can ask, for any
source/id, "what was the most recent snapshot at or before date X" — the
lookup the time-cut backtest depends on.

Nothing in this module ever opens a raw file in write mode a second time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import duckdb

from pharma_stats.config import MANIFEST_DB, RAW_DIR

_MANIFEST_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    source        VARCHAR NOT NULL,
    id            VARCHAR NOT NULL,
    snapshot_date DATE NOT NULL,
    fetched_at    TIMESTAMP NOT NULL,
    url           VARCHAR NOT NULL,
    sha256        VARCHAR NOT NULL,
    path          VARCHAR NOT NULL,
    PRIMARY KEY (source, id, snapshot_date)
)
"""


class ImmutabilityError(RuntimeError):
    """Raised when a write would change an existing raw snapshot's content."""


@dataclass(frozen=True)
class Snapshot:
    source: str
    id: str
    snapshot_date: date
    fetched_at: datetime
    url: str
    sha256: str
    path: Path

    @property
    def body(self) -> str:
        """The verbatim response body, as originally fetched."""
        return _read_envelope(self.path)["body"]

    def body_json(self) -> Any:
        """Convenience: parse the body as JSON."""
        return json.loads(self.body)


def _manifest_con(manifest_db: Path = MANIFEST_DB) -> duckdb.DuckDBPyConnection:
    manifest_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(manifest_db))
    con.execute(_MANIFEST_SCHEMA)
    return con


def _manifest_con_read_only(manifest_db: Path = MANIFEST_DB) -> Optional[duckdb.DuckDBPyConnection]:
    """A read_only connection lets DuckDB grant multiple simultaneous
    readers, instead of every caller fighting for an exclusive lock it
    never actually needed. get_as_of/latest/all_ids never write, so they
    always go through this, never _manifest_con's read-write connection —
    that's what previously made a pure-read backtest or matrix-build
    script collide with any other concurrent read as if it were a
    writer. DuckDB's read_only mode requires the file to already exist
    (it won't create one), so a not-yet-built manifest reports "no data"
    (None from get_as_of) via this returning None, rather than raising —
    consistent with how every other "not built yet" path in this project
    degrades."""
    if not manifest_db.exists():
        return None
    return duckdb.connect(str(manifest_db), read_only=True)


def _read_envelope(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(
    source: str,
    id: str,
    url: str,
    body: Union[str, bytes],
    *,
    fetched_at: Optional[datetime] = None,
    raw_dir: Path = RAW_DIR,
    manifest_db: Path = MANIFEST_DB,
) -> Snapshot:
    """Write one immutable snapshot and index it in the manifest.

    Idempotent: re-saving byte-identical content for the same
    source/id/day is a no-op that returns the existing snapshot. Saving
    *different* content for a source/id that already has a snapshot on
    that day raises ImmutabilityError, since the day-bucketed path can
    only hold one truth per day.
    """
    fetched_at = fetched_at or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    snapshot_date = fetched_at.date()

    body_text = body.decode("utf-8") if isinstance(body, bytes) else body
    sha256 = hashlib.sha256(body_text.encode("utf-8")).hexdigest()

    dir_path = raw_dir / source / snapshot_date.isoformat()
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{id}.json"

    envelope = {
        "url": url,
        "fetched_at": fetched_at.isoformat(),
        "sha256": sha256,
        "body": body_text,
    }

    if file_path.exists():
        existing = _read_envelope(file_path)
        if existing["sha256"] != sha256:
            raise ImmutabilityError(
                f"Refusing to overwrite immutable snapshot {file_path}: "
                f"existing sha256={existing['sha256']} new sha256={sha256}. "
                "Raw snapshots are write-once."
            )
        # Identical content already on disk; treat as a no-op.
        return _snapshot_from_disk(source, id, snapshot_date, file_path, existing)

    tmp_path = file_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    tmp_path.replace(file_path)  # atomic on POSIX

    con = _manifest_con(manifest_db)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO snapshots
                (source, id, snapshot_date, fetched_at, url, sha256, path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                source,
                id,
                snapshot_date,
                fetched_at,
                url,
                sha256,
                str(file_path),
            ],
        )
    finally:
        con.close()

    return Snapshot(
        source=source,
        id=id,
        snapshot_date=snapshot_date,
        fetched_at=fetched_at,
        url=url,
        sha256=sha256,
        path=file_path,
    )


def _snapshot_from_disk(source: str, id: str, snapshot_date: date, path: Path, envelope: dict) -> Snapshot:
    return Snapshot(
        source=source,
        id=id,
        snapshot_date=snapshot_date,
        fetched_at=datetime.fromisoformat(envelope["fetched_at"]),
        url=envelope["url"],
        sha256=envelope["sha256"],
        path=path,
    )


def get_as_of(
    source: str,
    id: str,
    as_of: Union[str, date],
    *,
    manifest_db: Path = MANIFEST_DB,
) -> Optional[Snapshot]:
    """Return the most recent snapshot for (source, id) at or before as_of.

    Returns None if no such snapshot exists. This is the primitive the
    time-cut backtest is built on: it lets any later analysis ask "what
    did we know about this trial as of date X" without leaking
    future-dated information.
    """
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)

    con = _manifest_con_read_only(manifest_db)
    if con is None:
        return None
    try:
        row = con.execute(
            """
            SELECT source, id, snapshot_date, fetched_at, url, sha256, path
            FROM snapshots
            WHERE source = ? AND id = ? AND snapshot_date <= ?
            ORDER BY snapshot_date DESC, fetched_at DESC
            LIMIT 1
            """,
            [source, id, as_of],
        ).fetchone()
    finally:
        con.close()

    if row is None:
        return None

    src, sid, snap_date, fetched_at, url, sha256, path = row
    return Snapshot(
        source=src,
        id=sid,
        snapshot_date=snap_date,
        fetched_at=fetched_at,
        url=url,
        sha256=sha256,
        path=Path(path),
    )


def latest(source: str, id: str, *, manifest_db: Path = MANIFEST_DB) -> Optional[Snapshot]:
    """Convenience: most recent snapshot regardless of date."""
    return get_as_of(source, id, date.today(), manifest_db=manifest_db)


def rebuild_manifest(raw_dir: Path = RAW_DIR, manifest_db: Path = MANIFEST_DB) -> int:
    """Rebuild the manifest from scratch by scanning raw/ on disk.

    The manifest is a derived index, not a source of truth — raw/ is.
    This lets the manifest be regenerated if it's ever lost, deleted, or
    out of sync (e.g. after a raw/ directory is copied between machines).
    Returns the number of snapshots indexed.
    """
    con = _manifest_con(manifest_db)
    count = 0
    try:
        con.execute("DELETE FROM snapshots")
        for source_dir in sorted(p for p in raw_dir.glob("*") if p.is_dir()):
            source = source_dir.name
            for date_dir in sorted(p for p in source_dir.glob("*") if p.is_dir()):
                try:
                    snapshot_date = date.fromisoformat(date_dir.name)
                except ValueError:
                    continue  # not a YYYY-MM-DD directory; skip
                for file_path in sorted(date_dir.glob("*.json")):
                    envelope = _read_envelope(file_path)
                    con.execute(
                        """
                        INSERT OR REPLACE INTO snapshots
                            (source, id, snapshot_date, fetched_at, url, sha256, path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            source,
                            file_path.stem,
                            snapshot_date,
                            datetime.fromisoformat(envelope["fetched_at"]),
                            envelope["url"],
                            envelope["sha256"],
                            str(file_path),
                        ],
                    )
                    count += 1
    finally:
        con.close()
    return count


def all_ids(source: str, *, manifest_db: Path = MANIFEST_DB) -> list[str]:
    """Distinct ids ever snapshotted for a source, per the manifest."""
    con = _manifest_con_read_only(manifest_db)
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT DISTINCT id FROM snapshots WHERE source = ? ORDER BY id", [source]
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]
