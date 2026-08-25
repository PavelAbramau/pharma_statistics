"""Provenance stage: raw/ <-> manifest.duckdb integrity, and the
get_as_of correctness probe that protects the time-cut backtest.

This is the stage the whole project's evidentiary claim rests on: every
downstream number traces back to a raw snapshot, and any analysis that
claims to know something "as of" a date must be able to trust that
get_as_of returns the historically-correct snapshot, not just the
latest one on disk.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import duckdb

from pharma_stats import snapshot as snap
from pharma_stats.audit.types import Check, fail, info, ok
from pharma_stats.config import MANIFEST_DB, RAW_DIR

STAGE = "provenance"


def run() -> list[Check]:
    checks: list[Check] = []
    checks += _integrity_checks()
    checks += _get_as_of_probe()
    return checks


def _iter_raw_files():
    if not RAW_DIR.exists():
        return
    for source_dir in sorted(p for p in RAW_DIR.glob("*") if p.is_dir()):
        for date_dir in sorted(p for p in source_dir.glob("*") if p.is_dir()):
            try:
                date.fromisoformat(date_dir.name)
            except ValueError:
                continue
            for f in sorted(date_dir.glob("*.json")):
                yield source_dir.name, date_dir.name, f


def _integrity_checks() -> list[Check]:
    con = duckdb.connect(str(MANIFEST_DB), read_only=True)
    try:
        manifest_rows = {
            (source, id_, snap_date.isoformat()): sha
            for source, id_, snap_date, sha in con.execute(
                "SELECT source, id, snapshot_date, sha256 FROM snapshots"
            ).fetchall()
        }
    finally:
        con.close()

    disk_keys: set[tuple[str, str, str]] = set()
    self_sha_mismatch: list[str] = []
    manifest_sha_mismatch: list[str] = []
    orphan_files: list[str] = []
    n_checked = 0

    for source, date_str, path in _iter_raw_files():
        envelope = json.loads(path.read_text(encoding="utf-8"))
        recomputed = hashlib.sha256(envelope["body"].encode("utf-8")).hexdigest()
        n_checked += 1
        if recomputed != envelope["sha256"]:
            self_sha_mismatch.append(str(path))

        key = (source, path.stem, date_str)
        disk_keys.add(key)
        manifest_sha = manifest_rows.get(key)
        if manifest_sha is None:
            orphan_files.append(str(path))
        elif manifest_sha != envelope["sha256"]:
            manifest_sha_mismatch.append(str(path))

    orphan_manifest_rows = [k for k in manifest_rows if k not in disk_keys]

    def result(name, expected, bad_list, total, detail_items):
        return (fail if bad_list else ok)(
            STAGE, name, expected, f"{len(bad_list)} / {total}",
            "; ".join(detail_items[:10]),
        )

    return [
        result(
            "raw file body hash matches its own stored sha256 envelope",
            "0 self-inconsistent files", self_sha_mismatch, n_checked, self_sha_mismatch,
        ),
        result(
            "every raw file has a manifest row",
            "0 orphan files", orphan_files, n_checked, orphan_files,
        ),
        result(
            "every manifest row has a raw file on disk",
            "0 orphan rows", orphan_manifest_rows, len(manifest_rows),
            [str(k) for k in orphan_manifest_rows],
        ),
        result(
            "manifest sha256 matches the raw file's sha256",
            "0 mismatches", manifest_sha_mismatch, n_checked, manifest_sha_mismatch,
        ),
        info(
            STAGE, "no (source,id,date) key holds two different hashes",
            expected="structurally guaranteed", actual="guaranteed",
            detail="one file per (source,id,date) path plus the manifest's PRIMARY KEY on "
                   "(source,id,snapshot_date) make this impossible without filesystem/DB "
                   "corruption; the sha-match checks above would also catch that case.",
        ),
    ]


def _get_as_of_probe() -> list[Check]:
    con = duckdb.connect(str(MANIFEST_DB), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT source, id, list(snapshot_date ORDER BY snapshot_date) AS dates
            FROM snapshots
            GROUP BY source, id
            HAVING count(DISTINCT snapshot_date) > 1
            ORDER BY id
            LIMIT 25
            """
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return [info(
            STAGE, "get_as_of correctness probe",
            expected=">=1 (source,id) with multiple snapshot dates to probe against",
            actual="0 such pairs in the manifest yet",
            detail="no probe is possible until re-fetches create dated history for the same id; rerun this audit later",
        )]

    n_ok, n_bad = 0, 0
    bad_detail: list[str] = []

    for source, id_, dates in rows:
        dates = sorted(dates)
        for i, d in enumerate(dates):
            got = snap.get_as_of(source, id_, d)
            ok_ = got is not None and got.snapshot_date == d
            n_ok += ok_
            n_bad += not ok_
            if not ok_:
                bad_detail.append(f"{source}/{id_} as_of={d}: got {got.snapshot_date if got else None}")

            if i + 1 < len(dates):
                probe_date = dates[i + 1] - timedelta(days=1)
                if probe_date >= d:
                    got2 = snap.get_as_of(source, id_, probe_date)
                    ok2 = got2 is not None and got2.snapshot_date == d
                    n_ok += ok2
                    n_bad += not ok2
                    if not ok2:
                        bad_detail.append(
                            f"{source}/{id_} as_of={probe_date}: expected {d}, "
                            f"got {got2.snapshot_date if got2 else None} "
                            "(this is the leak that would let a backtest see the future)"
                        )

    return [(fail if n_bad else ok)(
        STAGE, "get_as_of returns the historically-correct snapshot, not the latest",
        expected=f"{n_ok + n_bad} probes, 0 wrong", actual=f"{n_bad} wrong / {n_ok + n_bad}",
        detail="; ".join(bad_detail[:10]),
    )]
