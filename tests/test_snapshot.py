from datetime import date, datetime, timezone

import pytest

from pharma_stats import snapshot as snap


@pytest.fixture
def store(tmp_path):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"
    return raw_dir, manifest_db


def _dt(d: str) -> datetime:
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc)


def test_save_and_read_roundtrip(store):
    raw_dir, manifest_db = store
    s = snap.save_snapshot(
        "ctgov", "NCT001", "https://example.org/NCT001", '{"a": 1}',
        fetched_at=_dt("2020-01-05T00:00:00"), raw_dir=raw_dir, manifest_db=manifest_db,
    )
    assert s.sha256 == snap.hashlib.sha256(b'{"a": 1}').hexdigest()
    assert s.body == '{"a": 1}'
    assert s.body_json() == {"a": 1}
    assert (raw_dir / "ctgov" / "2020-01-05" / "NCT001.json").exists()


def test_raw_file_is_verbatim_envelope(store):
    raw_dir, manifest_db = store
    snap.save_snapshot(
        "ctgov", "NCT001", "https://example.org/NCT001", "raw body text",
        fetched_at=_dt("2020-01-05T00:00:00"), raw_dir=raw_dir, manifest_db=manifest_db,
    )
    import json
    on_disk = json.loads((raw_dir / "ctgov" / "2020-01-05" / "NCT001.json").read_text())
    assert on_disk["body"] == "raw body text"
    assert on_disk["url"] == "https://example.org/NCT001"
    assert "fetched_at" in on_disk and "sha256" in on_disk


def test_idempotent_same_content_same_day(store):
    raw_dir, manifest_db = store
    snap.save_snapshot("ctgov", "NCT001", "u", "same", fetched_at=_dt("2020-01-05T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)
    s2 = snap.save_snapshot("ctgov", "NCT001", "u", "same", fetched_at=_dt("2020-01-05T12:00:00"),
                             raw_dir=raw_dir, manifest_db=manifest_db)
    assert s2.body == "same"


def test_immutability_violation_raises(store):
    raw_dir, manifest_db = store
    snap.save_snapshot("ctgov", "NCT001", "u", "version A", fetched_at=_dt("2020-01-05T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)
    with pytest.raises(snap.ImmutabilityError):
        snap.save_snapshot("ctgov", "NCT001", "u", "version B", fetched_at=_dt("2020-01-05T12:00:00"),
                            raw_dir=raw_dir, manifest_db=manifest_db)


def test_get_as_of_returns_most_recent_at_or_before(store):
    raw_dir, manifest_db = store
    snap.save_snapshot("ctgov", "NCT001", "u", "v1", fetched_at=_dt("2020-01-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)
    snap.save_snapshot("ctgov", "NCT001", "u", "v2", fetched_at=_dt("2020-06-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)
    snap.save_snapshot("ctgov", "NCT001", "u", "v3", fetched_at=_dt("2020-12-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)

    assert snap.get_as_of("ctgov", "NCT001", date(2019, 12, 31), manifest_db=manifest_db) is None
    assert snap.get_as_of("ctgov", "NCT001", date(2020, 3, 1), manifest_db=manifest_db).body == "v1"
    assert snap.get_as_of("ctgov", "NCT001", date(2020, 6, 1), manifest_db=manifest_db).body == "v2"
    assert snap.get_as_of("ctgov", "NCT001", date(2021, 1, 1), manifest_db=manifest_db).body == "v3"


def test_get_as_of_returns_none_before_manifest_exists(store):
    """get_as_of/all_ids must never crash just because nothing's been
    fetched yet — DuckDB's read_only mode refuses to create a file that
    doesn't exist, so this has to be handled explicitly, not left to
    raise."""
    _, manifest_db = store
    assert not manifest_db.exists()
    assert snap.get_as_of("ctgov", "NCT001", date(2020, 1, 1), manifest_db=manifest_db) is None
    assert snap.all_ids("ctgov", manifest_db=manifest_db) == []


def test_get_as_of_does_not_take_a_write_lock(store):
    """Real incident (2026-09-04): get_as_of opened a read-write
    connection for a pure SELECT, so two purely-reading callers (e.g. a
    backtest and a report script) would collide on the manifest's
    exclusive lock as if one of them were writing. Two simultaneous
    get_as_of connections must be able to coexist — this is the
    regression test for the read_only fix, not just a happy-path check."""
    raw_dir, manifest_db = store
    snap.save_snapshot("ctgov", "NCT001", "u", "v1", fetched_at=_dt("2020-01-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)

    con_a = snap._manifest_con_read_only(manifest_db)
    try:
        # con_a is still open (simulating one long-lived reader) while a
        # second, independent get_as_of call runs concurrently — this
        # must succeed, not raise duckdb.IOException.
        result = snap.get_as_of("ctgov", "NCT001", date(2020, 6, 1), manifest_db=manifest_db)
        assert result.body == "v1"
    finally:
        con_a.close()


def test_rebuild_manifest_reindexes_from_disk(store):
    raw_dir, manifest_db = store
    snap.save_snapshot("ctgov", "NCT001", "u", "v1", fetched_at=_dt("2020-01-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)
    snap.save_snapshot("other", "X1", "u2", "v2", fetched_at=_dt("2020-02-01T00:00:00"),
                        raw_dir=raw_dir, manifest_db=manifest_db)

    manifest_db.unlink()
    count = snap.rebuild_manifest(raw_dir=raw_dir, manifest_db=manifest_db)
    assert count == 2
    assert snap.get_as_of("ctgov", "NCT001", date(2020, 12, 31), manifest_db=manifest_db).body == "v1"
    assert snap.get_as_of("other", "X1", date(2020, 12, 31), manifest_db=manifest_db).body == "v2"
