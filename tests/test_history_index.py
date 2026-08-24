from datetime import date, datetime, timezone

import duckdb

from pharma_stats.history.index import ensure_schema, module_filter_stats


def _seed(con, rows):
    ensure_schema(con)
    now = datetime.now(timezone.utc)
    for nct_id, version, modules in rows:
        con.execute(
            """
            INSERT INTO history_index
                (nct_id, version, posted_date, submitted_date, status, study_type,
                 changed_modules, schema_hash, indexed_at)
            VALUES (?, ?, ?, ?, 'RECRUITING', 'INTERVENTIONAL', ?, 'x', ?)
            """,
            [nct_id, version, date(2020, 1, 1), date(2020, 1, 1), modules, now],
        )


def test_module_filter_stats_excludes_baseline_version():
    con = duckdb.connect(":memory:")
    _seed(con, [("NCT1", 0, [])])  # version 0 = baseline, never "signal"
    stats = module_filter_stats(con, frozenset({"Study Status"}))
    assert stats["total_versions"] == 1
    assert stats["non_baseline_versions"] == 0
    assert stats["signal_versions"] == 0


def test_module_filter_stats_counts_intersecting_versions_only():
    con = duckdb.connect(":memory:")
    _seed(con, [
        ("NCT1", 0, []),
        ("NCT1", 1, ["Contacts/Locations"]),          # cosmetic only -> not signal
        ("NCT1", 2, ["Study Status"]),                 # signal
        ("NCT1", 3, ["Study Status", "Contacts/Locations"]),  # signal (mixed)
        ("NCT1", 4, ["Conditions"]),                    # not in filter -> not signal
    ])
    stats = module_filter_stats(con, frozenset({"Study Status", "Study Design"}))
    assert stats["non_baseline_versions"] == 4
    assert stats["signal_versions"] == 2
    assert stats["reduction_ratio"] == 0.5


def test_module_filter_stats_reduction_ratio_zero_when_no_versions():
    con = duckdb.connect(":memory:")
    ensure_schema(con)
    stats = module_filter_stats(con, frozenset({"Study Status"}))
    assert stats["total_versions"] == 0
    assert stats["reduction_ratio"] == 0.0
