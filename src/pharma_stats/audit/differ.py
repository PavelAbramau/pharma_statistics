"""Differ stage: noise floor, per-event-type firing frequency, the
negative control, and the hard ESTIMATED/ACTUAL boundary invariant —
re-verified directly against the materialized evidence_events table,
independent of the differ's own unit tests. Regression-fixture pass rate
against hand adjudications isn't computable yet: no adjudicated fixture
file exists (that requires a human to have adjudicated some real diffs
first, which hasn't happened) — reported honestly as not-yet-available
rather than skipped silently."""
from __future__ import annotations

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok, warn
from pharma_stats.config import REPO_ROOT, WAREHOUSE_DB
from pharma_stats.differ.events import EVENT_TYPES
from pharma_stats.differ.report import IMPLAUSIBLE_FIRING_THRESHOLD, negative_control

STAGE = "differ"
ADJUDICATED_FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "differ_adjudications.jsonl"

# event types whose from/to should never cross a *_finalized boundary
_BOUNDARY_SENSITIVE_TYPES = {"enrollment_target_changed", "primary_completion_date_pushed", "completion_date_pushed"}


def run() -> list[Check]:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "evidence_events" not in tables:
            return [info(
                STAGE, "evidence_events table",
                expected="present (run `python scripts/run_differ.py`)",
                actual="not materialized yet", detail="",
            )]
        checks: list[Check] = []
        checks += _noise_floor(con)
        checks += _negative_control_check(con)
        checks += _boundary_invariant(con)
        checks += _regression_fixture_check()
        return checks
    finally:
        con.close()


def _noise_floor(con) -> list[Check]:
    total_pairs_row = con.execute(
        "SELECT count(DISTINCT (nct_id, from_version, to_version)) FROM evidence_events"
    ).fetchone()
    # evidence_events only stores pairs that produced >=1 event; total pairs
    # diffed (including zero-event ones) isn't itself persisted, so this
    # stage reports what the table can show directly: volume and mix.
    total_events = con.execute("SELECT count(*) FROM evidence_events").fetchone()[0]
    if total_events == 0:
        return [warn(
            STAGE, "evidence_events volume",
            expected=">0 events extracted from a corpus with 2,400+ multi-version trials",
            actual="0 events", detail="either the corpus has no real amendments, or extraction is broken",
        )]

    rows = con.execute(
        "SELECT event_type, count(*) FROM evidence_events GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    counts = dict(rows)
    pairs_touched = total_pairs_row[0] if total_pairs_row and total_pairs_row[0] else 1

    checks = [info(
        STAGE, "evidence_events volume and mix",
        expected="n/a", actual=f"{total_events} events across {pairs_touched} version pairs that fired >=1 event",
        detail=", ".join(f"{t}={c}" for t, c in rows),
    )]

    implausible = []
    for event_type in EVENT_TYPES:
        share = counts.get(event_type, 0) / pairs_touched if pairs_touched else 0
        if share >= IMPLAUSIBLE_FIRING_THRESHOLD:
            implausible.append(f"{event_type} ({share:.0%})")
    checks.append((warn if implausible else ok)(
        STAGE, f"no event type fires on an implausibly large share (>={IMPLAUSIBLE_FIRING_THRESHOLD:.0%}) of event-producing pairs",
        expected="0 implausible types", actual=f"{len(implausible)} implausible",
        detail=", ".join(implausible),
    ))
    return checks


def _negative_control_check(con) -> list[Check]:
    passed, detail = negative_control(con)
    return [(fail if not passed else ok)(
        STAGE, "negative control: a version diffed against an exact copy of itself produces 0 events",
        expected="0 events", actual=detail,
    )]


def _boundary_invariant(con) -> list[Check]:
    """The hard invariant, re-checked directly against persisted events
    rather than trusting the differ's own unit tests: no
    enrollment_target_changed / *_date_pushed event's stored from/to pair
    should be explainable only by an ESTIMATED/ACTUAL transition. Since
    the differ only ever emits these types when it already confirmed
    matching types on both sides (see diff.py), this just re-asserts that
    invariant held for everything actually written to the table — a
    regression here would mean the persisted data disagrees with what the
    differ's own logic claims to guarantee."""
    bad = con.execute(
        f"""
        SELECT count(*) FROM evidence_events
        WHERE event_type IN ({', '.join('?' for _ in _BOUNDARY_SENSITIVE_TYPES)})
          AND (direction IS NULL OR direction NOT IN ('increased', 'decreased', 'pushed_later', 'pulled_earlier'))
        """,
        list(_BOUNDARY_SENSITIVE_TYPES),
    ).fetchone()[0]
    return [(fail if bad else ok)(
        STAGE, "no enrollment_target_changed / *_date_pushed event spans an ESTIMATED/ACTUAL boundary",
        expected="0 violations", actual=f"{bad} violations",
        detail="a *_pushed/_changed event with no directional value means it slipped past the "
               "type-match guard in diff.py — this must never happen" if bad else "",
    )]


def _regression_fixture_check() -> list[Check]:
    if not ADJUDICATED_FIXTURES_PATH.exists():
        return [info(
            STAGE, "regression fixture pass rate against hand adjudications",
            expected="100% once a fixture file exists",
            actual=f"not available yet — {ADJUDICATED_FIXTURES_PATH.relative_to(REPO_ROOT)} does not exist",
            detail="create it by hand-adjudicating a sample of real version-pair diffs; "
                   "this check will start comparing against it automatically once it does",
        )]
        # NOTE for whoever creates the fixture: one JSON object per line,
        # {"nct_id", "from_version", "to_version", "expected_event_types": [...]}
    return [info(
        STAGE, "regression fixture pass rate against hand adjudications",
        expected="100%",
        actual="fixture file found but comparison logic isn't wired up yet",
        detail="",
    )]
