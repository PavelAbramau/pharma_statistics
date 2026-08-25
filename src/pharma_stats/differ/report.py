"""Noise-floor / firing-frequency report and the negative control.

Per the user: this must exist and be reviewed *before* any event ever
reaches the labelling app.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from pharma_stats.config import REPORTS_DIR
from pharma_stats.differ.diff import diff_versions
from pharma_stats.differ.events import EVENT_TYPES
from pharma_stats.differ.extract import NoiseFloorStats, _study_from_body

# Above this firing-frequency share, a type is flagged for a human to
# sanity-check rather than trusted at face value — an event that fires on
# most version pairs is probably catching something cosmetic, not signal.
IMPLAUSIBLE_FIRING_THRESHOLD = 0.5


def negative_control(con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Diff a real, arbitrary fetched body against an exact copy of
    itself. Must produce zero events — this is the strongest, cheapest
    correctness guarantee available: no version-to-version noise can
    survive an identity comparison."""
    from pharma_stats import snapshot as snap

    row = con.execute(
        "SELECT nct_id, version FROM history_index ORDER BY nct_id, version LIMIT 1"
    ).fetchone()
    if row is None:
        return False, "no history_index rows to sample from"
    nct_id, version = row
    s = snap.latest("ctgov", f"{nct_id}:v{version}")
    if s is None:
        # scan for the first version that actually has a fetched body
        for nct_id, version in con.execute(
            "SELECT nct_id, version FROM history_index ORDER BY nct_id, version"
        ).fetchall():
            s = snap.latest("ctgov", f"{nct_id}:v{version}")
            if s is not None:
                break
    if s is None:
        return False, "no fetched version body available to self-diff"

    study = _study_from_body(s.body_json())
    events = diff_versions(nct_id, version, version, study, study, datetime.now(timezone.utc).date())
    ok = len(events) == 0
    detail = f"{nct_id} v{version} diffed against itself: {len(events)} events (expected 0)"
    return ok, detail


def render_noise_floor_report(stats: NoiseFloorStats, negative_control_result: tuple[bool, str]) -> str:
    lines = [
        "# Differ noise-floor report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Trials with 2+ fetched version bodies: {stats.total_trials_with_2plus_versions}",
        f"- Adjacent version pairs diffed: {stats.total_pairs}",
        f"- Pairs producing zero events: {stats.pairs_with_zero_events} "
        f"({stats.zero_event_fraction:.1%})",
        f"- Total events extracted: {stats.total_events}",
        "",
        "## Negative control",
        "",
        f"{'PASS' if negative_control_result[0] else 'FAIL'}: {negative_control_result[1]}",
        "",
        "## Firing frequency by event type",
        "",
        "(share of ALL diffed pairs that fired at least one event of this type; "
        f">={IMPLAUSIBLE_FIRING_THRESHOLD:.0%} is flagged for a human sanity-check)",
        "",
    ]
    for event_type in EVENT_TYPES:
        share = stats.firing_frequency(event_type)
        count = stats.events_by_type.get(event_type, 0)
        flag = " ⚠ implausibly high" if share >= IMPLAUSIBLE_FIRING_THRESHOLD else ""
        lines.append(f"- {event_type}: {count} events, fired on {share:.2%} of pairs{flag}")
    return "\n".join(lines)


def write_report(stats: NoiseFloorStats, negative_control_result: tuple[bool, str], out_dir: Path = None) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "differ_noise_floor.md"
    path.write_text(render_noise_floor_report(stats, negative_control_result), encoding="utf-8")
    return path
