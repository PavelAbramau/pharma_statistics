"""Synthetic program cost index — a relative ranking signal, never a
dollar estimate (the absolute figure will be wrong; only the ordering is
meant to be used). See docs/decisions/0003-synthetic-cost-benchmark.md
for the full construction and its one real leakage caveat (site count).

Formula: for each trial, phase_weight[phase] x enrollment x elapsed_months,
summed across a program's trials. enrollment/phase/start_date are always
resolved through the time-cut versioned-history path (never a
current-state read) — this module's whole point is a MONTHLY
TIME-VARYING feature, so every factor must be genuinely knowable as of
the month it describes.

Site count is deliberately NOT part of the time-cut-safe monthly series:
contactsLocationsModule (CT.gov's location list) only exists on the
current-state fetch (confirmed by inspecting real snapshots on disk —
raw/ctgov/2026-08-22/NCT04717414:v48.json has no contactsLocationsModule
at all; raw/ctgov/2026-09-01/NCT04717414.json, the current-state fetch,
has 185 locations). It is not a static universe-membership property
(docs/decisions/0001) — site count grows as a trial recruits — so
reading it into a historical month's value would leak today's final
count into every past month. It's exposed separately, as a constant
per-program multiplier, for the present-day ranking snapshot only
(program_cost_index_snapshot) — never inside monthly_cost_index_series,
which is the one registered in audit/leakage.md for backtest use.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Optional

import duckdb

from pharma_stats import snapshot as snap
from pharma_stats.labelling.provisional_programs import _study_from_body

# Historical note (2026-09-04): this retry wrapper was originally papering
# over a real bug — snapshot.get_as_of/.latest opened a read-write,
# exclusive-locking connection for a pure SELECT, so two purely-reading
# callers (this module and, say, a report script) collided with each
# other as if one were writing. That's now fixed at the root:
# get_as_of/all_ids always open manifest.duckdb read_only, and DuckDB
# lets any number of read_only connections coexist — see
# snapshot._manifest_con_read_only. Reader-vs-reader contention should no
# longer happen at all.
#
# What's left, and what this retry still guards against, is the
# unavoidable case: this module's lookup landing at the exact moment an
# actual writer (save_snapshot, rebuild_manifest) holds its brief,
# genuinely-exclusive write lock. That's rare and short-lived, so a
# retry with backoff remains the right tool for it — a read-only lookup
# has no risk of a partial write, so retrying is always safe.
_LOCK_RETRY_ATTEMPTS = 20
_LOCK_RETRY_DELAY_SECONDS = 1.5


def _snap_latest_with_retry(snap_id: str):
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        try:
            return snap.latest("ctgov", snap_id)
        except duckdb.IOException:
            if attempt == _LOCK_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_LOCK_RETRY_DELAY_SECONDS)

# Sertkaya A, Wong HH, Jessup A, Beleche T. "Key cost drivers of
# pharmaceutical clinical trials in the United States." Clinical Trials.
# 2016;13(2):117-126. Average total oncology study cost by phase (used
# here purely as RELATIVE weights between phases — not a per-patient
# rate; see the decision record for why a rate can't be backed out of
# the published tables). Phase 4 has no published oncology figure in
# this source; treated as phase-3-equivalent (post-marketing oncology
# studies are typically large registries) — an explicit assumption, not
# a benchmark, flagged here and in the decision record.
PHASE_COST_WEIGHT: dict[str, float] = {
    "PHASE1": 4.5,
    "PHASE2": 11.2,
    "PHASE3": 22.1,
    "PHASE4": 22.1,  # assumption, not benchmarked — see module docstring
}
_PHASE_RANK = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}

# Sublinear: a modeling convention (fixed per-site setup/monitoring
# overhead, not a duplicated trial), NOT benchmark-derived — no published
# site-count-scaling figure exists. Documented so it's never mistaken for
# a cited number.
SITE_COUNT_EXPONENT = 0.5


@dataclass
class TrialStateAsOf:
    nct_id: str
    as_of: date
    version: int
    enrollment_count: Optional[int]
    phase: Optional[str]
    start_date: Optional[date]
    source: str  # "versioned:vN" — always versioned-history, never current-state


def highest_phase(phases: Optional[list]) -> Optional[str]:
    """The most advanced phase on a multi-phase trial (e.g. PHASE1/PHASE2)
    — reflects the trial's actual complexity/cost tier better than the
    first-listed phase."""
    ranked = [p for p in (phases or []) if p in _PHASE_RANK]
    if not ranked:
        return None
    return max(ranked, key=lambda p: _PHASE_RANK[p])


def _parse_date(struct: Optional[dict]) -> Optional[date]:
    if not struct or not struct.get("date"):
        return None
    parts = struct["date"].split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def trial_version_history(nct_id: str, con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Every indexed version's resolved state, read ONCE each (not once
    per month a caller might ask about) — {version, posted_date,
    enrollment_count, phase, start_date}, sorted by posted_date
    ascending. This is the expensive I/O (one snapshot file read per
    version); callers computing a monthly series must fetch this once
    per trial and reuse it, never re-resolve per month."""
    rows = con.execute(
        "SELECT version, posted_date FROM history_index WHERE nct_id = ? ORDER BY posted_date ASC",
        [nct_id],
    ).fetchall()
    out = []
    for version, posted_date in rows:
        s = _snap_latest_with_retry(f"{nct_id}:v{version}")
        if s is None:
            continue
        study = _study_from_body(s.body_json())
        ps = study.get("protocolSection", {})
        design_mod = ps.get("designModule", {})
        status_mod = ps.get("statusModule", {})
        out.append({
            "version": version, "posted_date": posted_date,
            "enrollment_count": (design_mod.get("enrollmentInfo") or {}).get("count"),
            "phase": highest_phase(design_mod.get("phases")),
            "start_date": _parse_date(status_mod.get("startDateStruct")),
        })
    return out


def _state_as_of_from_history(history: list[dict], as_of: date) -> Optional[dict]:
    """Latest history entry with posted_date <= as_of. history must
    already be sorted ascending by posted_date (trial_version_history's
    contract)."""
    applicable = None
    for entry in history:
        if entry["posted_date"] <= as_of:
            applicable = entry
        else:
            break
    return applicable


def resolve_trial_state_as_of(
    nct_id: str, as_of: date, con: duckdb.DuckDBPyConnection,
) -> Optional[TrialStateAsOf]:
    """Trial state resolvable as of a historical date — the highest
    indexed version whose posted_date <= as_of, read through the
    versioned-history snapshot path only. None if no such version exists
    (the trial's state at that date is genuinely unknowable, not
    approximated from a later or current snapshot — the same "missing
    data must never be servable as evidence" discipline the labelling
    app's history_coverage guard already applies).

    Single-lookup convenience — re-reads the trial's whole history on
    every call. Computing many months for the same trial must use
    trial_version_history once and _state_as_of_from_history per month
    instead (see monthly_cost_index_series) — this function would redo
    the same snapshot file reads on every call otherwise."""
    history = trial_version_history(nct_id, con)
    entry = _state_as_of_from_history(history, as_of)
    if entry is None:
        return None
    return TrialStateAsOf(
        nct_id=nct_id, as_of=as_of, version=entry["version"],
        enrollment_count=entry["enrollment_count"], phase=entry["phase"],
        start_date=entry["start_date"], source=f"versioned:v{entry['version']}",
    )


def elapsed_months(start_date: Optional[date], as_of: date) -> float:
    """0 if not yet started (or start unknown) as of as_of. Fractional
    months via day-count/30.44 (average month length) — good enough for
    a relative index, not meant to be calendar-exact."""
    if start_date is None or start_date > as_of:
        return 0.0
    return (as_of - start_date).days / 30.44


def _cost_from_entry(entry: Optional[dict], as_of: date) -> float:
    if entry is None or entry["phase"] is None or not entry["enrollment_count"]:
        return 0.0
    weight = PHASE_COST_WEIGHT.get(entry["phase"])
    if weight is None:
        return 0.0
    return weight * entry["enrollment_count"] * elapsed_months(entry["start_date"], as_of)


def trial_cost_index_as_of(nct_id: str, as_of: date, con: duckdb.DuckDBPyConnection) -> float:
    """phase_weight x enrollment x elapsed_months. 0 if the trial's state
    isn't resolvable as of this date, or phase/enrollment are unknown —
    never guessed, never backfilled from a later snapshot. Single-lookup
    convenience — see resolve_trial_state_as_of's caching caveat."""
    history = trial_version_history(nct_id, con)
    return _cost_from_entry(_state_as_of_from_history(history, as_of), as_of)


def program_cost_index_as_of(program: dict, as_of: date, con: duckdb.DuckDBPyConnection) -> float:
    """Sum of trial_cost_index_as_of across every trial on the program.
    This is the time-cut-safe figure — no site count (see module
    docstring) — the one used for monthly_cost_index_series."""
    total = 0.0
    for t in program.get("trials") or []:
        total += trial_cost_index_as_of(t["nct_id"], as_of, con)
    return total


def current_site_count(program: dict) -> Optional[int]:
    """Today's site count, current-state-fetch only (see module
    docstring) — a STATIC per-program value, never resolved as-of a
    historical date. Sum across trials; None if no trial on this program
    has ever had a current-state fetch (the common case — most trials in
    this project haven't)."""
    total = 0
    any_found = False
    for t in program.get("trials") or []:
        s = _snap_latest_with_retry(t["nct_id"])
        if s is None:
            continue
        study = _study_from_body(s.body_json())
        locations = (study.get("protocolSection", {}).get("contactsLocationsModule") or {}).get("locations") or []
        total += len(locations)
        any_found = True
    return total if any_found else None


def site_count_factor(n_sites: Optional[int]) -> float:
    if not n_sites or n_sites < 1:
        return 1.0  # no site data -> neutral multiplier, never zero out the index
    return n_sites ** SITE_COUNT_EXPONENT


def program_cost_index_snapshot(program: dict, con: duckdb.DuckDBPyConnection, *, as_of: Optional[date] = None) -> dict:
    """Today's full ranking index, ALL FOUR factors (enrollment, duration,
    phase, site count) — for present-day cross-program ranking only.
    Includes today's site count deliberately (see module docstring); this
    is not the time-cut-safe series and must never be fed into a
    historical backtest at a past date."""
    as_of = as_of or date.today()
    base = program_cost_index_as_of(program, as_of, con)
    n_sites = current_site_count(program)
    return {
        "program_id": program["program_id"],
        "as_of": as_of.isoformat(),
        "base_index": base,
        "site_count": n_sites,
        "cost_index": base * site_count_factor(n_sites),
    }


def monthly_cost_index_series(
    program: dict, con: duckdb.DuckDBPyConnection, *, end: Optional[date] = None,
) -> list[dict]:
    """The time-cut-safe monthly series — enrollment x duration x phase
    only, no site count. Each point's cost_index uses ONLY versioned-
    history data with posted_date <= that month, so it's genuinely
    knowable as of the month it describes (knowability_date == as_of).
    Starts at the earliest trial start_date resolvable from ANY version
    on file; ends at `end` (default today).

    Fetches each trial's version history ONCE (trial_version_history),
    then reuses it across every month — resolving per-month via a fresh
    snapshot file read for every trial x month pair was the dominant
    cost of building this series across the whole universe (confirmed:
    ~90s for 20 programs, i.e. ~80 minutes for the full ~1093-program
    universe, before this change)."""
    end = end or date.today()
    histories = []
    for t in program.get("trials") or []:
        h = trial_version_history(t["nct_id"], con)
        if h:
            histories.append(h)
    if not histories:
        return []

    cursor = min(h[0]["posted_date"] for h in histories).replace(day=1)
    series = []
    while cursor <= end:
        total = sum(_cost_from_entry(_state_as_of_from_history(h, cursor), cursor) for h in histories)
        series.append({
            "as_of": cursor.isoformat(),
            "cost_index": total,
            "knowability_date": cursor.isoformat(),
        })
        year, month = cursor.year, cursor.month
        cursor = date(year + (month // 12), (month % 12) + 1, 1)
    return series
