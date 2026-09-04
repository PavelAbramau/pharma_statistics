"""Resolve a full TrialSummary as of a historical date — the general
version of finance/cost_model.py's resolve_trial_state_as_of (which only
extracts enrollment/phase/start_date for the cost index). The silence-
score heuristic needs every TrialSummary field, so this replicates
provisional_programs.summarize_trial's exact field extraction against
the AS-OF-resolved version body instead of _best_trial_snapshot's
"latest available" body.

Never a current-state read: only versioned-history bodies with
posted_date <= as_of are ever touched, matching docs/decisions/0001.

Two real bugs found and fixed while building this (2026-09-04, see
docs/decisions/0006 and 0008):

1. The backfill orchestrator's selective body-fetch only fetches a
   version's body when its changed_modules intersects the signal labels,
   and never fetches version 0 at all. 283/500 (56.6%) sampled
   version>0 rows have no fetched body. A naive "fetch the exact version
   implied by posted_date <= as_of" lookup returns None whenever the
   nearest metadata version has no body, even when an earlier, still-
   valid version's body is on disk. Fixed with a sparse carry-forward
   list (build_trial_cache), same pattern finance/cost_model.py's
   trial_version_history already used correctly.

2. Building that carry-forward list is the expensive part (one snapshot
   file read per version) — a first draft called it fresh for every
   (trial, month) pair building a monthly panel, ~250 months per trial,
   which is why a 10-program backtest run didn't finish in 2 minutes.
   build_trial_cache must be called ONCE per trial and reused across
   every month a caller asks about (features/panel.py does this); this
   module's own resolve_trial_summary_as_of remains a single-lookup
   convenience that pays the full cost per call, same caveat
   resolve_trial_state_as_of carries in cost_model.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import duckdb

from pharma_stats.finance.cost_model import _snap_latest_with_retry
from pharma_stats.labelling.provisional_programs import (
    TrialSummary,
    _history_rows,
    _parse_ct_date,
    _parse_month_date,
    _study_from_body,
)


@dataclass
class TrialCache:
    nct_id: str
    states: list[dict]  # [{version, posted_date, body}], sorted, FETCHED bodies only
    full_history: list[dict]  # every history_index row (for as-of truncation), sorted


def build_trial_cache(nct_id: str, con: duckdb.DuckDBPyConnection) -> TrialCache:
    """The expensive I/O (one snapshot file read per fetched version, one
    history_index query) — call ONCE per trial and reuse across every
    as-of query a caller makes, never per month."""
    rows = con.execute(
        "SELECT version, posted_date FROM history_index WHERE nct_id = ? ORDER BY posted_date ASC",
        [nct_id],
    ).fetchall()
    states = []
    for version, posted_date in rows:
        s = _snap_latest_with_retry(f"{nct_id}:v{version}")
        if s is None:
            continue
        states.append({"version": version, "posted_date": posted_date, "body": s.body_json()})
    full_history = _history_rows(nct_id, con)
    return TrialCache(nct_id=nct_id, states=states, full_history=full_history)


def resolve_from_cache(cache: TrialCache, as_of: date) -> Optional[TrialSummary]:
    """Pure, no I/O — the cheap part, safe to call once per month."""
    applicable = None
    for entry in cache.states:
        if entry["posted_date"] <= as_of:
            applicable = entry
        else:
            break
    if applicable is None:
        return None

    study = _study_from_body(applicable["body"])
    ps = study.get("protocolSection", {})
    status_mod = ps.get("statusModule", {})
    design_mod = ps.get("designModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    cond_mod = ps.get("conditionsModule", {})

    enrollment = design_mod.get("enrollmentInfo") or {}
    primary_completion, primary_completion_type = _parse_ct_date(status_mod.get("primaryCompletionDateStruct"))
    completion, completion_type = _parse_ct_date(status_mod.get("completionDateStruct"))
    last_update, _ = _parse_ct_date(status_mod.get("lastUpdatePostDateStruct"))

    # history truncated to what was knowable as of this date too — a
    # feature that reads t.history (e.g. an amendment-cadence count) must
    # not see amendments posted after as_of either.
    as_of_iso = as_of.isoformat()
    history = [h for h in cache.full_history if h["posted_date"] and h["posted_date"] <= as_of_iso]

    return TrialSummary(
        nct_id=cache.nct_id,
        status=status_mod.get("overallStatus"),
        phases=list(design_mod.get("phases") or []),
        why_stopped=status_mod.get("whyStopped"),
        enrollment_count=enrollment.get("count"),
        enrollment_type=enrollment.get("type"),
        conditions=list(cond_mod.get("conditions") or []),
        sponsor=(sponsor_mod.get("leadSponsor") or {}).get("name"),
        start_date=_parse_month_date((status_mod.get("startDateStruct") or {}).get("date")),
        primary_completion_date=primary_completion,
        primary_completion_type=primary_completion_type,
        completion_date=completion,
        completion_type=completion_type,
        last_update_post_date=last_update,
        status_verified_date=_parse_month_date(status_mod.get("statusVerifiedDate")),
        has_results=study.get("hasResults"),
        source_snapshot=f"versioned:v{applicable['version']}",
        history=history,
    )


def _fetched_version_states(nct_id: str, con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Back-compat single-lookup helper (as_of_probe.py's sampling scan)
    — see build_trial_cache for the version callers computing more than
    one as-of date per trial must use instead."""
    return build_trial_cache(nct_id, con).states


def resolve_trial_summary_as_of(
    nct_id: str, as_of: date, con: duckdb.DuckDBPyConnection,
) -> Optional[TrialSummary]:
    """Single-lookup convenience — pays the full build_trial_cache cost
    on every call. Computing a monthly series for the same trial must
    call build_trial_cache once and reuse resolve_from_cache per month
    instead (see features/panel.py)."""
    return resolve_from_cache(build_trial_cache(nct_id, con), as_of)
