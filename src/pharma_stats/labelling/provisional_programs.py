"""Provisional program view for the labelling app.

The five-entity warehouse (Asset / Program / Trial / EvidenceEvent / Org)
and controlled-vocabulary normalisation are both "not started" per
README.md. The labelling app cannot wait for them — labelling is the
project's wall-clock bottleneck. So this module builds a **v0, clearly
provisional** program rollup directly from what already exists:
``asset_candidates`` (discovery output) and the raw CT.gov snapshots on
disk (``raw/ctgov/...``), via :mod:`pharma_stats.snapshot`.

Provisional program == candidate asset. Indication and line-of-therapy
are NOT guessed here (CLAUDE.md: don't guess controlled-vocabulary values
ahead of the normalisation step) — every provisional program carries
``indication_code="UNSPECIFIED"`` and ``line_of_therapy="unspecified"``
and ``provisional=True``. When the real five-entity warehouse exists, a
single asset here will split into one row per (asset, indication, line)
and this module goes away. Nothing it produces is gold data; it is a
rebuildable derived view, same as everything else under ``data/``.

The silence score computed here is a hand-built heuristic over
currently-available signals (status, whyStopped text, enrollment
actual-vs-estimated, verification lapse, staleness). It is NOT the
project's eventual model output — it exists only to stratify the
labelling queue so early labels aren't all drawn from one end of the
score distribution. Treat it as an opinionated queue-ordering knob, not
a prediction.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import duckdb

from pharma_stats import snapshot as snap
from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.history.index import RECOMMENDED_SIGNAL_LABELS
from pharma_stats.labelling import trial_scope as ts

HISTORY_COVERAGE_LEVELS = ["full", "partial", "none"]  # in order of trust, most to least

SCORE_BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]

ARCHETYPES = [
    "unknown_status",
    "registry_terminated_vague_reason",
    "registry_terminated_stated_reason",
    "completed_no_results",
    "actively_amended",
    "other",
]

ACTIVE_LIKE_STATUSES = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
}
TERMINAL_STOP_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}

# --- silence score weights ---------------------------------------------
# Each component below is computed as a normalised 0-1 sub-score first
# (see the _*_subscore functions), then multiplied by its weight here.
# Weights are the ONLY place scale lives — a component function must never
# bake a cap/divisor into its own return value. They sum to 100 so the
# final score reads as "percentage of maximum plausible silence."
STALENESS_WEIGHT = 45
VERIFICATION_LAPSE_WEIGHT = 8
STATUS_AMBIGUITY_WEIGHT = 27
ENROLLMENT_SIGNAL_WEIGHT = 20
assert STALENESS_WEIGHT + VERIFICATION_LAPSE_WEIGHT + STATUS_AMBIGUITY_WEIGHT + ENROLLMENT_SIGNAL_WEIGHT == 100

STALENESS_TAU_DAYS = 365.0  # exponential saturation time-constant — see _staleness_subscore
VERIFICATION_LAPSE_DAYS_CAP = 365  # 1 year since last re-verification = maximally lapsed
# Status implies the record should carry a last-update date but none is on
# file — genuinely unknown, not "fresh" (0) and not "maximally stale" (1).
MISSING_DATE_STALENESS_SUBSCORE = 0.5
# An EXPLAINED terminal stop (a stated, non-vague why_stopped) is
# accounted-for silence: the sponsor said why it stopped moving, which is
# different evidence than a record that just went quiet. Discount, don't
# zero — a long-explained termination should still rank below an active
# program gone silent for the same stretch, not read as identical to a
# just-terminated one with the same why_stopped.
EXPLAINED_TERMINAL_STALENESS_DISCOUNT = 0.3

_VAGUE_WHY_STOPPED_PATTERNS = (
    "business reason", "business decision", "strategic reason",
    "study terminated", "sponsor decision", "no further information",
    "study discontinued", "terminated by sponsor", "administrative reason",
    "portfolio", "sponsor request", "study closed", "n/a", "unknown",
)

PROVISIONAL_PROGRAMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS provisional_programs (
    program_id          VARCHAR PRIMARY KEY,
    candidate_id         VARCHAR NOT NULL,
    proposed_name        VARCHAR,
    synonyms              VARCHAR[],
    indication_code       VARCHAR,
    line_of_therapy       VARCHAR,
    provisional           BOOLEAN,
    sponsors_over_time     JSON,
    trial_count            INTEGER,
    nct_ids                 VARCHAR[],
    excluded_shared_trials   JSON,
    history_coverage          VARCHAR,
    trial_coverage             JSON,
    silence_score            INTEGER,
    score_breakdown           JSON,
    band                       INTEGER,
    archetypes                 VARCHAR[],
    primary_archetype           VARCHAR,
    latest_status                 VARCHAR,
    latest_nct_id                  VARCHAR,
    trials                          JSON,
    timeline                         JSON,
    review_status                     VARCHAR,
    discovery_strategy                 VARCHAR,
    match_strength                      VARCHAR,
    matched_term                         VARCHAR,
    trial_scope                           JSON,
    scope_category                         VARCHAR,
    spans_heme_and_solid                    BOOLEAN,
    trial_has_mesh                           JSON,
    trial_text_hint                           JSON,
    non_oncology_hint                          BOOLEAN,
    non_industry_sponsor_hint                   BOOLEAN,
    excluded_expanded_access_trials              VARCHAR[],
    contacts_locations_amendment_cadence          DOUBLE,
    built_at                                     TIMESTAMP
)
"""


def _parse_ct_date(struct: Optional[dict]) -> tuple[Optional[date], Optional[str]]:
    if not struct or not struct.get("date"):
        return None, None
    raw = struct["date"]
    parts = raw.split("-")
    try:
        if len(parts) == 1:
            d = date(int(parts[0]), 1, 1)
        elif len(parts) == 2:
            d = date(int(parts[0]), int(parts[1]), 1)
        else:
            d = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None, struct.get("type")
    return d, struct.get("type")


def _parse_month_date(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    parts = raw.split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


@dataclass
class TrialSummary:
    nct_id: str
    status: Optional[str]
    phases: list[str]
    why_stopped: Optional[str]
    enrollment_count: Optional[int]
    enrollment_type: Optional[str]
    conditions: list[str]
    sponsor: Optional[str]
    start_date: Optional[date]
    primary_completion_date: Optional[date]
    primary_completion_type: Optional[str]
    completion_date: Optional[date]
    completion_type: Optional[str]
    last_update_post_date: Optional[date]
    status_verified_date: Optional[date]
    has_results: Optional[bool]
    source_snapshot: str  # "versioned:vN" or "latest"
    history: list[dict] = field(default_factory=list)  # from history_index, if any

    def to_json(self) -> dict:
        return {
            "nct_id": self.nct_id,
            "status": self.status,
            "phases": self.phases,
            "why_stopped": self.why_stopped,
            "enrollment_count": self.enrollment_count,
            "enrollment_type": self.enrollment_type,
            "conditions": self.conditions,
            "sponsor": self.sponsor,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "primary_completion_date": self.primary_completion_date.isoformat()
            if self.primary_completion_date else None,
            "primary_completion_type": self.primary_completion_type,
            "completion_date": self.completion_date.isoformat() if self.completion_date else None,
            "completion_type": self.completion_type,
            "last_update_post_date": self.last_update_post_date.isoformat()
            if self.last_update_post_date else None,
            "status_verified_date": self.status_verified_date.isoformat()
            if self.status_verified_date else None,
            "has_results": self.has_results,
            "source_snapshot": self.source_snapshot,
            "history": self.history,
            "ctgov_url": f"https://clinicaltrials.gov/study/{self.nct_id}",
        }


def _study_from_body(body: Any) -> dict:
    """Handle both raw shapes on disk: versioned history bodies are
    ``{"studyVersion": N, "study": {...}}``; plain discovery-fetch
    snapshots are the study object itself."""
    if isinstance(body, dict) and "study" in body and "protocolSection" in body["study"]:
        return body["study"]
    return body


def _best_trial_snapshot(
    nct_id: str, con: duckdb.DuckDBPyConnection,
) -> Optional[tuple[dict, str]]:
    """Prefer the highest indexed version's raw body (if any full history
    has been backfilled for this trial); otherwise fall back to whatever
    single "current state" snapshot discovery fetched."""
    row = con.execute(
        "SELECT max(version) FROM history_index WHERE nct_id = ?", [nct_id]
    ).fetchone()
    max_version = row[0] if row else None
    if max_version is not None:
        snap_id = f"{nct_id}:v{max_version}"
        s = snap.latest("ctgov", snap_id)
        if s is not None:
            return _study_from_body(s.body_json()), f"versioned:v{max_version}"

    s = snap.latest("ctgov", nct_id)
    if s is not None:
        return _study_from_body(s.body_json()), "latest"
    return None


# Functions in THIS file permitted to read a current-state-only snapshot
# (snapshot.latest/get_as_of called for something other than the
# versioned-history path _best_trial_snapshot resolves). See
# docs/decisions/0001-current-state-fetch-scope.md: current-state reads
# are allowed ONLY for static universe-membership properties (disease
# category, sponsor class, start date), never for a silence/model
# feature. audit/universe.py's _current_state_read_boundary check
# statically enforces that every other function in this file stays off
# snap.latest/get_as_of entirely. Widening this list is a deliberate,
# reviewable decision — update the doc above when you do.
CURRENT_STATE_READ_WHITELIST = {"_condition_browse_data"}


def _condition_browse_data(nct_id: str) -> tuple[list[dict], list[dict]]:
    """(meshes, ancestors) from CT.gov's conditionBrowseModule. This lives
    under derivedSection on a "current state" fetch (``snap.latest("ctgov",
    nct_id)``) ONLY — versioned-history diff bodies (the ones
    _best_trial_snapshot prefers for amendment tracking) never carry
    derivedSection at all, so this is deliberately independent of that
    lookup. Most trials in this project have never had a current-state
    fetch, so ([], []) — "no MeSH data" — is the common case; trial_scope
    treats that as "ambiguous", never as a reason to guess from text.
    Permitted current-state read — see CURRENT_STATE_READ_WHITELIST above
    and docs/decisions/0001-current-state-fetch-scope.md."""
    s = snap.latest("ctgov", nct_id)
    if s is None:
        return [], []
    study = _study_from_body(s.body_json())
    cbm = (study.get("derivedSection") or {}).get("conditionBrowseModule") or {}
    return list(cbm.get("meshes") or []), list(cbm.get("ancestors") or [])


def _history_rows(nct_id: str, con: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = con.execute(
        """
        SELECT version, posted_date, status, changed_modules
        FROM history_index WHERE nct_id = ? ORDER BY version
        """,
        [nct_id],
    ).fetchall()
    return [
        {
            "version": v,
            "posted_date": pd.isoformat() if pd else None,
            "status": st,
            "changed_modules": mods,
        }
        for v, pd, st, mods in rows
    ]


def summarize_trial(nct_id: str, con: duckdb.DuckDBPyConnection) -> Optional[TrialSummary]:
    found = _best_trial_snapshot(nct_id, con)
    if found is None:
        return None
    study, source = found
    ps = study.get("protocolSection", {})
    status_mod = ps.get("statusModule", {})
    design_mod = ps.get("designModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    cond_mod = ps.get("conditionsModule", {})

    enrollment = design_mod.get("enrollmentInfo") or {}
    primary_completion, primary_completion_type = _parse_ct_date(
        status_mod.get("primaryCompletionDateStruct")
    )
    completion, completion_type = _parse_ct_date(status_mod.get("completionDateStruct"))
    last_update, _ = _parse_ct_date(status_mod.get("lastUpdatePostDateStruct"))

    return TrialSummary(
        nct_id=nct_id,
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
        source_snapshot=source,
        history=_history_rows(nct_id, con),
    )


def _is_vague(why_stopped: Optional[str]) -> bool:
    if not why_stopped or len(why_stopped.strip()) < 15:
        return True
    lowered = why_stopped.lower()
    return any(p in lowered for p in _VAGUE_WHY_STOPPED_PATTERNS) and len(why_stopped) < 80


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _latest_trial(trials: list[TrialSummary]) -> TrialSummary:
    def key(t: TrialSummary):
        return t.last_update_post_date or date.min
    return max(trials, key=key)


def _staleness_subscore(t: TrialSummary, as_of: date) -> float:
    """0-1: how long since this trial's record was last touched, for EVERY
    status — days-since-last-update is the project's core signal and must
    not be gated on a status string (that was the dead-zone bug: COMPLETED
    trials, which are neither ACTIVE_LIKE nor TERMINAL_STOP, hard-scored
    zero here regardless of staleness). A terminal status with a stated,
    non-vague reason discounts the raw value (accounted-for silence);
    anything else — active, completed, unexplained/vague terminal — scores
    on the raw days alone.

    Exponential saturation (1 - e^-days/tau), not a linear clamp to a fixed
    cap: this project's trials range from days to ~20 years stale (2012+
    universe), and a hard cap turns every "at least this stale" trial —
    which was most of the 2-year-cap's ceiling — into the exact same value,
    a second dead zone at 1.0 instead of 0.0. The exponential keeps
    differentiating arbitrarily-stale trials from each other; it never
    exactly reaches 1.0 for finite days, so ties only happen at genuinely
    identical day-counts."""
    if t.last_update_post_date is None:
        return MISSING_DATE_STALENESS_SUBSCORE
    days = max((as_of - t.last_update_post_date).days, 0)
    raw = 1.0 - math.exp(-days / STALENESS_TAU_DAYS)
    if t.status in TERMINAL_STOP_STATUSES and t.why_stopped and not _is_vague(t.why_stopped):
        return raw * EXPLAINED_TERMINAL_STALENESS_DISCOUNT
    return raw


def _verification_lapse_subscore(t: TrialSummary, as_of: date) -> float:
    """0-1: a distinct, much smaller "sponsor stopped attesting" signal —
    only meaningful for a record CT.gov expects to be periodically
    re-verified while it's still nominally active. Correlates ~0.9 with
    staleness in this project's own EDA (both measure "time since this
    record was touched"), so it carries a small weight rather than being
    zeroed for non-active trials or dropped outright — see module docstring."""
    if t.status not in ACTIVE_LIKE_STATUSES or not t.status_verified_date:
        return 0.0
    days = (as_of - t.status_verified_date).days
    return _clamp(days / VERIFICATION_LAPSE_DAYS_CAP, 0.0, 1.0)


def _status_ambiguity_subscore(t: TrialSummary, as_of: date) -> float:
    """0-1 for a SINGLE trial — compute_silence_score takes the max across
    every trial on the asset, not just the latest, so a multi-trial asset's
    evidence isn't thrown away. Ratios preserved from the original point
    scale (25/20/10/10 out of a 25-point max)."""
    if t.status == "UNKNOWN":
        return 1.0
    if t.status in TERMINAL_STOP_STATUSES and not t.why_stopped:
        return 0.8
    if t.status in TERMINAL_STOP_STATUSES and _is_vague(t.why_stopped):
        return 0.4
    if t.status == "COMPLETED" and not t.has_results and t.completion_date and \
            (as_of - t.completion_date).days > 730:
        return 0.4
    return 0.0


def _enrollment_signal_subscore(t: TrialSummary) -> float:
    """0-1 for a SINGLE trial — record never updated actual enrollment
    despite stopping. Aggregated via max across all trials, same as
    status ambiguity."""
    if t.status in TERMINAL_STOP_STATUSES and t.enrollment_type == "ESTIMATED":
        return 1.0
    if t.status == "COMPLETED" and t.enrollment_type == "ESTIMATED":
        return 1 / 3
    return 0.0


def compute_silence_score(
    trials: list[TrialSummary], as_of: date,
) -> tuple[Optional[int], dict]:
    """Returns (score, breakdown). score is None — not an arbitrary
    midpoint — when there are no resolvable trial snapshots at all; the
    caller must exclude None-score programs from banding rather than park
    them at a fake 50."""
    if not trials:
        return None, {"note": "no trial snapshot available for this candidate's nct_ids"}

    # Staleness and verification lapse describe "is the sponsor still
    # touching this record" — that's an asset-is-alive-if-any-trial-is
    # question, so both look only at the most recently touched trial.
    # Status ambiguity and enrollment signal are per-trial evidence that a
    # multi-trial asset shouldn't lose by only looking at one trial, so
    # those aggregate (max) across every trial on file.
    latest = _latest_trial(trials)
    breakdown = {
        "staleness": round(_staleness_subscore(latest, as_of) * STALENESS_WEIGHT, 1),
        "verification_lapse": round(_verification_lapse_subscore(latest, as_of) * VERIFICATION_LAPSE_WEIGHT, 1),
        "status_ambiguity": round(
            max(_status_ambiguity_subscore(t, as_of) for t in trials) * STATUS_AMBIGUITY_WEIGHT, 1
        ),
        "enrollment_signal": round(
            max(_enrollment_signal_subscore(t) for t in trials) * ENROLLMENT_SIGNAL_WEIGHT, 1
        ),
    }
    score = round(min(100.0, sum(breakdown.values())))
    return int(score), breakdown


def classify_archetypes(trials: list[TrialSummary]) -> list[str]:
    if not trials:
        return ["other"]
    tags = set()
    if any(t.status == "UNKNOWN" for t in trials):
        tags.add("unknown_status")
    for t in trials:
        if t.status in TERMINAL_STOP_STATUSES:
            if _is_vague(t.why_stopped):
                tags.add("registry_terminated_vague_reason")
            else:
                tags.add("registry_terminated_stated_reason")
        if t.status == "COMPLETED" and not t.has_results:
            tags.add("completed_no_results")
    # actively_amended: >=2 distinct posted_dates within the trailing 400 days
    for t in trials:
        recent_dates = {
            h["posted_date"] for h in t.history
            if h["posted_date"] and
            (date.today() - date.fromisoformat(h["posted_date"])).days <= 400
        }
        if len(recent_dates) >= 2:
            tags.add("actively_amended")
            break
    if not tags:
        tags.add("other")
    return sorted(tags)


def contacts_locations_amendment_cadence(trials: list[TrialSummary], as_of: date) -> float:
    """Amendments per year, across all of a program's trials, where
    changed_modules includes "Contacts/Locations" — frequent site-roster
    churn (sites added/removed/updated) is active trial management, the
    opposite signal from silence, and was previously discarded entirely
    (Contacts/Locations amendments only ever showed up as generic
    "untyped" noise, never counted). Exposed as a candidate feature
    alongside silence_score/archetypes, not folded into either — that
    would be a separate, deliberate tuning decision, not made here.
    0.0 if no such amendment exists at all (not None — "no churn
    observed" is itself a real, computable value, unlike history
    coverage which can be genuinely unknown)."""
    count = 0
    first_date: Optional[date] = None
    for t in trials:
        for h in t.history:
            mods = set(h.get("changed_modules") or [])
            if "Contacts/Locations" not in mods or not h.get("posted_date"):
                continue
            count += 1
            d = date.fromisoformat(h["posted_date"])
            if first_date is None or d < first_date:
                first_date = d
    if count == 0 or first_date is None or first_date >= as_of:
        return 0.0
    elapsed_years = (as_of - first_date).days / 365.25
    return count / elapsed_years


def _primary_archetype(tags: list[str]) -> str:
    for a in ARCHETYPES:
        if a in tags:
            return a
    return "other"


def _band_for_score(score: Optional[int]) -> Optional[int]:
    if score is None:
        return None  # no resolvable trials — excluded from banding, not parked at a midpoint
    for i, (lo, hi) in enumerate(SCORE_BANDS):
        if lo <= score < hi:
            return i
    return len(SCORE_BANDS) - 1


def _has_evidence_events_table(con: duckdb.DuckDBPyConnection) -> bool:
    return bool(con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'evidence_events'"
    ).fetchone())


def _typed_events_for_trial(nct_id: str, con: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = con.execute(
        """
        SELECT event_date, event_type, field, direction, from_value, to_value, detail, to_version
        FROM evidence_events WHERE nct_id = ? ORDER BY event_date
        """,
        [nct_id],
    ).fetchall()
    cols = ["date", "event_type", "field", "direction", "from_value", "to_value", "detail", "version"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        if d["date"] is not None and not isinstance(d["date"], str):
            d["date"] = d["date"].isoformat()
        out.append(d)
    return out


def _trial_history_coverage(nct_id: str, con: duckdb.DuckDBPyConnection) -> str:
    """"full" / "partial" / "none" — whether we actually have enough
    fetched history to trust an empty event timeline as "nothing
    happened" rather than "we never looked".

    Diagnosed 2026-08-27: a trial with zero history_index rows renders
    the exact same empty timeline as a trial that was genuinely never
    amended — the single strongest silence signal this project has — and
    the two are visually indistinguishable in the review screen. Missing
    data must never be servable as evidence, so this is computed from the
    same facts the backfill orchestrator itself uses to decide what still
    needs fetching (history/orchestrator.py's `to_fetch` query), not
    guessed at:

    - no history_index row at all -> "none": we have never indexed this
      trial's amendment history. (A version-history refresh is cheap —
      one request — so this should be rare once backfill has run.)
    - indexed, single version only (max version 0) -> "full": this is a
      complete, positive fact — the trial has never been amended since
      registration — not an absence of information.
    - indexed, multi-version: "full" only if every version whose
      changed_modules intersects the signal-label set has had its body
      actually fetched (bodies_fetched_through_version covers it); a
      version with no signal-relevant module change needs no body fetch
      to know nothing tracked happened, so it doesn't count against
      coverage. Otherwise "partial".
    - a backfill_queue row with status='error' caps this at "partial" —
      the last refresh attempt failed, so even existing history_index
      rows might be stale (missing newer versions we don't know about).
    """
    versions = con.execute(
        "SELECT version, changed_modules FROM history_index WHERE nct_id = ?", [nct_id]
    ).fetchall()
    if not versions:
        return "none"

    bq = con.execute(
        "SELECT status, bodies_fetched_through_version FROM backfill_queue WHERE nct_id = ?", [nct_id]
    ).fetchone()
    errored = bool(bq and bq[0] == "error")

    max_version = max(v for v, _ in versions)
    if max_version == 0:
        return "partial" if errored else "full"

    signal_versions = [
        v for v, mods in versions
        if v > 0 and mods and set(mods) & RECOMMENDED_SIGNAL_LABELS
    ]
    max_signal_version = max(signal_versions) if signal_versions else 0
    bodies_fetched = bq[1] if bq and bq[1] is not None else -1

    complete = max_signal_version == 0 or bodies_fetched >= max_signal_version
    if errored or not complete:
        return "partial"
    return "full"


def _program_history_coverage(trial_coverages: list[str]) -> str:
    if not trial_coverages or all(c == "none" for c in trial_coverages):
        return "none"
    if all(c == "full" for c in trial_coverages):
        return "full"
    return "partial"


def _is_expanded_access(nct_id: str, con: duckdb.DuckDBPyConnection) -> bool:
    """True if any indexed version records study_type=EXPANDED_ACCESS —
    an FDA/sponsor expanded-access (compassionate use) program, not a
    clinical trial: no phase, no enrollment, statuses like AVAILABLE /
    NO_LONGER_AVAILABLE that would otherwise register as anomalous
    permanent silence. Confirmed against real data: NCT06099639's
    history_index rows all carry studyType=EXPANDED_ACCESS, and its
    protocolSection.designModule has no phases/enrollmentInfo at all —
    not missing data, a structurally different record type. Checked via
    history_index.study_type (already indexed from CT.gov's /history
    endpoint, no extra fetch) rather than the raw snapshot body."""
    row = con.execute(
        "SELECT count(*) FROM history_index WHERE nct_id = ? AND study_type = 'EXPANDED_ACCESS'",
        [nct_id],
    ).fetchone()
    return bool(row and row[0] > 0)


def build_program(
    candidate_row: dict, con: duckdb.DuckDBPyConnection, as_of: Optional[date] = None,
    combo_trial_shared_with: Optional[dict] = None, has_evidence_events: Optional[bool] = None,
) -> dict:
    as_of = as_of or date.today()
    combo_trial_shared_with = combo_trial_shared_with or {}
    if has_evidence_events is None:
        has_evidence_events = _has_evidence_events_table(con)

    all_nct_ids = candidate_row["nct_ids"]
    excluded_shared_trials = [
        {"nct_id": n, "shared_with": combo_trial_shared_with[n]}
        for n in all_nct_ids if n in combo_trial_shared_with
    ]
    expanded_access_ids = {n for n in all_nct_ids if _is_expanded_access(n, con)}
    excluded_expanded_access_trials = sorted(expanded_access_ids)
    included_ids = [
        n for n in all_nct_ids
        if n not in combo_trial_shared_with and n not in expanded_access_ids
    ]

    trials: list[TrialSummary] = []
    trial_scope: dict[str, str] = {}
    trial_has_mesh: dict[str, bool] = {}
    trial_text_hint: dict[str, Optional[str]] = {}
    for nct_id in included_ids:
        t = summarize_trial(nct_id, con)
        if t is not None:
            trials.append(t)
        # computed independently of whether summarize_trial resolved a
        # snapshot at all — a trial we know nothing about must classify
        # ambiguous, not be silently absent from the scope rollup
        meshes, ancestors = _condition_browse_data(nct_id)
        conditions = t.conditions if t is not None else []
        trial_scope[nct_id] = ts.classify_trial(meshes, ancestors, conditions)
        has_mesh = ts.has_mesh_data(meshes, ancestors)
        trial_has_mesh[nct_id] = has_mesh
        # text fallback ONLY when MeSH is absent — sort-queue hint, never
        # a scope decision (see trial_scope.text_hint_category)
        trial_text_hint[nct_id] = None if has_mesh else ts.text_hint_category(conditions)

    # computed over ALL included trials, not just ones summarize_trial
    # could resolve — a trial with no snapshot at all is exactly the
    # "we never looked" case this exists to catch, not something to skip
    trial_coverage = {n: _trial_history_coverage(n, con) for n in included_ids}
    history_coverage = _program_history_coverage(list(trial_coverage.values()))

    scope_values = list(trial_scope.values())
    scope_category = ts.classify_asset(scope_values)

    # sponsor_class_overrides.json takes precedence over CT.gov's raw,
    # sponsor-selected leadSponsor.class everywhere downstream — both the
    # non_industry_sponsor_hint scope signal and what the review screen
    # displays read effective_class off this enriched list, never the
    # candidate row's raw sponsors_over_time directly.
    sponsors_over_time = ts.apply_sponsor_class_overrides(candidate_row["sponsors_over_time"])

    score, breakdown = compute_silence_score(trials, as_of)
    archetypes = classify_archetypes(trials)
    latest = _latest_trial(trials) if trials else None

    timeline = []
    for t in trials:
        typed_events = _typed_events_for_trial(t.nct_id, con) if has_evidence_events else []
        if typed_events:
            for e in typed_events:
                timeline.append({
                    "nct_id": t.nct_id, "version": e["version"], "date": e["date"],
                    "status": None, "changed_modules": None,
                    "event_type": e["event_type"], "field": e["field"], "direction": e["direction"],
                    "from_value": e["from_value"], "to_value": e["to_value"],
                    "label": e["detail"],
                })
        else:
            for h in t.history:
                mods = set(h["changed_modules"] or [])
                if mods & RECOMMENDED_SIGNAL_LABELS:
                    # A signal-relevant module changed (Study Status, Study
                    # Design, Outcome Measures, Sponsor/Collaborators, Arms
                    # and Interventions) but the differ produced no typed
                    # event for it — a real gap worth checking (differ
                    # hasn't run recently, or genuinely found nothing to
                    # extract from a module that usually carries signal).
                    amendment_kind = "not_yet_extracted"
                    label = "amendment (signal-relevant module changed, not extracted yet)"
                else:
                    # Diagnosed 2026-09-02: NCT06731907 showed ~25 of these
                    # as "no extracted events yet", visually identical to a
                    # real gap. Every changed module here is cosmetic/out-
                    # of-scope (e.g. Contacts/Locations) — the differ
                    # correctly extracted nothing because there was nothing
                    # to extract, not because it failed to look.
                    amendment_kind = "no_signal_modules_changed"
                    label = "amendment (no signal-relevant module changed — correctly filtered)"
                timeline.append({
                    "nct_id": t.nct_id, "version": h["version"],
                    "date": h["posted_date"], "status": h["status"],
                    "changed_modules": h["changed_modules"],
                    "amendment_kind": amendment_kind,
                    "label": label,
                })
        timeline.append({
            "nct_id": t.nct_id, "version": None,
            "date": t.last_update_post_date.isoformat() if t.last_update_post_date else None,
            "status": t.status, "changed_modules": None,
            "label": f"known snapshot ({t.source_snapshot})",
        })
    timeline.sort(key=lambda e: e["date"] or "")

    return {
        "program_id": candidate_row["candidate_id"],
        "candidate_id": candidate_row["candidate_id"],
        "proposed_name": candidate_row["proposed_name"],
        "synonyms": candidate_row["synonyms"] or [],
        "indication_code": "UNSPECIFIED",
        "line_of_therapy": "unspecified",
        "provisional": True,
        "sponsors_over_time": sponsors_over_time,
        "trial_count": len(trials),
        "nct_ids": included_ids,
        "excluded_shared_trials": excluded_shared_trials,
        "excluded_expanded_access_trials": excluded_expanded_access_trials,
        # trial-level MeSH scope classification (trial_scope.py) — a hint,
        # never a silent decision. See scripts/apply_auto_scope_exclusions.py
        # for the only place a "heme_only" verdict here turns into an
        # actual gold record, and only after the validation sample clears
        # trial_scope.AGREEMENT_THRESHOLD.
        "trial_scope": trial_scope,
        "scope_category": scope_category,
        "spans_heme_and_solid": ts.spans_heme_and_solid(scope_values),
        # coverage + fallback bookkeeping — see audit/universe.py's MeSH
        # coverage gate and trial_scope.text_hint_category
        "trial_has_mesh": trial_has_mesh,
        "trial_text_hint": trial_text_hint,
        "non_oncology_hint": ts.is_non_oncology_asset(scope_values),
        "non_industry_sponsor_hint": ts.is_non_industry_sponsor(sponsors_over_time),
        "history_coverage": history_coverage,
        "trial_coverage": trial_coverage,
        "silence_score": score,
        "score_breakdown": breakdown,
        "band": _band_for_score(score),
        "archetypes": archetypes,
        "primary_archetype": _primary_archetype(archetypes),
        "contacts_locations_amendment_cadence": contacts_locations_amendment_cadence(trials, as_of),
        "latest_status": latest.status if latest else None,
        "latest_nct_id": latest.nct_id if latest else None,
        "trials": [t.to_json() for t in trials],
        "timeline": timeline,
        "review_status": candidate_row["review_status"],
        "discovery_strategy": candidate_row.get("discovery_strategy"),
        "match_strength": candidate_row.get("match_strength"),
        "matched_term": candidate_row.get("matched_term"),
        "built_at": datetime.now().isoformat(),
    }


def _asset_candidates_columns(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {
        r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'asset_candidates'"
        ).fetchall()
    }


def build_all_programs(
    con: duckdb.DuckDBPyConnection, as_of: Optional[date] = None,
) -> list[dict]:
    from pharma_stats.discovery.candidates import genuine_combo_trial_ids

    # discovery_strategy/match_strength/matched_term postdate this column's
    # introduction — an asset_candidates table built by an older run of
    # scripts/build_candidate_universe.py won't have them yet. Degrade to
    # NULL rather than hard-crash the app; re-running discovery backfills
    # the real values.
    have_discovery_cols = {"discovery_strategy", "match_strength", "matched_term"} <= _asset_candidates_columns(con)
    discovery_select = (
        "discovery_strategy, match_strength, matched_term" if have_discovery_cols
        else "NULL AS discovery_strategy, NULL AS match_strength, NULL AS matched_term"
    )

    rows = con.execute(
        f"""
        SELECT candidate_id, proposed_name, synonyms, sponsors_over_time,
               nct_ids, review_status, strategies, ambiguous,
               {discovery_select}
        FROM asset_candidates
        ORDER BY candidate_id
        """
    ).fetchall()
    cols = ["candidate_id", "proposed_name", "synonyms", "sponsors_over_time",
            "nct_ids", "review_status", "strategies", "ambiguous",
            "discovery_strategy", "match_strength", "matched_term"]
    candidate_rows = []
    for r in rows:
        row = dict(zip(cols, r))
        if isinstance(row["sponsors_over_time"], str):
            row["sponsors_over_time"] = json.loads(row["sponsors_over_time"])
        candidate_rows.append(row)

    combo_ids = genuine_combo_trial_ids(candidate_rows)
    shared_with: dict[str, list[str]] = {}
    for nct_id in combo_ids:
        names = sorted({c["proposed_name"] for c in candidate_rows if nct_id in (c["nct_ids"] or [])})
        shared_with[nct_id] = names

    has_events = _has_evidence_events_table(con)
    programs = []
    for row in candidate_rows:
        # per-candidate view: exclude *this candidate's own name* from its own "shared with" list
        own_shared = {
            n: [name for name in shared_with[n] if name != row["proposed_name"]]
            for n in (row["nct_ids"] or []) if n in shared_with
        }
        programs.append(build_program(
            row, con, as_of=as_of, combo_trial_shared_with=own_shared, has_evidence_events=has_events,
        ))
    return programs


def materialize(warehouse_db=None, as_of: Optional[date] = None) -> int:
    """(Re)build the provisional_programs table from asset_candidates +
    raw CT.gov snapshots. Safe to rerun any time; fully derived, like
    everything else under data/."""
    warehouse_db = warehouse_db or WAREHOUSE_DB  # resolved at call time, not import time
    con = duckdb.connect(str(warehouse_db))
    try:
        # fully derived/rebuildable — drop rather than migrate in place so
        # schema changes (e.g. a new column) never fail against a table
        # built by an older version of this module
        con.execute("DROP TABLE IF EXISTS provisional_programs")
        con.execute(PROVISIONAL_PROGRAMS_SCHEMA)
        programs = build_all_programs(con, as_of=as_of)
        con.executemany(
            """
            INSERT INTO provisional_programs
                (program_id, candidate_id, proposed_name, synonyms, indication_code,
                 line_of_therapy, provisional, sponsors_over_time, trial_count, nct_ids,
                 excluded_shared_trials, history_coverage, trial_coverage, silence_score,
                 score_breakdown, band, archetypes, primary_archetype, latest_status,
                 latest_nct_id, trials, timeline, review_status,
                 discovery_strategy, match_strength, matched_term,
                 trial_scope, scope_category, spans_heme_and_solid,
                 trial_has_mesh, trial_text_hint,
                 non_oncology_hint, non_industry_sponsor_hint,
                 excluded_expanded_access_trials, contacts_locations_amendment_cadence, built_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    p["program_id"], p["candidate_id"], p["proposed_name"], p["synonyms"],
                    p["indication_code"], p["line_of_therapy"], p["provisional"],
                    json.dumps(p["sponsors_over_time"]), p["trial_count"], p["nct_ids"],
                    json.dumps(p["excluded_shared_trials"]), p["history_coverage"],
                    json.dumps(p["trial_coverage"]),
                    p["silence_score"], json.dumps(p["score_breakdown"]), p["band"],
                    p["archetypes"], p["primary_archetype"], p["latest_status"],
                    p["latest_nct_id"], json.dumps(p["trials"]), json.dumps(p["timeline"]),
                    p["review_status"],
                    p["discovery_strategy"], p["match_strength"], p["matched_term"],
                    json.dumps(p["trial_scope"]), p["scope_category"], p["spans_heme_and_solid"],
                    json.dumps(p["trial_has_mesh"]), json.dumps(p["trial_text_hint"]),
                    p["non_oncology_hint"], p["non_industry_sponsor_hint"],
                    p["excluded_expanded_access_trials"], p["contacts_locations_amendment_cadence"],
                    p["built_at"],
                )
                for p in programs
            ],
        )
        return len(programs)
    finally:
        con.close()


def load_materialized(warehouse_db=None) -> list[dict]:
    warehouse_db = warehouse_db or WAREHOUSE_DB  # resolved at call time, not import time
    if not Path(warehouse_db).exists():
        return []
    con = duckdb.connect(str(warehouse_db), read_only=True)
    try:
        existing = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'provisional_programs'"
        ).fetchone()
        if existing is None:
            return []
        table_cols = {
            r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'provisional_programs'"
            ).fetchall()
        }
        if "trial_has_mesh" not in table_cols or "contacts_locations_amendment_cadence" not in table_cols:
            # built by an older version of this module, before that column
            # existed — treat as unmaterialized rather than hard-crash; the
            # caller's usual "not materialized yet" path rebuilds it fresh
            return []
        rows = con.execute(
            """
            SELECT program_id, candidate_id, proposed_name, synonyms, indication_code,
                   line_of_therapy, provisional, sponsors_over_time, trial_count, nct_ids,
                   excluded_shared_trials, history_coverage, trial_coverage, silence_score,
                   score_breakdown, band, archetypes, primary_archetype, latest_status,
                   latest_nct_id, trials, timeline, review_status,
                   discovery_strategy, match_strength, matched_term,
                   trial_scope, scope_category, spans_heme_and_solid,
                   trial_has_mesh, trial_text_hint,
                   non_oncology_hint, non_industry_sponsor_hint,
                   excluded_expanded_access_trials, contacts_locations_amendment_cadence, built_at
            FROM provisional_programs ORDER BY program_id
            """
        ).fetchall()
    finally:
        con.close()
    cols = [
        "program_id", "candidate_id", "proposed_name", "synonyms", "indication_code",
        "line_of_therapy", "provisional", "sponsors_over_time", "trial_count", "nct_ids",
        "excluded_shared_trials", "history_coverage", "trial_coverage", "silence_score",
        "score_breakdown", "band", "archetypes", "primary_archetype", "latest_status",
        "latest_nct_id", "trials", "timeline", "review_status",
        "discovery_strategy", "match_strength", "matched_term",
        "trial_scope", "scope_category", "spans_heme_and_solid",
        "trial_has_mesh", "trial_text_hint",
        "non_oncology_hint", "non_industry_sponsor_hint",
        "excluded_expanded_access_trials", "contacts_locations_amendment_cadence", "built_at",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("sponsors_over_time", "score_breakdown", "trials", "timeline",
                   "excluded_shared_trials", "trial_coverage", "trial_scope",
                   "trial_has_mesh", "trial_text_hint"):
            if isinstance(d[k], str):
                d[k] = json.loads(d[k])
        out.append(d)
    return out
