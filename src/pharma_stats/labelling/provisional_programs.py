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
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import duckdb

from pharma_stats import snapshot as snap
from pharma_stats.config import WAREHOUSE_DB

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
    built_at                           TIMESTAMP
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


def compute_silence_score(
    trials: list[TrialSummary], as_of: date,
) -> tuple[int, dict]:
    if not trials:
        return 50, {"note": "no trial snapshot available for this candidate's nct_ids"}

    latest = _latest_trial(trials)
    breakdown: dict[str, float] = {}

    # 1. staleness: days since the sponsor last touched a still-nominally-active record
    if latest.status in ACTIVE_LIKE_STATUSES and latest.last_update_post_date:
        days = (as_of - latest.last_update_post_date).days
        breakdown["staleness"] = round(_clamp(days / 18, 0, 40), 1)
    elif latest.status in ACTIVE_LIKE_STATUSES:
        breakdown["staleness"] = 20.0  # active status but no update date on file
    else:
        breakdown["staleness"] = 0.0

    # 2. status ambiguity
    if any(t.status == "UNKNOWN" for t in trials):
        breakdown["status_ambiguity"] = 25.0
    elif latest.status in TERMINAL_STOP_STATUSES and not latest.why_stopped:
        breakdown["status_ambiguity"] = 20.0
    elif latest.status in TERMINAL_STOP_STATUSES and _is_vague(latest.why_stopped):
        breakdown["status_ambiguity"] = 10.0
    elif latest.status in TERMINAL_STOP_STATUSES:
        breakdown["status_ambiguity"] = 0.0
    elif latest.status == "COMPLETED" and not latest.has_results and latest.completion_date and \
            (as_of - latest.completion_date).days > 730:
        breakdown["status_ambiguity"] = 10.0
    else:
        breakdown["status_ambiguity"] = 0.0

    # 3. enrollment signal: record never updated actual enrollment despite stopping
    if latest.status in TERMINAL_STOP_STATUSES and latest.enrollment_type == "ESTIMATED":
        breakdown["enrollment_signal"] = 15.0
    elif latest.status == "COMPLETED" and latest.enrollment_type == "ESTIMATED":
        breakdown["enrollment_signal"] = 5.0
    else:
        breakdown["enrollment_signal"] = 0.0

    # 4. verification lapse: how long since the sponsor last re-verified an active record
    if latest.status in ACTIVE_LIKE_STATUSES and latest.status_verified_date:
        days = (as_of - latest.status_verified_date).days
        breakdown["verification_lapse"] = round(_clamp(days / 24, 0, 15), 1)
    else:
        breakdown["verification_lapse"] = 0.0

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


def _primary_archetype(tags: list[str]) -> str:
    for a in ARCHETYPES:
        if a in tags:
            return a
    return "other"


def _band_for_score(score: int) -> int:
    for i, (lo, hi) in enumerate(SCORE_BANDS):
        if lo <= score < hi:
            return i
    return len(SCORE_BANDS) - 1


def build_program(
    candidate_row: dict, con: duckdb.DuckDBPyConnection, as_of: Optional[date] = None,
) -> dict:
    as_of = as_of or date.today()
    trials: list[TrialSummary] = []
    for nct_id in candidate_row["nct_ids"]:
        t = summarize_trial(nct_id, con)
        if t is not None:
            trials.append(t)

    score, breakdown = compute_silence_score(trials, as_of)
    archetypes = classify_archetypes(trials)
    latest = _latest_trial(trials) if trials else None

    timeline = []
    for t in trials:
        for h in t.history:
            timeline.append({
                "nct_id": t.nct_id, "version": h["version"],
                "date": h["posted_date"], "status": h["status"],
                "changed_modules": h["changed_modules"],
                "label": "amendment (untyped — EvidenceEvent extraction not built yet)",
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
        "sponsors_over_time": candidate_row["sponsors_over_time"],
        "trial_count": len(trials),
        "nct_ids": candidate_row["nct_ids"],
        "silence_score": score,
        "score_breakdown": breakdown,
        "band": _band_for_score(score),
        "archetypes": archetypes,
        "primary_archetype": _primary_archetype(archetypes),
        "latest_status": latest.status if latest else None,
        "latest_nct_id": latest.nct_id if latest else None,
        "trials": [t.to_json() for t in trials],
        "timeline": timeline,
        "review_status": candidate_row["review_status"],
        "built_at": datetime.now().isoformat(),
    }


def build_all_programs(
    con: duckdb.DuckDBPyConnection, as_of: Optional[date] = None,
) -> list[dict]:
    rows = con.execute(
        """
        SELECT candidate_id, proposed_name, synonyms, sponsors_over_time,
               nct_ids, review_status
        FROM asset_candidates
        ORDER BY candidate_id
        """
    ).fetchall()
    cols = ["candidate_id", "proposed_name", "synonyms", "sponsors_over_time", "nct_ids", "review_status"]
    programs = []
    for r in rows:
        row = dict(zip(cols, r))
        if isinstance(row["sponsors_over_time"], str):
            row["sponsors_over_time"] = json.loads(row["sponsors_over_time"])
        programs.append(build_program(row, con, as_of=as_of))
    return programs


def materialize(warehouse_db=None, as_of: Optional[date] = None) -> int:
    """(Re)build the provisional_programs table from asset_candidates +
    raw CT.gov snapshots. Safe to rerun any time; fully derived, like
    everything else under data/."""
    warehouse_db = warehouse_db or WAREHOUSE_DB  # resolved at call time, not import time
    con = duckdb.connect(str(warehouse_db))
    try:
        con.execute(PROVISIONAL_PROGRAMS_SCHEMA)
        programs = build_all_programs(con, as_of=as_of)
        con.execute("DELETE FROM provisional_programs")
        con.executemany(
            """
            INSERT INTO provisional_programs
                (program_id, candidate_id, proposed_name, synonyms, indication_code,
                 line_of_therapy, provisional, sponsors_over_time, trial_count, nct_ids,
                 silence_score, score_breakdown, band, archetypes, primary_archetype,
                 latest_status, latest_nct_id, trials, timeline, review_status, built_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    p["program_id"], p["candidate_id"], p["proposed_name"], p["synonyms"],
                    p["indication_code"], p["line_of_therapy"], p["provisional"],
                    json.dumps(p["sponsors_over_time"]), p["trial_count"], p["nct_ids"],
                    p["silence_score"], json.dumps(p["score_breakdown"]), p["band"],
                    p["archetypes"], p["primary_archetype"], p["latest_status"],
                    p["latest_nct_id"], json.dumps(p["trials"]), json.dumps(p["timeline"]),
                    p["review_status"], p["built_at"],
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
        rows = con.execute(
            """
            SELECT program_id, candidate_id, proposed_name, synonyms, indication_code,
                   line_of_therapy, provisional, sponsors_over_time, trial_count, nct_ids,
                   silence_score, score_breakdown, band, archetypes, primary_archetype,
                   latest_status, latest_nct_id, trials, timeline, review_status, built_at
            FROM provisional_programs ORDER BY program_id
            """
        ).fetchall()
    finally:
        con.close()
    cols = [
        "program_id", "candidate_id", "proposed_name", "synonyms", "indication_code",
        "line_of_therapy", "provisional", "sponsors_over_time", "trial_count", "nct_ids",
        "silence_score", "score_breakdown", "band", "archetypes", "primary_archetype",
        "latest_status", "latest_nct_id", "trials", "timeline", "review_status", "built_at",
    ]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        for k in ("sponsors_over_time", "score_breakdown", "trials", "timeline"):
            if isinstance(d[k], str):
                d[k] = json.loads(d[k])
        out.append(d)
    return out
