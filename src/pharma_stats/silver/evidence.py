"""Evidence gathering for the silver auto-labeller.

Scope, stated honestly: this first wiring answers every decomposed
question from CT.gov's own registry data ONLY (whyStopped text, status,
dates, evidence_events). silver/retrieval_agent.py's SEC EDGAR search is
NOT wired into this evidence bundle yet — that's a separate integration.
Expect the discontinuation-statement and stop-reason questions to abstain
often when whyStopped is empty or vague, and expect the successor-asset
question to abstain ALWAYS in this run (see silver/prompts.py) — there is
no citable evidence source for it yet. That's not a bug: abstention on
evidence that doesn't exist is the point.
"""
from __future__ import annotations

# Cap on timeline events sent to the model — a dozen arm_added/arm_removed
# lines contribute nothing to either decomposed question and just burn
# input tokens. why_stopped and current status are always included
# regardless (they live on the per-trial record in build_evidence, never
# in this trimmed timeline), so this cap never risks losing them.
MAX_TIMELINE_EVENTS = 15
_PRIORITY_EVENT_TYPES = {"status_changed", "enrollment_target_changed", "completion_date_pushed"}
_TERMINAL_STATUS_VALUES = {"TERMINATED", "WITHDRAWN", "SUSPENDED", "COMPLETED"}


def _event_priority(e: dict) -> int:
    """0 = a status_changed event landing on a terminal status (the
    single strongest silence/kill signal there is), 1 = the other
    decision-relevant event types, 2 = everything else (arm changes,
    generic amendments, ...)."""
    if e.get("event_type") == "status_changed" and (e.get("to_value") or "") in _TERMINAL_STATUS_VALUES:
        return 0
    if e.get("event_type") in _PRIORITY_EVENT_TYPES:
        return 1
    return 2


def _trim_timeline(events: list[dict]) -> list[dict]:
    """Keep the MAX_TIMELINE_EVENTS most decision-relevant events, not
    just the most recent N chronologically — a terminal transition from
    18 months ago must not be pushed out by a dozen recent arm_added
    lines. Within a priority tier, most-recent-first; the kept set is
    re-sorted chronologically ascending afterward for readable output."""
    if len(events) <= MAX_TIMELINE_EVENTS:
        return sorted(events, key=lambda e: e.get("date") or "")

    tiers: dict[int, list[dict]] = {}
    for e in events:
        tiers.setdefault(_event_priority(e), []).append(e)

    kept: list[dict] = []
    for tier in sorted(tiers):
        kept.extend(sorted(tiers[tier], key=lambda e: e.get("date") or "", reverse=True))
    kept = kept[:MAX_TIMELINE_EVENTS]
    return sorted(kept, key=lambda e: e.get("date") or "")


def citation_locator(nct_id: str, source_snapshot: "str | None") -> str:
    """The snapshot key a claim about this trial can actually be verified
    against. provisional_programs._best_trial_snapshot prefers the
    highest-indexed VERSIONED body when one has been backfilled — most
    trials in this project have — so citing the bare nct_id when the
    evidence really came from "versioned:vN" fails the citation gate
    (snapshot.latest resolves a different, often nonexistent, snapshot).
    source_snapshot comes straight from TrialSummary.to_json()."""
    if source_snapshot and source_snapshot.startswith("versioned:v"):
        version = source_snapshot.split("versioned:v", 1)[1]
        return f"ctgov:{nct_id}:v{version}"
    return f"ctgov:{nct_id}"


def build_evidence(program: dict) -> dict:
    trials = []
    for t in program.get("trials", []):
        trials.append({
            "nct_id": t["nct_id"],
            "status": t.get("status"),
            "why_stopped": t.get("why_stopped"),
            "completion_date": t.get("completion_date"),
            "last_update_post_date": t.get("last_update_post_date"),
            "start_date": t.get("start_date"),
            "source_snapshot": t.get("source_snapshot"),
        })
    raw_events = [e for e in program.get("timeline", []) if e.get("event_type")]
    trimmed = _trim_timeline(raw_events)
    timeline = [
        {"nct_id": e.get("nct_id"), "date": e.get("date"), "event_type": e.get("event_type"),
         "label": e.get("label")}
        for e in trimmed
    ]
    return {"trials": trials, "timeline": timeline}


def source_snapshot_for(evidence: dict, nct_id: str) -> "str | None":
    """Look up which snapshot a given nct_id's facts came from, so a
    citation built from a model's answer (which only gives back the
    nct_id) can point at the right one — see citation_locator."""
    for t in evidence.get("trials") or []:
        if t["nct_id"] == nct_id:
            return t.get("source_snapshot")
    return None


def evidence_text(evidence: dict) -> str:
    """Plain-text rendering for the prompt. Every fact here is citable as
    a raw_snapshot citation — see citation_locator() for the exact key,
    which depends on whether this trial's facts came from a versioned or
    bare snapshot. The quoted substring must appear in that trial's
    actual raw CT.gov body for the citation gate (silver/citations.py)
    to pass."""
    lines = []
    for t in evidence.get("trials") or []:
        lines.append(
            f"[{t['nct_id']}] status={t['status']}, why_stopped={t['why_stopped']!r}, "
            f"start_date={t['start_date']}, completion_date={t['completion_date']}, "
            f"last_update={t['last_update_post_date']}"
        )
    for e in evidence.get("timeline") or []:
        lines.append(f"[{e['nct_id']}] {e['date']}: {e['event_type']} — {e['label']}")
    return "\n".join(lines) if lines else "(no trial data on file for this program)"


def trials_initiated_since(program: dict, since_date: str) -> bool:
    """Q1, answered deterministically from warehouse data — no model call
    needed or wanted for a fact we already know exactly. "Initiated"
    means a trial start_date on or after since_date."""
    for t in program.get("trials", []):
        start = t.get("start_date")
        if start and start >= since_date:
            return True
    return False
