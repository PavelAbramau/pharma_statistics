"""Local labelling app: FastAPI server + single-page vanilla-JS frontend.

Launch with ``python scripts/run_labelling_app.py``. Everything is local:
no model-assisted labelling, no network calls other than the outbound
reference links the reviewer clicks themselves.
"""
from __future__ import annotations

import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import stats as label_stats
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.labelling.vocab import (
    APP_VERSION, CONFIDENCE_LEVELS, IN_SCOPE_VALUES, IS_ADC_VALUES, KILL_REASONS,
    PROGRAM_STATUSES, SCOPE_OUT_REASONS,
)

STATIC_DIR = Path(__file__).with_name("static")

_lock = threading.Lock()

app = FastAPI(title="ADC program labelling")


def _build_or_load_programs() -> list[dict]:
    programs = pp.load_materialized()
    if not programs:
        print("No provisional_programs table found — materializing from asset_candidates "
              "+ raw CT.gov snapshots (one-time, may take a minute)...")
        pp.materialize()
        programs = pp.load_materialized()
    return programs


_state: dict[str, Any] = {}


def _init_state() -> None:
    programs = _build_or_load_programs()
    programs_by_id = {p["program_id"]: p for p in programs}
    records = store.load_records()
    reviewed = store.reviewed_program_ids(records)  # any gate (1, 2, or 3) — all terminal

    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=reviewed)
        q.save_session(session)

    _state.update(programs=programs, programs_by_id=programs_by_id, session=session)


_init_state()


def _sweep_stale_pending(ttl_seconds: int = 1800) -> None:
    session = _state["session"]
    now = datetime.now(timezone.utc)
    stale_tokens = []
    for token, entry in session["pending_serve"].items():
        served_at = datetime.fromisoformat(entry["served_at"])
        if (now - served_at).total_seconds() > ttl_seconds:
            stale_tokens.append(token)
    for token in stale_tokens:
        entry = session["pending_serve"].pop(token)
        if not entry["is_repeat_probe"]:
            q.requeue(session, entry["program_id"])


def _validation_sample_ids() -> set[str]:
    """Program ids held out from auto-exclusion for the blind agreement
    check (universe.py's audit stage). Read fresh each call — this file
    is tiny and can change underneath a running app via
    scripts/apply_auto_scope_exclusions.py."""
    return {item["program_id"] for item in ts.load_validation_sample()}


def _program_public(program: dict, *, reveal: bool) -> dict:
    name = program["proposed_name"]
    # The 30-program blind validation sample must look exactly like any
    # other candidate — no scope hint, no pre-fill, nothing that tips off
    # the reviewer that the classifier already has an opinion here.
    in_validation_sample = program["program_id"] in _validation_sample_ids()
    out = {
        "program_id": program["program_id"],
        "candidate_id": program["candidate_id"],
        "proposed_name": name,
        "synonyms": program["synonyms"],
        "indication_code": program["indication_code"],
        "line_of_therapy": program["line_of_therapy"],
        "provisional": program["provisional"],
        "sponsors_over_time": program["sponsors_over_time"],
        "trial_count": program["trial_count"],
        "nct_ids": program["nct_ids"],
        "excluded_shared_trials": program["excluded_shared_trials"],
        # data-quality, not a model opinion — always visible, never gated by blind mode
        "history_coverage": program["history_coverage"],
        "trial_coverage": program["trial_coverage"],
        "latest_status": program["latest_status"],
        "trials": program["trials"],
        "timeline": program["timeline"],
        "review_status": program["review_status"],
        # discovery provenance — why this candidate exists at all. Also not
        # a model opinion, and directly informs the Gate 1 is_adc judgement,
        # so always visible regardless of blind mode.
        "discovery_strategy": program.get("discovery_strategy"),
        "match_strength": program.get("match_strength"),
        "matched_term": program.get("matched_term"),
        # scope is evaluated at trial level, not asset level (CLAUDE.md) —
        # this asset-wide provisional program bundles every trial, so Gate
        # 2 needs to see whether they actually agree before the reviewer
        # picks one in_scope answer for all of them. Withheld entirely for
        # the blind validation sample (see in_validation_sample above).
        "trial_scope": None if in_validation_sample else program.get("trial_scope"),
        "scope_category": None if in_validation_sample else program.get("scope_category"),
        "spans_heme_and_solid": False if in_validation_sample else program.get("spans_heme_and_solid", False),
        # cheap MeSH-derived hints for the other two scope_reasons — these
        # aren't part of the heme_only auto-exclusion/validation mechanism,
        # so they're never withheld: pre-fill and sort, never delete.
        "non_oncology_hint": program.get("non_oncology_hint", False),
        "non_industry_sponsor_hint": program.get("non_industry_sponsor_hint", False),
        "links": {
            "pubmed_search": "https://pubmed.ncbi.nlm.nih.gov/?term="
            + urllib.parse.quote(name),
            "web_search_discontinued": "https://www.google.com/search?q="
            + urllib.parse.quote(f'"{name}" discontinued'),
        },
    }
    if reveal:
        out["silence_score"] = program["silence_score"]
        out["score_breakdown"] = program["score_breakdown"]
        out["band"] = program["band"]
        out["archetypes"] = program["archetypes"]
        out["primary_archetype"] = program["primary_archetype"]
    return out


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/vocab")
def vocab():
    return {
        "statuses": PROGRAM_STATUSES,
        "kill_reasons": KILL_REASONS,
        "confidence_levels": CONFIDENCE_LEVELS,
        "is_adc_values": IS_ADC_VALUES,
        "in_scope_values": IN_SCOPE_VALUES,
        "scope_out_reasons": SCOPE_OUT_REASONS,
        "app_version": APP_VERSION,
    }


@app.get("/api/next")
def next_program(blind: bool = True):
    with _lock:
        session = _state["session"]
        _sweep_stale_pending()

        records = store.load_records()
        # repeat-probe self-consistency re-serves only ever come from fully
        # (gate 3) labelled programs — a gate-1/2 triage rejection has no
        # status/kill_reason to compare against, see stats.self_consistency
        fully_labelled = store.fully_labelled_program_ids(records)

        program = None
        is_repeat = False
        # Bounded, not infinite: skipped-for-coverage programs get
        # requeued (a future backfill may fix them), so an unbounded loop
        # here could spin forever if nothing in the queue has full
        # coverage yet.
        max_attempts = len(session["order"]) + len(fully_labelled) + 2
        for _ in range(max_attempts):
            program_id, is_repeat = q.pop_next(session, fully_labelled)
            if program_id is None:
                q.save_session(session)
                return JSONResponse({"done": True, "message": "Queue exhausted — every provisional program has been labelled or flagged."})
            program = _state["programs_by_id"].get(program_id)
            if program is None:
                # Program vanished from the materialized set since the session
                # was built (e.g. warehouse rebuilt); drop it and try the next one.
                continue
            if program["history_coverage"] != "full":
                # Refuse to serve: an empty/incomplete event timeline here
                # is visually indistinguishable from "genuinely never
                # amended", the strongest silence signal there is. Missing
                # data must never be servable as evidence. Requeue rather
                # than drop — a later backfill pass may complete it.
                if not is_repeat:
                    q.requeue(session, program_id)
                program = None
                continue
            break

        if program is None:
            q.save_session(session)
            return JSONResponse({
                "done": True, "insufficient_coverage": True,
                "message": "No remaining programs have full history_coverage. "
                           "Run a backfill pass before continuing — see the gold_set audit stage.",
            })

        serve_token = q.make_serve_token(session, program, is_repeat)
        q.save_session(session)

        return {
            "done": False,
            "serve_token": serve_token,
            "session_id": session["session_id"],
            "program": _program_public(program, reveal=not blind),
        }


class LabelPayload(BaseModel):
    serve_token: str
    action: str  # "label" | "skip"
    gate_reached: Optional[int] = None  # 1, 2, or 3 — see vocab.py
    status: Optional[str] = None
    kill_reason: Optional[str] = None
    confidence: Optional[str] = None
    is_adc: Optional[str] = None
    in_scope: Optional[str] = None
    scope_reason: Optional[str] = None
    evidence_note: Optional[str] = None
    label_evidence_date: Optional[str] = None
    public_confirmation_date: Optional[str] = None
    never_publicly_confirmed: bool = False
    blind: bool = True
    seconds_spent: Optional[float] = None
    status_revised_after_external_search: bool = False


@app.post("/api/labels")
def submit_label(payload: LabelPayload):
    with _lock:
        session = _state["session"]
        pending = session["pending_serve"].pop(payload.serve_token, None)
        if pending is None:
            raise HTTPException(400, "unknown or expired serve_token — try fetching the next program again")

        program_id = pending["program_id"]
        program = _state["programs_by_id"].get(program_id)

        body = payload.model_dump()
        body["program_id"] = program_id
        body["candidate_id"] = program["candidate_id"] if program else None
        body["proposed_name"] = program["proposed_name"] if program else None
        # discovery provenance is stamped from the server's own record of
        # this candidate, never trusted from the client — it's the whole
        # point of the Gate-1 rejection pattern counter that this is the
        # discovery pipeline's own account of why the candidate exists.
        body["discovery_strategy"] = program["discovery_strategy"] if program else None
        body["match_strength"] = program["match_strength"] if program else None
        body["matched_term"] = program["matched_term"] if program else None
        body["is_repeat_probe"] = pending["is_repeat_probe"]
        body["history_coverage_at_serve_time"] = pending.get("history_coverage")

        try:
            store.validate_label_payload(body)
        except store.ValidationError as e:
            # put it back — a rejected save must not lose the reviewer's place
            session["pending_serve"][payload.serve_token] = pending
            q.save_session(session)
            raise HTTPException(422, str(e))

        record = store.build_record(body, session_id=session["session_id"], served_stratum=pending)
        store.append_record(record)

        # skip ("come back later") is the only outcome that gets requeued —
        # every gate outcome (1, 2, or 3) is terminal: the program has been
        # reviewed and must never be re-served (see store.reviewed_program_ids).
        if payload.action == "skip" and not pending["is_repeat_probe"]:
            q.requeue(session, program_id)

        q.save_session(session)

        reveal = None
        if program is not None:
            reveal = {
                "silence_score": program["silence_score"],
                "score_breakdown": program["score_breakdown"],
                "band": program["band"],
                "archetypes": program["archetypes"],
                "primary_archetype": program["primary_archetype"],
            }
        return {"ok": True, "reveal": reveal}


@app.get("/api/session")
def session_stats():
    with _lock:
        session = _state["session"]
        programs = _state["programs"]
        records = store.load_records()

        fully_labelled = store.fully_labelled_program_ids(records)

        return {
            "session_id": session["session_id"],
            "app_version": APP_VERSION,
            "total_programs": len(programs),
            # "labelled" means gate 3 only — triage rejections never count
            # toward this or the stratum/target numbers below.
            "labelled_count": len(fully_labelled),
            "remaining_fresh_in_queue": len(session["order"]),
            **label_stats.gate_counts(records),
            **label_stats.blind_counts(records),
            "median_seconds_per_label": label_stats.median_seconds_per_label(records),
            "stratum_progress": label_stats.stratum_progress(programs, fully_labelled),
            "self_consistency": label_stats.self_consistency(records),
            "gate1_rejection_pattern_counts": label_stats.gate1_rejection_pattern_counts(records),
        }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
