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
from pharma_stats.labelling.vocab import APP_VERSION, CONFIDENCE_LEVELS, KILL_REASONS, PROGRAM_STATUSES

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
    labelled = store.labelled_program_ids(records)
    invalid = store.invalid_flagged_ids(records)

    session = q.load_session()
    if session is None:
        session = q.new_session(programs, exclude_ids=labelled | invalid)
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


def _program_public(program: dict, *, reveal: bool) -> dict:
    name = program["proposed_name"]
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
        "latest_status": program["latest_status"],
        "trials": program["trials"],
        "timeline": program["timeline"],
        "review_status": program["review_status"],
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
        "app_version": APP_VERSION,
    }


@app.get("/api/next")
def next_program(blind: bool = True):
    with _lock:
        session = _state["session"]
        _sweep_stale_pending()

        records = store.load_records()
        labelled = store.labelled_program_ids(records)

        program = None
        is_repeat = False
        while True:
            program_id, is_repeat = q.pop_next(session, labelled)
            if program_id is None:
                q.save_session(session)
                return JSONResponse({"done": True, "message": "Queue exhausted — every provisional program has been labelled or flagged."})
            program = _state["programs_by_id"].get(program_id)
            if program is not None:
                break
            # Program vanished from the materialized set since the session
            # was built (e.g. warehouse rebuilt); drop it and try the next one.

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
    action: str  # "label" | "skip" | "flag_invalid"
    status: Optional[str] = None
    kill_reason: Optional[str] = None
    confidence: Optional[str] = None
    evidence_note: Optional[str] = None
    label_evidence_date: Optional[str] = None
    public_confirmation_date: Optional[str] = None
    never_publicly_confirmed: bool = False
    blind: bool = True
    seconds_spent: Optional[float] = None


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
        body["is_repeat_probe"] = pending["is_repeat_probe"]

        try:
            store.validate_label_payload(body)
        except store.ValidationError as e:
            # put it back — a rejected save must not lose the reviewer's place
            session["pending_serve"][payload.serve_token] = pending
            q.save_session(session)
            raise HTTPException(422, str(e))

        record = store.build_record(body, session_id=session["session_id"], served_stratum=pending)
        store.append_record(record)

        if payload.action in ("skip",) and not pending["is_repeat_probe"]:
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

        labelled = store.labelled_program_ids(records)
        invalid = store.invalid_flagged_ids(records)

        return {
            "session_id": session["session_id"],
            "app_version": APP_VERSION,
            "total_programs": len(programs),
            "labelled_count": len(labelled),
            "invalid_flagged_count": len(invalid),
            "remaining_fresh_in_queue": len(session["order"]),
            **label_stats.blind_counts(records),
            "median_seconds_per_label": label_stats.median_seconds_per_label(records),
            "stratum_progress": label_stats.stratum_progress(programs, labelled),
            "self_consistency": label_stats.self_consistency(records),
        }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
