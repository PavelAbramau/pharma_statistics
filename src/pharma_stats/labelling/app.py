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

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import queue_order as qo
from pharma_stats.labelling import stats as label_stats
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.labelling import triage_serve
from pharma_stats.labelling.vocab import (
    APP_VERSION, CONFIDENCE_LEVELS, CONFIRMATION_EVIDENCE_TYPES, GATE2_SCOPE_OUT_REASONS,
    IN_SCOPE_VALUES, IS_ADC_VALUES, KILL_REASONS, PROGRAM_STATUSES, SCOPE_OUT_REASONS,
)
from pharma_stats.triage import evidence as tev
from pharma_stats.triage import grounding
from pharma_stats.triage import staging as triage_staging
from pharma_stats.triage import validation as tval

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
    triage_serve.ingest_reopens(session, set(programs_by_id))
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


def _triage_holdout_ids() -> set[str]:
    return {d["program_id"] for d in tval.load_validation_sample()}


def _serve_plan_for(program: dict, *, reopened: bool, gold_records: list[dict]) -> triage_serve.ServePlan:
    heme_holdout = _validation_sample_ids()
    triage_holdout = _triage_holdout_ids()
    heme_auto_ok, _ = triage_serve.heme_only_auto_exclude_allowed(_state["programs"], gold_records)
    model_ok, _ = triage_serve.model_layer_gate_passed(gold_records)
    staged = triage_staging.latest_by_program(triage_staging.load_records())
    pid = program["program_id"]
    return triage_serve.serve_plan(
        program,
        reopened=reopened,
        heme_holdout=pid in heme_holdout,
        triage_holdout=pid in triage_holdout,
        heme_auto_ok=heme_auto_ok,
        model_gate_ok=model_ok,
        staged_record=staged.get(pid),
    )


def _program_public(
    program: dict, *, reveal: bool, start_gate: int = 1,
    reopened: bool = False, triage_context: Optional[dict] = None,
) -> dict:
    name = program["proposed_name"]
    # The 30-program blind validation sample must look exactly like any
    # other candidate — no scope hint, no pre-fill, nothing that tips off
    # the reviewer that the classifier already has an opinion here.
    in_validation_sample = program["program_id"] in _validation_sample_ids()
    hide_triage = in_validation_sample or program["program_id"] in _triage_holdout_ids()
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
        "start_gate": 1 if hide_triage else start_gate,
        "reopened": reopened,
        "triage_context": None if hide_triage else triage_context,
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
        "gate2_scope_out_reasons": GATE2_SCOPE_OUT_REASONS,
        "confirmation_evidence_types": CONFIRMATION_EVIDENCE_TYPES,
        "app_version": APP_VERSION,
    }


@app.get("/api/next")
def next_program(blind: bool = True):
    with _lock:
        session = _state["session"]
        _sweep_stale_pending()

        records = store.load_records()
        reviewed = store.reviewed_program_ids(records)
        # repeat-probe self-consistency re-serves only ever come from fully
        # (gate 3) labelled programs — a gate-1/2 triage rejection has no
        # status/kill_reason to compare against, see stats.self_consistency
        fully_labelled = store.fully_labelled_program_ids(records)

        triage_serve.ingest_reopens(session, set(_state["programs_by_id"]))
        reopen_queue = session.setdefault("reopen_queue", [])

        program = None
        is_repeat = False
        serving_reopen = False
        plan = None
        # Bounded, not infinite: skipped-for-coverage programs get
        # requeued (a future backfill may fix them), so an unbounded loop
        # here could spin forever if nothing in the queue has full
        # coverage yet.
        active_key = q._active_list_key(session)
        max_attempts = len(session.get(active_key, [])) + len(reopen_queue) + len(fully_labelled) + 2
        for _ in range(max_attempts):
            serving_reopen = False
            if reopen_queue:
                program_id = reopen_queue.pop(0)
                is_repeat = False
                session["total_served"] = session.get("total_served", 0) + 1
                serving_reopen = True
            else:
                program_id, is_repeat = q.pop_next(session, fully_labelled)
            if program_id is None:
                q.save_session(session)
                return JSONResponse({"done": True, "message": "Queue exhausted — every provisional program has been labelled or flagged."})
            program = _state["programs_by_id"].get(program_id)
            if program is None:
                # Program vanished from the materialized set since the session
                # was built (e.g. warehouse rebuilt); drop it and try the next one.
                continue
            if program_id in reviewed and not serving_reopen:
                # Auto-triage (or a later gold line) landed since this
                # session was built — drop, don't requeue.
                continue
            if program["history_coverage"] != "full":
                # Refuse to serve: an empty/incomplete event timeline here
                # is visually indistinguishable from "genuinely never
                # amended", the strongest silence signal there is. Missing
                # data must never be servable as evidence. Requeue rather
                # than drop — a later backfill pass may complete it.
                if not is_repeat and not serving_reopen:
                    q.requeue(session, program_id)
                elif serving_reopen:
                    reopen_queue.append(program_id)
                program = None
                continue
            plan = _serve_plan_for(program, reopened=serving_reopen, gold_records=records)
            if plan.skip and not serving_reopen:
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
            "program": _program_public(
                program, reveal=not blind,
                start_gate=plan.start_gate if plan else 1,
                reopened=serving_reopen,
                triage_context=plan.context if plan else None,
            ),
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
    confirmation_evidence_type: Optional[str] = None
    never_publicly_confirmed: bool = False
    third_party_first_noted_date: Optional[str] = None
    third_party_source: Optional[str] = None
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

        heme_auto_ok, _ = triage_serve.heme_only_auto_exclude_allowed(programs, records)
        model_ok, _ = triage_serve.model_layer_gate_passed(records)
        remaining = list(session.get("reopen_queue") or []) + list(session.get("order") or [])
        composition = triage_serve.queue_composition(
            programs, remaining,
            gold_records=records,
            heme_auto_ok=heme_auto_ok,
            model_gate_ok=model_ok,
            heme_holdout_ids=_validation_sample_ids(),
            triage_holdout_ids=_triage_holdout_ids(),
            reopen_ids=session.get("reopen_queue") or [],
        )

        return {
            "session_id": session["session_id"],
            "app_version": APP_VERSION,
            "active_queue": session.get("active_queue", "main"),
            "main_queue_remaining": len(session.get("order") or []),
            "validation_queue_remaining": len(session.get("validation_order") or []),
            "total_programs": len(programs),
            # "labelled" means gate 3 only — triage rejections never count
            # toward this or the stratum/target numbers below.
            "labelled_count": len(fully_labelled),
            "remaining_fresh_in_queue": composition["manual_queue"],
            **label_stats.gate_counts(records),
            **label_stats.blind_counts(records),
            "median_seconds_per_label": label_stats.median_seconds_per_label(records),
            "stratum_progress": label_stats.stratum_progress(programs, fully_labelled),
            "self_consistency": label_stats.self_consistency(records),
            "gate1_rejection_pattern_counts": label_stats.gate1_rejection_pattern_counts(records),
            "queue_enter_gate1": composition["enter_gate1"],
            "queue_enter_gate2": composition["enter_gate2"],
            "queue_enter_gate3": composition["enter_gate3"],
            "hours_left_to_target": composition["hours_left_to_target"],
            "remaining_to_target": composition["remaining_to_target"],
            "gate3_target": composition["gate3_target"],
        }


class SwitchQueuePayload(BaseModel):
    queue: str  # "main" | "validation"


@app.post("/api/switch_queue")
def switch_queue(payload: SwitchQueuePayload):
    with _lock:
        session = _state["session"]
        try:
            q.switch_queue(session, payload.queue)
        except ValueError as e:
            raise HTTPException(422, str(e))
        q.save_session(session)
        return {
            "ok": True, "active_queue": session["active_queue"],
            "main_queue_remaining": len(session.get("order") or []),
            "validation_queue_remaining": len(session.get("validation_order") or []),
        }


@app.get("/api/bulk_reject_candidates")
def bulk_reject_candidates(limit: int = 40):
    """Likely-reject bucket only (see labelling/queue_order.py) — obvious
    non-ADCs the disposition ordering already fast-tracks, offered here
    for one-reason multi-select rejection instead of one card at a time.
    Never includes a blind-holdout program_id."""
    with _lock:
        session = _state["session"]
        records = store.load_records()
        reviewed = store.reviewed_program_ids(records)
        heme_auto_ok, _ = triage_serve.heme_only_auto_exclude_allowed(_state["programs"], records)
        model_gate_ok, _ = triage_serve.model_layer_gate_passed(records)
        staged = triage_staging.latest_by_program(triage_staging.load_records())
        remaining_ids = list(session.get("reopen_queue") or []) + list(session.get("order") or [])
        remaining_ids = [pid for pid in remaining_ids if pid not in reviewed]

        con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
        try:
            candidates = qo.likely_reject_candidates(
                _state["programs"], remaining_ids,
                heme_auto_ok=heme_auto_ok, model_gate_ok=model_gate_ok,
                heme_holdout_ids=_validation_sample_ids(), triage_holdout_ids=_triage_holdout_ids(),
                staged_by_program=staged, con=con, limit=limit,
            )
        finally:
            con.close()
        return {"candidates": candidates}


class BulkRejectPayload(BaseModel):
    program_ids: list[str]
    reason: str


@app.post("/api/bulk_reject")
def bulk_reject(payload: BulkRejectPayload):
    """Gate-1 is_adc=no for every listed program_id, one shared reason.
    decided_by=human — a person selected these, triage didn't — so this
    is a normal gold record, not an auto-decision; the gate in
    triage/promote.py is unrelated and unaffected."""
    reason = (payload.reason or "").strip()
    if not reason:
        raise HTTPException(422, "reason is required")
    if not payload.program_ids:
        raise HTTPException(422, "program_ids must not be empty")

    with _lock:
        session = _state["session"]
        records = store.load_records()
        reviewed = store.reviewed_program_ids(records)
        blind_ids = _validation_sample_ids() | _triage_holdout_ids()

        accepted: list[str] = []
        skipped: list[str] = []
        for pid in payload.program_ids:
            program = _state["programs_by_id"].get(pid)
            if program is None or pid in reviewed or pid in blind_ids:
                skipped.append(pid)
                continue
            body = {
                "action": "label", "program_id": pid,
                "candidate_id": program["candidate_id"], "proposed_name": program["proposed_name"],
                "gate_reached": 1, "decided_by": "human", "is_adc": "no",
                "evidence_note": reason,
                "discovery_strategy": program.get("discovery_strategy"),
                "match_strength": program.get("match_strength"),
                "matched_term": program.get("matched_term"),
                "blind": False,
            }
            try:
                store.validate_label_payload(body)
            except store.ValidationError:
                skipped.append(pid)
                continue
            record = store.build_record(body, session_id=session["session_id"], served_stratum={})
            store.append_record(record)
            reviewed.add(pid)
            accepted.append(pid)

        rejected_set = set(accepted)
        if rejected_set:
            session["order"] = [pid for pid in session.get("order", []) if pid not in rejected_set]
            if session.get("reopen_queue"):
                session["reopen_queue"] = [pid for pid in session["reopen_queue"] if pid not in rejected_set]
            q.save_session(session)

        return {"ok": True, "n_rejected": len(accepted), "rejected": accepted, "skipped": skipped}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
