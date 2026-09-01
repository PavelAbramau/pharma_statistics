"""Commit Layer 1 auto-rejections to staging + gold, and drop them from
the live labelling session.

Layer 2/3 never commit here — they wait on triage.validation.check_gate.
is_adc=yes / in_scope=yes is never written (Gate 3 stays human); those
ids stay in the queue at start_gate=3.

    python scripts/apply_triage_to_queue.py [--dry-run]
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.labelling import triage_serve
from pharma_stats.triage import deterministic as det
from pharma_stats.triage import staging

SESSION_ID = "auto:layer1_triage"


def _gold_body(program: dict, result: det.Layer1Result) -> dict:
    body = {
        "action": "label",
        "program_id": program["program_id"],
        "candidate_id": program.get("candidate_id"),
        "proposed_name": program.get("proposed_name"),
        "decided_by": "auto",
        "triage_layer": 1,
        "triage_rule": result.rule,
        "discovery_strategy": program.get("discovery_strategy"),
        "match_strength": program.get("match_strength"),
        "matched_term": program.get("matched_term"),
        "is_adc": result.is_adc,
        "in_scope": result.in_scope,
        "scope_reason": result.scope_reason,
    }
    if result.is_adc == "no":
        body["gate_reached"] = 1
    else:
        body["gate_reached"] = 2
        body["in_scope"] = "no"
        body["scope_reason"] = result.scope_reason
    return body


def layer1_commit_candidates(
    programs: list[dict],
    *,
    gold_records: Optional[list[dict]] = None,
    heme_auto_ok: bool = False,
    holdout_ids: Optional[set[str]] = None,
) -> list[tuple[dict, det.Layer1Result]]:
    """Unreviewed Layer 1 committable rejections that should actually be
    written this run. heme_only is withheld unless heme_auto_ok; holdout
    ids (heme validation sample) stay in the manual queue regardless."""
    gold_records = gold_records if gold_records is not None else store.load_records()
    reviewed = store.reviewed_program_ids(gold_records)
    holdout_ids = holdout_ids or set()
    out = []
    for p in programs:
        pid = p["program_id"]
        if pid in reviewed or pid in holdout_ids:
            continue
        result = det.evaluate(p)
        if result is None or not result.committable:
            continue
        if result.scope_reason == "heme_only" and not heme_auto_ok:
            continue
        out.append((p, result))
    return out


def apply_layer1(
    programs: Optional[list[dict]] = None,
    *,
    dry_run: bool = False,
    run_id: Optional[str] = None,
) -> dict:
    programs = programs if programs is not None else pp.load_materialized()
    gold_records = store.load_records()
    heme_auto_ok, heme_reason = triage_serve.heme_only_auto_exclude_allowed(programs, gold_records)
    holdout_ids = {item["program_id"] for item in ts.load_validation_sample()}
    candidates = layer1_commit_candidates(
        programs, gold_records=gold_records, heme_auto_ok=heme_auto_ok, holdout_ids=holdout_ids,
    )
    run_id = run_id or f"triage:layer1:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    n_adc_no = sum(1 for _, r in candidates if r.is_adc == "no")
    n_scope_no = len(candidates) - n_adc_no

    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "n_commit": len(candidates),
            "n_is_adc_no": n_adc_no,
            "n_in_scope_no": n_scope_no,
            "heme_auto_ok": heme_auto_ok,
            "heme_reason": heme_reason,
            "names": [(p["proposed_name"], r.rule) for p, r in candidates],
        }

    n_written = 0
    for program, result in candidates:
        staging.append_record(staging.build_record({
            "program_id": program["program_id"],
            "proposed_name": program.get("proposed_name"),
            "is_adc": result.is_adc,
            "in_scope": "no" if result.committable else result.in_scope,
            "scope_reason": result.scope_reason if result.scope_reason else (
                "not_an_adc" if result.is_adc == "no" else None
            ),
            "layer": 1,
            "rule": result.rule,
        }, run_id=run_id))
        body = _gold_body(program, result)
        store.validate_label_payload(body)
        record = store.build_record(body, session_id=SESSION_ID, served_stratum={})
        store.append_record(record)
        n_written += 1

    skip_ids = {p["program_id"] for p, _ in candidates}
    session = q.load_session()
    if session is not None:
        session["order"] = [pid for pid in session["order"] if pid not in skip_ids]
        q.save_session(session)

    return {
        "dry_run": False,
        "run_id": run_id,
        "n_commit": n_written,
        "n_is_adc_no": n_adc_no,
        "n_in_scope_no": n_scope_no,
        "heme_auto_ok": heme_auto_ok,
        "heme_reason": heme_reason,
    }
