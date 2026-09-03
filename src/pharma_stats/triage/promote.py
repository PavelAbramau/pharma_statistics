"""Bulk promotion of PENDING Layer 2/3 staged decisions to gold — the
only place that happens, and only after triage/validation.check_gate()
passes on the drawn sample. Layer 1/1.5 already commit directly (see
triage/apply.py) since they're deterministic, no-model-uncertainty
decisions; Layer 2/3 wait here on purpose.

Only is_adc=no is ever accepted: it's the only committable outcome
Layer 2/3 produce (they never resolve in_scope — that's Layer 1's
exclusive domain, see deterministic.py). An is_adc=yes verdict is never
written to gold by this module; it stays "pending" in staging for the
labelling queue to show as auto-derived context at Gate 3 (still a
human's call).

"Fully reversible": nothing here overwrites or deletes anything.
Accepting appends NEW gold records (decided_by=auto) — append-only, same
as any human decision; a later human review is simply a newer record and
wins per store.latest_by_program, same override rule as everywhere else.
Staging's own "status" is likewise never rewritten in place: accepting or
rejecting appends a NEW staging record for the same program_id with
status set, so the full history (including the original pending verdict)
survives untouched.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pharma_stats.labelling import store as gold_store
from pharma_stats.triage import staging, validation as val


class PromotionError(RuntimeError):
    pass


def pending_committable_layer23(
    staged_records: list[dict], gold_records: list[dict],
) -> list[dict]:
    """Latest staged Layer 2/3 record per program, filtered to: still
    pending, not a manual_overflow flag, is_adc=no (the only committable
    outcome — see module docstring), and not already reviewed by a human
    since staging (pool integrity, checked again individually at write
    time by staging.append_record)."""
    latest = staging.latest_by_program(staged_records)
    reviewed = gold_store.reviewed_program_ids(gold_records)
    return [
        r for r in latest.values()
        if r.get("layer") in (2, 3) and r.get("status") == "pending"
        and not r.get("manual_overflow") and r.get("is_adc") == "no"
        and r["program_id"] not in reviewed
    ]


def _gold_body(record: dict) -> dict:
    return {
        "action": "label", "program_id": record["program_id"],
        "proposed_name": record.get("proposed_name"),
        "decided_by": "auto", "triage_layer": record.get("layer"),
        "triage_rule": record.get("rule"), "triage_model": record.get("model"),
        "triage_prompt_version": record.get("prompt_version"),
        "gate_reached": 1, "is_adc": "no",
    }


def _restage_with_status(record: dict, status: str, *, run_id: str) -> None:
    updated = dict(record)
    updated["status"] = status
    updated["event_id"] = str(uuid.uuid4())
    updated["timestamp"] = datetime.now(timezone.utc).isoformat()
    updated["run_id"] = run_id
    staging.append_record(updated)


def accept_all_pending(
    *, sample: Optional[list[dict]] = None, gold_records: Optional[list[dict]] = None,
    staged_records: Optional[list[dict]] = None, run_id: Optional[str] = None,
) -> dict:
    """Checks the validation gate itself — this is the ONE place that can
    write Layer 2/3 verdicts to gold, so the gate is enforced here, not
    left to the caller's discipline. Raises PromotionError (writes
    nothing) if the gate hasn't passed."""
    gold_records = gold_records if gold_records is not None else gold_store.load_records()
    staged_records = staged_records if staged_records is not None else staging.load_records()
    sample = sample if sample is not None else val.load_validation_sample()

    agreement = val.compute_agreement(sample, gold_records)
    passed, reason = val.check_gate(agreement)
    if not passed:
        raise PromotionError(
            f"validation gate has not passed ({reason}) — refusing to accept anything into gold. "
            "Call reject_all_pending() instead, or judge more of the validation sample first."
        )

    run_id = run_id or f"triage:promote:accept:{uuid.uuid4().hex[:8]}"
    candidates = pending_committable_layer23(staged_records, gold_records)

    n_written = 0
    for r in candidates:
        body = _gold_body(r)
        gold_store.validate_label_payload(body)
        gold_record = gold_store.build_record(body, session_id=run_id, served_stratum={})
        # Restage as accepted BEFORE writing gold: staging.append_record's
        # pool-integrity check (assert_not_reviewed) reads live gold state on
        # every write, so writing gold first would make it see this very
        # record and mistake its own auto-write for a human override.
        _restage_with_status(r, "accepted", run_id=run_id)
        gold_store.append_record(gold_record)
        n_written += 1

    return {"run_id": run_id, "n_accepted": n_written, "gate_reason": reason}


def reject_all_pending(
    *, staged_records: Optional[list[dict]] = None, run_id: Optional[str] = None,
    reason: str = "validation gate failed",
) -> dict:
    """Never writes gold. Marks every pending Layer 2/3 decision (both
    is_adc=yes and is_adc=no — the whole batch this validation run was
    meant to clear) as rejected, so nothing downstream mistakes an
    unvalidated verdict for a trusted one. The candidates stay exactly
    where they were: in the human queue, no auto-decision applied."""
    staged_records = staged_records if staged_records is not None else staging.load_records()
    latest = staging.latest_by_program(staged_records)
    candidates = [
        r for r in latest.values()
        if r.get("layer") in (2, 3) and r.get("status") == "pending" and not r.get("manual_overflow")
    ]
    run_id = run_id or f"triage:promote:reject:{uuid.uuid4().hex[:8]}"
    for r in candidates:
        updated = dict(r)
        updated["manual_overflow_reason"] = reason  # reused field: "why is this not trusted"
        _restage_with_status(updated, "rejected", run_id=run_id)
    return {"run_id": run_id, "n_rejected": len(candidates), "reason": reason}
