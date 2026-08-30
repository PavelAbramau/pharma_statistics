"""Append-only gold label store.

One JSON line per labelling action, ever. Revisions are new lines with a
later timestamp, never edits — the history is data. Nothing in this
module opens gold/labels.jsonl in a mode that can lose an existing line.

Every "label" record carries gate_reached (1/2/3, see vocab.py) — the
review screen is three sequential gates, and a record's gate_reached says
how far that particular review got. Gates 1-2 are triage rejections (a
molecule that isn't an ADC, or an ADC that's out of this project's scope):
real evidence about discovery precision, but never a "label" in the sense
that stratum coverage, the 150-label target, or the label_sufficiency
bootstrap mean it. Only gate_reached=3 records are that. Every function
here that produces a set of "labelled" ids is named to make which one it
means unambiguous — see fully_labelled_program_ids vs reviewed_program_ids.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pharma_stats.config import GOLD_DIR
from pharma_stats.labelling.vocab import (
    APP_VERSION, IN_SCOPE_VALUES, IS_ADC_VALUES, KILL_REASONS, PROGRAM_STATUSES,
    SCOPE_OUT_REASONS,
)

LABELS_PATH = GOLD_DIR / "labels.jsonl"


class ValidationError(ValueError):
    pass


def validate_label_payload(payload: dict) -> None:
    action = payload.get("action")
    if action not in ("label", "skip"):
        raise ValidationError(f"unknown action {action!r}")

    if action != "label":
        return

    gate = payload.get("gate_reached")
    if gate not in (1, 2, 3):
        raise ValidationError(f"gate_reached must be 1, 2, or 3, got {gate!r}")

    decided_by = payload.get("decided_by") or "human"
    if decided_by not in ("human", "auto"):
        raise ValidationError(f"decided_by must be 'human' or 'auto', got {decided_by!r}")

    if decided_by == "auto":
        # A pure scope-eligibility call (trial_scope.auto_scope_decision):
        # this project is solid-tumours-only regardless of molecule type,
        # so a confidently heme_only asset is out of scope whether or not
        # it's even an ADC. is_adc is therefore neither required nor
        # asserted here — it stays entirely the reviewer's call, made
        # whenever/if a human looks at this asset (e.g. via the held-out
        # validation sample).
        if gate != 2:
            raise ValidationError("decided_by=auto is only valid for gate_reached=2")
        if payload.get("in_scope") != "no":
            raise ValidationError("decided_by=auto records must be in_scope=no")
        scope_reason = payload.get("scope_reason")
        if scope_reason not in SCOPE_OUT_REASONS:
            raise ValidationError(
                f"decided_by=auto requires scope_reason from {SCOPE_OUT_REASONS}, got {scope_reason!r}"
            )
        is_adc = payload.get("is_adc")
        if is_adc is not None and is_adc not in IS_ADC_VALUES:
            raise ValidationError(f"is_adc must be one of {IS_ADC_VALUES} or absent, got {is_adc!r}")
        return

    is_adc = payload.get("is_adc")
    if is_adc not in IS_ADC_VALUES:
        raise ValidationError(f"is_adc must be one of {IS_ADC_VALUES}, got {is_adc!r}")

    if gate == 1:
        if is_adc == "yes":
            raise ValidationError("gate_reached=1 with is_adc=yes must proceed to gate 2, not save here")
        return  # terminal triage rejection: no in_scope, no status, no coverage requirement

    # gates 2 and 3 are only reachable when gate 1 passed
    if is_adc != "yes":
        raise ValidationError("gate_reached>=2 requires is_adc=yes (gate 1 must have passed)")

    in_scope = payload.get("in_scope")
    if in_scope not in IN_SCOPE_VALUES:
        raise ValidationError(f"in_scope must be one of {IN_SCOPE_VALUES}, got {in_scope!r}")

    if gate == 2:
        if in_scope == "yes":
            raise ValidationError("gate_reached=2 with in_scope=yes must proceed to gate 3, not save here")
        scope_reason = payload.get("scope_reason")
        if scope_reason not in SCOPE_OUT_REASONS:
            raise ValidationError(
                f"gate_reached=2 with in_scope=no requires scope_reason from {SCOPE_OUT_REASONS}, "
                f"got {scope_reason!r}"
            )
        return  # terminal scope rejection: no status, no coverage requirement

    # gate == 3: the full label. Only reachable when gate 2 passed.
    if in_scope != "yes":
        raise ValidationError("gate_reached=3 requires in_scope=yes (gate 2 must have passed)")

    # Defense in depth: the app's /api/next already refuses to *serve*
    # anything short of full history_coverage, so this should never fire
    # in practice — but re-checking here closes the race where coverage
    # data is rebuilt/degrades in the window between serve and save.
    # Missing data must never be servable as evidence, full stop. Only
    # gate 3 needs this: gates 1-2 judge molecule identity and scope, not
    # the amendment-history silence signal, so they aren't blocked on it.
    coverage = payload.get("history_coverage_at_serve_time")
    if coverage != "full":
        raise ValidationError(
            f"refusing to save: history_coverage_at_serve_time={coverage!r}, not 'full' — "
            "this program's evidence was incomplete when served, which should be impossible"
        )

    status = payload.get("status")
    if status not in PROGRAM_STATUSES:
        raise ValidationError(f"status must be one of {PROGRAM_STATUSES}, got {status!r}")

    confidence = payload.get("confidence")
    if confidence not in ("high", "medium", "low"):
        raise ValidationError("confidence (high/medium/low) is required")

    if status == "dead_confirmed":
        kill_reason = payload.get("kill_reason")
        if kill_reason not in KILL_REASONS:
            raise ValidationError(
                f"dead_confirmed requires a kill_reason from {KILL_REASONS}"
            )
        never_confirmed = bool(payload.get("never_publicly_confirmed"))
        confirmation_date = payload.get("public_confirmation_date")
        if not confirmation_date and not never_confirmed:
            raise ValidationError(
                "dead_confirmed requires public_confirmation_date or "
                "never_publicly_confirmed=true"
            )
        if not payload.get("label_evidence_date"):
            raise ValidationError("dead_confirmed requires label_evidence_date")


def build_record(payload: dict, *, session_id: str, served_stratum: dict) -> dict:
    """Stamp a validated client payload into a durable JSONL record."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "app_version": APP_VERSION,
        "action": payload["action"],
        "program_id": payload["program_id"],
        "candidate_id": payload.get("candidate_id"),
        "proposed_name": payload.get("proposed_name"),
        "gate_reached": payload.get("gate_reached"),
        "decided_by": payload.get("decided_by") or "human",
        "is_adc": payload.get("is_adc"),
        "in_scope": payload.get("in_scope"),
        "scope_reason": payload.get("scope_reason"),
        # discovery provenance, stamped server-side from the served candidate's
        # own data (never client-supplied) — see app.py's submit_label. The
        # "which pattern misfired" signal only means anything if it's the
        # discovery pipeline's own record of why this candidate exists, not
        # something the reviewer typed.
        "discovery_strategy": payload.get("discovery_strategy"),
        "match_strength": payload.get("match_strength"),
        "matched_term": payload.get("matched_term"),
        "status": payload.get("status"),
        "kill_reason": payload.get("kill_reason"),
        "confidence": payload.get("confidence"),
        "evidence_note": payload.get("evidence_note") or "",
        "label_evidence_date": payload.get("label_evidence_date"),
        "public_confirmation_date": payload.get("public_confirmation_date"),
        "never_publicly_confirmed": bool(payload.get("never_publicly_confirmed")),
        "blind": bool(payload.get("blind")),
        "is_repeat_probe": bool(payload.get("is_repeat_probe")),
        "stratum_band": served_stratum.get("band"),
        "stratum_archetype": served_stratum.get("archetype"),
        "silence_score_at_label_time": served_stratum.get("silence_score"),
        "history_coverage_at_serve_time": served_stratum.get("history_coverage"),
        "status_revised_after_external_search": bool(payload.get("status_revised_after_external_search")),
        "seconds_spent": payload.get("seconds_spent"),
    }


def append_record(record: dict, path: Optional[Path] = None) -> None:
    path = path or LABELS_PATH  # resolved at call time, not import time, so it stays testable
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_records(path: Optional[Path] = None) -> list[dict]:
    path = path or LABELS_PATH
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_by_program(records: list[dict]) -> dict[str, dict]:
    """Most recent record (by timestamp) per program_id, "label" actions
    only, any gate — the current-best-known outcome for each program.
    Excludes repeat-probe re-serves from being treated as the primary
    outcome twice; both copies still live in the JSONL for self-consistency
    analysis."""
    latest: dict[str, dict] = {}
    for r in records:
        if r["action"] != "label" or r.get("is_repeat_probe"):
            continue
        pid = r["program_id"]
        if pid not in latest or r["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = r
    return latest


def reviewed_program_ids(records: list[dict]) -> set[str]:
    """Every program with a label record of ANY gate — gate 1 and 2
    rejections are just as terminal as a full gate-3 label (the program is
    never re-served once decided), so this is what the queue's exclusion
    set must use. NOT what stratum coverage / the label target / the
    sufficiency bootstrap should use — see fully_labelled_program_ids."""
    return set(latest_by_program(records).keys())


def fully_labelled_program_ids(records: list[dict]) -> set[str]:
    """Programs whose latest label record reached gate 3 — the real gold
    labels. This is what stratum coverage, the label-count target, and the
    label_sufficiency bootstrap must count; a gate 1/2 triage rejection was
    never a program to begin with, so it must never inflate these."""
    return {pid for pid, r in latest_by_program(records).items() if r.get("gate_reached") == 3}
