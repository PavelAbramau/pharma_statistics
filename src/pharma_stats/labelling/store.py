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
    APP_VERSION, CONFIRMATION_EVIDENCE_TYPES, GATE2_SCOPE_OUT_REASONS, IN_SCOPE_VALUES,
    IS_ADC_VALUES, KILL_REASONS, PROGRAM_STATUSES, TRIAGE_LAYERS,
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
        # decided_by=auto is a source, not a specific rule — it covers both
        # scripts/apply_auto_scope_exclusions.py's heme_only exclusions and
        # pharma_stats.triage's five-rule Layer 1 (plus Layers 2/3's model
        # and web-search is_adc verdicts). Every such record must say which
        # layer decided it (see vocab.TRIAGE_LAYERS) so "how was this
        # decided" is always answerable from the record alone.
        triage_layer = payload.get("triage_layer")
        if triage_layer not in TRIAGE_LAYERS:
            raise ValidationError(
                f"decided_by=auto requires triage_layer from {TRIAGE_LAYERS}, got {triage_layer!r}"
            )

        if gate == 1:
            # An auto is_adc=no verdict (INN denylist hit, or a batched/
            # web-search layer confidently ruling it out). is_adc=yes is
            # never saved terminal at gate 1 — same rule as a human review
            # (see below) — because it isn't informative on its own: the
            # candidate still needs an in_scope verdict, which either
            # combines into one gate-2 auto record (see below) or, if no
            # scope rule fires either, isn't written at all and stays in
            # the normal manual queue.
            is_adc = payload.get("is_adc")
            if is_adc != "no":
                raise ValidationError(
                    f"decided_by=auto, gate_reached=1 requires is_adc='no', got {is_adc!r} — "
                    "an auto is_adc=yes verdict is never saved at gate 1 alone"
                )
            return

        if gate == 2:
            # A pure scope-eligibility call. is_adc is required here (unlike
            # the old heme_only-only behaviour) so a triage-produced record
            # always states both verdicts it combined — but "unknown"/None
            # stays legal for scripts/apply_auto_scope_exclusions.py's
            # heme_only path, which never asks the molecule-identity
            # question at all (see trial_scope.auto_scope_decision).
            if payload.get("in_scope") != "no":
                raise ValidationError("decided_by=auto gate_reached=2 records must be in_scope=no")
            scope_reason = payload.get("scope_reason")
            if scope_reason not in GATE2_SCOPE_OUT_REASONS:
                raise ValidationError(
                    f"decided_by=auto requires scope_reason from {GATE2_SCOPE_OUT_REASONS}, got {scope_reason!r}"
                )
            is_adc = payload.get("is_adc")
            if is_adc is not None and is_adc not in IS_ADC_VALUES:
                raise ValidationError(f"is_adc must be one of {IS_ADC_VALUES} or absent, got {is_adc!r}")
            return

        raise ValidationError("decided_by=auto is only valid for gate_reached in (1, 2)")

    is_adc = payload.get("is_adc")
    if is_adc not in IS_ADC_VALUES:
        raise ValidationError(f"is_adc must be one of {IS_ADC_VALUES}, got {is_adc!r}")

    if gate == 1:
        if is_adc == "yes":
            raise ValidationError("gate_reached=1 with is_adc=yes must proceed to gate 2, not save here")
        # is_adc=no always also means in_scope=no / not_an_adc — the fields
        # are independent but a non-ADC cannot be in this project's filter.
        # Omitted pair is allowed here; build_record fills it in. A provided
        # pair must be the consistent one (never is_adc=no + in_scope=yes).
        if is_adc == "no":
            in_scope = payload.get("in_scope")
            scope_reason = payload.get("scope_reason")
            if in_scope is not None or scope_reason is not None:
                if in_scope != "no" or scope_reason != "not_an_adc":
                    raise ValidationError(
                        "is_adc=no must carry in_scope=no and scope_reason=not_an_adc "
                        f"(or omit both); got in_scope={in_scope!r}, scope_reason={scope_reason!r}"
                    )
        return  # terminal triage rejection: no status, no coverage requirement

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
        if scope_reason not in GATE2_SCOPE_OUT_REASONS:
            raise ValidationError(
                f"gate_reached=2 with in_scope=no requires scope_reason from {GATE2_SCOPE_OUT_REASONS}, "
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

    # Third-party (commercial database / industry tracker) sighting of the
    # discontinuation — a DIFFERENT signal than public_confirmation_date
    # (the sponsor's/a filing's own statement). Independent of status: it's
    # evidence about what a tracker recorded, not the reviewer's own
    # judgement, so it's validated as a pair (a date needs its source to be
    # usable as evidence) rather than gated on status.
    third_party_date = payload.get("third_party_first_noted_date")
    third_party_source = payload.get("third_party_source")
    if bool(third_party_date) != bool(third_party_source):
        raise ValidationError(
            "third_party_first_noted_date and third_party_source must be given together "
            "(a date with no source, or a source with no date, isn't usable evidence)"
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
        # Only meaningful when a confirmation date actually exists — a
        # never_publicly_confirmed record has no confirmation to type.
        # pipeline_page_removal is a legitimate value here, not an error —
        # see vocab.py: it's the ambiguous case, tagged rather than resolved.
        if confirmation_date:
            evidence_type = payload.get("confirmation_evidence_type")
            if evidence_type not in CONFIRMATION_EVIDENCE_TYPES:
                raise ValidationError(
                    f"public_confirmation_date requires confirmation_evidence_type from "
                    f"{CONFIRMATION_EVIDENCE_TYPES}, got {evidence_type!r}"
                )


def build_record(payload: dict, *, session_id: str, served_stratum: dict) -> dict:
    """Stamp a validated client payload into a durable JSONL record."""
    is_adc = payload.get("is_adc")
    in_scope = payload.get("in_scope")
    scope_reason = payload.get("scope_reason")
    # A molecule rejection is also a filter rejection. Persist both fields
    # even when the client only sent is_adc=no (legacy gate-1 saves).
    if payload.get("action") == "label" and is_adc == "no" and in_scope is None:
        in_scope = "no"
        scope_reason = "not_an_adc"
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
        "is_adc": is_adc,
        "in_scope": in_scope,
        "scope_reason": scope_reason,
        # decided_by=auto provenance (see vocab.TRIAGE_LAYERS) — which
        # layer decided this, and exactly which rule/model/prompt version,
        # so "how was this decided" is always answerable from the record
        # alone. None for a human decision.
        "triage_layer": payload.get("triage_layer"),
        "triage_rule": payload.get("triage_rule"),
        "triage_model": payload.get("triage_model"),
        "triage_prompt_version": payload.get("triage_prompt_version"),
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
        "confirmation_evidence_type": payload.get("confirmation_evidence_type"),
        "never_publicly_confirmed": bool(payload.get("never_publicly_confirmed")),
        # Commercial-database/tracker sighting — distinct from
        # public_confirmation_date (sponsor/filing statement). Lets lead
        # time be computed against either benchmark without conflating them.
        "third_party_first_noted_date": payload.get("third_party_first_noted_date"),
        "third_party_source": payload.get("third_party_source"),
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
