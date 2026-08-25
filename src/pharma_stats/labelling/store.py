"""Append-only gold label store.

One JSON line per labelling action, ever. Revisions are new lines with a
later timestamp, never edits — the history is data. Nothing in this
module opens gold/labels.jsonl in a mode that can lose an existing line.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pharma_stats.config import GOLD_DIR
from pharma_stats.labelling.vocab import APP_VERSION, KILL_REASONS, PROGRAM_STATUSES

LABELS_PATH = GOLD_DIR / "labels.jsonl"


class ValidationError(ValueError):
    pass


def validate_label_payload(payload: dict) -> None:
    action = payload.get("action")
    if action not in ("label", "skip", "flag_invalid"):
        raise ValidationError(f"unknown action {action!r}")

    if action != "label":
        return

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
    only — the current-best-known label for each program. Excludes
    repeat-probe re-serves from being treated as the primary label twice;
    both copies still live in the JSONL for self-consistency analysis."""
    latest: dict[str, dict] = {}
    for r in records:
        if r["action"] != "label" or r.get("is_repeat_probe"):
            continue
        pid = r["program_id"]
        if pid not in latest or r["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = r
    return latest


def labelled_program_ids(records: list[dict]) -> set[str]:
    return set(latest_by_program(records).keys())


def invalid_flagged_ids(records: list[dict]) -> set[str]:
    return {r["program_id"] for r in records if r["action"] == "flag_invalid"}
