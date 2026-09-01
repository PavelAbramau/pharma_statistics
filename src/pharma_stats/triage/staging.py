"""Staging table for triage auto-decisions — NOT gold, NOT silver.

Nothing produced by pharma_stats.triage is ever committed to
gold/labels.jsonl automatically. Every decision lands here first; a human
reads the run report and bulk-accepts or rejects afterward (a separate,
later acceptance step — not implemented yet, since nothing should be
built to auto-apply before the first real report exists to read). Same
append-only spirit as gold/silver, and the same per-write pool-integrity
check (pool.assert_not_reviewed) as a second, independent guard beyond
whatever the caller already checked at pool-selection time.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pharma_stats.config import TRIAGE_DIR
from pharma_stats.triage.pool import assert_not_reviewed

STAGING_PATH = TRIAGE_DIR / "staged_decisions.jsonl"

STATUS_VALUES = ("pending", "accepted", "rejected")


def build_record(decision: dict, *, run_id: str) -> dict:
    """decision keys: program_id, proposed_name, is_adc, in_scope,
    scope_reason, layer (1/2/3), rule, model, prompt_version, from_recall,
    quote, manual_overflow, manual_overflow_reason."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "pending",  # pending | accepted | rejected — set only by a later bulk-decision step
        "decided_by": "auto",
        "program_id": decision["program_id"],
        "proposed_name": decision.get("proposed_name"),
        "is_adc": decision.get("is_adc"),
        "in_scope": decision.get("in_scope"),
        "scope_reason": decision.get("scope_reason"),
        "layer": decision.get("layer"),
        "rule": decision.get("rule"),
        "model": decision.get("model"),
        "prompt_version": decision.get("prompt_version"),
        "from_recall": decision.get("from_recall"),
        "quote": decision.get("quote"),
        "evidence_source": decision.get("evidence_source"),
        "confidence": decision.get("confidence"),
        "grounding_forced_recall": bool(decision.get("grounding_forced_recall")),
        "manual_overflow": bool(decision.get("manual_overflow")),
        "manual_overflow_reason": decision.get("manual_overflow_reason"),
    }


def append_record(record: dict, path: Optional[Path] = None) -> None:
    assert_not_reviewed(record["program_id"])
    path = path or STAGING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_records(path: Optional[Path] = None) -> list[dict]:
    path = path or STAGING_PATH
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_by_program(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in records:
        pid = r["program_id"]
        if pid not in latest or r["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = r
    return latest
