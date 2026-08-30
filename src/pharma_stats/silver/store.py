"""Append-only SILVER label store — structurally separate from gold.

Absolute constraint: nothing in this module, or anything that calls it,
may write to gold/labels.jsonl. Enforced three ways:

1. This module lives under a different top-level directory (silver/,
   a sibling of gold/, not a subdirectory of it) and never imports
   pharma_stats.labelling.store — there is no code path from here into
   the gold writer at all.
2. append_record refuses (raises) to write any record whose "labeller"
   field isn't exactly "auto" — build_record stamps that unconditionally
   and does not accept it as a payload override.
3. audit/gold_set.py's "zero auto-sourced records in gold" check asserts
   no record with labeller="auto" ever appears in gold/labels.jsonl,
   independent of whether anything upstream tried to.

The gold set is the evaluation set for whatever this package produces —
it must stay independent of it, always.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pharma_stats.config import SILVER_DIR

SILVER_LABELS_PATH = SILVER_DIR / "labels.jsonl"


def build_record(payload: dict, *, session_id: str) -> dict:
    """Stamp a silver payload into a durable record. labeller is always
    "auto" — there is no parameter to make it anything else."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "labeller": "auto",
        "program_id": payload["program_id"],
        "proposed_name": payload.get("proposed_name"),
        "status": payload.get("status"),
        "kill_reason": payload.get("kill_reason"),
        "public_confirmation_date": payload.get("public_confirmation_date"),
        "label_evidence_date": payload.get("label_evidence_date"),
        "never_publicly_confirmed": bool(payload.get("never_publicly_confirmed")),
        # the four decomposed-question answers (silver/questions.py), each
        # with its citations — kept verbatim so a human can audit exactly
        # what evidence produced this label, not just the final status
        "answers": payload.get("answers"),
        "abstained": bool(payload.get("abstained")),
        "abstain_reason": payload.get("abstain_reason"),
        # exactly which deterministic rule branch fired (questions.apply_deterministic_rules)
        "rule_path": payload.get("rule_path"),
        # self-consistency sampling (silver/sampling.py): k samples, how
        # many agreed, and on what — the abstention signal
        "self_consistency": payload.get("self_consistency"),
        # Red Team objection agent's verdict against this label, if run
        "red_team_objection": payload.get("red_team_objection"),
    }


def append_record(record: dict, path: Optional[Path] = None) -> None:
    if record.get("labeller") != "auto":
        raise ValueError(
            "silver.store.append_record refuses to write a record without labeller='auto' "
            f"(got {record.get('labeller')!r}) — this store only ever holds auto-sourced labels"
        )
    path = path or SILVER_LABELS_PATH  # resolved at call time, not import time, so it stays testable
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_records(path: Optional[Path] = None) -> list[dict]:
    path = path or SILVER_LABELS_PATH
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
