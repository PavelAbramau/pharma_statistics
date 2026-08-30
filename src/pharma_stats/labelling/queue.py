"""Stratified labelling queue + resumable session state.

Design constraint from the user: ranking programs by silence score and
labelling top-down yields a gold set with no negatives and an
uninterpretable precision figure. So the queue is built by interleaving
across score band (0-20/20-40/40-60/60-80/80-100) and archetype
(registry-terminated w/ stated reason, w/ vague reason, UNKNOWN status,
completed-no-results, actively-amended, other) — round-robin across
every (band, archetype) cell — rather than sorted by score.

Session state (queue position, pending serves, repeat-probe counter) is
persisted to a small JSON file under data/ so a session survives closing
the laptop. It is NOT the source of truth for labels — gold/labels.jsonl
is — so losing this file only costs requeue order, never a label.
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR
from pharma_stats.labelling.provisional_programs import ARCHETYPES, SCORE_BANDS

SESSION_PATH = DATA_DIR / "labelling_session.json"

REPEAT_PROBE_EVERY = 10  # ~10% self-consistency re-serve rate


def build_stratified_order(programs: list[dict], exclude_ids: set[str]) -> list[str]:
    cells: dict[tuple[int, str], list[str]] = {
        (b, a): [] for b in range(len(SCORE_BANDS)) for a in ARCHETYPES
    }
    for p in programs:
        if p["program_id"] in exclude_ids:
            continue
        # band is None when compute_silence_score had no resolvable trial
        # snapshot at all (provisional_programs.py) — never a real score, so
        # never one of the 5 real bands (stats.stratum_progress excludes
        # these from label-target/coverage counting for the same reason).
        # It must still be SERVABLE, though: this is exactly the case the
        # app's own "history_coverage != full -> requeue" guard exists to
        # handle gracefully once a later backfill resolves the snapshot, so
        # it gets its own synthetic bucket rather than being dropped from
        # the queue entirely.
        band_key = p["band"] if p["band"] is not None else "unscored"
        key = (band_key, p["primary_archetype"])
        cells.setdefault(key, []).append(p["program_id"])

    rng = random.Random(0)
    for members in cells.values():
        rng.shuffle(members)

    cell_keys = list(cells.keys())
    order: list[str] = []
    i = 0
    remaining = True
    while remaining:
        remaining = False
        for key in cell_keys:
            bucket = cells[key]
            if i < len(bucket):
                order.append(bucket[i])
                remaining = True
        i += 1
    return order


def new_session(programs: list[dict], exclude_ids: set[str]) -> dict:
    return {
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "order": build_stratified_order(programs, exclude_ids),
        "total_served": 0,
        "pending_serve": {},
        "served_log": [],
    }


def load_session(path: Optional[Path] = None) -> Optional[dict]:
    path = path or SESSION_PATH  # resolved at call time, not import time, so it stays testable
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session: dict, path: Optional[Path] = None) -> None:
    path = path or SESSION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2), encoding="utf-8")
    tmp.replace(path)


def pop_next(session: dict, labelled_ids: set[str]) -> tuple[Optional[str], bool]:
    """Returns (program_id, is_repeat_probe). program_id is None if the
    fresh queue is exhausted and no repeat is due."""
    total = session["total_served"]
    is_repeat = total > 0 and (total + 1) % REPEAT_PROBE_EVERY == 0
    program_id = None

    if is_repeat and labelled_ids:
        last_repeat = session.get("_last_repeat_id")
        pool = [pid for pid in labelled_ids if pid != last_repeat] or list(labelled_ids)
        program_id = random.choice(pool)
        session["_last_repeat_id"] = program_id
    else:
        is_repeat = False
        if session["order"]:
            program_id = session["order"].pop(0)

    if program_id is not None:
        session["total_served"] = total + 1
    return program_id, is_repeat


def make_serve_token(
    session: dict, program: dict, is_repeat: bool,
) -> str:
    token = str(uuid.uuid4())
    entry = {
        "program_id": program["program_id"],
        "band": program["band"],
        "archetype": program["primary_archetype"],
        "silence_score": program["silence_score"],
        "history_coverage": program["history_coverage"],
        "is_repeat_probe": is_repeat,
        "served_at": datetime.now(timezone.utc).isoformat(),
    }
    session["pending_serve"][token] = entry
    session["served_log"].append({**entry, "serve_token": token})
    return token


def requeue(session: dict, program_id: str) -> None:
    """'Skip, come back to this' — put it back at the end of the fresh
    queue rather than discarding it."""
    session["order"].append(program_id)
