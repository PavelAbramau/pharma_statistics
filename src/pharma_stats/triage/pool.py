"""Candidate pool selection with a hard pool-integrity guarantee: an
auto-decision must never be produced for, or override, a program a human
has already decided — any gate, not just gate 3. Asserted, not assumed,
in two places: once in bulk at pool selection (select_candidate_pool), and
again per-program immediately before any decision is staged
(assert_not_reviewed) — the second check exists because a long-running
Layer 2/3 pass can span hours, during which the human keeps labelling in
the app; a program reviewed AFTER pool selection but BEFORE this
pipeline's staging write must still be caught.
"""
from __future__ import annotations

from typing import Optional

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store as gold_store


class PoolIntegrityError(RuntimeError):
    pass


def select_candidate_pool(
    programs: Optional[list[dict]] = None, gold_records: Optional[list[dict]] = None,
) -> tuple[list[dict], dict]:
    """(pool, stats). pool excludes every program_id with ANY gold label
    record (gate 1, 2, or 3 — see store.reviewed_program_ids). Raises
    PoolIntegrityError if the exclusion somehow failed — this should be
    structurally impossible given the list comprehension below, but the
    user asked to assert it rather than assume it, so it's checked."""
    programs = programs if programs is not None else pp.load_materialized()
    gold_records = gold_records if gold_records is not None else gold_store.load_records()

    reviewed = gold_store.reviewed_program_ids(gold_records)
    pool = [p for p in programs if p["program_id"] not in reviewed]

    overlap = {p["program_id"] for p in pool} & reviewed
    if overlap:
        raise PoolIntegrityError(
            f"{len(overlap)} candidate(s) in the selected pool already have a gold record — "
            f"pool selection must exclude every reviewed program_id, no exceptions. "
            f"First few: {sorted(overlap)[:10]}"
        )

    latest = gold_store.latest_by_program(gold_records)
    gate_counts: dict = {1: 0, 2: 0, 3: 0}
    for r in latest.values():
        gate = r.get("gate_reached")
        if gate in gate_counts:
            gate_counts[gate] += 1

    stats = {
        "total_materialized": len(programs),
        "total_gold_records": len(gold_records),
        "total_reviewed_programs": len(reviewed),
        "gate1_rejected_count": gate_counts[1],
        "gate2_rejected_count": gate_counts[2],
        "gate3_labelled_count": gate_counts[3],
        "pool_size": len(pool),
        "overlap_count": len(overlap),  # always 0 if we get here — printed for transparency
    }
    return pool, stats


def assert_not_reviewed(program_id: str, gold_records: Optional[list[dict]] = None) -> None:
    """Defense-in-depth: call this immediately before staging ANY
    auto-decision for program_id, not just once at pool-selection time —
    see module docstring for why the gap matters."""
    gold_records = gold_records if gold_records is not None else gold_store.load_records()
    reviewed = gold_store.reviewed_program_ids(gold_records)
    if program_id in reviewed:
        raise PoolIntegrityError(
            f"refusing to stage an auto-decision for {program_id!r} — a human has reviewed it "
            "in gold/labels.jsonl since pool selection; auto must never override manual"
        )
