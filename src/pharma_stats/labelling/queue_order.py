"""Disposition-ordered queue for a full coverage pass over the residue —
"ordered for speed, not stratification" (the reviewer's own words): every
remaining program that still needs a human gate, grouped so similar
decisions can be batched, rather than interleaved by score band/archetype
(queue.build_stratified_order, which stays the default for normal
labelling sessions).

Three buckets, served in this order:
  1. likely_reject  — no confident triage verdict yet, but the evidence
     text names a non-ADC modality or an oral/tablet/capsule route
     (grounding.has_small_molecule_or_oral_signal). First, because these
     are the fastest calls in the whole queue.
  2. ambiguous       — everything else still needing a human gate: no
     text signal either way, or a genuine ADC (is_adc=yes) whose scope
     isn't resolved yet.
  3. confirmed_adc   — triage already resolved is_adc=yes AND
     in_scope=yes. Last, on purpose: these enter at gate 3 directly (see
     triage_serve.serve_plan) with the triage verdict shown as
     auto-derived, overridable context, and the reviewer asked for these
     when "warmed up," not first.

Within each bucket, order is deterministic (stable sort on program_id) —
there's no scoring signal worth stratifying on here, unlike the normal
queue.
"""
from __future__ import annotations

from typing import Optional

import duckdb

from pharma_stats.labelling import triage_serve
from pharma_stats.triage import evidence as tev
from pharma_stats.triage import grounding

BUCKET_ORDER = ("likely_reject", "ambiguous", "confirmed_adc")


def disposition_bucket(
    program: dict, plan: triage_serve.ServePlan, *, con: Optional[duckdb.DuckDBPyConnection],
) -> tuple[Optional[str], Optional[str]]:
    """(bucket, signal_snippet). bucket is None when the plan says don't
    serve at all (plan.skip — already auto-excluded). signal_snippet is
    only ever set for likely_reject — the evidence text that names a
    non-ADC modality or oral/tablet/capsule route, shown to the reviewer
    as the reason this candidate was fast-tracked."""
    if plan.skip:
        return None, None
    if plan.start_gate == 3:
        return "confirmed_adc", None
    ctx = plan.context or {}
    if ctx.get("is_adc") == "yes":
        # Genuine ADC, scope not yet resolved — a real gate-2 decision,
        # not a fast reject.
        return "ambiguous", None
    if con is None:
        return "ambiguous", None
    evidence = tev.build_layer2_evidence(program, con)
    snippet = grounding.matching_small_molecule_or_oral_snippet(evidence.get("text_snippets"))
    if snippet is not None:
        return "likely_reject", snippet
    return "ambiguous", None


def build_disposition_order(
    programs: list[dict],
    remaining_ids: list[str],
    *,
    gold_records: list[dict],
    heme_auto_ok: bool,
    model_gate_ok: bool,
    heme_holdout_ids: set,
    triage_holdout_ids: set,
    staged_by_program: dict,
    con: Optional[duckdb.DuckDBPyConnection],
) -> tuple[list[str], dict]:
    """(ordered_ids, counts). ordered_ids covers only remaining_ids not
    already auto-excluded (plan.skip) — those never entered the manual
    queue in the first place, same as the normal session."""
    by_id = {p["program_id"]: p for p in programs}
    buckets: dict[str, list[str]] = {k: [] for k in BUCKET_ORDER}

    for pid in remaining_ids:
        program = by_id.get(pid)
        if program is None:
            continue
        plan = triage_serve.serve_plan(
            program,
            heme_holdout=pid in heme_holdout_ids,
            triage_holdout=pid in triage_holdout_ids,
            heme_auto_ok=heme_auto_ok,
            model_gate_ok=model_gate_ok,
            staged_record=staged_by_program.get(pid),
        )
        bucket, _snippet = disposition_bucket(program, plan, con=con)
        if bucket is None:
            continue
        buckets[bucket].append(pid)

    for bucket in buckets.values():
        bucket.sort()

    ordered = [pid for bucket in BUCKET_ORDER for pid in buckets[bucket]]
    counts = {bucket: len(ids) for bucket, ids in buckets.items()}
    return ordered, counts


def likely_reject_candidates(
    programs: list[dict],
    remaining_ids: list[str],
    *,
    heme_auto_ok: bool,
    model_gate_ok: bool,
    heme_holdout_ids: set,
    triage_holdout_ids: set,
    staged_by_program: dict,
    con: duckdb.DuckDBPyConnection,
    limit: int,
) -> list[dict]:
    """[{program_id, proposed_name, signal_snippet}] for the bulk-reject
    panel — the likely_reject bucket only, with the matched evidence
    snippet shown as the reason. Blind holdouts are never included here:
    those must only ever appear through the normal three-gate card."""
    by_id = {p["program_id"]: p for p in programs}
    out: list[dict] = []
    for pid in remaining_ids:
        if pid in heme_holdout_ids or pid in triage_holdout_ids:
            continue
        program = by_id.get(pid)
        if program is None:
            continue
        plan = triage_serve.serve_plan(
            program,
            heme_holdout=False, triage_holdout=False,
            heme_auto_ok=heme_auto_ok, model_gate_ok=model_gate_ok,
            staged_record=staged_by_program.get(pid),
        )
        bucket, snippet = disposition_bucket(program, plan, con=con)
        if bucket != "likely_reject":
            continue
        out.append({"program_id": pid, "proposed_name": program.get("proposed_name"), "signal_snippet": snippet})
        if len(out) >= limit:
            break
    return out
