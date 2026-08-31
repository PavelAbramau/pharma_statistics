"""End-to-end orchestration: pool selection -> Layer 1 -> evidence ->
Layer 2/3 -> staging -> validation sampling. Wires together pool.py,
evidence.py, layer2.py, layer3.py, staging.py, validation.py — no new
policy lives here beyond how they're sequenced.
"""
from __future__ import annotations

from typing import Optional

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.silver import model_client
from pharma_stats.triage import deterministic as det
from pharma_stats.triage import evidence as tev
from pharma_stats.triage import layer2
from pharma_stats.triage import layer3
from pharma_stats.triage import pool as tpool
from pharma_stats.triage import staging


def build_residue_evidence(candidates: list[dict], con: duckdb.DuckDBPyConnection) -> tuple[list[dict], list[dict]]:
    """(residue_programs, evidences). residue = Layer 1 couldn't resolve
    at all (evaluate() returned None) — anything Layer 1 DID resolve was
    already free and never needs a model call."""
    residue = [p for p in candidates if det.evaluate(p) is None]
    evidences = [tev.build_layer2_evidence(p, con) for p in residue]
    return residue, evidences


def partition_by_text_evidence(evidences: list[dict]) -> tuple[list[dict], list[dict]]:
    """(with_text -> Layer 2, no_text -> straight to Layer 3). A
    recall-only answer on an unnamed dev code with no supporting text is
    the highest-hallucination-risk case in this design: from_recall
    routes to Layer 3 regardless (layer2.route_to_layer3), so a
    no-text candidate would end up there anyway — this skips the wasted
    Layer 2 call rather than spend it producing a guess we'd distrust."""
    with_text = [e for e in evidences if e.get("text_snippets")]
    no_text = [e for e in evidences if not e.get("text_snippets")]
    return with_text, no_text


def cap_layer3_queue(
    candidates: list[dict], cap: int = layer3.MAX_LAYER3_CANDIDATES,
) -> tuple[list[dict], list[dict]]:
    """(within_cap, overflow). Overflow is NEVER auto-decided or silently
    dropped — see stage_manual_overflow, which flags each one explicitly
    in the staging table for the human queue."""
    if len(candidates) <= cap:
        return candidates, []
    return candidates[:cap], candidates[cap:]


def stage_manual_overflow(overflow: list[dict], *, run_id: str, reason: str) -> int:
    for ev in overflow:
        record = staging.build_record({
            "program_id": ev["program_id"], "proposed_name": ev.get("name"),
            "manual_overflow": True, "manual_overflow_reason": reason,
        }, run_id=run_id)
        staging.append_record(record)
    return len(overflow)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _count_tokens_best_effort(text: str, *, system: Optional[str], model: str) -> tuple[int, bool]:
    """(tokens, exact). Tries the real tokenizer (model_client.count_tokens
    — needs ANTHROPIC_API_KEY and network); falls back to the local
    ~4-chars/token approximation, clearly marked inexact, rather than
    failing the whole dry run over a missing key."""
    try:
        return model_client.count_tokens(text, system=system, model=model), True
    except model_client.ModelClientError:
        return _estimate_tokens(text) + (_estimate_tokens(system) if system else 0), False


def dry_run_report(
    pool: list[dict], con: duckdb.DuckDBPyConnection, *, model: str = model_client.DEFAULT_MODEL,
    layer2_limit: Optional[int] = None,
) -> dict:
    """Full Step-A report: Layer 1 resolution, evidence partitioning,
    Layer 2 cost (real prompts, best-effort real tokenizer), Layer 3
    queue composition and cap check. Zero completion calls; count_tokens
    calls only if a key happens to be available (see
    _count_tokens_best_effort), and the report says plainly whether the
    counts are exact or approximated."""
    layer1_summary = det.summarize(pool)
    residue, evidences = build_residue_evidence(pool, con)
    if layer2_limit is not None:
        evidences = evidences[:layer2_limit]
    with_text, no_text = partition_by_text_evidence(evidences)

    groups = layer2.group_into_batches(with_text)
    exact_tokenizer = True
    total_typical_in = total_worst_in = 0
    group_token_counts = []
    for g in groups:
        prompt = layer2.build_batch_prompt(g)
        tokens, exact = _count_tokens_best_effort(prompt, system=layer2._SYSTEM, model=model)
        exact_tokenizer = exact_tokenizer and exact
        group_token_counts.append(tokens)
        total_typical_in += tokens * layer2.INITIAL_K
        total_worst_in += tokens * layer2.ESCALATED_K

    assumed_out_per_candidate = 60  # small structured JSON answer
    total_typical_out = sum(len(g) for g in groups) * assumed_out_per_candidate * layer2.INITIAL_K
    total_worst_out = sum(len(g) for g in groups) * assumed_out_per_candidate * layer2.ESCALATED_K
    layer2_typical_cost = model_client.estimate_cost(total_typical_in, total_typical_out, model, batch=True)
    layer2_worst_cost = model_client.estimate_cost(total_worst_in, total_worst_out, model, batch=True)

    # Layer 3 queue: the no_text bypass is KNOWN now; how many of with_text
    # will additionally route there (unsure/disagreement/recall) is NOT
    # knowable without actually running Layer 2 — reported as a range
    # bound (0 known-minimum from no_text, up to len(with_text) worst case)
    # rather than a fabricated point estimate.
    layer3_known_minimum, layer3_overflow_at_minimum = cap_layer3_queue(no_text)
    est_input_per_l3_call = 200 + 1200  # prompt+system + typical web_search result payload
    est_output_per_l3_call = 150
    layer3_per_call_cost = model_client.estimate_cost(est_input_per_l3_call, est_output_per_l3_call, model, batch=True)

    return {
        "model": model, "exact_tokenizer": exact_tokenizer,
        "layer1": layer1_summary,
        "residue_size": len(residue),
        "layer2_candidates": len(with_text), "layer2_groups": len(groups),
        "layer2_group_tokens": group_token_counts,
        "layer2_typical_cost_usd": layer2_typical_cost, "layer2_worst_cost_usd": layer2_worst_cost,
        "layer3_known_minimum_queue": len(no_text),
        "layer3_cap": layer3.MAX_LAYER3_CANDIDATES,
        "layer3_overflow_at_minimum": len(layer3_overflow_at_minimum),
        "layer3_per_call_cost_usd": layer3_per_call_cost,
        "layer3_minimum_cost_usd": layer3_per_call_cost * min(len(no_text), layer3.MAX_LAYER3_CANDIDATES),
        "layer3_cost_if_all_with_text_also_route_usd": layer3_per_call_cost * min(
            len(no_text) + len(with_text), layer3.MAX_LAYER3_CANDIDATES,
        ),
    }
