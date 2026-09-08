"""The "fine view" opportunity matrix: target antigen (HGNC symbol) x
specific MeSH indication term — finer-grained than matrix.py's payload x
tumour-system matrix, at the cost of much sparser cells (more axis
values, same ~230-program population). Reuses matrix.py's live/dead
proxy classification and quadrant rule verbatim (program_live_dead_status,
classify_quadrant, Cell) — only the two axis functions differ.

Target resolution is layered:
  1. attributes.target.derive_target (offline dictionaries + trial-text
     extraction, including the trial_text_majority tier).
  2. A residual-resolution map (program_id -> HGNC symbol) from
     scripts/resolve_target_residual.py's ChEMBL + Open Targets pass —
     passed in explicitly rather than queried live here, so building this
     matrix never depends on network access or repeats the expensive
     residual-resolution API pass.

Indication axis: attributes.indication.program_indication_mesh_term (the
most common specific MeSH condition term across a program's trials) —
NOT the real OncoTree indication_code (see that module's docstring).

Population, same exclusion discipline as matrix.py: is_adc=yes AND
in_scope=yes (gold-first, triage-staged fallback), with a resolved target
on EITHER axis-1 source above, AND a resolved indication_mesh_term.
Unresolved-on-either-axis programs are excluded entirely, never bucketed
into a catch-all cell (same reasoning as matrix.py's payload/tumour-system
exclusions: a catch-all would misrepresent "we don't know" as "this
combination is common").
"""
from __future__ import annotations

from typing import Optional

from pharma_stats.attributes import indication as ind
from pharma_stats.attributes import target as target_attr
from pharma_stats.attributes.matrix import Cell, program_live_dead_status
from pharma_stats.labelling import store
from pharma_stats.triage import evidence as tev


def resolve_target_with_residual(
    program: dict, text_snippets: list, residual: dict[str, dict],
) -> tuple[Optional[str], str]:
    """derive_target's offline tiers first; the residual ChEMBL+Open
    Targets map only for what's still unresolved after that. Source
    string distinguishes the tiers for the population-funnel report."""
    name = program.get("proposed_name")
    synonyms = program.get("synonyms") or []
    t, source = target_attr.derive_target(name, synonyms, text_snippets)
    if t is not None:
        return t, source
    r = residual.get(program["program_id"])
    if r is not None:
        return r["target"], "chembl_opentargets_verified"
    return None, "unresolved"


def build_target_indication_matrix(
    programs: list[dict], con, *, residual: Optional[dict[str, dict]] = None,
) -> tuple[dict[tuple[str, str], Cell], dict[str, dict], dict[str, int]]:
    """(cells, program_attributes, population_stats) -- same shape as
    matrix.build_matrix. Requires a live DuckDB connection (con) to pull
    each program's trial-text evidence for target derivation (the
    payload x tumour-system matrix doesn't need this — target derivation
    is the only axis function here that reads trial text)."""
    residual = residual or {}
    gold_records = store.load_records()
    gold_latest = store.latest_by_program(gold_records)

    cells: dict[tuple[str, str], Cell] = {}
    program_attributes: dict[str, dict] = {}
    n_scope_confirmed = 0
    n_excluded_target_unresolved = 0
    n_excluded_indication_unresolved = 0
    target_source_counts: dict[str, int] = {}

    for p in programs:
        pstatus = program_live_dead_status(p, gold_latest)
        if pstatus is None:
            continue
        n_scope_confirmed += 1

        ev = tev.build_layer2_evidence(p, con)
        target, source = resolve_target_with_residual(p, ev.get("text_snippets"), residual)
        target_source_counts[source] = target_source_counts.get(source, 0) + 1
        if target is None:
            n_excluded_target_unresolved += 1
            continue

        indication_term = ind.program_indication_mesh_term(p)
        if indication_term is None:
            n_excluded_indication_unresolved += 1
            continue

        key = (target, indication_term)
        cell = cells.setdefault(key, Cell(payload=target, tumour_system=indication_term))
        entry = {
            "program_id": p["program_id"], "proposed_name": p.get("proposed_name"),
            "status": pstatus.status, "kill_reason": pstatus.kill_reason, "basis": pstatus.basis,
        }
        if pstatus.is_dead:
            cell.dead_programs.append(entry)
        else:
            cell.live_programs.append(entry)

        program_attributes[p["program_id"]] = {
            "target": target, "target_source": source, "indication_mesh_term": indication_term,
            "is_dead": pstatus.is_dead, "basis": pstatus.basis,
            "status": pstatus.status, "kill_reason": pstatus.kill_reason,
        }

    population_stats = {
        "n_scope_confirmed": n_scope_confirmed,
        "n_excluded_target_unresolved": n_excluded_target_unresolved,
        "n_excluded_indication_unresolved": n_excluded_indication_unresolved,
        "n_in_population": len(program_attributes),
        "target_source_counts": target_source_counts,
    }
    return cells, program_attributes, population_stats
