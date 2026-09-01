"""Evidence bundle for Layer 2's batched model call.

Per candidate: name, up to 5 synonyms, lead sponsor, up to 3 condition
strings (all already on the materialized program dict), PLUS trial free
text — the intervention description field and the first ~300 characters
of briefSummary — pulled straight from the best available raw CT.gov
snapshot (reusing provisional_programs._best_trial_snapshot /
_study_from_body, the exact same resolution logic the rest of the project
uses; this module doesn't reinvent snapshot lookup).

Why the text matters: sponsors frequently state the modality explicitly
in these fields ("... is an antibody-drug conjugate targeting ..."), which
converts Layer 2's question from a memory question ("do I recognize this
name") into an extraction question ("does this text say what it is") —
dramatically more reliable for the unnamed dev codes that make up most of
Layer 1's residue. Never the trial dump or event timeline — none of that
helps decide whether a molecule is an ADC, and it would only inflate the
prompt (see silver/evidence.py's own timeline trim for the same
principle, applied there for the same reason on a different question).
"""
from __future__ import annotations

from typing import Optional

import duckdb

from pharma_stats.labelling.provisional_programs import _best_trial_snapshot, _study_from_body
from pharma_stats.triage.grounding import snippet_mentions_candidate, truncate_at_word

BRIEF_SUMMARY_CHARS = 300
MAX_TRIALS_FOR_TEXT = 3   # cap raw-snapshot reads per candidate — token/latency efficiency
MAX_TEXT_SNIPPETS = 5     # cap total snippets sent to the model, across all trials
MAX_SYNONYMS = 5
MAX_CONDITIONS = 3


def _trial_text_fields(nct_id: str, con: duckdb.DuckDBPyConnection) -> tuple[Optional[str], list[str]]:
    """(brief_summary, intervention_descriptions) for one trial, from its
    best available raw snapshot. Never captured in TrialSummary — nothing
    else in the project needs this text, only Layer 2's evidence bundle."""
    resolved = _best_trial_snapshot(nct_id, con)
    if resolved is None:
        return None, []
    body, _source = resolved
    study = _study_from_body(body)
    protocol = study.get("protocolSection") or {}
    brief_summary = (protocol.get("descriptionModule") or {}).get("briefSummary")
    interventions = (protocol.get("armsInterventionsModule") or {}).get("interventions") or []
    descriptions = [i["description"] for i in interventions if i.get("description")]
    return brief_summary, descriptions


def build_layer2_evidence(program: dict, con: duckdb.DuckDBPyConnection) -> dict:
    """The exact, capped bundle sent to Layer 2 for one candidate —
    nothing else. sponsors_over_time is sorted by name, not recency (see
    discovery/candidates.py), so "lead sponsor" here is whichever has the
    latest last_seen date — the same "most current" logic the sponsor
    audit report uses, not just sponsors[-1]."""
    sponsors = program.get("sponsors_over_time") or []
    lead_sponsor = None
    if sponsors:
        dated = [s for s in sponsors if s.get("last_seen")]
        pool = dated or sponsors
        lead_sponsor = max(pool, key=lambda s: s.get("last_seen") or "")["sponsor"]

    conditions: list[str] = []
    for t in program.get("trials") or []:
        for c in t.get("conditions") or []:
            if c not in conditions:
                conditions.append(c)
        if len(conditions) >= MAX_CONDITIONS:
            break

    text_snippets: list[str] = []
    name = program.get("proposed_name")
    synonyms = (program.get("synonyms") or [])[:MAX_SYNONYMS]
    trials = program.get("trials") or []
    for t in trials[:MAX_TRIALS_FOR_TEXT]:
        brief_summary, descriptions = _trial_text_fields(t["nct_id"], con)
        candidates = []
        if brief_summary:
            candidates.append(truncate_at_word(brief_summary, BRIEF_SUMMARY_CHARS))
        candidates.extend(d for d in descriptions if d)
        for snippet in candidates:
            if snippet_mentions_candidate(snippet, name, synonyms):
                text_snippets.append(snippet)
        if len(text_snippets) >= MAX_TEXT_SNIPPETS:
            break

    return {
        "program_id": program["program_id"],
        "name": name,
        "synonyms": synonyms,
        "lead_sponsor": lead_sponsor,
        "conditions": conditions[:MAX_CONDITIONS],
        "text_snippets": text_snippets[:MAX_TEXT_SNIPPETS],
    }
