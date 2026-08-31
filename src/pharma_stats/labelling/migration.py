"""Analysis + append-only backfill for the is_adc / in_scope split.

Two jobs:

1. Backfill latest is_adc=no rows that never recorded in_scope (or that
   recorded the inconsistent pair is_adc=no + in_scope=yes). New lines
   only — gold/ is append-only. See `rows_needing_scope_backfill` and
   `build_scope_backfill_record`.
2. Surface latest is_adc=no rows whose proposed_name or synonyms carry a
   known ADC naming suffix (discovery.patterns.SUFFIX_TERMS). Those are
   the ones most likely to have been called "not an ADC" when the real
   answer is "an ADC, just out of scope" — re-decide them by hand.

The heme/solid trial-scope classification lives in labelling/trial_scope.py.
"""
from __future__ import annotations

from typing import Optional

from pharma_stats.discovery.patterns import SUFFIX_TERMS, matches_pattern
from pharma_stats.labelling import store

BACKFILL_SESSION_ID = "migrate:is_adc_in_scope_split"


def has_adc_naming_suffix(name: str) -> Optional[str]:
    """The matched SUFFIX_TERMS entry (strong ADC naming signal), or None."""
    lowered = (name or "").lower()
    for term in SUFFIX_TERMS:
        if term in lowered:
            return term
    return None


def _names_for(record: dict, program: Optional[dict] = None) -> list[str]:
    names = [record.get("proposed_name") or ""]
    if program:
        names.extend(program.get("synonyms") or [])
    return [n for n in names if n]


def first_suffix_hit(names: list[str]) -> Optional[tuple[str, str]]:
    """(matched_suffix, name_it_fired_on) or None."""
    for name in names:
        suffix = has_adc_naming_suffix(name)
        if suffix:
            return suffix, name
    return None


def first_pattern_hit(names: list[str]) -> Optional[tuple[str, str, str]]:
    """(strength, term, name_it_fired_on) or None — suffix or literal."""
    for name in names:
        hit = matches_pattern(name)
        if hit:
            return hit[0], hit[1], name
    return None


def flag_invalid_migration_candidates(records: list[dict]) -> list[dict]:
    """Rows saved under the old flag_invalid action (no is_adc field at
    all — that's what marks them as pre-migration, since every row saved
    under the current gate model always carries one) whose proposed_name
    hits a known ADC naming suffix. Re-decide these first under the
    corrected schema."""
    out = []
    for r in records:
        if r.get("action") != "flag_invalid" or r.get("is_adc") is not None:
            continue
        suffix = has_adc_naming_suffix(r.get("proposed_name") or "")
        if suffix:
            out.append({
                "event_id": r.get("event_id"),
                "program_id": r.get("program_id"),
                "proposed_name": r.get("proposed_name"),
                "matched_suffix": suffix,
            })
    return out


def not_an_adc_suffix_candidates(
    records: list[dict], programs_by_id: Optional[dict] = None,
) -> list[dict]:
    """Latest is_adc=no rows whose proposed_name or synonyms hit a known
    INN/USAN ADC suffix. These stay is_adc=no in the backfill — they need
    a human re-decision, not an automatic flip."""
    programs_by_id = programs_by_id or {}
    out = []
    for r in store.latest_by_program(records).values():
        if r.get("is_adc") != "no":
            continue
        names = _names_for(r, programs_by_id.get(r["program_id"]))
        hit = first_suffix_hit(names)
        if not hit:
            continue
        suffix, matched_on = hit
        out.append({
            "event_id": r.get("event_id"),
            "program_id": r.get("program_id"),
            "proposed_name": r.get("proposed_name"),
            "matched_suffix": suffix,
            "matched_on": matched_on,
        })
    out.sort(key=lambda row: (row["proposed_name"] or "").lower())
    return out


def not_an_adc_literal_candidates(
    records: list[dict], programs_by_id: Optional[dict] = None,
) -> list[dict]:
    """Latest is_adc=no rows with a weaker literal ADC/conjugate hit (and
    no INN suffix). Not the requested re-decision list — a near-miss
    shortlist so a real ADC without a USAN suffix (e.g. an exatecan
    conjugate) isn't invisible."""
    programs_by_id = programs_by_id or {}
    out = []
    for r in store.latest_by_program(records).values():
        if r.get("is_adc") != "no":
            continue
        names = _names_for(r, programs_by_id.get(r["program_id"]))
        if first_suffix_hit(names):
            continue
        hit = first_pattern_hit(names)
        if not hit:
            continue
        strength, term, matched_on = hit
        out.append({
            "event_id": r.get("event_id"),
            "program_id": r.get("program_id"),
            "proposed_name": r.get("proposed_name"),
            "match_strength": strength,
            "matched_term": term,
            "matched_on": matched_on,
        })
    out.sort(key=lambda row: (row["proposed_name"] or "").lower())
    return out


def rows_needing_scope_backfill(records: list[dict]) -> list[dict]:
    """Latest is_adc=no rows that don't yet carry in_scope=no / not_an_adc."""
    out = []
    for r in store.latest_by_program(records).values():
        if r.get("is_adc") != "no":
            continue
        if r.get("in_scope") == "no" and r.get("scope_reason") == "not_an_adc":
            continue
        out.append(r)
    out.sort(key=lambda row: (row.get("proposed_name") or "").lower())
    return out


def build_scope_backfill_record(old: dict) -> dict:
    """Append-only correction: keep the reviewer's is_adc=no, write the
    matching in_scope=no / not_an_adc pair. Does not flip the molecule
    judgement — suffix hits still need a later human line."""
    decided_by = old.get("decided_by") or "human"
    body = {
        "action": "label",
        "program_id": old["program_id"],
        "candidate_id": old.get("candidate_id"),
        "proposed_name": old.get("proposed_name"),
        "gate_reached": 1,
        "decided_by": decided_by,
        "is_adc": "no",
        "in_scope": "no",
        "scope_reason": "not_an_adc",
        "discovery_strategy": old.get("discovery_strategy"),
        "match_strength": old.get("match_strength"),
        "matched_term": old.get("matched_term"),
        "evidence_note": old.get("evidence_note") or "",
        "blind": bool(old.get("blind")),
    }
    if decided_by == "auto":
        body["triage_layer"] = old.get("triage_layer") or 1
        body["triage_rule"] = old.get("triage_rule") or "schema_split_backfill"
    store.validate_label_payload(body)
    return store.build_record(
        body,
        session_id=BACKFILL_SESSION_ID,
        served_stratum={
            "band": old.get("stratum_band"),
            "archetype": old.get("stratum_archetype"),
            "silence_score": old.get("silence_score_at_label_time"),
            "history_coverage": old.get("history_coverage_at_serve_time"),
        },
    )
