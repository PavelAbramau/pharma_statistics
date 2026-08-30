"""One-off analysis supporting the is_adc / in_scope schema split: any row
already reviewed under the old, retired ``flag_invalid`` action (a bare
"not an ADC" flag with no payload) needs re-deciding under the current
gate schema (see vocab.py). `flag_invalid_migration_candidates` surfaces
the highest-priority subset: rows whose proposed_name carries a known
strong ADC naming suffix (discovery.patterns.SUFFIX_TERMS) — these are the
ones most likely to have been called "not an ADC" when the real answer is
"an ADC, just out of scope for some other reason".

The heme/solid trial-scope classification that used to live here has been
superseded by the MeSH-based classifier in labelling/trial_scope.py —
see that module and discovery/mesh_categories.py.
"""
from __future__ import annotations

from typing import Optional

from pharma_stats.discovery.patterns import SUFFIX_TERMS


def has_adc_naming_suffix(name: str) -> Optional[str]:
    """The matched SUFFIX_TERMS entry (strong ADC naming signal), or None."""
    lowered = (name or "").lower()
    for term in SUFFIX_TERMS:
        if term in lowered:
            return term
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
