"""Deterministic dead_historical rule for the matrix-only universe
extension (discovery/matrix_extension.py) — see docs/decisions/0008. This
cohort (ADC trials from 2000-2012, and haematological-indication ADC
trials) gets NO manual labelling, ever: it exists solely to populate the
B5 opportunity matrix's density tables, never gold/labels.jsonl, never
provisional_programs, never the detector's training data.

A program qualifies for dead_historical iff ALL of:

  1. No trial started in the last 10 years (as of `today`) — the most
     recent trial start date across the program is more than 10 years
     ago. A program still enrolling recently is not "historical" by this
     rule's own name, regardless of how it might resolve otherwise.
  2. Not a known-approved compound
     (discovery/known_approved_adcs.py's small, hand-curated,
     conservative list — a real external fact this project has no live
     regulatory-database access to verify).
  3. No successor: no OTHER program, from the same sponsor, on the same
     resolved target, with a LATER earliest-trial-start-date — checked
     against a population index the caller builds from both this
     extension cohort and the main gold-confirmed/materialized universe.
     Vacuously satisfied ("no known successor") when the target itself
     doesn't resolve (attributes.target.derive_target unresolved) —
     lineage can't be checked without knowing the lineage's identity, and
     CLAUDE.md's over-inclusion-over-precision default applies here too
     (silently withholding dead_historical, rather than guessing, is the
     safe failure direction).

Failing rule 1 or matching rule 2 means "not classified" — this module
answers exactly one question (dead_historical yes/no), never a different
status; a program that fails this rule is simply left out of the
extension's density counts, not silently guessed into "live" or any real
gold status.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

TEN_YEARS_DAYS = 3650


def _names_lower(names: list[str]) -> set[str]:
    return {n.strip().lower() for n in names if n}


def is_known_approved(names: list[str], approved_names: frozenset[str]) -> bool:
    return bool(_names_lower(names) & approved_names)


def has_trial_in_last_10_years(most_recent_trial_start: Optional[date], *, today: date) -> bool:
    """True (never "historical") if there's no trial-start date at all to
    reason from — an unknown recency is not evidence of historical
    inactivity, never guessed as old."""
    if most_recent_trial_start is None:
        return True
    return most_recent_trial_start >= today - timedelta(days=TEN_YEARS_DAYS)


def has_successor(
    own_program_id: str, sponsor: str, target: str, own_earliest_start: Optional[date],
    population_index: dict[tuple[str, str], list[tuple[str, Optional[date]]]],
) -> bool:
    """True if the same (sponsor, target) key has another program (not
    this one) whose own earliest trial start is LATER than this
    program's — the asset's lineage continued under this sponsor, so
    this specific program reads as superseded-in-spirit, not dead."""
    if own_earliest_start is None:
        return False
    for other_id, other_start in population_index.get((sponsor, target), []):
        if other_id == own_program_id or other_start is None:
            continue
        if other_start > own_earliest_start:
            return True
    return False


def classify_dead_historical(
    program_id: str,
    names: list[str],
    sponsors: list[str],
    target: Optional[str],
    trial_start_dates: list[date],
    *,
    today: date,
    population_index: dict[tuple[str, str], list[tuple[str, Optional[date]]]],
    approved_names: frozenset[str],
) -> Optional[str]:
    """"dead_historical" or None. See module docstring for the three
    rules; all three must be satisfied (recency fails it, approval fails
    it, a real successor fails it)."""
    if not trial_start_dates:
        return None  # nothing to reason about recency from at all

    most_recent = max(trial_start_dates)
    if has_trial_in_last_10_years(most_recent, today=today):
        return None

    if is_known_approved(names, approved_names):
        return None

    if target:
        earliest = min(trial_start_dates)
        for sponsor in sponsors:
            if has_successor(program_id, sponsor, target, earliest, population_index):
                return None

    return "dead_historical"
