"""Matrix-only universe extension: ADC trials from 2000-2012 (backward in
time) and haematological-indication ADC trials (sideways in tumour type)
— see docs/decisions/0008. CLAUDE.md's locked scope (industry sponsors,
2012-present, solid tumours only) is relaxed on exactly these two axes,
and ONLY for this module's population. No preclinical data — everything
here is still a real CT.gov trial. These programs never enter
gold/labels.jsonl, provisional_programs, or the detector's training data;
they exist solely to populate density-table cells in the B5 opportunity
matrix's coarse (payload x tumour-system) view, tagged with the
deterministic, lower-confidence `dead_historical` status
(labelling/dead_historical.py) instead of a gold label.

Discovery reuses discovery/candidates.py's Strategy 1 (suffix/literal
pattern matching over query.cond=cancer) — the highest-recall, most
reviewable of its three strategies — with two differences: (1) no 2012
floor, studies starting as early as 2000 are kept
(SCOPE_START_DATE_HISTORICAL), and (2) each study's raw
`conditionsModule.conditions` strings are captured (candidates.py's
_study_context doesn't need them for the main pipeline; this one does,
for the text-based heme/tumour-system proxy below). Seed and sponsor
expansion (candidates.py's strategies 2/3) are deliberately NOT run here
— bounding this to one pass keeps a lower-priority, matrix-only side
population from costing as much API time as the real universe build; this
is a documented recall trade-off, not an oversight.

Tumour-system/heme classification is a TEXT heuristic on raw condition
strings (discovery/heme_scope_text.py), not the MeSH-ID pipeline
attributes/tumour_system.py uses for the main matrix — that pipeline
needs a current-state CT.gov fetch per trial (conditionBrowseModule; see
discovery/mesh_categories.py's module docstring), and paying that cost
for a matrix-only, explicitly lower-confidence cohort was judged not
worth it. Every extension program is therefore excluded from the FINE
(target x specific MeSH indication) matrix view entirely — that axis has
no non-MeSH proxy — and only ever appears in the COARSE view, tagged
tumour_system_basis="text_heuristic" so no downstream table can conflate
it with the real MeSH-based resolution used everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from pharma_stats.attributes import payload as pl
from pharma_stats.attributes import target as target_attr
from pharma_stats.attributes.matrix import Cell
from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.discovery.candidates import (
    FIELDS,
    CandidateAsset,
    Mention,
    _candidate_interventions,
    _study_context,
    _study_start_date,
    build_candidate_table,
)
from pharma_stats.discovery.candidates import SCOPE_START_DATE as MAIN_SCOPE_START_DATE
from pharma_stats.discovery.heme_scope_text import classify_condition_text
from pharma_stats.discovery.known_approved_adcs import known_approved_name_set
from pharma_stats.discovery.patterns import LITERAL_TERMS, SUFFIX_TERMS, matches_pattern
from pharma_stats.labelling import dead_historical as dh
from pharma_stats.triage.deterministic import evaluate_is_adc

SCOPE_START_DATE_HISTORICAL = date(2000, 1, 1)


def _in_extension_window(study: dict) -> bool:
    """Same over-inclusion posture as candidates._in_scope_by_date: an
    unparseable/missing start date is kept, never excluded on an unknown."""
    start = _study_start_date(study)
    return start is None or start >= SCOPE_START_DATE_HISTORICAL


def _study_conditions(study: dict) -> list[str]:
    return list(study["protocolSection"].get("conditionsModule", {}).get("conditions", []))


def discover_extension_mentions(
    client: CtgovClient, *, max_per_query: int = 3000,
) -> tuple[list[Mention], dict[str, list[str]]]:
    """(mentions, conditions_by_nct) — pattern-matching pass only (no seed/
    sponsor expansion), no 2012 floor. conditions_by_nct is keyed by every
    matched trial's nct_id, for heme_scope_text classification after
    clustering (CandidateAsset itself carries no condition data)."""
    queries: list[tuple[str, str]] = [("intr", t) for t in SUFFIX_TERMS] + [
        ("term", t) for t in LITERAL_TERMS
    ]
    seen_studies: set[str] = set()
    mentions: list[Mention] = []
    conditions_by_nct: dict[str, list[str]] = {}

    for kind, term in queries:
        kwargs = dict(cond="cancer", fields=FIELDS, max_studies=max_per_query)
        if kind == "intr":
            kwargs["intr"] = term
        else:
            kwargs["term"] = term

        for study in client.search_studies(**kwargs):
            ctx = _study_context(study)
            if ctx["nct_id"] in seen_studies or not _in_extension_window(study):
                continue
            for entry in _candidate_interventions(study):
                names = [entry.get("name", "")] + list(entry.get("otherNames", []))
                hit = None
                for n in names:
                    if not n:
                        continue
                    m = matches_pattern(n)
                    if m and (hit is None or m[0] == "suffix"):
                        hit = m
                if hit is None:
                    continue
                seen_studies.add(ctx["nct_id"])
                conditions_by_nct[ctx["nct_id"]] = _study_conditions(study)
                mentions.append(Mention(
                    intervention_name=entry.get("name", ""),
                    other_names=list(entry.get("otherNames", [])),
                    intervention_type=entry.get("type", ""),
                    strategy="pattern_match_extension",
                    match_strength=hit[0],
                    match_term=hit[1],
                    **ctx,
                ))
    return mentions, conditions_by_nct


@dataclass
class ExtensionProgram:
    program_id: str
    proposed_name: str
    synonyms: list[str]
    sponsors: list[str]
    nct_ids: list[str]
    trial_start_dates: list[date]
    target: Optional[str]
    payload_chemotype: str
    tumour_bucket: Optional[str]  # one of tumour_system.TUMOUR_SYSTEM_VALUES, or "heme", or None
    extension_reasons: list[str] = field(default_factory=list)  # "pre_2012" and/or "heme"
    dead_historical: Optional[str] = None


def _parse_date(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def build_extension_programs(
    candidates: list[CandidateAsset], conditions_by_nct: dict[str, list[str]],
    *, existing_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[list[ExtensionProgram], dict]:
    """Resolve payload/target/tumour-bucket, apply the locked-scope
    filters this module still keeps (ADC-only, industry-only), and tag
    each surviving candidate with why it's in the extension (pre_2012,
    heme, or both) — candidates entirely inside the main pipeline's own
    2012+/solid window are dropped here, not double-counted; they belong
    to the real universe (scripts/build_candidate_universe.py), not this
    one. Returns (programs, funnel_stats).

    `existing_candidate_ids` should be every candidate_id already in the
    main pipeline's asset_candidates/provisional_programs (pass
    `{p.get("candidate_id") for p in provisional_programs.load_materialized()}`)
    — this module's own clustering (discovery.candidates.build_candidate_table)
    generates the SAME deterministic candidate_id hash for the same
    compound, so re-discovering an asset the main pipeline already found
    (a real case: a heme-only compound with SOME trials in the 2012+
    window gets caught by both passes) must be excluded here, not just
    tagged "mx-ext-" — otherwise labelling.dead_historical's successor
    check can spuriously treat the SAME asset's two independently-
    discovered copies as two different programs (found on real data,
    2026-09-08: Indatuximab Ravtansine/BT062 wrongly read as its own
    successor)."""
    stats = {
        "n_candidates_discovered": len(candidates),
        "n_excluded_not_adc": 0,
        "n_excluded_non_industry": 0,
        "n_excluded_already_in_main_universe": 0,
        "n_excluded_neither_reason": 0,
        "n_in_extension": 0,
    }
    programs: list[ExtensionProgram] = []

    for c in candidates:
        program_id = f"mx-ext-{c.candidate_id}"
        pseudo = {"proposed_name": c.proposed_name, "synonyms": c.synonyms}
        is_adc, _rule = evaluate_is_adc(pseudo)
        if is_adc != "yes":
            stats["n_excluded_not_adc"] += 1
            continue

        sponsors = [s["sponsor"] for s in c.sponsors_over_time if s.get("sponsor")]
        classes = {s.get("class") for s in c.sponsors_over_time}
        if not sponsors or classes != {"INDUSTRY"}:
            stats["n_excluded_non_industry"] += 1
            continue

        if c.candidate_id in existing_candidate_ids:
            stats["n_excluded_already_in_main_universe"] += 1
            continue

        first = _parse_date(c.first_trial_start_date)
        reasons = []
        if first is not None and first < MAIN_SCOPE_START_DATE:
            reasons.append("pre_2012")

        conditions: list[str] = []
        for nct in c.nct_ids:
            conditions.extend(conditions_by_nct.get(nct, []))
        tumour_bucket = classify_condition_text(conditions)
        if tumour_bucket == "heme":
            reasons.append("heme")

        if not reasons:
            stats["n_excluded_neither_reason"] += 1
            continue

        target, _source = target_attr.derive_target(c.proposed_name, c.synonyms, text_snippets=None)
        chemotype = pl.derive_payload_chemotype(c.proposed_name, c.synonyms)
        trial_start_dates = [d for d in [_parse_date(c.first_trial_start_date),
                                          _parse_date(c.last_trial_start_date)] if d is not None]

        stats["n_in_extension"] += 1
        programs.append(ExtensionProgram(
            program_id=program_id, proposed_name=c.proposed_name, synonyms=c.synonyms,
            sponsors=sponsors, nct_ids=c.nct_ids, trial_start_dates=trial_start_dates,
            target=target, payload_chemotype=chemotype, tumour_bucket=tumour_bucket,
            extension_reasons=reasons,
        ))

    return programs, stats


def build_population_index(
    extension_programs: list[ExtensionProgram], main_programs: list[dict],
) -> dict[tuple[str, str], list[tuple[str, Optional[date]]]]:
    """(sponsor, target) -> [(program_id, earliest_trial_start_date)] —
    across BOTH this extension cohort and the main gold-confirmed/
    materialized universe, for dead_historical's successor check. main
    programs without a resolvable target contribute nothing (can't be
    anyone's successor if we don't know what they target).

    Target resolution here is offline-tiers only (derive_target with no
    trial-text snippets) — cheaper than the fine matrix view's full
    resolution (attributes/target_indication_matrix.py, which also reads
    trial text via a DB connection), at the cost of missing some real
    targets that only resolve from trial text. That under-resolution
    biases this check toward MISSING a real successor (target doesn't
    resolve -> that main-universe program can't appear in the index at
    all), which in turn biases dead_historical toward over-firing rather
    than under-firing — a real, disclosed approximation (docs/decisions
    /0008), not a silent one."""
    index: dict[tuple[str, str], list[tuple[str, Optional[date]]]] = {}

    for p in extension_programs:
        if not p.target:
            continue
        earliest = min(p.trial_start_dates) if p.trial_start_dates else None
        for sponsor in p.sponsors:
            index.setdefault((sponsor, p.target), []).append((p.program_id, earliest))

    for mp in main_programs:
        name = mp.get("proposed_name")
        synonyms = mp.get("synonyms") or []
        target, _source = target_attr.derive_target(name, synonyms, text_snippets=None)
        if not target:
            continue
        starts = [t.get("start_date") for t in mp.get("trials") or [] if t.get("start_date")]
        earliest = min(_parse_date(s) for s in starts) if starts else None
        for s in mp.get("sponsors_over_time") or []:
            sponsor = s.get("sponsor")
            if sponsor:
                index.setdefault((sponsor, target), []).append((mp["program_id"], earliest))

    return index


def classify_extension_cohort(
    programs: list[ExtensionProgram],
    population_index: dict[tuple[str, str], list[tuple[str, Optional[date]]]],
    *, today: Optional[date] = None,
) -> None:
    """Mutates each ExtensionProgram's .dead_historical in place per
    labelling.dead_historical's deterministic rule."""
    today = today or date.today()
    approved = known_approved_name_set()
    for p in programs:
        names = [p.proposed_name] + p.synonyms
        p.dead_historical = dh.classify_dead_historical(
            p.program_id, names, p.sponsors, p.target, p.trial_start_dates,
            today=today, population_index=population_index, approved_names=approved,
        )


def fold_into_coarse_matrix(
    cells: dict[tuple[str, str], Cell], programs: list[ExtensionProgram],
) -> dict:
    """Adds each dead_historical extension program to the SAME coarse
    (payload x tumour_system) cell dict the main matrix (attributes/
    matrix.py's build_matrix) already produced — mutating in place, and
    creating a new Cell for a (payload, tumour_system) combination the
    gold-quality population never populated (most commonly payload x
    "heme", a 9th tumour_system value that never appears in the main
    matrix at all, since attributes/tumour_system.py's 8 groups are
    solid-only by construction). Only ever appends to
    dead_historical_programs — n_live/n_dead/quadrant classification for
    every pre-existing cell are untouched.

    Only dead_historical programs are folded in (not_classified/live
    programs from this cohort have no honest status to report in a
    live/dead density table and are left out, counted separately in the
    caller's coverage report instead)."""
    n_folded = 0
    n_excluded_payload_undisclosed = 0
    n_excluded_tumour_unresolved = 0
    for p in programs:
        if p.dead_historical != "dead_historical":
            continue
        if p.payload_chemotype == "undisclosed":
            n_excluded_payload_undisclosed += 1
            continue
        if p.tumour_bucket is None:
            n_excluded_tumour_unresolved += 1
            continue
        key = (p.payload_chemotype, p.tumour_bucket)
        cell = cells.setdefault(key, Cell(payload=p.payload_chemotype, tumour_system=p.tumour_bucket))
        cell.dead_historical_programs.append({
            "program_id": p.program_id, "proposed_name": p.proposed_name,
            "status": "dead_historical", "kill_reason": None, "basis": "deterministic_rule",
            "extension_reasons": p.extension_reasons,
        })
        n_folded += 1
    return {
        "n_dead_historical_folded_in": n_folded,
        "n_excluded_payload_undisclosed": n_excluded_payload_undisclosed,
        "n_excluded_tumour_unresolved": n_excluded_tumour_unresolved,
    }
