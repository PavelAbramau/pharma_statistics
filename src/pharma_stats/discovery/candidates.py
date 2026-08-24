"""Builds the ADC candidate-asset table from CT.gov via three independent,
union-ed identification strategies:

1. pattern_match  — naming-suffix / literal-term pattern matching over
   intervention names encountered in oncology studies.
2. seed_expansion — every trial that shares an intervention with a seed
   asset (see seed_assets.json).
3. sponsor_expansion — for any industry sponsor that produced a
   pattern_match/seed_expansion hit, every other DRUG/BIOLOGICAL
   intervention in that sponsor's oncology trials that looks like an ADC
   or an unnamed development-stage code.

This module deliberately does NOT assign payload, target, or indication —
those come from the controlled-vocabulary normalisation step. It only
identifies candidate assets, their trials, and the raw name/synonym
strings encountered for each.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.discovery.patterns import (
    CANDIDATE_INTERVENTION_TYPES,
    is_denylisted,
    looks_like_dev_code,
    matches_pattern,
    LITERAL_TERMS,
    SUFFIX_TERMS,
)

SEED_ASSETS_PATH = Path(__file__).with_name("seed_assets.json")

SCOPE_START_DATE = date(2012, 1, 1)

FIELDS = [
    "NCTId", "BriefTitle", "OverallStatus", "StudyType", "StartDate",
    "LeadSponsorName", "LeadSponsorClass", "InterventionName",
    "InterventionOtherName", "InterventionType", "Condition",
]


def load_seed_assets() -> list[dict]:
    payload = json.loads(SEED_ASSETS_PATH.read_text())
    return payload["seed_assets"]


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _study_start_date(study: dict) -> Optional[date]:
    raw = study["protocolSection"].get("statusModule", {}).get("startDateStruct", {}).get("date")
    if not raw:
        return None
    try:
        parts = raw.split("-")
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _in_scope_by_date(study: dict) -> bool:
    """Exclude only studies with a clearly-parseable start date before the
    2012 scope boundary. Studies with no/unparseable start date are kept —
    favour over-inclusion; the human audit and later normalisation can
    drop them."""
    start = _study_start_date(study)
    return start is None or start >= SCOPE_START_DATE


@dataclass
class Mention:
    nct_id: str
    intervention_name: str
    other_names: list[str]
    intervention_type: str
    strategy: str
    lead_sponsor: Optional[str]
    lead_sponsor_class: Optional[str]
    study_start_date: Optional[date]
    overall_status: Optional[str]
    brief_title: str
    match_strength: Optional[str] = None  # "suffix" | "literal" | "seed" | "dev_code" | None


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _candidate_interventions(study: dict) -> Iterator[dict]:
    interventions = (
        study["protocolSection"].get("armsInterventionsModule", {}).get("interventions", [])
    )
    for entry in interventions:
        if entry.get("type") in CANDIDATE_INTERVENTION_TYPES:
            yield entry


def _study_context(study: dict) -> dict:
    ident = study["protocolSection"]["identificationModule"]
    status = study["protocolSection"].get("statusModule", {})
    sponsor = study["protocolSection"].get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    return dict(
        nct_id=ident["nctId"],
        brief_title=ident.get("briefTitle", ""),
        lead_sponsor=sponsor.get("name"),
        lead_sponsor_class=sponsor.get("class"),
        study_start_date=_study_start_date(study),
        overall_status=status.get("overallStatus"),
    )


# -- Strategy 1: pattern matching -------------------------------------------

def iter_pattern_matches(client: CtgovClient, *, max_per_query: int = 5000) -> Iterator[Mention]:
    queries: list[tuple[str, str]] = [("intr", t) for t in SUFFIX_TERMS] + [
        ("term", t) for t in LITERAL_TERMS
    ]
    seen_studies: set[str] = set()

    for kind, term in queries:
        kwargs = dict(cond="cancer", fields=FIELDS, max_studies=max_per_query)
        if kind == "intr":
            kwargs["intr"] = term
        else:
            kwargs["term"] = term

        for study in client.search_studies(**kwargs):
            ctx = _study_context(study)
            if ctx["nct_id"] in seen_studies or not _in_scope_by_date(study):
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
                yield Mention(
                    intervention_name=entry.get("name", ""),
                    other_names=list(entry.get("otherNames", [])),
                    intervention_type=entry.get("type", ""),
                    strategy="pattern_match",
                    match_strength=hit[0],
                    **ctx,
                )


# -- Strategy 2: seed expansion ----------------------------------------------

def iter_seed_matches(client: CtgovClient, seed_assets: list[dict]) -> Iterator[Mention]:
    for seed in seed_assets:
        seed_terms = [seed["name"]] + list(seed.get("synonyms", []))
        seed_normalized = {_normalize(t) for t in seed_terms}

        for term in seed_terms:
            for study in client.search_studies(
                intr=term, cond="cancer", fields=FIELDS, max_studies=2000
            ):
                if not _in_scope_by_date(study):
                    continue
                ctx = _study_context(study)
                for entry in _candidate_interventions(study):
                    names = [entry.get("name", "")] + list(entry.get("otherNames", []))
                    normalized_names = {_normalize(n) for n in names if n}
                    if not (normalized_names & seed_normalized):
                        continue
                    yield Mention(
                        intervention_name=entry.get("name", ""),
                        other_names=list(entry.get("otherNames", [])),
                        intervention_type=entry.get("type", ""),
                        strategy="seed_expansion",
                        match_strength="seed",
                        **ctx,
                    )


# -- Strategy 3: sponsor expansion -------------------------------------------

def iter_sponsor_matches(
    client: CtgovClient, sponsor_names: set[str], known_normalized_names: set[str],
    *, max_per_sponsor: int = 800,
) -> Iterator[Mention]:
    for sponsor in sorted(sponsor_names):
        for study in client.search_studies(
            spons=sponsor, cond="cancer", fields=FIELDS, max_studies=max_per_sponsor,
        ):
            if not _in_scope_by_date(study):
                continue
            ctx = _study_context(study)
            if ctx["lead_sponsor"] != sponsor:
                continue  # collaborator match, not lead — not this sponsor's own asset
            for entry in _candidate_interventions(study):
                name = entry.get("name", "")
                if not name or is_denylisted(name):
                    continue
                normalized = _normalize(name)
                if normalized in known_normalized_names:
                    continue  # already captured by strategy 1/2

                pattern_hit = matches_pattern(name)
                if pattern_hit is None:
                    for n in entry.get("otherNames", []):
                        if n and matches_pattern(n):
                            pattern_hit = matches_pattern(n)
                            break
                dev_code = looks_like_dev_code(name)
                if not (pattern_hit or dev_code):
                    continue

                yield Mention(
                    intervention_name=name,
                    other_names=list(entry.get("otherNames", [])),
                    intervention_type=entry.get("type", ""),
                    strategy="sponsor_expansion",
                    match_strength=pattern_hit[0] if pattern_hit else "dev_code",
                    **ctx,
                )


# -- Clustering ---------------------------------------------------------------

@dataclass
class CandidateAsset:
    candidate_id: str
    proposed_name: str
    synonyms: list[str]
    sponsors_over_time: list[dict]
    trial_count: int
    nct_ids: list[str]
    first_trial_start_date: Optional[str]
    last_trial_start_date: Optional[str]
    strategies: list[str]
    ambiguous: bool
    dev_code_only: bool
    review_status: str = "unreviewed"


def _pick_proposed_name(raw_names: list[str]) -> str:
    """Pick the cleanest generic-looking name from a synonym cluster.

    Sponsors sometimes put a whole sentence, dosing regimen, or combo
    description in the intervention-name field (e.g. "Sacituzumab
    govitecan is a humanized mAb with a hydrolysable linker through
    which..."). Longest-name-wins would pick that; INN-style generic
    names are typically 1-4 words, so we score for closeness to that
    length instead of raw length.
    """
    def score(n: str) -> tuple:
        suffix_hit = matches_pattern(n)
        has_suffix = 1 if (suffix_hit and suffix_hit[0] == "suffix") else 0
        lowered = n.lower()
        combo_penalty = -(
            n.count(",") + n.count("+") + (" and " in lowered)
            + bool(re.search(r"\d+\s*(mg|kg|mcg|mg/kg)\b", lowered))
        )
        word_count = len(n.strip().split())
        closeness_to_inn_length = -abs(word_count - 3)
        return (has_suffix, combo_penalty, closeness_to_inn_length, -len(n))

    return max(raw_names, key=score)


def _merge_substring_clusters(uf: UnionFind, all_keys: set[str]) -> None:
    """Second clustering pass: many sponsors put a whole combo-regimen or
    arm description in the intervention-name field instead of a bare drug
    name (e.g. "Arm A: Belantamab Mafodotin", "Belantamab mafodotin,
    dexamethasone, ixazomib, pomalidomide"). Exact-string / otherNames
    union alone leaves these as their own singleton clusters even though
    they're clearly the same asset. This merges any key into a
    multi-word "anchor" key it whole-word-contains — deterministic
    substring containment, not fuzzy/probabilistic matching.
    """
    keys_list = sorted(all_keys)
    anchors = [k for k in keys_list if len(k.split()) >= 2]
    for anchor in anchors:
        boundary_re = None
        for k in keys_list:
            if k == anchor or anchor not in k:
                continue
            if boundary_re is None:
                boundary_re = re.compile(r"(?<!\w)" + re.escape(anchor) + r"(?!\w)")
            if boundary_re.search(k):
                uf.union(anchor, k)


def build_candidate_table(mentions: list[Mention]) -> list[CandidateAsset]:
    uf = UnionFind()
    all_keys: set[str] = set()
    for m in mentions:
        keys = sorted({_normalize(n) for n in [m.intervention_name] + m.other_names if n})
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)
        for k in keys:
            uf.find(k)  # ensure registered even if singleton
        all_keys.update(keys)

    _merge_substring_clusters(uf, all_keys)

    clusters: dict[str, list[Mention]] = {}
    for m in mentions:
        keys = [_normalize(n) for n in [m.intervention_name] + m.other_names if n]
        if not keys:
            continue
        root = uf.find(keys[0])
        clusters.setdefault(root, []).append(m)

    candidates: list[CandidateAsset] = []
    for root, group in clusters.items():
        root_hash = hashlib.sha256(root.encode("utf-8")).hexdigest()[:10]
        raw_names = sorted({m.intervention_name for m in group if m.intervention_name})
        for m in group:
            raw_names += [n for n in m.other_names if n]
        raw_names = sorted(set(raw_names))
        proposed = _pick_proposed_name(raw_names)
        synonyms = [n for n in raw_names if n != proposed]

        sponsors: dict[str, dict] = {}
        for m in group:
            if not m.lead_sponsor:
                continue
            rec = sponsors.setdefault(
                m.lead_sponsor, {"sponsor": m.lead_sponsor, "class": m.lead_sponsor_class,
                                  "first_seen": None, "last_seen": None},
            )
            d = m.study_start_date
            if d:
                if rec["first_seen"] is None or d < date.fromisoformat(rec["first_seen"]):
                    rec["first_seen"] = d.isoformat()
                if rec["last_seen"] is None or d > date.fromisoformat(rec["last_seen"]):
                    rec["last_seen"] = d.isoformat()

        nct_ids = sorted({m.nct_id for m in group})
        dates = [m.study_start_date for m in group if m.study_start_date]
        strategies = sorted({m.strategy for m in group})

        strengths = {m.match_strength for m in group}
        ambiguous = strengths <= {"literal"}  # only weak literal hits, nothing corroborating
        dev_code_only = strengths <= {"dev_code"}  # only an unnamed-compound-code guess, no ADC signal at all

        candidates.append(
            CandidateAsset(
                candidate_id=f"cand_{root_hash}_{root[:30].replace(' ', '_')}",
                proposed_name=proposed,
                synonyms=synonyms,
                sponsors_over_time=sorted(sponsors.values(), key=lambda r: r["sponsor"]),
                trial_count=len(nct_ids),
                nct_ids=nct_ids,
                first_trial_start_date=min(dates).isoformat() if dates else None,
                last_trial_start_date=max(dates).isoformat() if dates else None,
                strategies=strategies,
                ambiguous=ambiguous,
                dev_code_only=dev_code_only,
            )
        )

    candidates.sort(key=lambda c: c.proposed_name.lower())
    return candidates
