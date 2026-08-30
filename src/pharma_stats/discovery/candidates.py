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
    match_term: Optional[str] = None  # the specific SUFFIX_TERMS/LITERAL_TERMS entry (or seed
    # name) that matched — None for a bare dev_code guess, which carries no name-pattern evidence


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
                    match_term=hit[1],
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
                        match_term=term,  # the specific seed name/synonym this query searched for
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
                    match_term=pattern_hit[1] if pattern_hit else None,
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
    # The single weakest piece of discovery evidence behind this candidate
    # (see _primary_evidence) — the one most implicated if a human review
    # later finds it isn't really an ADC. Used to group Gate 1 rejections
    # by matching pattern so patterns.py tuning has a real signal instead
    # of "some rejections happened".
    discovery_strategy: Optional[str] = None
    match_strength: Optional[str] = None
    matched_term: Optional[str] = None
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


# Class-descriptor phrases (from patterns.LITERAL_TERMS) name a *modality*,
# not a specific compound — "ADC" is not a synonym for any one drug. Left
# in the union-find identity graph, a bare "ADC" (or "immunoconjugate",
# etc.) that two unrelated trials both happen to list as an otherName acts
# as a hub node: since union-find keys are global strings, not scoped to
# one mention, both trials' real compounds get transitively merged through
# that shared node. Diagnosed 2026-08-25: one such hub ("adc") merged 1,056
# trials spanning dozens of genuinely distinct real ADCs — plus assorted
# unrelated non-ADC dev-code compounds — into a single candidate wrongly
# labelled "Enfortumab Vedotin (EV)". These terms may still appear in a
# candidate's raw synonym list for display; they must never drive identity.
_GENERIC_CLUSTER_TERMS = {t.lower() for t in LITERAL_TERMS}


def _is_generic_descriptor(normalized_key: str) -> bool:
    return normalized_key in _GENERIC_CLUSTER_TERMS


def _is_too_short_to_be_specific(normalized_key: str) -> bool:
    """Short, purely-alphabetic shorthand ("BV", "SG", "TE", "EV", "Pola")
    is common informal ADC-literature shorthand, but the same 2-4 letter
    code gets reused by unrelated groups for *different* compounds far
    more often than a full generic name collides. Diagnosed 2026-08-25:
    after excluding generic descriptors (above), a second supercluster
    formed via exactly these short codes acting as the same kind of
    transitive hub across dozens of unrelated real ADCs. A real
    compound's cluster is still reachable through its full generic name
    or a digit-bearing dev code carried in the same mention, so this only
    removes an untrustworthy shortcut, not real identity signal.
    """
    return len(normalized_key) <= 4 and normalized_key.isalpha()


def _is_untrusted_identity_key(normalized_key: str) -> bool:
    """A third hub mechanism, diagnosed 2026-08-25 by tracing the actual
    union chain that bridged Kadcyla to Adcetris: kadcyla ~ trastuzumab
    emtansine ~ trastuzumab deruxtecan ~ sacituzumab govitecan ~
    pembrolizumab ~ adcetris. pembrolizumab is a real, specific, correctly
    named drug — but it is not an ADC, and it is co-listed as an otherName
    across dozens of unrelated ADC combination trials, so it links all of
    its combo-partners to each other. is_denylisted() (patterns.py) was
    already built for exactly this ("obviously not an ADC") but was only
    ever applied to strategy 3 acceptance, not to clustering identity for
    any strategy — a known-non-ADC name in a mention's otherNames still
    got treated as a synonym of whatever real ADC it was combined with.
    """
    return (
        _is_generic_descriptor(normalized_key)
        or _is_too_short_to_be_specific(normalized_key)
        or is_denylisted(normalized_key)
    )


def _looks_specific(key: str) -> bool:
    """True if `key` looks like it names one particular compound (an
    INN/USAN suffix hit, or dev-code shaped) rather than a generic
    target/modality descriptor (e.g. "trop2 adc", "her2 adc") that many
    unrelated real compounds could legitimately share. Used to gate which
    multi-word keys are trusted as substring-merge anchors — anchoring on
    a generic descriptor would transitively fuse distinct assets that
    merely target the same antigen or share a modality, the same failure
    mode as the bare "adc" hub above, one level up."""
    hit = matches_pattern(key)
    if hit and hit[0] == "suffix":
        return True
    return looks_like_dev_code(key.replace(" ", "").replace(",", ""))


def _suffixes_in(key: str) -> set:
    return {term for term in SUFFIX_TERMS if term in key}


_COMBO_SEPARATOR_RE = re.compile(r"\s*(?:/|\+|,|&|\band\b|\bplus\b|\bwith\b)\s*")


def _names_multiple_specific_compounds(key: str) -> bool:
    """True if `key` splits (on /, +, ',', '&', "and", "plus", "with")
    into 2+ segments that each independently look like a specific
    compound (see _looks_specific) — i.e. this string names a
    *combination* of two-or-more real, distinct assets rather than being
    one asset's alternate name or a combo-regimen with one active drug
    and inert backbone agents (dexamethasone, ixazomib, ... — those
    segments don't look "specific" on their own, so a regimen like
    "belantamab mafodotin, dexamethasone, ixazomib" still passes as a
    single asset here)."""
    segments = [s for s in _COMBO_SEPARATOR_RE.split(key) if s.strip()]
    if len(segments) < 2:
        return False
    return sum(1 for s in segments if _looks_specific(s.strip())) >= 2


def _merge_substring_clusters(uf: UnionFind, all_keys: set[str]) -> None:
    """Second clustering pass: many sponsors put a whole combo-regimen or
    arm description in the intervention-name field instead of a bare drug
    name (e.g. "Arm A: Belantamab Mafodotin", "Belantamab mafodotin,
    dexamethasone, ixazomib, pomalidomide"). Exact-string / otherNames
    union alone leaves these as their own singleton clusters even though
    they're clearly the same asset. This merges any key into a
    multi-word "anchor" key it whole-word-contains — deterministic
    substring containment, not fuzzy/probabilistic matching. Anchors are
    restricted to keys that look like they name one specific compound
    (see _looks_specific) so a generic descriptor phrase can't drive a
    merge of unrelated assets.

    Genuine two-ADC combination arms are a fourth failure mode, diagnosed
    2026-08-25 by tracing the chain that bridged Kadcyla to Trodelvy:
    a trial literally named "Sacituzumab Govitecan / Trastuzumab
    Deruxtecan" whole-word-contains BOTH real anchors, so both got unioned
    into it — fusing two genuinely distinct, unrelated real ADCs. This
    case is real (per the user: combination trials involving two ADCs
    need many-to-many trial<->asset modelling, which doesn't exist yet in
    this candidate-clustering step) and must NOT be resolved by merging.
    So a key that itself names 2+ specific compounds is excluded from
    both sides of this pass entirely — it can't anchor a merge, and
    nothing merges into it — leaving it as its own small unclustered
    candidate for human adjudication rather than silently fused into
    either real asset (or, worse, transitively bridging them together).
    """
    keys_list = [k for k in sorted(all_keys) if not _names_multiple_specific_compounds(k)]
    anchors = [k for k in keys_list if len(k.split()) >= 2 and _looks_specific(k)]
    for anchor in anchors:
        anchor_suffixes = _suffixes_in(anchor)
        boundary_re = None
        for k in keys_list:
            if k == anchor or anchor not in k:
                continue
            if _suffixes_in(k) - anchor_suffixes:
                continue  # k names a different suffix-bearing compound too — a combo, not a synonym
            if boundary_re is None:
                boundary_re = re.compile(r"(?<!\w)" + re.escape(anchor) + r"(?!\w)")
            if boundary_re.search(k):
                uf.union(anchor, k)


def _identity_keys(m: "Mention") -> list[str]:
    """Normalised names usable as clustering identity — excludes generic
    modality/class descriptors and untrustworthy short abbreviations
    (see _is_untrusted_identity_key)."""
    keys = sorted({_normalize(n) for n in [m.intervention_name] + m.other_names if n})
    return [k for k in keys if not _is_untrusted_identity_key(k)]


def _mention_spans_multiple_compounds(keys: list[str]) -> bool:
    """True if this one mention's own keys carry 2+ *distinct* suffix
    signatures — i.e. it names more than one real, specific compound
    rather than several aliases of one. Diagnosed 2026-08-25 by tracing
    the chain that (still) bridged Kadcyla to Enhertu after the
    denylist/parenthetical fix above: a comparator-arm mention
    (intervention "Trastuzumab (Herceptin)", otherNames ["Trastuzumab
    Emtansine", "Trastuzumab Deruxtecan"]) has its bare-antibody name
    correctly excluded by is_denylisted now, but the *other two* keys are
    both real, valid, correctly-identified — and correctly DIFFERENT —
    compounds, sitting right next to each other in the same mention's own
    key list. The base per-mention union step would fuse them directly
    regardless of anything the substring-merge pass does. There's no
    reliable way to tell, from text alone, which of this mention's
    suffix-less keys (dev codes, brand names) belongs to which of the 2+
    named compounds — so when this fires, the mention contributes no
    union information at all; each real compound still clusters
    correctly through its other, single-compound mentions elsewhere.

    Counts distinct suffixes seen across ALL of the mention's keys taken
    together, not per-key: a single combo-string key like "Sacituzumab
    Govitecan / Trastuzumab Deruxtecan" carries two suffixes by itself
    and must trip this just as surely as two separate single-suffix keys
    would. (A per-key version missed exactly this, letting that one
    combo string bridge two real ADCs through a shared brand-name
    neighbour that itself carries no suffix.)
    """
    all_suffixes_seen: set = set()
    for k in keys:
        all_suffixes_seen |= _suffixes_in(k)
    return len(all_suffixes_seen) >= 2


# Confidence ranking, weakest first: a dev_code guess carries zero
# name-pattern evidence at all; a literal term ("conjugate", "adc") is
# explicitly weak per patterns.py; seed_expansion matched a known real
# ADC's own name; suffix is the curated INN naming convention — the
# strongest signal this pipeline has. The WEAKEST entry present in a
# candidate's mentions is the one to blame if it turns out not to be a
# real ADC, since anything with a seed/suffix hit is essentially never a
# false positive.
_STRENGTH_RANK = {"dev_code": 0, "literal": 1, "seed": 2, "suffix": 3}


def _primary_evidence(group: list[Mention]) -> tuple[str, Optional[str], Optional[str]]:
    """(strategy, match_strength, matched_term) for the weakest evidence in
    the cluster — see _STRENGTH_RANK."""
    best = min(
        group,
        key=lambda m: (_STRENGTH_RANK.get(m.match_strength, 0), m.strategy, m.match_term or ""),
    )
    return best.strategy, best.match_strength, best.match_term


def build_candidate_table(mentions: list[Mention]) -> list[CandidateAsset]:
    uf = UnionFind()
    all_keys: set[str] = set()
    for m in mentions:
        keys = _identity_keys(m)
        if not _mention_spans_multiple_compounds(keys):
            for a, b in zip(keys, keys[1:]):
                uf.union(a, b)
        for k in keys:
            uf.find(k)  # ensure registered even if singleton
        all_keys.update(keys)

    _merge_substring_clusters(uf, all_keys)

    clusters: dict[str, list[Mention]] = {}
    for m in mentions:
        keys = _identity_keys(m)
        if not keys:
            continue  # nothing but a generic descriptor (e.g. bare "ADC") — not an identifiable compound
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
        discovery_strategy, match_strength, matched_term = _primary_evidence(group)

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
                discovery_strategy=discovery_strategy,
                match_strength=match_strength,
                matched_term=matched_term,
            )
        )

    candidates.sort(key=lambda c: c.proposed_name.lower())
    return candidates


def genuine_combo_trial_ids(candidates: list[dict]) -> set[str]:
    """Trial ids shared by 2+ candidates that are ALL independently
    verified (found by pattern_match or seed_expansion, not merely a
    weak literal-term match) — the signature of a real trial testing
    multiple distinct assets together, not a clustering error (see the
    _is_untrusted_identity_key family above for that failure mode).

    Used by both the audit's universe stage and the labelling app's
    provisional program builder — the two need the exact same answer to
    "is this shared trial real or noise", so this lives in one place.
    Takes plain dicts (candidate_id/proposed_name/nct_ids/strategies/
    ambiguous) rather than CandidateAsset so DB-row callers don't need to
    round-trip through the dataclass.
    """
    nct_to_candidates: dict[str, list[dict]] = {}
    for c in candidates:
        for nct_id in c.get("nct_ids") or []:
            nct_to_candidates.setdefault(nct_id, []).append(c)

    def is_verified(c: dict) -> bool:
        return bool(set(c.get("strategies") or []) & {"pattern_match", "seed_expansion"}) and not c.get("ambiguous")

    return {
        nct_id for nct_id, cs in nct_to_candidates.items()
        if len(cs) > 1 and all(is_verified(c) for c in cs)
    }
