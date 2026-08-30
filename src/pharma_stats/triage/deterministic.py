"""Layer 1: deterministic Gate 1/2 triage — no API calls, ever.

Rules, in order. Rules 0-3 answer is_adc; rules 4-6 answer in_scope. The
two questions are independent: an is_adc=yes candidate can still be
resolved out-of-scope (rules 4-5 apply "whether or not it's even an ADC" —
this project is solid-tumours/industry-only regardless of molecule type),
a candidate whose is_adc rule doesn't fire can still be resolved
out-of-scope on its own via rules 4-5, and — new — a candidate can get a
positive in_scope=yes even before is_adc is known (rule 6), since scope
eligibility (sponsor/tumour-type/date) doesn't depend on molecule
identity. What's never produced here is a terminal is_adc=yes-alone
record — that combination has no representation in the gold schema (see
evaluate()'s docstring).

Reuses discovery/patterns.py's SUFFIX_TERMS/is_denylisted (the same
vocabulary discovery itself uses to find candidates), discovery.candidates
.load_seed_assets/tests/fixtures/known_adcs.txt (the same curated,
human-verified name lists used elsewhere in the project — see rule 0's
docstring), and labelling/trial_scope.py's sponsor-class-override-aware
helpers (the same MeSH/sponsor machinery already driving the review
screen's scope hints) — nothing here is a new, competing vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pharma_stats.config import REPO_ROOT
from pharma_stats.discovery.candidates import load_seed_assets
from pharma_stats.discovery.patterns import SUFFIX_TERMS, is_denylisted
from pharma_stats.labelling import trial_scope as ts

# Target+modality "class label" with no real molecule identifier at all —
# "Her3-ADC", "Cohort 18 (Nectin-4 ADC)". A short alnum/hyphenated token
# immediately adjacent to the specific token "ADC". Treated as is_adc=yes
# (the name itself says it's an ADC) but tagged distinctly from a real
# INN-suffix ID, since asset identity for these is inherently ambiguous —
# a human/later normalisation pass, not this triage, resolves which real
# compound it is.
#
# Deliberately "ADC" only, NOT "conjugate": bare "conjugate" reproduces a
# known false-positive trap discovery/patterns.py's LITERAL_TERMS already
# documents (fires on polymer-drug conjugates, siRNA conjugates, vaccine
# conjugates — none of them antibody-drug conjugates). Confirmed against
# real data: "Etirinotecan Pegol ... Polymer Conjugate" (a PEGylated
# irinotecan, no antibody at all) and "TLR9 Agonist/STAT3 siRNA Conjugate"
# both matched, and are not ADCs, when "conjugate" was included here.
_GENERIC_CLASS_LABEL_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9\-]{1,20}[\s-]*ADC\b", re.IGNORECASE,
)

KNOWN_ADCS_PATH = REPO_ROOT / "tests" / "fixtures" / "known_adcs.txt"
IN_SCOPE_MIN_START_DATE = "2012-01-01"  # CLAUDE.md's locked date range floor


@lru_cache(maxsize=1)
def known_adc_names() -> frozenset:
    """Exact-match name/synonym set from two curated, human-trusted
    sources: discovery/seed_assets.json (the seed list discovery itself
    expands from) and tests/fixtures/known_adcs.txt (an independently
    sourced recall-probe fixture — see its own header for provenance).
    Both are hand-reviewed, not inferred, so an exact hit here is treated
    as a real answer, no API call needed. Deliberately does NOT imply
    in_scope — known_adcs.txt intentionally includes haematologic and
    discontinued assets (over-inclusion for recall testing); this rule
    only ever answers is_adc, never scope. Cached for the life of the
    process — these files don't change mid-run."""
    names: set[str] = set()
    for seed in load_seed_assets():
        names.add(seed["name"].strip().lower())
        for syn in seed.get("synonyms") or []:
            names.add(syn.strip().lower())
    if KNOWN_ADCS_PATH.exists():
        for line in KNOWN_ADCS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.split("|"):
                part = part.strip().lower()
                if part:
                    names.add(part)
    return frozenset(names)


@dataclass
class Layer1Result:
    is_adc: Optional[str]      # "yes" / "no" / None
    in_scope: Optional[str]    # "yes" / "no" / None
    scope_reason: Optional[str]
    rule: str
    # True iff this maps to a valid terminal gold record under the current
    # schema (is_adc=no at gate 1, or in_scope=no at gate 2 — see
    # labelling/store.validate_label_payload). False for is_adc=yes +
    # in_scope=yes, and for is_adc=None + in_scope=yes: real, correct
    # answers, but the gold schema has no terminal record for in_scope=yes
    # — gate 2 with in_scope=yes must proceed to gate 3, which stays a
    # human's job. Reported as resolved, just not written anywhere yet.
    committable: bool


def _all_names(program: dict) -> list[str]:
    return [program.get("proposed_name") or ""] + list(program.get("synonyms") or [])


def _known_list_hit(names: list[str]) -> Optional[str]:
    known = known_adc_names()
    for name in names:
        if name and name.strip().lower() in known:
            return name
    return None


def _inn_suffix_hit(names: list[str]) -> Optional[str]:
    for name in names:
        lowered = name.lower()
        for term in SUFFIX_TERMS:
            if term in lowered:
                return term
    return None


def _is_generic_class_label(names: list[str]) -> bool:
    return any(_GENERIC_CLASS_LABEL_RE.search(name) for name in names)


def evaluate_is_adc(program: dict) -> tuple[Optional[str], Optional[str]]:
    """(is_adc, rule) from rules 0-3 (the molecule-identity rules) only.
    (None, None) if nothing fires — Layer 2/3 or the human queue decides."""
    names = _all_names(program)

    known_hit = _known_list_hit(names)
    if known_hit:
        return "yes", f"layer1_known_list:{known_hit}"

    suffix = _inn_suffix_hit(names)
    if suffix:
        return "yes", f"layer1_inn_suffix:{suffix}"

    if any(is_denylisted(n) for n in names if n):
        return "no", "layer1_denylist"

    if _is_generic_class_label(names):
        return "yes", "layer1_generic_class_label"

    return None, None


def evaluate_in_scope_rejection(program: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """(in_scope, scope_reason, rule) from rules 4-5 — the only two
    deterministic scope-REJECTION rules. Only ever ("no", reason, rule) or
    (None, None, None)."""
    sponsors = ts.apply_sponsor_class_overrides(program.get("sponsors_over_time") or [])
    if ts.is_all_sponsors_non_industry(sponsors):
        return "no", "non_industry", "layer1_sponsor_class_other"

    scope_values = list((program.get("trial_scope") or {}).values())
    if ts.classify_asset(scope_values) == "heme_only":
        return "no", "heme_only", "layer1_mesh_heme_only"

    return None, None, None


def evaluate_in_scope_positive(program: dict) -> Optional[str]:
    """Rule 6: a positive in_scope=yes verdict — sponsor class
    confidently INDUSTRY (every sponsor this asset has ever had, post
    sponsor_class_overrides.json), every trial MeSH-classifies solid (not
    heme, not ambiguous, not non_oncology), and the earliest trial started
    on or after 2012 (CLAUDE.md's locked date range). A REAL rule, not
    "no rejection rule fired" — deliberately independent of is_adc, since
    scope eligibility doesn't depend on molecule identity: this can (and
    is meant to) fire even while is_adc is still pending Layer 2/3.
    Returns "layer1_positive_in_scope" or None (doesn't meet the bar —
    falls back to elimination in evaluate() if is_adc=yes is separately
    known and no rejection fired either)."""
    sponsors = ts.apply_sponsor_class_overrides(program.get("sponsors_over_time") or [])
    if not ts.is_all_sponsors_industry(sponsors):
        return None

    scope_values = list((program.get("trial_scope") or {}).values())
    if not scope_values or not all(v == "solid" for v in scope_values):
        return None

    starts = [t.get("start_date") for t in program.get("trials") or [] if t.get("start_date")]
    if not starts or min(starts) < IN_SCOPE_MIN_START_DATE:
        return None

    return "layer1_positive_in_scope"


def evaluate(program: dict) -> Optional[Layer1Result]:
    """Full Layer 1 pass for one candidate, combining is_adc + in_scope.

    Returns None only when NOTHING resolved at all. Every other case
    returns a result, including is_adc=None + in_scope="yes" (rule 6 can
    fire independently of molecule identity) and is_adc="yes" +
    in_scope="yes" (rule 6 if it fires; otherwise by elimination — no
    rejection rule fired either, still a correct answer, just without a
    named positive rule behind it). Callers that only care about what can
    be written to gold today should filter on .committable — see
    Layer1Result's docstring for exactly which combinations qualify."""
    is_adc, is_adc_rule = evaluate_is_adc(program)

    if is_adc == "no":
        return Layer1Result(
            is_adc="no", in_scope=None, scope_reason=None, rule=is_adc_rule, committable=True,
        )

    reject_scope, scope_reason, reject_rule = evaluate_in_scope_rejection(program)
    positive_rule = evaluate_in_scope_positive(program)

    if is_adc == "yes":
        if reject_scope == "no":
            return Layer1Result(
                is_adc="yes", in_scope="no", scope_reason=scope_reason,
                rule=f"{is_adc_rule}+{reject_rule}", committable=True,
            )
        if positive_rule:
            return Layer1Result(
                is_adc="yes", in_scope="yes", scope_reason=None,
                rule=f"{is_adc_rule}+{positive_rule}", committable=False,
            )
        return Layer1Result(
            is_adc="yes", in_scope="yes", scope_reason=None,
            rule=f"{is_adc_rule}+layer1_elimination", committable=False,
        )

    # is_adc undetermined by rules 0-3: scope can still resolve on its own,
    # independent of molecule identity — either direction.
    if reject_scope == "no":
        return Layer1Result(
            is_adc=None, in_scope="no", scope_reason=scope_reason, rule=reject_rule, committable=True,
        )
    if positive_rule:
        return Layer1Result(
            is_adc=None, in_scope="yes", scope_reason=None, rule=positive_rule, committable=False,
        )

    return None


def summarize(programs: list[dict]) -> dict:
    """Per-outcome counts across a whole candidate set — the "how many of
    the N are resolved by Layer 1" report. resolved = anything evaluate()
    returned a result for (a real answer, per the user's own definition
    of "resolved"); committable = the subset writable to gold today."""
    counts = {
        "total": len(programs),
        "is_adc_no": 0,
        "is_adc_yes_in_scope_no": 0,
        "is_adc_yes_in_scope_yes_not_committable": 0,
        "in_scope_yes_is_adc_pending_not_committable": 0,
        "in_scope_no_only": 0,
        "unresolved": 0,
    }
    by_rule: dict[str, int] = {}
    for p in programs:
        result = evaluate(p)
        if result is None:
            counts["unresolved"] += 1
            continue
        by_rule[result.rule] = by_rule.get(result.rule, 0) + 1
        if result.is_adc == "no":
            counts["is_adc_no"] += 1
        elif result.is_adc == "yes" and result.in_scope == "no":
            counts["is_adc_yes_in_scope_no"] += 1
        elif result.is_adc == "yes" and result.in_scope == "yes":
            counts["is_adc_yes_in_scope_yes_not_committable"] += 1
        elif result.is_adc is None and result.in_scope == "yes":
            counts["in_scope_yes_is_adc_pending_not_committable"] += 1
        else:
            counts["in_scope_no_only"] += 1
    counts["resolved"] = counts["total"] - counts["unresolved"]
    counts["resolved_rate"] = counts["resolved"] / counts["total"] if counts["total"] else 0.0
    counts["committable"] = (
        counts["is_adc_no"] + counts["is_adc_yes_in_scope_no"] + counts["in_scope_no_only"]
    )
    counts["committable_rate"] = counts["committable"] / counts["total"] if counts["total"] else 0.0
    counts["by_rule"] = dict(sorted(by_rule.items(), key=lambda kv: -kv[1]))
    return counts
