"""Post-hoc quote grounding and evidence-snippet hygiene for Layer 2.

The model is asked to quote trial text and set from_recall honestly. It
does not always: BAT8010 returned is_adc=yes, from_recall=False, and a
quote that was the study's boilerplate objective sentence — nothing about
modality. This module is the programmatic check the prompt cannot be
trusted to perform.

Also: evidence snippets about a *different* molecule in a combination
arm (SY-5609 answered from an Inavolisib oral-dosing sentence) and
mid-word truncation of briefSummary (Lonsurf → "tipirac").
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pharma_stats.discovery.patterns import LITERAL_TERMS, SUFFIX_TERMS

# Payload / linker / ADC-class tokens that actually speak to modality.
# SUFFIX_TERMS covers the INN payload suffixes; LITERAL_TERMS covers
# "antibody-drug conjugate" / "adc" / "conjugate"; the extras are
# chemotype names a quote might use instead of the INN suffix.
_PAYLOAD_EXTRAS = (
    "exatecan", "mmae", "mmaf", "dxd", "sn-38", "sn38", "pbd",
    "calicheamicin", "auristatin", "maytansine", "maytansinoid",
    "camptothecin", "tubulysin", "duocarmycin", "amanitin", "eribulin",
    "immunoconjugate", "payload", "linker",
)

_YES_PHRASES = tuple(LITERAL_TERMS) + tuple(SUFFIX_TERMS) + _PAYLOAD_EXTRAS

# Route of administration that an ADC does not use. Infusion / IV /
# "solution for injection" is the *typical* ADC presentation — it cannot
# ground a no. Oral / tablet / capsule can.
_NO_ROA_RE = re.compile(
    r"\b(orally|oral|tablet|tablets|capsule|capsules|p\.?o\.?)\b",
    re.IGNORECASE,
)

# Explicit non-ADC modality. "antibody" is included only when the quote
# is not also an ADC/conjugate statement (checked separately).
_NO_MODALITY_RE = re.compile(
    r"\b(monoclonal antibody|\bmabs?\b|bispecific|small[- ]molecule|"
    r"inhibitors?|kinase|vaccine|car-t|cell therap|siRNA|sirna|"
    r"igg[1-4]|checkpoint|protac)\b",
    re.IGNORECASE,
)

_ADC_OR_CONJUGATE_RE = re.compile(
    r"\b(adc|antibody[- ]drug|immunoconjugate|conjugate|payload|linker)\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+\-.]{1,}")


def truncate_at_word(text: str, max_chars: int) -> str:
    """Cut at a word boundary so 'tipiracil' never becomes 'tipirac'."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if text[max_chars].isalnum() and cut[-1].isalnum():
        sp = cut.rsplit(None, 1)
        cut = sp[0] if sp else cut
    return cut.rstrip()


def candidate_name_tokens(name: Optional[str], synonyms: Optional[list] = None) -> list[str]:
    """Tokens worth requiring in a snippet: the proposed name, each
    synonym, and slash/comma-split parts, dropping tiny tokens that
    would match noise (e.g. 'IO')."""
    raw = [name or ""] + list(synonyms or [])
    out: list[str] = []
    seen: set[str] = set()
    for r in raw:
        if not r or not r.strip():
            continue
        pieces = [r.strip()] + [p.strip() for p in re.split(r"[/,;]", r) if p.strip()]
        for p in pieces:
            key = p.lower()
            if key in seen:
                continue
            if len(p) < 4 and not re.search(r"\d", p):
                continue
            seen.add(key)
            out.append(p)
    return out


def snippet_mentions_candidate(snippet: str, name: Optional[str], synonyms: Optional[list] = None) -> bool:
    """True iff the snippet names this candidate (or a synonym), not a
    combination-arm partner. Case-insensitive substring; tokens under
    4 chars that contain a digit (dev codes like 'XL114') still match."""
    if not snippet:
        return False
    lowered = snippet.lower()
    for tok in candidate_name_tokens(name, synonyms):
        if tok.lower() in lowered:
            return True
    return False


def quote_grounds_yes(quote: Optional[str]) -> bool:
    """A yes is text-grounded only if the quote actually mentions an ADC
    modality / payload / linker — not the study's objective boilerplate."""
    if not quote or not str(quote).strip():
        return False
    q = quote.lower()
    for phrase in _YES_PHRASES:
        if phrase.lower() in q:
            return True
    return bool(re.search(r"\badc\b", q))


def quote_grounds_no(quote: Optional[str]) -> bool:
    """A no is text-grounded only if the quote states a non-ADC modality
    or an oral/tablet/capsule route. 'Solution for infusion' is the
    default ADC presentation and is non-probative."""
    if not quote or not str(quote).strip():
        return False
    if _NO_ROA_RE.search(quote):
        return True
    # "not an ADC" / "not an antibody-drug conjugate"
    if re.search(r"\bnot an?\s+(adc|antibody[- ]drug)\b", quote, re.IGNORECASE):
        return True
    if _NO_MODALITY_RE.search(quote):
        # A quote that also says it's an ADC/conjugate is not a no-grounding
        if _ADC_OR_CONJUGATE_RE.search(quote) and not re.search(
            r"\bnot\b", quote, re.IGNORECASE,
        ):
            return False
        return True
    # bare "antibody" without conjugate/ADC — a mAb statement
    if re.search(r"\bantibod(?:y|ies)\b", quote, re.IGNORECASE) and not _ADC_OR_CONJUGATE_RE.search(quote):
        return True
    return False


def matching_small_molecule_or_oral_snippet(text_snippets: Optional[list]) -> Optional[str]:
    """First evidence snippet naming a non-ADC modality or an oral/tablet/
    capsule route — the same signals quote_grounds_no checks a single
    quote against, applied here across a candidate's whole evidence
    bundle to flag likely-reject candidates for queue ordering (see
    labelling/queue_order.py) and to show the reviewer why in the bulk-
    reject panel. None if nothing matches — never used to decide
    anything on its own."""
    for snippet in text_snippets or []:
        if not snippet:
            continue
        if _NO_ROA_RE.search(snippet) or _NO_MODALITY_RE.search(snippet):
            return snippet
    return None


def has_small_molecule_or_oral_signal(text_snippets: Optional[list]) -> bool:
    return matching_small_molecule_or_oral_snippet(text_snippets) is not None


def evidence_source(is_adc: str, from_recall: bool, quote: Optional[str]) -> str:
    """text / recall / no_usable_evidence. Unsure + empty quote + not
    recall is neither text nor recall — the previous default-to-text
    path is what made unsure rows print confidence=text with a blank quote."""
    if from_recall:
        return "recall"
    if quote and str(quote).strip():
        return "text"
    return "no_usable_evidence"


def confidence_label(*, disagreement: bool, votes: list) -> str:
    """unanimous / escalated-and-resolved / escalated-and-split.

    Unanimous is about THIS candidate's round-1 votes, not whether the
    20-candidate group was escalated because a sibling disagreed.
    Escalated-and-resolved: round-1 split, final tally has a unique
    plurality. Escalated-and-split: round-1 split and the final tally
    is still a tie (or empty)."""
    if not disagreement:
        return "unanimous"
    if not votes:
        return "escalated-and-split"
    counts = Counter(votes)
    ranked = counts.most_common(2)
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return "escalated-and-resolved"
    return "escalated-and-split"


def apply_grounding(is_adc: str, from_recall: bool, quote: Optional[str]) -> tuple[bool, bool]:
    """(from_recall, grounding_forced). A yes/no that claimed text
    grounding without a probative quote is forced to from_recall=True
    (and therefore Layer 3) — the quote is kept for the audit trail."""
    if from_recall or is_adc not in ("yes", "no"):
        return from_recall, False
    ok = quote_grounds_yes(quote) if is_adc == "yes" else quote_grounds_no(quote)
    if ok:
        return False, False
    return True, True
