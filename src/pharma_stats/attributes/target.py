"""B0: target antigen derivation, three sources in priority order (per
the user's explicit design):

  1. antibody-stem dictionary — well-known INN antibody names whose
     target is public, settled pharmacology (trastuzumab -> HER2, etc.),
     matched against the candidate's own name/synonyms. Highest
     confidence: these are approved-or-late-stage compounds with a
     documented target, not an inference.
  2. trial text — explicit target mentions in the evidence text
     (brief summary + intervention descriptions) already extracted by
     triage/evidence.py. Requires exactly ONE distinct target to match;
     multiple distinct hits (e.g. a combination-arm trial) means don't
     guess which one is THIS candidate's target.
  3. candidate name/synonyms themselves — for target-named dev codes
     that state the antigen directly (rare, but real).

Every symbol here is HGNC (CLAUDE.md's controlled vocabulary for target).
Unresolved candidates are never guessed — they're returned as (None,
"unresolved") for the caller to route to a review queue, same pattern as
triage's unmapped-value handling.

Both dictionaries are reviewable, hand-built, and deliberately
conservative: a real target left out here surfaces honestly as
"unresolved" (safe), whereas a wrong entry would silently mislabel a
program (unsafe) — so entries were only added for well-documented,
unambiguous antibody/target or target/alias pairs. Extend by hand,
same convention as discovery/patterns.py.
"""
from __future__ import annotations

import re
from typing import Optional

# Antibody INN stem (lowercase, as it appears inside the full generic
# name, e.g. "trastuzumab" inside "trastuzumab deruxtecan") -> HGNC target
# symbol. Comment names the trade/commonly-known compound this was
# verified against.
ANTIBODY_STEM_TO_TARGET: dict[str, str] = {
    "trastuzumab": "ERBB2",       # Herceptin / Kadcyla / Enhertu family — HER2
    "pertuzumab": "ERBB2",        # Perjeta — HER2
    "disitamab": "ERBB2",         # disitamab vedotin / RC48 — HER2
    "zanidatamab": "ERBB2",       # zanidatamab (bispecific) — HER2
    "sacituzumab": "TACSTD2",     # Trodelvy — Trop-2
    "datopotamab": "TACSTD2",     # datopotamab deruxtecan / Dato-DXd — Trop-2
    "enfortumab": "NECTIN4",      # Padcev — Nectin-4
    "tisotumab": "F3",            # Tivdak — Tissue Factor
    "mirvetuximab": "FOLR1",      # Elahere — folate receptor alpha
    "rinatabart": "FOLR1",        # rinatabart sesutecan / GEN1184 — folate receptor alpha
    "inotuzumab": "CD22",         # Besponsa
    "moxetumomab": "CD22",        # moxetumomab pasudotox
    "pinatuzumab": "CD22",        # pinatuzumab vedotin
    "gemtuzumab": "CD33",         # Mylotarg
    "vadastuximab": "CD33",       # vadastuximab talirine / SGN-CD33A
    "brentuximab": "TNFRSF8",     # Adcetris — CD30
    "polatuzumab": "CD79B",       # Polivy
    "iladatuzumab": "CD79B",      # iladatuzumab vedotin
    "loncastuximab": "CD19",      # Zynlonta
    "coltuximab": "CD19",         # coltuximab ravtansine
    "denintuzumab": "CD19",       # denintuzumab mafodotin / SGN-CD19A
    "vorsetuzumab": "CD70",       # vorsetuzumab mafodotin / SGN-75
    "belantamab": "TNFRSF17",     # Blenrep — BCMA
    "naratuximab": "CD37",        # naratuximab emtansine
    "indatuximab": "SDC1",        # indatuximab ravtansine — CD138/syndecan-1
    "telisotuzumab": "MET",       # telisotuzumab vedotin / Teliso-V — c-Met
    "patritumab": "ERBB3",        # patritumab deruxtecan — HER3
    "rovalpituzumab": "DLL3",     # rovalpituzumab tesirine / Rova-T
    "ladiratuzumab": "SLC39A6",   # ladiratuzumab vedotin — LIV-1
    "lifastuzumab": "SLC34A2",    # lifastuzumab vedotin — NaPi2b
    "upifitamab": "SLC34A2",      # upifitamab rilsodotin / UpRi — NaPi2b
    "zilovertamab": "ROR1",       # zilovertamab vedotin
    "mecbotamab": "AXL",          # mecbotamab vedotin
    "vandortuzumab": "STEAP1",    # vandortuzumab vedotin
    "glembatumumab": "GPNMB",     # glembatumumab vedotin / CDX-011
    "cofetuzumab": "PTK7",        # cofetuzumab pelidotin
    "praluzatamab": "ALCAM",      # praluzatamab ravtansine — CD166
    "tusamitamab": "CEACAM5",     # tusamitamab ravtansine
    "vobramitamab": "CD276",      # vobramitamab duocarmazine / MGC018 — B7-H3
}

# HGNC symbol -> case-insensitive alias patterns worth scanning trial
# text / candidate names for. Word-boundary-anchored throughout; symbols
# that collide with common English words (MET, KIT) require a more
# specific phrasing rather than the bare symbol.
_TARGET_ALIASES: dict[str, list[str]] = {
    "ERBB2": [r"her-?2(?:/neu)?", r"human epidermal growth factor receptor[- ]?2", r"\berbb2\b"],
    "ERBB3": [r"\bher-?3\b", r"\berbb3\b"],
    "TACSTD2": [r"\btrop-?2\b", r"\btacstd2\b"],
    "NECTIN4": [r"\bnectin-?4\b"],
    "F3": [r"\btissue factor\b"],
    "FOLR1": [r"folate receptor[- ]alpha", r"\bfr-?alpha\b", r"\bfolr1\b", r"frα"],
    "CD22": [r"\bcd22\b"],
    "CD33": [r"\bcd33\b"],
    "TNFRSF8": [r"\bcd30\b"],
    "CD79B": [r"\bcd79b\b"],
    "CD19": [r"\bcd19\b"],
    "CD70": [r"\bcd70\b"],
    "KIT": [r"\bcd117\b", r"\bc-?kit\b"],
    "SDC1": [r"\bcd138\b", r"syndecan-?1"],
    "TNFRSF17": [r"\bbcma\b"],
    "CD37": [r"\bcd37\b"],
    "MET": [r"\bc-?met\b", r"\bmet receptor\b"],
    "DLL3": [r"\bdll3\b"],
    "MSLN": [r"\bmesothelin\b", r"\bmsln\b"],
    "CEACAM5": [r"\bceacam5\b", r"carcinoembryonic antigen"],
    "CD276": [r"\bb7-?h3\b", r"\bcd276\b"],
    "SLC39A6": [r"\bliv-?1\b", r"\bslc39a6\b"],
    "SLC34A2": [r"\bnapi2b\b", r"\bslc34a2\b"],
    "ROR1": [r"\bror1\b"],
    "AXL": [r"\baxl\b"],
    "STEAP1": [r"\bsteap1\b"],
    "STEAP2": [r"\bsteap2\b"],
    "GPNMB": [r"\bgpnmb\b"],
    "PTK7": [r"\bptk7\b"],
    "ITGB6": [r"integrin beta-?6", r"\bitgb6\b"],
    "ALCAM": [r"\bcd166\b", r"\balcam\b"],
    "GPA33": [r"\bgpa33\b", r"\ba33 antigen\b"],
    "CDH6": [r"\bcdh6\b"],
    "CDH17": [r"\bcdh17\b"],
    "FOLH1": [r"\bpsma\b", r"\bfolh1\b"],
    "CLDN18": [r"claudin[- ]?18\.2", r"\bcldn18\.2\b", r"\bcldn18\b"],
    "EGFR": [r"\begfr\b", r"epidermal growth factor receptor"],
    "MUC16": [r"\bmuc16\b"],
    "MUC1": [r"\bmuc1\b"],
    "SLAMF7": [r"\bcd319\b", r"\bslamf7\b"],
}

_COMPILED_ALIASES: dict[str, list[re.Pattern]] = {
    symbol: [re.compile(p, re.IGNORECASE) for p in patterns]
    for symbol, patterns in _TARGET_ALIASES.items()
}


def _matches_in_text(text: str) -> set[str]:
    hits = set()
    for symbol, patterns in _COMPILED_ALIASES.items():
        if any(p.search(text) for p in patterns):
            hits.add(symbol)
    return hits


def target_from_antibody_stem(name: Optional[str], synonyms: Optional[list] = None) -> Optional[str]:
    for candidate in [name or ""] + list(synonyms or []):
        lowered = candidate.lower()
        for stem, target in ANTIBODY_STEM_TO_TARGET.items():
            if stem in lowered:
                return target
    return None


def target_from_trial_text(text_snippets: Optional[list]) -> Optional[str]:
    """None on zero OR multiple distinct hits — ambiguous is not a
    resolution, same policy as the payload/scope grounding checks."""
    hits: set[str] = set()
    for snippet in text_snippets or []:
        if snippet:
            hits |= _matches_in_text(snippet)
    return hits.pop() if len(hits) == 1 else None


def target_from_name(name: Optional[str], synonyms: Optional[list] = None) -> Optional[str]:
    hits: set[str] = set()
    for candidate in [name or ""] + list(synonyms or []):
        hits |= _matches_in_text(candidate)
    return hits.pop() if len(hits) == 1 else None


def derive_target(
    name: Optional[str], synonyms: Optional[list] = None, text_snippets: Optional[list] = None,
) -> tuple[Optional[str], str]:
    """(hgnc_symbol_or_None, source). source is "antibody_stem",
    "trial_text", "name", or "unresolved" — unresolved means route to a
    review queue, never guess."""
    t = target_from_antibody_stem(name, synonyms)
    if t:
        return t, "antibody_stem"
    t = target_from_trial_text(text_snippets)
    if t:
        return t, "trial_text"
    t = target_from_name(name, synonyms)
    if t:
        return t, "name"
    return None, "unresolved"
