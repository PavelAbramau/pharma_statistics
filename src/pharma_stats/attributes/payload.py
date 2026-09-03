"""B0: payload chemotype derivation from the INN/USAN suffix.

Reuses discovery.patterns.SUFFIX_TERMS (already a reviewed, hand-built
dictionary — see that module's docstring) and adds only the
suffix -> CLAUDE.md-controlled-vocabulary mapping, since patterns.py's
own inline comments name the payload family in prose, not as a value any
code can read. Extend SUFFIX_TO_CHEMOTYPE by hand alongside
patterns.SUFFIX_TERMS whenever a new suffix is added there.

A candidate with no suffix hit is "undisclosed", never guessed — this
covers every bare dev code (e.g. "XMT-1592"), which is most of the
Layer 2/3-confirmed-but-not-yet-named population; naming happens later
in a compound's life, well after this project would need to classify it.
"""
from __future__ import annotations

from typing import Optional

from pharma_stats.discovery import patterns

# CLAUDE.md's fixed payload chemotype vocabulary. One entry per
# patterns.SUFFIX_TERMS suffix — "other" for a real, known payload class
# that isn't one of CLAUDE.md's named categories (e.g. NAMPT inhibitors),
# never for "I'm not sure".
SUFFIX_TO_CHEMOTYPE: dict[str, str] = {
    "vedotin": "auristatin",            # MMAE
    "mafodotin": "auristatin",          # MMAF
    "deruxtecan": "camptothecin_topo1",  # DXd
    "govitecan": "camptothecin_topo1",  # SN-38
    "tirumotecan": "camptothecin_topo1",  # DXd-class
    "rezetecan": "camptothecin_topo1",   # DXd-class
    "sesutecan": "camptothecin_topo1",   # topo1-class
    "pamirtecan": "camptothecin_topo1",  # topo1-class
    "emtansine": "maytansinoid",        # DM1
    "soravtansine": "maytansinoid",     # DM4
    "ravtansine": "maytansinoid",       # DM4
    "tesirine": "pbd_dimer",
    "ozogamicin": "calicheamicin",
    "duocarmazine": "duocarmycin",
    "rilsodotin": "other",              # NAMPTi payload — not a CLAUDE.md category
}

CHEMOTYPE_VALUES = (
    "camptothecin_topo1", "auristatin", "maytansinoid", "calicheamicin",
    "pbd_dimer", "duocarmycin", "amanitin", "tubulysin", "eribulin",
    "immune_agonist", "radioisotope", "other", "undisclosed",
)


def derive_payload_chemotype(name: Optional[str], synonyms: Optional[list] = None) -> str:
    """"undisclosed" whenever no suffix fires — including every bare dev
    code — never a guess from name shape or trial text."""
    for candidate in [name or ""] + list(synonyms or []):
        match = patterns.matches_pattern(candidate)
        if match and match[0] == "suffix":
            return SUFFIX_TO_CHEMOTYPE.get(match[1], "other")
    return "undisclosed"
