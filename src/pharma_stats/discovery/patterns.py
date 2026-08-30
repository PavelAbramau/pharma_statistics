"""Text patterns used to recognise a CT.gov intervention as an ADC candidate.

Two confidence tiers:

- "suffix" (strong): the USAN/INN naming convention for the antibody+linker
  +payload combination puts a fixed suffix on the generic name (e.g.
  "trastuzumab deruxtecan", "enfortumab vedotin"). A suffix hit is a strong
  signal — false positives are rare.
- "literal" (weak): the intervention name/other-names literally contains a
  descriptive phrase like "antibody-drug conjugate" or "conjugate". These
  fire on unrelated things too (vaccine conjugates, chemistry terms), so a
  literal-only match with no corroborating suffix or seed hit is flagged
  ambiguous for human review rather than trusted outright.

This list is deliberately not exhaustive of every ADC ever named — recall
comes from combining this with seed-list and sponsor expansion, plus the
human audit pass. Extend SUFFIX_TERMS/LITERAL_TERMS by hand as new naming
conventions are noticed.
"""
import re
from typing import Optional

# INN/USAN suffixes for antibody-drug conjugates, by payload family.
# The user's original list: vedotin, deruxtecan, govitecan, emtansine, tesirine.
# Extended with other established ADC suffixes for recall.
SUFFIX_TERMS = [
    "vedotin",       # MMAE (auristatin)
    "deruxtecan",    # DXd (camptothecin/topo1)
    "govitecan",     # SN-38 (camptothecin/topo1)
    "emtansine",     # DM1 (maytansinoid)
    "tesirine",      # PBD dimer
    "mafodotin",     # MMAF (auristatin)
    "ozogamicin",    # calicheamicin
    "soravtansine",  # DM4 (maytansinoid)
    "ravtansine",    # DM4 (maytansinoid) — catches "soravtansine" too, kept separate for recall
    "duocarmazine",  # duocarmycin
    "tirumotecan",   # DXd-class (e.g. sacituzumab tirumotecan / MK-2870)
    # Added 2026-08-26 after a recall-gap audit against an independently
    # sourced known-ADC list found these four missing entirely:
    "rezetecan",     # DXd-class (e.g. trastuzumab rezetecan / SHR-A1811)
    "rilsodotin",    # NAMPTi payload (e.g. upifitamab rilsodotin / XMT-1536)
    "sesutecan",     # topo1-class (e.g. rinatabart sesutecan / GEN1184)
    "pamirtecan",    # topo1-class (e.g. trastuzumab pamirtecan / BNT323/DB-1303)
]

# Descriptive terms that indicate "this is an ADC" without a fixed suffix.
# Ambiguous on their own (see module docstring); used as weak signal.
LITERAL_TERMS = [
    "antibody-drug conjugate",
    "antibody drug conjugate",
    "immunoconjugate",
    "adc",
    "conjugate",
]

# Intervention types worth inspecting at all. Everything else (PROCEDURE,
# DEVICE, RADIATION, DIETARY_SUPPLEMENT, BEHAVIORAL, ...) is not a molecule.
CANDIDATE_INTERVENTION_TYPES = {"DRUG", "BIOLOGICAL"}

# Common combo-partner / backbone oncology drugs that are emphatically not
# ADCs. Used only to cut obvious noise in sponsor-based expansion (strategy
# 3), where we'd otherwise flag every checkpoint inhibitor and chemo
# backbone a sponsor has ever tested. Not applied to strategies 1/2.
NON_ADC_DENYLIST = {
    "placebo", "best supportive care",
    "pembrolizumab", "nivolumab", "ipilimumab", "atezolizumab", "durvalumab",
    "avelumab", "cemiplimab", "tremelimumab",
    "paclitaxel", "docetaxel", "nab-paclitaxel", "carboplatin", "cisplatin",
    "oxaliplatin", "gemcitabine", "doxorubicin", "cyclophosphamide",
    "capecitabine", "5-fluorouracil", "5-fu", "fluorouracil", "irinotecan",
    "etoposide", "vinorelbine", "eribulin",
    "olaparib", "niraparib", "rucaparib", "talazoparib",
    "bevacizumab", "ramucirumab",
    "trastuzumab", "pertuzumab", "rituximab", "cetuximab", "panitumumab",
    "letrozole", "anastrozole", "exemestane", "fulvestrant", "tamoxifen",
    "palbociclib", "ribociclib", "abemaciclib",
    "dexamethasone", "prednisone", "leucovorin", "folinic acid",
}

# Looks like an unnamed development-stage compound code, e.g. "XMT-1592",
# "STRO-002", "SKB264", "ABBV-011". Deliberately loose — this is a recall
# aid for strategy 3, not a precision filter; the audit report is where
# false positives get pruned by a human.
_DEV_CODE_RE = re.compile(r"^[A-Za-z]{2,8}[-\s]?\d{2,6}[A-Za-z]{0,3}$")

# Common trailing formulation/dosage-form qualifiers sponsors append to an
# otherwise bare dev code (e.g. "SKB410 for injection"). Diagnosed
# 2026-08-26: this made the recall probe report SKB410 as "missed by
# discovery" when the real cause was that the qualifier alone broke the
# dev-code match, keeping strategy 3 from ever accepting the mention.
_TRAILING_FORMULATION_RE = re.compile(
    r"\s+for\s+(injection|infusion|intravenous\s+use)\s*$", re.IGNORECASE,
)


def looks_like_dev_code(name: str) -> bool:
    stripped = name.strip()
    if _DEV_CODE_RE.match(stripped):
        return True
    return bool(_DEV_CODE_RE.match(_TRAILING_FORMULATION_RE.sub("", stripped)))


def matches_pattern(name: str) -> Optional[tuple[str, str]]:
    """Check one candidate string against the suffix/literal term lists.

    Returns (strength, matched_term) for the strongest match found
    ("suffix" beats "literal"), or None if nothing matches.
    """
    lowered = name.lower()
    for term in SUFFIX_TERMS:
        if term in lowered:
            return ("suffix", term)
    for term in LITERAL_TERMS:
        # word-boundary match for short/ambiguous tokens like "adc"
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return ("literal", term)
    return None


_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def is_denylisted(name: str) -> bool:
    """Exact match against NON_ADC_DENYLIST, also after stripping one
    trailing brand-name parenthetical (e.g. "Trastuzumab (Herceptin)" ->
    "trastuzumab"). Diagnosed 2026-08-25: a comparator arm named exactly
    that way slipped past the plain exact-match check and went on to
    bridge two genuinely distinct real ADCs (Kadcyla and Enhertu) into
    one candidate via clustering identity — see candidates.py."""
    normalized = name.strip().lower()
    if normalized in NON_ADC_DENYLIST:
        return True
    return _TRAILING_PAREN_RE.sub("", normalized).strip() in NON_ADC_DENYLIST
