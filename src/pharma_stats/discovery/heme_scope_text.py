"""Text-based heme / tumour-system proxy for the matrix-only universe
extension (discovery/matrix_extension.py, docs/decisions/0008) — NOT the
MeSH-ID pipeline attributes/tumour_system.py uses for the main opportunity
matrix. Keyword matching against raw CT.gov condition strings (the
`conditionsModule.conditions` list the basic search API already returns),
never MeSH IDs — deliberately lower confidence, since the extension cohort
never gets a current-state fetch (conditionBrowseModule requires one; see
discovery/mesh_categories.py's module docstring, and matrix_extension.py's
for why that cost isn't paid here).

The heme side reuses `labelling.trial_scope.text_hint_category` — this
project's own reviewed, word-boundary-safe text-hint keyword set
(`discovery.mesh_categories.HEME_TEXT_HINT_KEYWORDS`), already used
elsewhere as a sort-priority (never authoritative) heme signal for
exactly the same reason it's safe to lean on harder here: this cohort is
already explicitly lower-confidence by design (docs/decisions/0008). The
solid-system side has no existing text-based equivalent to reuse (the
main pipeline only ever resolves tumour system from MeSH), so those 8
keyword lists are hand-curated here, same convention as
discovery/patterns.py and mesh_categories.py — extend by hand as new
condition phrasings are noticed, never inferred from a model.

Every hit routes to one of attributes.tumour_system.TUMOUR_SYSTEM_VALUES
or "heme"; anything else is None (excluded, never bucketed into a
catch-all) — same discipline as tumour_system.py.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from pharma_stats.labelling.trial_scope import text_hint_category

_KEYWORDS_BY_SYSTEM: dict[str, tuple[str, ...]] = {
    "breast": ("breast",),
    "gi_hepatobiliary": (
        "colorectal", "colon cancer", "colonic", "rectal", "gastric",
        "stomach", "esophageal", "oesophageal", "pancreatic",
        "hepatocellular", "liver cancer", "cholangiocarcinoma", "biliary",
        "gastrointestinal", "gastroesophageal",
    ),
    "thoracic_lung": ("lung cancer", "nsclc", "sclc", "mesothelioma", "bronchogenic", "pulmonary"),
    "genitourinary": (
        "renal cell", "kidney cancer", "bladder", "urothelial", "prostate",
        "testicular", "penile",
    ),
    "gynecologic": (
        "ovarian", "cervical", "endometrial", "uterine", "fallopian",
        "vulvar", "vaginal",
    ),
    "skin_ocular_melanoma": ("melanoma", "uveal", "merkel cell", "cutaneous"),
    "sarcoma_soft_tissue": ("sarcoma", "gist", "soft tissue"),
    "endocrine_neuroendocrine_germcell": (
        "neuroendocrine", "thyroid cancer", "germ cell", "carcinoid",
        "adrenocortical",
    ),
}


def classify_condition_text(conditions: list[str]) -> Optional[str]:
    """"heme" or one of attributes.tumour_system.TUMOUR_SYSTEM_VALUES, by
    majority vote of keyword hits across ALL of a program's raw condition
    strings (a basket trial mixing heme and solid conditions resolves to
    whichever has more hits, same tie-break spirit as
    tumour_system.program_tumour_system's Counter.most_common) — None if
    nothing specific matches (generic "solid tumors"/"neoplasms" text,
    or no conditions at all). Each condition string votes once, for
    whichever bucket it hits first (heme checked first, via
    text_hint_category)."""
    votes: Counter = Counter()
    for raw in conditions or []:
        if text_hint_category([raw]) == "heme":
            votes["heme"] += 1
            continue
        text = raw.lower()
        for system, keywords in _KEYWORDS_BY_SYSTEM.items():
            if any(kw in text for kw in keywords):
                votes[system] += 1
                break
    if not votes:
        return None
    return votes.most_common(1)[0][0]
