"""Reviewable MeSH-ID -> category dictionary for trial-level scope
classification (heme / solid / non_oncology / generic_basket).

This is the minimum viable version of indication normalisation — a stopgap
until the real OncoTree mapping exists (README.md: "not started"). It will
be EXTENDED by the OncoTree work later, not replaced by it: OncoTree gives
a specific indication code per program; this gives a coarse heme-vs-solid
scope signal per trial, which is all the locked-scope filter ("solid
tumours only") actually needs.

Source of truth: CT.gov's own conditionBrowseModule (derivedSection),
which ties each trial's conditions to NLM MeSH descriptor IDs plus that
descriptor's MeSH-tree ancestors. Classifying by ID (not by string-matching
condition text) is what makes this trustworthy — "Non-Small Cell Lung
Cancer" and "NSCLC" are the same ID regardless of how a sponsor typed it.

IMPORTANT — coverage: conditionBrowseModule only exists on CT.gov's
"current state" study fetch, never on the versioned-history diff bodies
this project fetches for amendment tracking. As of 2026-08-29, only 26 of
~2900 trial IDs in asset_candidates have ever had a current-state fetch,
so this dictionary is seeded from exactly the ~101 MeSH IDs actually
observed across those 26 trials (see scripts/build_candidate_universe.py
history — every ID below was read from a real CT.gov response, not
guessed from general MeSH knowledge, to avoid silently mis-categorising on
a mistyped ID). Trials with MeSH data outside this set, or with no
conditionBrowseModule at all, classify as "ambiguous" (see
labelling/trial_scope.py) rather than being guessed at. Extend this file
by hand as more current-state fetches land more MeSH IDs — that is the
review step this whole mechanism exists to keep honest.

Categories:
- "heme"           — Hemic and Lymphatic Diseases branch: leukemia,
                      lymphoma, myeloma/immunoproliferative disorders.
- "solid"          — any specific solid-organ/tissue neoplasm branch.
- "non_oncology"   — not a cancer condition at all (none observed yet;
                      the category exists for whatever a mis-swept
                      candidate's condition turns out to be).
- "generic_basket" — a root/umbrella oncology term ("Neoplasms",
                      "Neoplasm Metastasis", ...) that by itself says
                      nothing about histology or site — the signature of
                      an all-comers basket trial, not a specific
                      diagnosis. Never resolves to heme or solid alone.
"""
from __future__ import annotations

# fmt: off
MESH_ID_CATEGORY: dict[str, str] = {
    # -- heme: Hemic and Lymphatic Diseases / Immunoproliferative branch --
    "D006402": "heme",  # Hematologic Diseases
    "D006425": "heme",  # Hemic and Lymphatic Diseases
    "D008206": "heme",  # Lymphatic Diseases
    "D007938": "heme",  # Leukemia
    "D007945": "heme",  # Leukemia, Lymphoid
    "D007951": "heme",  # Leukemia, Myeloid
    "D015470": "heme",  # Leukemia, Myeloid, Acute
    "D008223": "heme",  # Lymphoma
    "D016393": "heme",  # Lymphoma, B-Cell
    "D008228": "heme",  # Lymphoma, Non-Hodgkin
    "D016403": "heme",  # Lymphoma, Large B-Cell, Diffuse
    "D054198": "heme",  # Precursor Cell Lymphoblastic Leukemia-Lymphoma
    "D008232": "heme",  # Lymphoproliferative Disorders
    "D007160": "heme",  # Immunoproliferative Disorders (myeloma's branch)
    # Immune System Diseases is broader than heme in general MeSH usage
    # (autoimmune disease lives here too) — classified heme here because
    # every observed occurrence was an ancestor of a heme malignancy, not
    # a genuinely non-oncology immune condition. Revisit if that changes.
    "D007154": "heme",  # Immune System Diseases

    # -- solid: organ/site-specific neoplasm branches --
    "D001660": "solid",  # Biliary Tract Diseases
    "D001661": "solid",  # Biliary Tract Neoplasms
    "D001941": "solid",  # Breast Diseases
    "D001943": "solid",  # Breast Neoplasms
    "D064726": "solid",  # Triple Negative Breast Neoplasms
    "D002277": "solid",  # Carcinoma
    "D000230": "solid",  # Adenocarcinoma
    "D006528": "solid",  # Carcinoma, Hepatocellular
    "D002283": "solid",  # Carcinoma, Bronchogenic
    "D002289": "solid",  # Carcinoma, Non-Small-Cell Lung
    "D001984": "solid",  # Bronchial Neoplasms
    "D008171": "solid",  # Lung Diseases
    "D008175": "solid",  # Lung Neoplasms
    "D012140": "solid",  # Respiratory Tract Diseases
    "D012142": "solid",  # Respiratory Tract Neoplasms
    "D013899": "solid",  # Thoracic Neoplasms
    "D002292": "solid",  # Carcinoma, Renal Cell
    "D007674": "solid",  # Kidney Diseases
    "D007680": "solid",  # Kidney Neoplasms
    "D002295": "solid",  # Carcinoma, Transitional Cell
    "D000093284": "solid",  # Non-Muscle Invasive Bladder Neoplasms
    "D001745": "solid",  # Urinary Bladder Diseases
    "D001749": "solid",  # Urinary Bladder Neoplasms
    "D014515": "solid",  # Ureteral Diseases
    "D014516": "solid",  # Ureteral Neoplasms
    "D014570": "solid",  # Urologic Diseases
    "D014571": "solid",  # Urologic Neoplasms
    "D000091642": "solid",  # Urogenital Diseases
    "D014565": "solid",  # Urogenital Neoplasms
    "D052776": "solid",  # Female Urogenital Diseases
    "D005261": "solid",  # Female Urogenital Diseases and Pregnancy Complications
    "D052801": "solid",  # Male Urogenital Diseases
    "D015179": "solid",  # Colorectal Neoplasms
    "D003108": "solid",  # Colonic Diseases
    "D007410": "solid",  # Intestinal Diseases
    "D007414": "solid",  # Intestinal Neoplasms
    "D012002": "solid",  # Rectal Diseases
    "D005767": "solid",  # Gastrointestinal Diseases
    "D005770": "solid",  # Gastrointestinal Neoplasms
    "D004066": "solid",  # Digestive System Diseases
    "D004067": "solid",  # Digestive System Neoplasms
    "D008107": "solid",  # Liver Diseases
    "D008113": "solid",  # Liver Neoplasms
    "D005184": "solid",  # Fallopian Tube Diseases
    "D005185": "solid",  # Fallopian Tube Neoplasms
    "D000291": "solid",  # Adnexal Diseases
    "D010049": "solid",  # Ovarian Diseases
    "D010051": "solid",  # Ovarian Neoplasms
    "D005831": "solid",  # Genital Diseases, Female
    "D005833": "solid",  # Genital Neoplasms, Female
    "D000091662": "solid",  # Genital Diseases
    "D006058": "solid",  # Gonadal Disorders
    "D000098943": "solid",  # Uveal Melanoma
    "D014603": "solid",  # Uveal Diseases
    "D014604": "solid",  # Uveal Neoplasms
    "D005128": "solid",  # Eye Diseases
    "D005134": "solid",  # Eye Neoplasms
    "D008545": "solid",  # Melanoma
    "D018326": "solid",  # Nevi and Melanomas
    "D004701": "solid",  # Endocrine Gland Neoplasms
    "D004700": "solid",  # Endocrine System Diseases
    "D009375": "solid",  # Neoplasms, Glandular and Epithelial
    "D009380": "solid",  # Neoplasms, Nerve Tissue
    "D017599": "solid",  # Neuroectodermal Tumors
    "D018358": "solid",  # Neuroendocrine Tumors
    "D009373": "solid",  # Neoplasms, Germ Cell and Embryonal
    # Fibrohistiocytic / soft-tissue sarcoma branch — "Histiocytoma" here
    # means the connective-tissue tumour family, NOT a heme/immune
    # histiocytic disorder (e.g. Langerhans cell histiocytosis) — do not
    # merge this with the heme block above.
    "D012509": "solid",  # Sarcoma
    "D009372": "solid",  # Neoplasms, Connective Tissue
    "D018204": "solid",  # Neoplasms, Connective and Soft Tissue
    "D018218": "solid",  # Neoplasms, Fibrous Tissue
    "D005354": "solid",  # Fibrosarcoma
    "D051642": "solid",  # Histiocytoma
    "D051677": "solid",  # Histiocytoma, Malignant Fibrous
    "D012871": "solid",  # Skin Diseases
    "D017437": "solid",  # Skin and Connective Tissue Diseases

    # -- generic_basket: umbrella terms that say nothing about site/histology --
    "D009369": "generic_basket",  # Neoplasms
    "D009370": "generic_basket",  # Neoplasms by Histologic Type
    "D009371": "generic_basket",  # Neoplasms by Site
    "D009385": "generic_basket",  # Neoplastic Processes
    "D010335": "generic_basket",  # Pathologic Processes
    "D013568": "generic_basket",  # Pathological Conditions, Signs and Symptoms
    "D009362": "generic_basket",  # Neoplasm Metastasis

    # -- non_oncology: none observed yet in this project's data. Add here
    # (with the real CT.gov ID) the first time a genuinely non-cancer
    # condition MeSH ID turns up on a discovered candidate.
}
# fmt: on


def category_for(mesh_id: str) -> str | None:
    """The curated category for one MeSH ID, or None if it isn't in the
    dictionary yet — callers must treat None as "don't guess", never as
    a silent default."""
    return MESH_ID_CATEGORY.get(mesh_id)


# Text-level overrides: forced to "ambiguous" regardless of MeSH signal,
# because these specific clinical categories are genuinely dual-nature
# (a heme malignancy behaving/treated like a solid-organ disease) and a
# MeSH-branch verdict alone would misrepresent the real clinical judgement
# call involved. Matched as case-insensitive substrings against the raw
# condition strings (deliberately still string matching for exactly this
# named list — everything else in this module classifies by MeSH ID).
AMBIGUOUS_OVERRIDE_PHRASES: list[str] = [
    # CNS lymphoma: a lymphoma (heme) presenting/managed as a CNS
    # (solid-organ) disease.
    "cns lymphoma",
    "central nervous system lymphoma",
    "primary central nervous system lymphoma",
    # Cutaneous T-cell lymphoma and its named subtypes: a lymphoma (heme)
    # that is clinically a skin (solid-organ) disease.
    "cutaneous t-cell lymphoma",
    "cutaneous t cell lymphoma",
    "mycosis fungoides",
    "sezary syndrome",
    "sézary syndrome",
    # Myeloma with bone involvement: myeloma (heme) with a named
    # solid-organ (skeletal) manifestation.
    "myeloma bone",
    "myeloma with bone",
    "myeloma-related bone",
    # All-comers / tumor-agnostic basket-trial phrasing — a real signal
    # text carries that a generic-only MeSH hit (generic_basket) alone
    # might miss if the sponsor used non-MeSH-indexed wording.
    "all comers",
    "all-comers",
    "tumor agnostic",
    "tumor-agnostic",
    "tumour agnostic",
    "tumour-agnostic",
    "regardless of tumor type",
    "regardless of tumour type",
    "any tumor type",
    "any solid tumor type",
]
