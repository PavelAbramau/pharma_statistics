"""B5 opportunity-matrix row axis: coarse organ/tissue "tumour system"
grouping, replacing the specific-MeSH-term indication axis
(attributes/indication.py) that made the target x indication matrix
inevitably sparse — ~300 in-scope programs spread across target x
specific-MeSH-term can never be dense; that's arithmetic, not a data
gap. See attributes/matrix.py's module docstring for the full rebuild
rationale and docs/decisions/0005.

Reuses discovery/mesh_categories.py's hand-curated, CT.gov-observed
"solid" MeSH ID set (already reviewed, ID-keyed, real data only — see
that module's docstring) and adds only the ID -> system grouping on top,
same extend-by-hand convention: add new IDs to mesh_categories.py first,
then group them here.

This project has no real MeSH tree-number data on disk — CT.gov's
conditionBrowseModule returns id+term only, never a tree number — so
"minimum tree depth" is enforced the only way this data supports: a term
that says nothing about body site/histology is too shallow to be a
matrix cell, full stop, not a lenient bucket. Two things are therefore
NEVER assigned a system (system_for returns None), and callers must treat
None identically both ways — exclude the program from the axis entirely,
never bucket it into a catch-all "other"/"unknown" row:
  - anything mesh_categories.MESH_ID_CATEGORY marks "generic_basket"
    (root/umbrella terms: Neoplasms, Neoplasm Metastasis, Neoplasms by
    Site, ...) or anything not "solid" at all (heme, non_oncology, or
    simply not yet in that dictionary)
  - the two site-agnostic histology terms that DO live in "solid" itself
    (Carcinoma, Adenocarcinoma) — real, specific pathology, but not a
    body site; grouping either into one of the 8 systems below would
    misrepresent it as, say, a GI or lung program
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from pharma_stats.discovery.mesh_categories import MESH_ID_CATEGORY
from pharma_stats.labelling.provisional_programs import _condition_browse_data

# CT.gov "solid" MeSH ID (mesh_categories.py) -> one of 8 coarse tumour-
# system groups. Every key here is also "solid" in
# mesh_categories.MESH_ID_CATEGORY — system_for() checks that at call
# time rather than trusting the two dictionaries stay in sync by
# convention alone. Extend by hand alongside mesh_categories.py whenever
# a new "solid" MeSH ID is curated there.
MESH_ID_TO_SYSTEM: dict[str, str] = {
    # -- breast --
    "D001941": "breast", "D001943": "breast", "D064726": "breast",

    # -- gi_hepatobiliary --
    "D001660": "gi_hepatobiliary", "D001661": "gi_hepatobiliary",
    "D006528": "gi_hepatobiliary", "D008107": "gi_hepatobiliary",
    "D008113": "gi_hepatobiliary", "D015179": "gi_hepatobiliary",
    "D003108": "gi_hepatobiliary", "D007410": "gi_hepatobiliary",
    "D007414": "gi_hepatobiliary", "D012002": "gi_hepatobiliary",
    "D005767": "gi_hepatobiliary", "D005770": "gi_hepatobiliary",
    "D004066": "gi_hepatobiliary", "D004067": "gi_hepatobiliary",

    # -- thoracic_lung --
    "D002283": "thoracic_lung", "D002289": "thoracic_lung",
    "D001984": "thoracic_lung", "D008171": "thoracic_lung",
    "D008175": "thoracic_lung", "D012140": "thoracic_lung",
    "D012142": "thoracic_lung", "D013899": "thoracic_lung",

    # -- genitourinary (kidney/bladder/urothelial/male genital) --
    "D002292": "genitourinary", "D007674": "genitourinary",
    "D007680": "genitourinary", "D002295": "genitourinary",
    "D000093284": "genitourinary", "D001745": "genitourinary",
    "D001749": "genitourinary", "D014515": "genitourinary",
    "D014516": "genitourinary", "D014570": "genitourinary",
    "D014571": "genitourinary", "D000091642": "genitourinary",
    "D014565": "genitourinary", "D052801": "genitourinary",

    # -- gynecologic (ovarian/fallopian/female genital) --
    "D005184": "gynecologic", "D005185": "gynecologic",
    "D000291": "gynecologic", "D010049": "gynecologic",
    "D010051": "gynecologic", "D005831": "gynecologic",
    "D005833": "gynecologic", "D000091662": "gynecologic",
    "D052776": "gynecologic", "D005261": "gynecologic",
    "D006058": "gynecologic",

    # -- skin_ocular_melanoma --
    "D000098943": "skin_ocular_melanoma", "D014603": "skin_ocular_melanoma",
    "D014604": "skin_ocular_melanoma", "D005128": "skin_ocular_melanoma",
    "D005134": "skin_ocular_melanoma", "D008545": "skin_ocular_melanoma",
    "D018326": "skin_ocular_melanoma", "D012871": "skin_ocular_melanoma",
    "D017437": "skin_ocular_melanoma",

    # -- sarcoma_soft_tissue --
    "D012509": "sarcoma_soft_tissue", "D009372": "sarcoma_soft_tissue",
    "D018204": "sarcoma_soft_tissue", "D018218": "sarcoma_soft_tissue",
    "D005354": "sarcoma_soft_tissue", "D051642": "sarcoma_soft_tissue",
    "D051677": "sarcoma_soft_tissue",

    # -- endocrine_neuroendocrine_germcell --
    "D004701": "endocrine_neuroendocrine_germcell",
    "D004700": "endocrine_neuroendocrine_germcell",
    "D009375": "endocrine_neuroendocrine_germcell",
    "D009380": "endocrine_neuroendocrine_germcell",
    "D017599": "endocrine_neuroendocrine_germcell",
    "D018358": "endocrine_neuroendocrine_germcell",
    "D009373": "endocrine_neuroendocrine_germcell",

    # D002277 (Carcinoma) and D000230 (Adenocarcinoma) deliberately
    # excluded — real "solid" category, but site-agnostic histology, not
    # a body system; see module docstring.
}

TUMOUR_SYSTEM_VALUES = (
    "breast", "gi_hepatobiliary", "thoracic_lung", "genitourinary",
    "gynecologic", "skin_ocular_melanoma", "sarcoma_soft_tissue",
    "endocrine_neuroendocrine_germcell",
)


def system_for(mesh_id: Optional[str]) -> Optional[str]:
    """The curated tumour-system group for one MeSH ID, or None if it's
    generic, non-solid, or not yet curated. Callers must treat None as
    "exclude from the axis," never as a silent "other" bucket."""
    if not mesh_id or MESH_ID_CATEGORY.get(mesh_id) != "solid":
        return None
    return MESH_ID_TO_SYSTEM.get(mesh_id)


def program_tumour_system(program: dict) -> Optional[str]:
    """Most frequent resolvable tumour system across a program's trials'
    leaf MeSH conditions — same _condition_browse_data source as
    attributes/indication.py's program_indication_mesh_term, one level
    coarser. None whenever no trial resolves to any of the 8 systems (no
    MeSH data at all, MeSH present but only generic/site-agnostic terms,
    or a non-solid/uncurated ID) — the program routes to exclusion, never
    a guessed or catch-all group."""
    counts: Counter = Counter()
    for t in program.get("trials") or []:
        meshes, _ancestors = _condition_browse_data(t["nct_id"])
        for m in meshes:
            system = system_for(m.get("id"))
            if system:
                counts[system] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
