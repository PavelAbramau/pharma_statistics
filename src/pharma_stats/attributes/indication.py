"""B0/B5 indication axis: the most common specific MeSH condition term
across a program's trials — NOT the real OncoTree indication_code
(provisional_programs.py hard-codes that "UNSPECIFIED"; the real
five-entity normalisation hasn't been built). Named indication_mesh_term,
never indication_code, so nothing downstream mistakes this for the real
thing.

Read from CT.gov's current-state fetch only (conditionBrowseModule),
reusing provisional_programs._condition_browse_data — the same sanctioned
current-state read trial_scope's heme/solid classification already uses.
Disease category is a static, universe-membership property per
docs/decisions/0001 (it doesn't change over a program's life the way
status/enrollment do), so this is within that decision's existing
exception, not a new one.

Deliberately the leaf-level MeSH term (conditionBrowseModule.meshes), not
the coarser ancestors list mesh_categories.py uses for heme/solid — a
target x indication matrix needs the specific condition ("Breast
Neoplasms"), not just "solid tumor."
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from pharma_stats.labelling.provisional_programs import _condition_browse_data


def program_indication_mesh_term(program: dict) -> Optional[str]:
    """Most frequent specific MeSH condition term across a program's
    trials. None if no trial on this program has ever had a
    current-state fetch (the common case for most of the universe)."""
    counts: Counter = Counter()
    for t in program.get("trials") or []:
        meshes, _ancestors = _condition_browse_data(t["nct_id"])
        for m in meshes:
            term = m.get("term")
            if term:
                counts[term] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]
