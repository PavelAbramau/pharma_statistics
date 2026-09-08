"""Hand-curated, reviewable list of ADCs with a real regulatory approval
(FDA and/or other major regulator), used ONLY to keep the matrix-only
universe extension's deterministic dead_historical rule
(labelling/dead_historical.py, docs/decisions/0008) from mislabelling a
known-successful compound as dead. This project has no live connection to
a regulatory database (CLAUDE.md: DuckDB over Parquet, no cloud) and this
cohort gets no manual labelling at all, so this is a small, conservative,
by-hand fact list — same convention as discovery/seed_assets.json and
tests/fixtures/known_adcs.txt, and the same honesty caveat: NOT
exhaustive, NOT re-verified against a live source, and NOT a substitute
for a real gold "approved" label. Extend by hand as needed.

Deliberately asymmetric risk, noted for whoever extends this: an entry
missing from this list can cause a real approved compound to be
misclassified dead_historical (an omission risk); a wrong entry can
wrongly exempt something that actually died (a fabrication risk). Keep
entries to compounds with widely-known, high-confidence brand-name
approvals only — when in doubt, leave it out and let the successor/
recency legs of the rule (or simply "not classified") do the work
instead.
"""
from __future__ import annotations

# name -> synonyms (brand names, dev codes, INN variants), lower-cased at
# lookup time by dead_historical.is_known_approved.
KNOWN_APPROVED_ADCS: dict[str, list[str]] = {
    "brentuximab vedotin": ["adcetris", "sgn-35"],
    "gemtuzumab ozogamicin": ["mylotarg"],
    "inotuzumab ozogamicin": ["besponsa"],
    "polatuzumab vedotin": ["polivy"],
    "loncastuximab tesirine": ["zynlonta", "adct-402"],
    "belantamab mafodotin": ["blenrep"],
    "trastuzumab emtansine": ["kadcyla", "t-dm1", "ado-trastuzumab emtansine"],
    "trastuzumab deruxtecan": ["enhertu", "ds-8201", "ds-8201a", "t-dxd"],
    "sacituzumab govitecan": ["trodelvy", "immu-132"],
    "enfortumab vedotin": ["padcev", "asg-22me"],
    "mirvetuximab soravtansine": ["elahere"],
    "tisotumab vedotin": ["tivdak"],
    "disitamab vedotin": ["aidixi", "rc48"],
    "moxetumomab pasudotox": ["lumoxiti"],
}


def known_approved_name_set() -> frozenset[str]:
    """Every name/synonym, lower-cased, flattened into one lookup set."""
    names: set[str] = set()
    for name, synonyms in KNOWN_APPROVED_ADCS.items():
        names.add(name.strip().lower())
        for syn in synonyms:
            names.add(syn.strip().lower())
    return frozenset(names)
