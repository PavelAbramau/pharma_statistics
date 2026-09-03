"""Layer 1.5: deterministic external-database lookups for candidates
Layer 1's local rules and Layer 2/3's model calls all left unsure.

These aren't cases where the model failed — Layer 3's own web search
already tried and came back empty for most of them (see
evidence_source="no_usable_evidence" in triage/validation.py's docstring).
The problem is that CT.gov's registry text never states the molecule's
modality for these candidates. More inference — a bigger model, more
samples, another search — won't manufacture a fact that isn't in the
evidence. A different EVIDENCE SOURCE might have it explicitly, as a
matter of curated record rather than something to infer.

Sources, in priority order, stopping at the first confident hit:

1. ChEMBL (chembl_lookup / evaluate_layer1_5) — molecule_type explicitly
   distinguishes "Antibody drug conjugate" from "Antibody" / "Small
   molecule". Free, no-auth REST API, deterministic. This genuinely
   should have been a Layer 1 rule from the start (see
   deterministic.py) — it's added here as a separate, later pass because
   the 232-candidate target already exists and re-running Layer 1's
   general pass over the whole universe is a bigger, separate change.

2. DrugBank — requested as a second source. NOT implemented: its query
   API ("Clinical Intelligence API") is a paid/academic-licensed product
   with no free public endpoint. Verified 2026-09-01: fetching
   docs.drugbank.com/v1/ returns HTTP 403 without credentials, and
   DrugBank's own release page describes API access as either a paid
   commercial license or an academic license requiring registration
   (bulk downloads under that license are also "temporarily paused" as
   of this check). drugbank_lookup() raises NotImplementedError rather
   than guessing at an endpoint shape nobody here has ever seen — wire it
   for real once real credentials and documented endpoints exist.

Every hit records the source database, the field queried, and the exact
retrieved value (Layer15Hit) — as traceable as a quoted snippet from a
model call, and reusable in exactly the shape
deterministic.Layer1Result/triage/apply.py already expect, so a ChEMBL
hit stages and (once reviewed) commits through the identical path an
INN-suffix or denylist hit does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharma_stats.clients.chembl import ChemblClient, ChemblError
from pharma_stats.triage.deterministic import Layer1Result

# Confident, curated mappings only. Verified live against real ChEMBL
# responses (2026-09-01) for exactly these three values — ChEMBL's schema
# has other molecule_type values this project has never observed (e.g.
# Oligonucleotide, Protein, Enzyme, Unknown) and this deliberately does
# NOT guess a mapping for them; an unrecognized value falls through to
# "not confident" rather than being silently classed either way.
CONFIDENT_MOLECULE_TYPE_MAP = {
    "Antibody drug conjugate": "yes",
    "Small molecule": "no",   # structurally cannot be an antibody-drug conjugate
    "Antibody": "no",         # THIS entity, per ChEMBL, is an unconjugated antibody
}


@dataclass
class Layer15Hit:
    source: str          # "chembl"
    field: str            # "molecule_type"
    value: str            # raw retrieved value, e.g. "Antibody drug conjugate"
    record_id: str        # e.g. ChEMBL id
    matched_name: str     # which of our name/synonym strings matched
    matched_via: str      # "pref_name" or "molecule_synonyms"


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").strip().lower().split())


def chembl_lookup(
    name: Optional[str], synonyms: list[str], *, client: ChemblClient,
) -> Optional[Layer15Hit]:
    """Query ChEMBL for each of our name/synonym strings in turn, stopping
    at the first EXACT (case/whitespace-insensitive) match against a
    returned molecule's pref_name or a molecule_synonyms entry. ChEMBL's
    own search endpoint is fuzzy relevance-ranked — a result being FIRST
    is not enough; this only trusts a molecule after verifying our name
    actually matches ONE OF that specific molecule's recorded names,
    never on ChEMBL's ranking alone."""
    our_names = {_norm(n) for n in ([name] + list(synonyms or [])) if n and n.strip()}
    if not our_names:
        return None

    tried: set[str] = set()
    for query in [name] + list(synonyms or []):
        if not query or not query.strip() or _norm(query) in tried:
            continue
        tried.add(_norm(query))
        try:
            molecules = client.search_molecule(query)
        except ChemblError:
            continue  # a lookup failure is "no hit" here, not a crash — same policy as a citation gate miss

        for m in molecules:
            mtype = m.get("molecule_type")
            if mtype not in CONFIDENT_MOLECULE_TYPE_MAP:
                continue
            pref = m.get("pref_name")
            if pref and _norm(pref) in our_names:
                return Layer15Hit(
                    source="chembl", field="molecule_type", value=mtype,
                    record_id=m.get("molecule_chembl_id") or "", matched_name=pref, matched_via="pref_name",
                )
            for syn in m.get("molecule_synonyms") or []:
                syn_name = syn.get("molecule_synonym")
                if syn_name and _norm(syn_name) in our_names:
                    return Layer15Hit(
                        source="chembl", field="molecule_type", value=mtype,
                        record_id=m.get("molecule_chembl_id") or "",
                        matched_name=syn_name, matched_via="molecule_synonyms",
                    )
    return None


def drugbank_lookup(name: Optional[str], synonyms: list[str]) -> Optional[Layer15Hit]:
    raise NotImplementedError(
        "DrugBank's query API ('Clinical Intelligence API') is a paid/academic-licensed "
        "product with no free public endpoint — verified 2026-09-01, docs.drugbank.com/v1/ "
        "returns HTTP 403 without credentials. Provide real credentials and the actual "
        "documented request/response shape before implementing this; do not guess it."
    )


def evaluate_layer1_5(
    program: dict, *, client: ChemblClient,
) -> tuple[Optional[Layer1Result], Optional[Layer15Hit]]:
    """(result, hit) — result is shaped exactly like
    deterministic.evaluate()'s output (same dataclass) so it stages and
    (once reviewed) commits through the identical path any other Layer 1
    rule does. Layer 1.5 only ever answers is_adc, never scope — a
    resolved is_adc=no is immediately committable (gate 1, terminal); a
    resolved is_adc=yes still needs a scope call, exactly like an
    INN-suffix hit that found nothing on the two deterministic
    scope-rejection rules. None, None if nothing resolves — sources beyond
    ChEMBL (see module docstring) or the human queue decide from there."""
    name = program.get("proposed_name")
    synonyms = program.get("synonyms") or []
    hit = chembl_lookup(name, synonyms, client=client)
    if hit is None:
        return None, None

    is_adc = CONFIDENT_MOLECULE_TYPE_MAP[hit.value]
    rule = f"layer1_5_chembl:{hit.value}:{hit.record_id}"
    if is_adc == "no":
        return Layer1Result(is_adc="no", in_scope=None, scope_reason=None, rule=rule, committable=True), hit
    return Layer1Result(is_adc="yes", in_scope=None, scope_reason=None, rule=rule, committable=False), hit
