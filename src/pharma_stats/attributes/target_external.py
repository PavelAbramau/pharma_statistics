"""Residual target-antigen resolution via external drug databases — the
last tier after attributes/target.py's offline dictionaries and trial-
text extraction have been exhausted. Kept as a SEPARATE, explicit,
network-dependent step (never folded into derive_target()) because:

  1. It requires live API calls (ChEMBL + Open Targets), unlike
     derive_target's pure-offline dictionary/regex tiers.
  2. Its confidence basis is different (a verified cross-database
     identity match + a structured mechanism-of-action record), not a
     text-pattern hit — callers must be able to tell the two apart.

Two-step design, run in this order per candidate:
  1. ChEMBL `molecule/search` for the candidate's own name/synonyms,
     accepting a hit ONLY on an EXACT (case/whitespace-insensitive) match
     against that molecule's own pref_name or a molecule_synonyms entry
     — never on ChEMBL's fuzzy relevance ranking alone. Same discipline
     as triage/layer1_5.chembl_lookup (verified live 2026-09-07: querying
     "TAK-188" or "TAK-500" returns 20 fuzzy-ranked hits whose FIRST
     result is a completely unrelated compound — rank-1 trust would
     silently mislabel the target).
  2. Open Targets `drug(chemblId)` mechanismsOfAction, keyed off the
     ChEMBL id verified in step 1 (skips re-verifying identity — the
     ChEMBL match already did that). Resolves to a target ONLY when
     exactly one mechanism row has actionType == "BINDING AGENT" (the
     antibody's own antigen-binding mechanism, as opposed to the
     cytotoxic payload's mechanism, e.g. "INHIBITOR" against tubulin/
     topoisomerase — verified live against tusamitamab ravtansine,
     CHEMBL4298098) AND that row names exactly one target. Multiple
     BINDING AGENT rows, multiple targets in one row, or zero such rows
     all mean "don't guess" -> unresolved, same policy as
     target_from_trial_text's ambiguity handling.

A genuine bispecific/dual-target ADC will legitimately have >1 target
here and correctly stay unresolved by this tier -- same as
attributes/target.py's own trial-text extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pharma_stats.clients.chembl import ChemblClient, ChemblError
from pharma_stats.clients.opentargets import OpenTargetsClient, OpenTargetsError

ANTIBODY_TARGET_ACTION_TYPE = "BINDING AGENT"


def _norm(s: Optional[str]) -> str:
    return " ".join((s or "").strip().lower().split())


@dataclass
class ExternalTargetHit:
    target: str            # HGNC symbol
    chembl_id: str
    matched_name: str       # which of our name/synonym strings matched
    matched_via: str        # "pref_name" or "molecule_synonyms"
    mechanism_of_action: str  # raw text, for the review trail


def _verified_chembl_id(
    name: Optional[str], synonyms: list[str], *, client: ChemblClient,
) -> Optional[tuple[str, str, str]]:
    """(chembl_id, matched_name, matched_via) on an EXACT match, else
    None. Mirrors triage/layer1_5.chembl_lookup's verification discipline
    exactly, but doesn't filter by molecule_type -- an early dev code can
    be in ChEMBL with its type still unclassified."""
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
            continue

        for m in molecules:
            cid = m.get("molecule_chembl_id")
            if not cid:
                continue
            pref = m.get("pref_name")
            if pref and _norm(pref) in our_names:
                return cid, pref, "pref_name"
            for syn in m.get("molecule_synonyms") or []:
                syn_name = syn.get("molecule_synonym")
                if syn_name and _norm(syn_name) in our_names:
                    return cid, syn_name, "molecule_synonyms"
    return None


def resolve_target_external(
    name: Optional[str], synonyms: Optional[list] = None, *,
    chembl_client: ChemblClient, opentargets_client: OpenTargetsClient,
) -> Optional[ExternalTargetHit]:
    """None if the candidate isn't found in ChEMBL at all (a real,
    common outcome for very early-phase/obscure dev codes -- verified
    live: several genuine Chinese Phase-1 ADC dev codes return zero
    ChEMBL search hits entirely, a hard data-availability ceiling this
    function cannot work around), or found but Open Targets has no
    single-target BINDING AGENT mechanism on file for it."""
    verified = _verified_chembl_id(name, synonyms or [], client=chembl_client)
    if verified is None:
        return None
    chembl_id, matched_name, matched_via = verified

    try:
        rows = opentargets_client.drug_mechanisms(chembl_id)
    except OpenTargetsError:
        return None

    binding_rows = [r for r in rows if r.get("actionType") == ANTIBODY_TARGET_ACTION_TYPE]
    if len(binding_rows) != 1:
        return None  # zero -> no antibody-target mechanism on file; >1 -> ambiguous, don't guess
    row = binding_rows[0]
    targets = row.get("targets") or []
    symbols = {t["approvedSymbol"] for t in targets if t.get("approvedSymbol")}
    if len(symbols) != 1:
        return None  # 0 or >1 distinct symbols on the single binding-agent row -- don't guess

    return ExternalTargetHit(
        target=symbols.pop(), chembl_id=chembl_id, matched_name=matched_name, matched_via=matched_via,
        mechanism_of_action=row.get("mechanismOfAction") or "",
    )
