"""Red Team objection agent: an adversarial check that runs AFTER
self-consistency sampling agrees on a label, not instead of it. Given a
silver label and its supporting evidence, generates the strongest
evidenced case that the label is wrong; a strong, evidenced objection
forces abstention regardless of how confidently the k samples agreed.

Wired to the Anthropic API (silver/model_client.py). Runs ONE call at
temperature=0 — a single authoritative adversarial pass, not sampled;
self-consistency sampling already happened upstream on the underlying
questions, so this step isn't re-sampling those, it's a second, different
kind of check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from pharma_stats.silver import citations as citation_gate
from pharma_stats.silver import evidence as evidence_mod
from pharma_stats.silver import model_client
from pharma_stats.silver.evidence import evidence_text
from pharma_stats.silver.questions import Citation

OBJECTION_STRENGTHS = ["weak", "moderate", "strong"]

RED_TEAM_SYSTEM = (
    "You are a skeptical reviewer trying to find the strongest evidenced flaw in a "
    "conclusion. Use ONLY the evidence given — never outside knowledge you cannot cite. "
    "If you genuinely cannot find a real objection in this evidence, say so honestly; "
    "do not manufacture one to seem thorough. Respond with a single JSON object and "
    "nothing else."
)


@dataclass
class Objection:
    strength: str  # one of OBJECTION_STRENGTHS
    argument: str
    citations: list[Citation] = field(default_factory=list)


def forces_abstention(objection: Objection) -> bool:
    """A strong objection with no evidence behind it doesn't count — the
    whole point is an EVIDENCED case, not just a model expressing doubt."""
    return objection.strength == "strong" and bool(objection.citations)


def _prompt(label: dict, ev_text: str) -> str:
    return f"""A silver auto-labeller concluded the following about a clinical program,
from the evidence below:

Label: {json.dumps(label)}

Evidence (CT.gov registry data for every trial on file for this program):
{ev_text}

Task: generate the STRONGEST evidenced case that this label is WRONG, using only
the evidence above.

Respond with exactly this JSON shape:
{{"strength": "weak" | "moderate" | "strong",
  "argument": "<your case, 1-3 sentences>",
  "nct_id": "<NCT id your argument is based on, or null>",
  "quote": "<the EXACT substring you are relying on, verbatim, or null>"}}"""


def generate_objection(
    label: dict, evidence: dict, *, model: str = model_client.RED_TEAM_MODEL,
) -> tuple[Objection, dict]:
    """Returns (Objection, log) — log carries the prompt/raw response/
    citation verdict for silver/store.py's full-reasoning record."""
    ev_text = evidence_text(evidence)
    prompt = _prompt(label, ev_text)
    raw = model_client.complete(prompt, system=RED_TEAM_SYSTEM, temperature=0.0, model=model)
    parsed = model_client.extract_json(raw)

    log = {"prompt": prompt, "system": RED_TEAM_SYSTEM, "model": model, "temperature": 0.0, "raw_response": raw}

    if not parsed or parsed.get("strength") not in OBJECTION_STRENGTHS:
        log["citation_verdict"] = None
        objection = Objection(strength="weak", argument="(unparseable model response)")
        log["objection"] = vars(objection) | {"citations": []}
        return objection, log

    strength = parsed["strength"]
    argument = parsed.get("argument") or ""
    nct_id, quote = parsed.get("nct_id"), parsed.get("quote")

    citations: list[Citation] = []
    if nct_id and quote:
        locator = evidence_mod.citation_locator(nct_id, evidence_mod.source_snapshot_for(evidence, nct_id))
        citation = Citation(source_type="raw_snapshot", locator=locator, quote=quote)
        try:
            passed = citation_gate.resolve_and_verify(citation)
            reason = None
        except citation_gate.CitationError as e:
            passed, reason = False, str(e)
        log["citation_verdict"] = {"passed": passed, "reason": reason, "citation": vars(citation)}
        if passed:
            citations.append(citation)
        else:
            # an unverifiable citation can't back a strong objection —
            # downgrade rather than let it force an abstention it hasn't earned
            strength = "weak"
    else:
        log["citation_verdict"] = None

    objection = Objection(strength=strength, argument=argument, citations=citations)
    log["objection"] = {"strength": objection.strength, "argument": objection.argument,
                         "citations": [vars(c) for c in objection.citations]}
    return objection, log
