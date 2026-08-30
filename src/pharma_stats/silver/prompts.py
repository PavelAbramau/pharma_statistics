"""Prompt construction, response parsing, and k=5 self-consistency
sampling for the two decomposed questions this first wiring can actually
answer from CT.gov evidence alone. See evidence.py's module docstring for
scope: Q1 (trial initiated since X) is deterministic
(evidence.trials_initiated_since — no model call, no need for one); Q4
(successor asset) always abstains here (ask_successor_asset makes no API
call at all — there is no citable evidence source for it yet).
"""
from __future__ import annotations

import json
from typing import Optional

from pharma_stats.silver import citations as citation_gate
from pharma_stats.silver import evidence as evidence_mod
from pharma_stats.silver import model_client
from pharma_stats.silver import sampling
from pharma_stats.silver.evidence import evidence_text
from pharma_stats.silver.questions import (
    NOT_DETERMINABLE, Citation, DiscontinuationStatementAnswer, StopReasonAnswer,
    SuccessorAssetAnswer,
)

K = 5
TEMPERATURE = 1.0

_EXTRACTION_SYSTEM = (
    "You are extracting a single fact from clinical trial registry evidence. "
    "Answer ONLY from the evidence given below — never from outside knowledge you "
    "cannot cite. If the evidence does not clearly support an answer, say so "
    "honestly rather than guessing. Respond with a single JSON object and nothing else."
)


def _discontinuation_prompt(program_name: str, evidence: str) -> str:
    return f"""Asset/program: {program_name}

Evidence (CT.gov registry data for every trial on file for this program):
{evidence}

Question: Does a public statement of discontinuation exist for this program? A
CT.gov "why_stopped" field on a TERMINATED or WITHDRAWN trial that gives a
substantive reason counts as a public statement. A blank why_stopped, a vague
one ("business reasons" with no detail), or a trial that is not
terminated/withdrawn does NOT count as a clear statement.

Respond with exactly this JSON shape:
{{"exists": true, "statement_date": "<date associated with that trial>", "nct_id": "<NCT id>", "quote": "<EXACT why_stopped substring, verbatim>"}}
or
{{"exists": false, "statement_date": null, "nct_id": null, "quote": null}}
or
{{"exists": "not_determinable", "statement_date": null, "nct_id": null, "quote": null}}

Use "not_determinable" if the evidence is ambiguous or insufficient — do not guess."""


def _stop_reason_prompt(program_name: str, evidence: str) -> str:
    return f"""Asset/program: {program_name}

Evidence (CT.gov registry data for every trial on file for this program):
{evidence}

Question: Does the stated stop reason (why_stopped) cite efficacy, safety, or
business/strategic grounds? Classify ONLY the reason actually stated in the
evidence — do not infer a reason that isn't written.

Respond with exactly this JSON shape:
{{"category": "efficacy" | "safety" | "business" | "not_determinable",
  "nct_id": "<the NCT id this reason is on, or null>",
  "quote": "<the EXACT why_stopped substring you are relying on, verbatim, or null>"}}

Use "not_determinable" if no trial has a clear, substantive stop reason stated."""


def _sample_once(prompt: str, model: str) -> tuple[Optional[dict], str]:
    raw = model_client.complete(prompt, system=_EXTRACTION_SYSTEM, temperature=TEMPERATURE, model=model)
    return model_client.extract_json(raw), raw


def _verify_citation(
    nct_id: Optional[str], quote: Optional[str], evidence: dict,
) -> tuple[Optional[Citation], dict]:
    if not nct_id or not quote:
        return None, {"passed": False, "reason": "no usable citation in majority sample"}
    locator = evidence_mod.citation_locator(nct_id, evidence_mod.source_snapshot_for(evidence, nct_id))
    citation = Citation(source_type="raw_snapshot", locator=locator, quote=quote)
    try:
        passed, reason = citation_gate.resolve_and_verify(citation), None
    except citation_gate.CitationError as e:
        passed, reason = False, str(e)
    return (citation if passed else None), {"passed": passed, "reason": reason, "citation": vars(citation)}


def ask_discontinuation_statement(
    program_name: str, evidence: dict, *, model: str = model_client.DEFAULT_MODEL,
) -> tuple[DiscontinuationStatementAnswer, dict]:
    prompt = _discontinuation_prompt(program_name, evidence_text(evidence))
    raw_responses: list[str] = []
    parsed: list[Optional[dict]] = []

    def _one() -> str:
        p, raw = _sample_once(prompt, model)
        raw_responses.append(raw)
        parsed.append(p)
        return json.dumps(p.get("exists") if p else None)

    votes = sampling.sample_answers(_one, k=K, temperature=TEMPERATURE)
    disagreement = sampling.should_abstain(votes)
    majority_raw, majority_count = sampling.majority_vote(votes) if votes else (json.dumps(None), 0)
    majority_value = json.loads(majority_raw)

    log = {
        "question": "discontinuation_statement", "prompt": prompt, "system": _EXTRACTION_SYSTEM,
        "model": model, "k": K, "temperature": TEMPERATURE,
        "raw_responses": raw_responses, "parsed_samples": parsed,
        "votes": [json.loads(v) for v in votes], "disagreement": disagreement, "majority_count": majority_count,
    }

    if disagreement or majority_value in (None, NOT_DETERMINABLE, False):
        log["citation_verdict"] = None
        result = NOT_DETERMINABLE if majority_value in (None, NOT_DETERMINABLE) or disagreement else False
        return DiscontinuationStatementAnswer(exists=result), log

    sample = next((p for p in parsed if p and p.get("exists") is True), None)
    citation, verdict = _verify_citation(
        sample.get("nct_id") if sample else None, sample.get("quote") if sample else None, evidence,
    )
    log["citation_verdict"] = verdict
    if citation is None:
        return DiscontinuationStatementAnswer(exists=NOT_DETERMINABLE), log
    return DiscontinuationStatementAnswer(
        exists=True, statement_date=sample.get("statement_date"), citations=[citation],
    ), log


def ask_stop_reason(
    program_name: str, evidence: dict, *, model: str = model_client.DEFAULT_MODEL,
) -> tuple[StopReasonAnswer, dict]:
    prompt = _stop_reason_prompt(program_name, evidence_text(evidence))
    raw_responses: list[str] = []
    parsed: list[Optional[dict]] = []

    def _one() -> str:
        p, raw = _sample_once(prompt, model)
        raw_responses.append(raw)
        parsed.append(p)
        return json.dumps(p.get("category") if p else None)

    votes = sampling.sample_answers(_one, k=K, temperature=TEMPERATURE)
    disagreement = sampling.should_abstain(votes)
    majority_raw, majority_count = sampling.majority_vote(votes) if votes else (json.dumps(None), 0)
    majority_value = json.loads(majority_raw)

    log = {
        "question": "stop_reason", "prompt": prompt, "system": _EXTRACTION_SYSTEM,
        "model": model, "k": K, "temperature": TEMPERATURE,
        "raw_responses": raw_responses, "parsed_samples": parsed,
        "votes": [json.loads(v) for v in votes], "disagreement": disagreement, "majority_count": majority_count,
    }

    if disagreement or majority_value in (None, NOT_DETERMINABLE):
        log["citation_verdict"] = None
        return StopReasonAnswer(category=NOT_DETERMINABLE), log

    sample = next((p for p in parsed if p and p.get("category") == majority_value), None)
    citation, verdict = _verify_citation(
        sample.get("nct_id") if sample else None, sample.get("quote") if sample else None, evidence,
    )
    log["citation_verdict"] = verdict
    if citation is None:
        return StopReasonAnswer(category=NOT_DETERMINABLE), log
    return StopReasonAnswer(category=majority_value, citations=[citation]), log


def ask_successor_asset(program_name: str, evidence: dict) -> tuple[SuccessorAssetAnswer, dict]:
    """Always abstains — no citable evidence source for successor-asset
    detection is wired in yet (would need target-antigen normalisation,
    which doesn't exist per CLAUDE.md, or the retrieval agent hooked into
    evidence-gathering, which it isn't). No model call is made."""
    log = {
        "question": "successor_asset", "prompt": None, "model": None,
        "raw_responses": [], "parsed_samples": [],
        "note": "no citable evidence source wired in yet — hard-coded not_determinable, no API call made",
    }
    return SuccessorAssetAnswer(exists=NOT_DETERMINABLE), log
