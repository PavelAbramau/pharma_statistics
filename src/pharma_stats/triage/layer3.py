"""Layer 3: single web_search query per candidate, for whatever Layer 2
routes onward (see layer2.route_to_layer3: unsure, persistent disagreement,
or an answer that only ever came from background knowledge). Capped —
this is the most expensive layer per candidate (a live web search), so the
cap is reported before ever running, and anything still unsure after this
goes to the human's manual queue untouched.

One query per candidate: '"<name>" antibody drug conjugate', via the
server-side web_search tool (Anthropic-hosted, no client-side fetch loop —
see the claude-api skill's Server Tools reference). Submitted through the
same Message Batches API as Layer 2, one request per candidate (batching
several candidates into one prompt doesn't make sense here — each needs
its own independent search).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from pharma_stats.silver import model_client

# Anthropic custom_id is ^[a-zA-Z0-9_-]{1,64}$. program_ids can contain
# slashes (e.g. trifluridine/tipiracil) and the old "l3:{pid}" prefix used
# a colon — both 400 the whole batch. Keep a deterministic mapping so
# collect_layer3_answers can still look results up by program_id.
_SAFE_PID = re.compile(r"^[a-zA-Z0-9_-]{1,61}$")

MAX_LAYER3_CANDIDATES = 400  # hard cap — reported before running, never silently exceeded
MODEL = model_client.DEFAULT_MODEL
MAX_USES = 1  # one search per candidate — this is a targeted lookup, not open research
PROMPT_VERSION = "layer3-v1"

_SYSTEM = (
    "You are confirming whether a named clinical-stage compound is an antibody-drug "
    "conjugate (ADC) — an antibody or antibody fragment covalently linked to a cytotoxic "
    "payload via a chemical linker. Use the web search result to answer; quote the exact "
    "sentence that supports your answer. If the search doesn't clearly settle it, answer "
    "\"unsure\" — do not guess from general knowledge. Respond with a single JSON object "
    "and nothing else, after your search."
)

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_USES}


@dataclass
class Layer3Answer:
    program_id: str
    name: str
    is_adc: str  # "yes" / "no" / "unsure"
    quote: Optional[str]
    source_url: Optional[str]


def build_query(name: str) -> str:
    return f'"{name}" antibody drug conjugate'


def build_prompt(name: str) -> str:
    return f"""Search for: {build_query(name)}

Question: Is "{name}" an antibody-drug conjugate (ADC)?

Respond with exactly this JSON shape:
{{"is_adc": "yes" | "no" | "unsure", "quote": "<exact sentence from the search result you relied on, or null>",
  "source_url": "<the URL the quote came from, or null>"}}"""


def _custom_id(program_id: str) -> str:
    if _SAFE_PID.fullmatch(program_id):
        return f"l3_{program_id}"
    digest = hashlib.sha256(program_id.encode("utf-8")).hexdigest()[:32]
    return f"l3_{digest}"


def submit_layer3_batch(candidates: list[dict], *, model: str = MODEL) -> str:
    """candidates: [{"program_id", "name"}, ...], already capped by the
    caller at MAX_LAYER3_CANDIDATES. One request per candidate, single
    web_search use each."""
    requests = [
        {
            "custom_id": _custom_id(c["program_id"]), "prompt": build_prompt(c["name"]),
            "system": _SYSTEM, "temperature": 0.0, "max_tokens": 1024, "model": model,
            "tools": [WEB_SEARCH_TOOL],
        }
        for c in candidates
    ]
    return model_client.submit_batch(requests, model=model)


def collect_layer3_answers(
    candidates: list[dict], batch_id: str, *, model: str = MODEL,
) -> tuple[dict[str, Layer3Answer], list]:
    results = model_client.collect_batch_results(batch_id, model=model)
    out: dict[str, Layer3Answer] = {}
    usages = []
    for c in candidates:
        pid = c["program_id"]
        hit = results.get(_custom_id(pid))
        if hit is None:
            out[pid] = Layer3Answer(pid, c["name"], "unsure", None, None)
            continue
        text, usage = hit
        usages.append(usage)
        parsed = model_client.extract_json(text)
        if not parsed or parsed.get("is_adc") not in ("yes", "no", "unsure"):
            out[pid] = Layer3Answer(pid, c["name"], "unsure", None, None)
            continue
        out[pid] = Layer3Answer(
            pid, c["name"], parsed["is_adc"], parsed.get("quote"), parsed.get("source_url"),
        )
    return out, usages


def run_layer3(candidates: list[dict], *, model: str = MODEL) -> tuple[dict[str, Layer3Answer], dict]:
    """Full Layer 3 pass, capped at MAX_LAYER3_CANDIDATES — raises rather
    than silently truncating if the caller didn't already cap and report
    the count (see scripts/run_triage_layer2.py's dry-run, which must
    print this count BEFORE any real run)."""
    if len(candidates) > MAX_LAYER3_CANDIDATES:
        raise model_client.ModelClientError(
            f"{len(candidates)} candidates exceeds MAX_LAYER3_CANDIDATES={MAX_LAYER3_CANDIDATES} — "
            "cap the input yourself and report the count before calling run_layer3"
        )
    batch_id = submit_layer3_batch(candidates, model=model)
    model_client.poll_batch_until_done(batch_id)
    answers, usages = collect_layer3_answers(candidates, batch_id, model=model)

    log = {
        "prompt_version": PROMPT_VERSION, "model": model, "n_candidates": len(candidates),
        "n_unsure": sum(1 for a in answers.values() if a.is_adc == "unsure"),
        "usage": {
            "calls": len(usages),
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "cost_usd": sum(
                model_client.estimate_cost(u.input_tokens, u.output_tokens, u.model, batch=True)
                for u in usages
            ),
        },
    }
    return answers, log
