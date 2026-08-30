"""Thin Anthropic API wrapper for the silver auto-labeller.

Model choice is deliberate, not the usual default: self-consistency
sampling (silver/sampling.py) needs genuine temperature-driven diversity
across k samples, but explicit `temperature` is REJECTED (400) on the
newest model generation — Claude Opus 5, Claude Sonnet 5, and Claude Fable
5 all remove sampling controls when adaptive thinking is on (their
default). Only Opus 4.6 / Sonnet 4.6 and older still accept `temperature`.
So DEFAULT_MODEL here is "claude-sonnet-4-6", not the usual claude-opus-5
default — the feature (temperature-controlled sampling) dictates the
model, not the other way around. Override with --model if you want
claude-opus-4-6 (higher quality, same temperature support) or
claude-haiku-4-5 (cheaper, temperature-compatible) — anything from the
Opus-5/Sonnet-5/Fable-5 generation will 400 on the k=5 sampling calls.

Requires ANTHROPIC_API_KEY in the environment. Never wired into anything
that writes to gold/labels.jsonl — see silver/store.py.
"""
from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_MODEL = "claude-sonnet-4-6"
RED_TEAM_MODEL = "claude-sonnet-4-6"


class ModelClientError(RuntimeError):
    pass


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise ModelClientError(
            'The "silver" extra is required: pip install -e ".[silver]"'
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ModelClientError(
            "ANTHROPIC_API_KEY is not set. Export it before running the silver labeller — "
            "this is real API spend, so it's never assumed or defaulted."
        )
    return anthropic.Anthropic()


def complete(
    prompt: str, *, system: str | None = None, temperature: float = 1.0,
    max_tokens: int = 1024, model: str = DEFAULT_MODEL,
) -> str:
    """One completion call. Returns the concatenated text content.
    Raises ModelClientError on missing credentials/dependency; lets the
    SDK's own typed exceptions (RateLimitError, APIStatusError, ...)
    propagate for the caller to handle/retry."""
    client = _client()
    kwargs: dict = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return "".join(block.text for block in response.content if block.type == "text")


def extract_json(text: str) -> Optional[dict]:
    """Lenient JSON extraction from a completion's text: first '{' to
    last '}'. Prompt-based JSON (not output_config structured outputs,
    whose exact current shape isn't verified here) — a parse failure
    returns None rather than raising, so callers can treat an
    unparseable sample as evidence of nothing rather than crash a k=5 run
    over one bad sample."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
