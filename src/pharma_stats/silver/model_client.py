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
Opus-5/Sonnet-5/Fable-5 generation will 400 on the k-sampling calls.

Requires ANTHROPIC_API_KEY in the environment. Never wired into anything
that writes to gold/labels.jsonl — see silver/store.py.

Resilience: a single anthropic.APITimeoutError used to kill an entire run
at program 1 of 5 (see scripts/run_silver_labelling.py's per-program try/
except, which now logs and continues instead of sys.exit(1)). Two layers
of defense here: the SDK client itself gets max_retries=5 and a longer
connect timeout (so transient network hiccups resolve before ever
reaching caller code), and every remaining anthropic.APIError is
re-wrapped as ModelClientError so callers have exactly one exception type
to catch, never the SDK's internal hierarchy.
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional

DEFAULT_MODEL = "claude-sonnet-4-6"
RED_TEAM_MODEL = "claude-sonnet-4-6"

# $ per 1M tokens, standard (non-batch) pricing. Batch API is 50% of these —
# see BATCH_DISCOUNT. Extend by hand as new models are used here; an
# unlisted model raises rather than silently estimating $0.
PRICING_PER_MILLION_TOKENS = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
BATCH_DISCOUNT = 0.5

# SDK client resilience — a longer connect timeout and more retries than
# the SDK default (10 min / max_retries=2), so a transient network hiccup
# doesn't take down a run spanning hours. See module docstring.
CLIENT_MAX_RETRIES = 5
CLIENT_TIMEOUT_SECONDS = 900.0
CLIENT_CONNECT_TIMEOUT_SECONDS = 30.0


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int
    model: str

    def cost_usd(self) -> float:
        return estimate_cost(self.input_tokens, self.output_tokens, self.model)


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
    return anthropic.Anthropic(
        max_retries=CLIENT_MAX_RETRIES,
        timeout=anthropic.Timeout(CLIENT_TIMEOUT_SECONDS, connect=CLIENT_CONNECT_TIMEOUT_SECONDS),
    )


def estimate_cost(input_tokens: int, output_tokens: int, model: str, *, batch: bool = False) -> float:
    if model not in PRICING_PER_MILLION_TOKENS:
        raise ModelClientError(
            f"no pricing on file for model {model!r} — add it to "
            f"model_client.PRICING_PER_MILLION_TOKENS before estimating cost for it"
        )
    rates = PRICING_PER_MILLION_TOKENS[model]
    cost = (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
    return cost * BATCH_DISCOUNT if batch else cost


def count_tokens(prompt: str, *, system: Optional[str] = None, model: str = DEFAULT_MODEL) -> int:
    """Exact input token count via the Messages API's count_tokens
    endpoint — never a char-count/tiktoken approximation, which silently
    drifts from what's actually billed. Used by --dry-run's cost estimate;
    requires ANTHROPIC_API_KEY (a token-counting call is free, but still a
    real API round trip)."""
    client = _client()
    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.count_tokens(**kwargs)
    except Exception as e:
        raise ModelClientError(f"count_tokens failed: {e}") from e
    return response.input_tokens


def complete(
    prompt: str, *, system: Optional[str] = None, temperature: float = 1.0,
    max_tokens: int = 1024, model: str = DEFAULT_MODEL,
) -> tuple[str, Usage]:
    """One completion call. Returns (text, Usage) — every caller gets
    actual token usage for cost logging, not just the text. Raises
    ModelClientError on missing credentials/dependency, and re-wraps every
    anthropic.APIError (timeouts, connection failures, rate limits, 5xx —
    the whole SDK exception hierarchy) as ModelClientError too, so a
    caller only ever needs to catch one exception type. 4xx errors that
    mean a real bug in the request (BadRequestError, AuthenticationError,
    NotFoundError) still surface distinctly in the wrapped message rather
    than being silently retried past — the SDK's own max_retries only
    retries what it considers retryable (429/5xx/connection errors)."""
    import anthropic

    client = _client()
    kwargs: dict = {
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    try:
        response = client.messages.create(**kwargs)
    except anthropic.APIError as e:
        raise ModelClientError(f"{type(e).__name__}: {e}") from e
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=model,
    )
    return text, usage


def submit_batch(requests: list[dict], *, model: str = DEFAULT_MODEL) -> str:
    """Real Message Batches API submission — 50% cost discount,
    asynchronous (usually <1hr, up to 24hr per Anthropic's own SLA).
    requests: [{"custom_id", "prompt", "system"?, "temperature"?,
    "max_tokens"?, "model"?, "tools"?}, ...]. Returns the batch id; poll
    with poll_batch_until_done, then collect_batch_results."""
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()
    batch_requests = []
    for r in requests:
        params: dict = {
            "model": r.get("model", model), "max_tokens": r.get("max_tokens", 1024),
            "temperature": r.get("temperature", 1.0),
            "messages": [{"role": "user", "content": r["prompt"]}],
        }
        if r.get("system"):
            params["system"] = r["system"]
        if r.get("tools"):
            params["tools"] = r["tools"]
        batch_requests.append(Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**params)))
    try:
        batch = client.messages.batches.create(requests=batch_requests)
    except anthropic.APIError as e:
        raise ModelClientError(f"{type(e).__name__}: {e}") from e
    return batch.id


def poll_batch_until_done(
    batch_id: str, *, poll_interval_seconds: float = 30.0, max_wait_seconds: float = 24 * 3600,
) -> dict:
    """Blocks until the batch's processing_status is "ended" (succeeded,
    errored, canceled, and expired results are all resolved individually
    per-request — see collect_batch_results). Raises ModelClientError if
    max_wait_seconds elapses first (Anthropic's own SLA is up to 24h)."""
    import time

    client = _client()
    waited = 0.0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            return {
                "processing_status": "ended",
                "succeeded": getattr(counts, "succeeded", None),
                "errored": getattr(counts, "errored", None),
                "canceled": getattr(counts, "canceled", None),
                "expired": getattr(counts, "expired", None),
            }
        if waited >= max_wait_seconds:
            raise ModelClientError(f"batch {batch_id} did not complete within {max_wait_seconds}s")
        time.sleep(poll_interval_seconds)
        waited += poll_interval_seconds


def collect_batch_results(batch_id: str, *, model: str = DEFAULT_MODEL) -> dict:
    """{custom_id: (text, Usage)} on success; a custom_id whose request
    errored/canceled/expired is omitted (caller treats a missing custom_id
    as "not determinable", never guesses a value for it). Results arrive
    in any order per Anthropic's own docs — never assume by position."""
    client = _client()
    out: dict = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            msg = result.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            usage = Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens, model=model)
            out[result.custom_id] = (text, usage)
    return out


def extract_json_array(text: str) -> Optional[list]:
    """Same lenient-extraction contract as extract_json, for a batched
    call whose response is a JSON array (one object per candidate) rather
    than a single object — see pharma_stats.triage.layer2."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> Optional[dict]:
    """Lenient JSON extraction from a completion's text: first '{' to
    last '}'. Prompt-based JSON (not output_config structured outputs,
    whose exact current shape isn't verified here) — a parse failure
    returns None rather than raising, so callers can treat an
    unparseable sample as evidence of nothing rather than crash a whole
    sampling run over one bad sample."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
