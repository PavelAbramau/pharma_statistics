"""Self-consistency sampling: the abstention signal.

Confidence comes from agreement across k independent samples at
non-zero temperature, not from a model's own stated confidence —
disagreement across samples IS the abstention trigger, not a fallback for
when it happens to look uncertain.

Wired to the Anthropic API (silver/model_client.py) via silver/prompts.py,
which builds question_fn as a closure over one fully-formed prompt at a
fixed temperature/model — sample_answers itself stays model-agnostic,
calling question_fn k times and nothing else.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_K = 5


def majority_vote(samples: list[T]) -> tuple[T, int]:
    """Pure aggregation over k already-drawn samples. Ties resolve to
    whichever should_abstain() decides — this just reports the top value
    and its count, it doesn't itself decide pass/abstain."""
    if not samples:
        raise ValueError("majority_vote requires at least one sample")
    value, count = Counter(samples).most_common(1)[0]
    return value, count


def should_abstain(samples: list[T]) -> bool:
    """Any disagreement across the k samples abstains. Deliberately strict
    (unanimity, not majority) by default — loosen only as an explicit,
    considered policy choice at wiring time, not silently."""
    if not samples:
        return True
    _, count = majority_vote(samples)
    return count < len(samples)


def sample_answers(question_fn: Callable[[], T], k: int = DEFAULT_K, temperature: float = 0.7) -> list[T]:
    """Draw k independent samples of one decomposed question by calling
    question_fn k times. `temperature` is accepted for interface/logging
    parity with the design (it's documented as part of the sampling
    contract) but not applied here — question_fn (built by
    silver/prompts.py) is a closure that already carries its own bound
    prompt/temperature/model, so the actual API call and its temperature
    live there, not in this generic aggregation loop."""
    return [question_fn() for _ in range(k)]
