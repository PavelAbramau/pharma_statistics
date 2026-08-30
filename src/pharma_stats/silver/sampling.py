"""Self-consistency sampling: the abstention signal.

Confidence comes from agreement across k independent samples at
non-zero temperature, not from a model's own stated confidence —
disagreement across samples IS the abstention trigger, not a fallback for
when it happens to look uncertain.

STUB: sample_answers needs an actual model-invocation mechanism, which
hasn't been chosen yet (scaffolding first, per the user's own call).
majority_vote and should_abstain are pure aggregation over already-drawn
samples and are fully implemented — they don't care how the samples were
produced, so they're ready for whatever sample_answers becomes.
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
    """Draw k independent samples of one decomposed question (see
    silver/questions.py) at non-zero temperature.

    NOT IMPLEMENTED: needs a model-invocation mechanism to be chosen
    first — see silver/__init__.py and the project decision log. Once
    chosen, this is the only function that needs to change; everything
    else in this module and silver/eval.py already works against its
    output shape (a list of T, one per sample)."""
    raise NotImplementedError(
        "sample_answers needs a model-invocation mechanism to be chosen before it can run"
    )
