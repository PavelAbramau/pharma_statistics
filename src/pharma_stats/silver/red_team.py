"""Red Team objection agent: an adversarial check that runs AFTER
self-consistency sampling agrees on a label, not instead of it. Given a
silver label and its supporting evidence, generates the strongest
evidenced case that the label is wrong; a strong, evidenced objection
forces abstention regardless of how confidently the k samples agreed.

STUB: generate_objection needs the same model-invocation decision as
silver/sampling.py. forces_abstention is pure policy over an already-
generated Objection and is fully implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pharma_stats.silver.questions import Citation

OBJECTION_STRENGTHS = ["weak", "moderate", "strong"]


@dataclass
class Objection:
    strength: str  # one of OBJECTION_STRENGTHS
    argument: str
    citations: list[Citation] = field(default_factory=list)


def forces_abstention(objection: Objection) -> bool:
    """A strong objection with no evidence behind it doesn't count — the
    whole point is an EVIDENCED case, not just a model expressing doubt."""
    return objection.strength == "strong" and bool(objection.citations)


def generate_objection(label: dict, evidence: dict) -> Objection:
    """NOT IMPLEMENTED: needs a model-invocation mechanism to be chosen
    first — see silver/sampling.py and silver/__init__.py."""
    raise NotImplementedError(
        "generate_objection needs a model-invocation mechanism to be chosen before it can run"
    )
