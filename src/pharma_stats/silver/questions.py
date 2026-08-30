"""The four decomposed questions the silver auto-labeller answers, instead
of one big "is this program dead" judgement call — plus the deterministic
rule that combines answers into a candidate silver label.

    1. Has the sponsor initiated any trial on this asset since date X?
    2. Does a public statement of discontinuation exist? (date + URL)
    3. Does the stated stop reason cite efficacy, safety, or business?
    4. Is there a successor asset from the same sponsor with the same target?

Each is narrow and independently checkable, and each can answer
NOT_DETERMINABLE — abstention is the point, not a fallback. Answering
these questions (by a human, by sampling a model, or by hand for tests) is
out of scope for this module; apply_deterministic_rules only combines
whatever answers it's given, and never guesses past a gap in them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

NOT_DETERMINABLE = "not_determinable"

Trit = Union[bool, str]  # True / False / NOT_DETERMINABLE

STOP_REASON_CATEGORIES = ["efficacy", "safety", "business", NOT_DETERMINABLE]
_KILL_REASON_BY_STOP_CATEGORY = {
    "efficacy": "futility_efficacy",
    "safety": "toxicity_safety",
    "business": "strategic_portfolio",
}


@dataclass
class Citation:
    """A specific evidenced source for an answer. Every answer that isn't
    NOT_DETERMINABLE must carry at least one of these — see
    silver/citations.py for the two-stage resolve-then-verify gate that
    keeps a fabricated one from ever backing a record."""
    source_type: str  # "raw_snapshot" | "fetched_url"
    locator: str        # "ctgov:NCT01234567[:vN]" or a URL
    quote: str           # the exact substring the answer relies on


@dataclass
class TrialInitiatedSinceAnswer:
    value: Trit
    since_date: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class DiscontinuationStatementAnswer:
    exists: Trit
    statement_date: Optional[str] = None
    url: Optional[str] = None
    citations: list[Citation] = field(default_factory=list)


@dataclass
class StopReasonAnswer:
    category: str  # one of STOP_REASON_CATEGORIES
    citations: list[Citation] = field(default_factory=list)


@dataclass
class SuccessorAssetAnswer:
    exists: Trit
    successor_name: Optional[str] = None
    citations: list[Citation] = field(default_factory=list)


@dataclass
class DecomposedAnswers:
    trial_initiated_since: TrialInitiatedSinceAnswer
    discontinuation_statement: DiscontinuationStatementAnswer
    stop_reason: StopReasonAnswer
    successor_asset: SuccessorAssetAnswer


def _abstain(reason: str, rule_path: str) -> dict:
    return {
        "status": None, "kill_reason": None, "public_confirmation_date": None,
        "never_publicly_confirmed": False, "abstain": True, "abstain_reason": reason,
        "rule_path": rule_path,
    }


def apply_deterministic_rules(answers: DecomposedAnswers) -> dict:
    """Pure function: decomposed answers -> a candidate silver label
    ({"status", "kill_reason", "public_confirmation_date",
    "never_publicly_confirmed", "abstain", "abstain_reason", "rule_path"}).
    rule_path names exactly which branch fired — part of "the deterministic
    rule path taken" a silver record must log. Any NOT_DETERMINABLE answer
    that a rule actually needs forces abstention — this never guesses past
    a gap in the evidence, and self-consistency disagreement
    (silver/sampling.py) is a second, independent abstention trigger
    layered on top of this one."""
    statement = answers.discontinuation_statement
    trial_since = answers.trial_initiated_since.value
    successor = answers.successor_asset

    if statement.exists == NOT_DETERMINABLE or trial_since == NOT_DETERMINABLE:
        return _abstain(
            "core evidence (discontinuation statement or new-trial check) not determinable",
            "abstain:core_evidence_not_determinable",
        )

    if statement.exists is True:
        reason = answers.stop_reason.category
        if reason == NOT_DETERMINABLE:
            return _abstain(
                "discontinuation confirmed but stop reason not determinable",
                "abstain:stop_reason_not_determinable",
            )
        return {
            "status": "dead_confirmed",
            "kill_reason": _KILL_REASON_BY_STOP_CATEGORY.get(reason, "unknown_silent"),
            "public_confirmation_date": statement.statement_date,
            "never_publicly_confirmed": False,
            "abstain": False,
            "rule_path": f"dead_confirmed:{reason}",
        }

    # statement.exists is False from here — no public discontinuation on file
    if successor.exists == NOT_DETERMINABLE and trial_since is False:
        return _abstain(
            "no new trial and no discontinuation statement, but successor-asset check "
            "not determinable — can't tell superseded from dormant",
            "abstain:successor_not_determinable",
        )

    if trial_since is False and successor.exists is True:
        return {
            "status": "superseded", "kill_reason": None, "public_confirmation_date": None,
            "never_publicly_confirmed": False, "abstain": False, "rule_path": "superseded",
        }

    if trial_since is False and successor.exists is False:
        return {
            "status": "dormant_suspected", "kill_reason": None, "public_confirmation_date": None,
            "never_publicly_confirmed": False, "abstain": False, "rule_path": "dormant_suspected",
        }

    if trial_since is True:
        return {
            "status": "active", "kill_reason": None, "public_confirmation_date": None,
            "never_publicly_confirmed": False, "abstain": False, "rule_path": "active",
        }

    return _abstain("answers don't cleanly resolve under the deterministic rule set", "abstain:unresolved")
