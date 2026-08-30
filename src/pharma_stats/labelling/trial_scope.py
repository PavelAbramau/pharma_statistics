"""Trial- and asset-level scope classification from CT.gov MeSH data.

Minimum-viable indication normalisation (see discovery/mesh_categories.py
for why this exists and its limits). Three layers:

1. classify_trial: one trial's heme/solid/non_oncology/ambiguous verdict,
   from its conditionBrowseModule meshes+ancestors (never from condition
   text alone — see the module docstring in mesh_categories.py for the
   one deliberate exception, the named AMBIGUOUS_OVERRIDE_PHRASES).
2. classify_asset / spans_heme_and_solid: rolling trial verdicts up to the
   asset (== provisional program) level.
3. auto_scope_decision + the validation/kill-switch machinery: turning a
   confident heme_only verdict into an actual pre-filled (never silently
   final) scope decision, with a held-out random sample and an agreement
   check gating whether auto-exclusion is trusted at all.

Nothing here writes to gold/labels.jsonl directly — see
scripts/apply_auto_scope_exclusions.py for the only place that does, and
only after checking validation_agreement() clears AGREEMENT_THRESHOLD.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR
from pharma_stats.discovery.mesh_categories import AMBIGUOUS_OVERRIDE_PHRASES, category_for

AGREEMENT_THRESHOLD = 0.95
VALIDATION_SAMPLE_SIZE = 30

# Program ids held out from auto-exclusion so their human review can be
# compared blind against the classifier's prediction (see
# validation_agreement). Not gold data — a small, rebuildable index, same
# spirit as data/labelling_session.json.
VALIDATION_SAMPLE_PATH = DATA_DIR / "auto_scope_validation_sample.json"


def load_validation_sample(path: Optional[Path] = None) -> list[dict]:
    path = path or VALIDATION_SAMPLE_PATH  # resolved at call time, not import time, so it stays testable
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_validation_sample(sample: list[dict], path: Optional[Path] = None) -> None:
    path = path or VALIDATION_SAMPLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, indent=2, sort_keys=True))


def classify_trial(
    meshes: list[dict], ancestors: list[dict], conditions: list[str],
) -> str:
    """One trial's scope verdict: "heme" / "solid" / "non_oncology" /
    "ambiguous". Never guesses: absent MeSH data, MeSH IDs not yet in the
    dictionary, or a genuine heme+solid split all resolve to "ambiguous"
    rather than a guessed answer — see the module docstring for exactly
    what "fall back to text" does and doesn't mean here."""
    condition_text = " | ".join(conditions or []).lower()
    for phrase in AMBIGUOUS_OVERRIDE_PHRASES:
        if phrase in condition_text:
            return "ambiguous"

    ids = [m["id"] for m in (meshes or [])] + [a["id"] for a in (ancestors or [])]
    if not ids:
        return "ambiguous"  # no MeSH data at all — never guess from text alone

    categories = {category_for(i) for i in ids} - {None}
    if not categories:
        return "ambiguous"  # MeSH present, but none of it is in our dictionary yet

    specific = categories - {"generic_basket"}
    if "heme" in specific and "solid" in specific:
        return "ambiguous"  # spans both branches — see mesh_categories.py
    if specific == {"heme"}:
        return "heme"
    if specific == {"solid"}:
        return "solid"
    if specific == {"non_oncology"}:
        return "non_oncology"
    if not specific:
        return "ambiguous"  # only generic root terms — the all-comers/basket signature
    return "ambiguous"  # any other mixture (e.g. non_oncology + heme) — don't guess


def classify_asset(trial_classifications: list[str]) -> str:
    """"heme_only" iff every trial classified heme and there's at least
    one trial; "needs_review" otherwise. Any solid, ambiguous, or
    non_oncology trial — or no trials at all — keeps the asset in the
    manual queue."""
    if trial_classifications and all(c == "heme" for c in trial_classifications):
        return "heme_only"
    return "needs_review"


def spans_heme_and_solid(trial_classifications: list[str]) -> bool:
    """The mixed-evidence case a whole-asset in_scope=no would wrongly
    kill: at least one heme trial AND at least one solid trial."""
    return "heme" in trial_classifications and "solid" in trial_classifications


def is_non_oncology_asset(trial_classifications: list[str]) -> bool:
    return bool(trial_classifications) and all(c == "non_oncology" for c in trial_classifications)


def is_non_industry_sponsor(sponsors_over_time: list[dict]) -> bool:
    """Flag only — CLAUDE.md's owner-is-a-dated-interval model means an
    asset can have had more than one sponsor class over time; any
    non-INDUSTRY sponsor on file is enough to surface it for review."""
    return any((s.get("class") or "").upper() != "INDUSTRY" for s in (sponsors_over_time or []))


def auto_scope_decision(asset_category: str) -> Optional[dict]:
    """The pre-fill for a confidently heme_only asset. is_adc is
    deliberately absent: this is a pure scope call (this project is
    solid-tumours-only regardless of molecule type), so it neither needs
    nor asserts a molecule judgement — that stays entirely the reviewer's,
    whenever/if they look at this asset."""
    if asset_category != "heme_only":
        return None
    return {"in_scope": "no", "scope_reason": "heme_only", "decided_by": "auto"}


def draw_validation_sample(
    candidate_ids: list[str], already_reserved: set[str],
    target_size: int = VALIDATION_SAMPLE_SIZE, seed: int = 0,
) -> list[str]:
    """Stable holdout: every previously-reserved id is kept UNCONDITIONALLY
    (even one that's since been reviewed and dropped out of
    `candidate_ids` — its agreement data must not be lost), then topped up
    to target_size from `candidate_ids` if more unreviewed heme_only
    assets exist than are currently reserved. Deterministic given the same
    inputs, so reruns without new candidates reproduce the same set."""
    kept = sorted(already_reserved)
    need = target_size - len(kept)
    if need <= 0:
        return kept
    pool = [c for c in candidate_ids if c not in already_reserved]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return kept + pool[:need]


def validation_agreement(sample: list[dict], records: list[dict]) -> dict:
    """Agreement rate between the classifier's prediction for each held-out
    sample member and the reviewer's own (blind — never shown the
    prediction) decision, once they've actually reached a scope call.
    `sample` is [{"program_id", "predicted_in_scope", "predicted_scope_reason"}, ...].
    Members not yet reviewed past Gate 1 aren't counted — there's nothing
    to compare yet, not a disagreement."""
    from pharma_stats.labelling.store import latest_by_program

    latest = latest_by_program(records)
    compared = 0
    agreements = 0
    for item in sample:
        r = latest.get(item["program_id"])
        if r is None or r.get("gate_reached") not in (2, 3):
            continue  # never reached a scope decision yet
        human_in_scope = r.get("in_scope")
        predicted_in_scope = item["predicted_in_scope"]
        compared += 1
        if human_in_scope != predicted_in_scope:
            continue
        if predicted_in_scope == "no" and r.get("scope_reason") != item.get("predicted_scope_reason"):
            continue
        agreements += 1
    rate = (agreements / compared) if compared else None
    return {"compared": compared, "agreements": agreements, "agreement_rate": rate}
