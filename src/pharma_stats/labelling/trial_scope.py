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

import html
import json
import random
import re
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR, REPO_ROOT
from pharma_stats.discovery.mesh_categories import (
    AMBIGUOUS_OVERRIDE_PHRASES, HEME_TEXT_HINT_KEYWORDS, category_for,
)

_HEME_TEXT_HINT_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in HEME_TEXT_HINT_KEYWORDS) + r")\b", re.IGNORECASE,
)

AGREEMENT_THRESHOLD = 0.95
VALIDATION_SAMPLE_SIZE = 30

# Below this, a heme_only/spans-both count is coverage noise, not a real
# result — see docs/decisions/0001-current-state-fetch-scope.md. Single
# source of truth for audit/universe.py's gate and
# scripts/apply_auto_scope_exclusions.py's own refusal to run below it.
MESH_COVERAGE_THRESHOLD = 0.90

# Program ids held out from auto-exclusion so their human review can be
# compared blind against the classifier's prediction (see
# validation_agreement). Not gold data — a small, rebuildable index, same
# spirit as data/labelling_session.json.
VALIDATION_SAMPLE_PATH = DATA_DIR / "auto_scope_validation_sample.json"

# Hand-adjudicated corrections to CT.gov's sponsor-selected leadSponsor.class
# field, keyed by exact sponsor name. That field is self-reported and gets
# miscoded in both directions (e.g. "Shanghai Institute Of Biological
# Products" — a state-owned commercial biologics manufacturer under
# Sinopharm/CNBG — is classed OTHER despite the academic-sounding name).
# Reviewable, hand-authored, at repo root (like gold/) — never guessed or
# auto-populated; see scripts/report_sponsor_class_candidates.py, which
# only ever proposes candidates for a human to adjudicate here. Takes
# precedence over the raw field everywhere via apply_sponsor_class_overrides.
SPONSOR_CLASS_OVERRIDES_PATH = REPO_ROOT / "sponsor_class_overrides.json"


def load_validation_sample(path: Optional[Path] = None) -> list[dict]:
    path = path or VALIDATION_SAMPLE_PATH  # resolved at call time, not import time, so it stays testable
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_validation_sample(sample: list[dict], path: Optional[Path] = None) -> None:
    path = path or VALIDATION_SAMPLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, indent=2, sort_keys=True))


def has_mesh_data(meshes: list[dict], ancestors: list[dict]) -> bool:
    """Whether this trial has ANY conditionBrowseModule signal at all —
    the coverage denominator for the MeSH-coverage gate (audit/universe.py)
    and the trigger for the text_hint_category fallback below. Distinct
    from classify_trial's "ambiguous", which also fires when MeSH IS
    present but isn't in the dictionary yet — that's a dictionary gap, not
    a coverage gap, and the two must not be conflated when reporting
    coverage."""
    return bool(meshes) or bool(ancestors)


def text_hint_category(conditions: list[str]) -> Optional[str]:
    """Free-text fallback for trials that still have no MeSH data after
    the current-state fetch (docs/decisions/0001-current-state-fetch-scope.md).
    Returns "heme" or None — NEVER "solid" or anything else, and NEVER
    consulted by classify_trial/classify_asset/auto_scope_decision. This
    is sort-queue-priority signal only: good enough to triage faster, not
    good enough to decide scope. A trial this flags stays "ambiguous" in
    trial_scope and can never be auto-excluded."""
    text = " | ".join(conditions or [])
    return "heme" if _HEME_TEXT_HINT_RE.search(text) else None


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


def load_sponsor_class_overrides(path: Optional[Path] = None) -> dict:
    path = path or SPONSOR_CLASS_OVERRIDES_PATH  # resolved at call time, not import time, so it stays testable
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_sponsor_class_overrides(
    sponsors_over_time: list[dict], overrides: Optional[dict] = None,
) -> list[dict]:
    """Enrich each sponsor entry with effective_class (the override's class
    if one is on file for this exact sponsor name, else CT.gov's raw
    class) and class_overridden (whether one was applied) — without ever
    mutating the raw "class" field, so the original CT.gov value stays on
    record. Every consumer of sponsor class (is_non_industry_sponsor, the
    review screen) must read effective_class, never class directly.

    Matched on the HTML-unescaped name: the same sponsor's name shows up
    both escaped and not (e.g. "Beijing Children&#x27;s Hospital" vs
    "Beijing Children's Hospital") across different raw CT.gov snapshots
    of the same trial — a real inconsistency in the registry's own data,
    not something this project introduces. Matching on the raw string
    would silently miss the override for whichever variant a human didn't
    happen to type in sponsor_class_overrides.json."""
    overrides = load_sponsor_class_overrides() if overrides is None else overrides
    overrides_by_unescaped_name = {html.unescape(name): payload for name, payload in overrides.items()}
    enriched = []
    for s in sponsors_over_time or []:
        s = dict(s)
        override = overrides_by_unescaped_name.get(html.unescape(s.get("sponsor") or ""))
        raw_class = (s.get("class") or "").upper() or None
        s["effective_class"] = override["override_class"].upper() if override else raw_class
        s["class_overridden"] = bool(override)
        enriched.append(s)
    return enriched


def is_non_industry_sponsor(sponsors_over_time: list[dict]) -> bool:
    """Flag only, for the review screen's pre-fill hint — CLAUDE.md's
    owner-is-a-dated-interval model means an asset can have had more than
    one sponsor class over time; any non-INDUSTRY sponsor on file is
    enough to surface it for a human to look at. Reads effective_class
    when present (post sponsor_class_overrides.json correction), falling
    back to the raw CT.gov class otherwise.

    Deliberately loose ("any") — never use this for an automatic
    in_scope=no decision. A real industry ADC with an investigator-
    initiated side-trial led by a hospital would have exactly one
    non-industry sponsor on file and still be wrongly rejected; see
    pharma_stats.triage.deterministic, which uses
    is_all_sponsors_non_industry instead."""
    return any(
        (s.get("effective_class") or s.get("class") or "").upper() != "INDUSTRY"
        for s in (sponsors_over_time or [])
    )


def is_all_sponsors_industry(sponsors_over_time: list[dict]) -> bool:
    """The positive mirror of is_all_sponsors_non_industry — every sponsor
    this asset has ever had is confidently INDUSTRY. Used for a positive
    in_scope=yes rule (pharma_stats.triage.deterministic), not just a
    "no rejection fired" default. False with no sponsors on file — no
    evidence is not evidence of eligibility either."""
    sponsors = sponsors_over_time or []
    if not sponsors:
        return False
    return all(
        (s.get("effective_class") or s.get("class") or "").upper() == "INDUSTRY"
        for s in sponsors
    )


def is_all_sponsors_non_industry(sponsors_over_time: list[dict]) -> bool:
    """Auto-decision strength — every sponsor this asset has ever had is
    non-INDUSTRY, not just some. Conservative on purpose: an asset with a
    real industry sponsor plus an academic collaborator on a side-trial
    must not be auto-rejected here (is_non_industry_sponsor's "any"
    semantics would wrongly catch it — confirmed against real data: 24
    INN-suffix-verified ADCs would be misclassified non_industry under
    "any" but not under "all"). False (never reject) when there are no
    sponsors on file at all — no evidence is not evidence of exclusion."""
    sponsors = sponsors_over_time or []
    if not sponsors:
        return False
    return all(
        (s.get("effective_class") or s.get("class") or "").upper() != "INDUSTRY"
        for s in sponsors
    )


def mesh_coverage(programs: list[dict]) -> dict:
    """Fraction of in-universe trials with any conditionBrowseModule data
    at all (trial_has_mesh, from provisional_programs.build_program) —
    the single source of truth for both audit/universe.py's coverage gate
    and scripts/apply_auto_scope_exclusions.py's own refusal to run below
    MESH_COVERAGE_THRESHOLD. Distinct from a trial classifying "ambiguous"
    for a dictionary gap instead of a coverage gap — see has_mesh_data."""
    total = covered = 0
    for p in programs:
        for has_mesh in (p.get("trial_has_mesh") or {}).values():
            total += 1
            covered += int(bool(has_mesh))
    rate = (covered / total) if total else 0.0
    return {"covered": covered, "total": total, "coverage_rate": rate}


def asset_scope_bucket(trial_classifications: list[str]) -> str:
    """Roll a trial-level heme/solid/non_oncology/ambiguous list up to one
    asset bucket. Distinct from classify_asset, which only ever returns
    heme_only vs needs_review — the auto-exclusion rule only cares about
    that binary. This function is for the coverage report: it says *why*
    an asset is needs_review (all-ambiguous vs mixed vs spanning both)."""
    if not trial_classifications:
        return "no_trials"
    unique = set(trial_classifications)
    if unique == {"heme"}:
        return "heme_only"
    if unique == {"solid"}:
        return "solid_only"
    if "heme" in unique and "solid" in unique:
        return "both"
    if unique == {"ambiguous"}:
        return "all_ambiguous"
    if unique == {"non_oncology"}:
        return "non_oncology_only"
    return "mixed_other"


def scope_distribution(programs: list[dict]) -> dict:
    """Universe-wide MeSH classification mix, trial-level and asset-level.

    `ambiguous` at trial level is NOT the inverse of mesh_coverage:
    coverage is "has any conditionBrowseModule data"; ambiguous also
    fires when MeSH is present but the dictionary can't classify it
    (gap, heme+solid mix, generic basket terms). If ambiguous dominates,
    heme_only auto-exclusion (which requires ALL trials to classify heme)
    is weaker than a high coverage rate makes it look — it under-excludes
    rather than over-excludes, but the qualifying set is coverage noise
    plus dictionary-gap noise, not a clean heme cohort.
    """
    trial_counts = {"heme": 0, "solid": 0, "non_oncology": 0, "ambiguous": 0, "total": 0}
    ambiguous_with_mesh = 0
    ambiguous_without_mesh = 0
    asset_counts = {
        "heme_only": 0, "solid_only": 0, "both": 0, "all_ambiguous": 0,
        "non_oncology_only": 0, "mixed_other": 0, "no_trials": 0,
    }
    for p in programs:
        scope = p.get("trial_scope") or {}
        has_mesh = p.get("trial_has_mesh") or {}
        classes = list(scope.values())
        for nct, cat in scope.items():
            if cat not in trial_counts:
                trial_counts[cat] = 0
            trial_counts[cat] += 1
            trial_counts["total"] += 1
            if cat == "ambiguous":
                if has_mesh.get(nct):
                    ambiguous_with_mesh += 1
                else:
                    ambiguous_without_mesh += 1
        bucket = asset_scope_bucket(classes)
        asset_counts[bucket] = asset_counts.get(bucket, 0) + 1
    trial_total = trial_counts["total"] or 0
    rates = {
        k: (trial_counts.get(k, 0) / trial_total if trial_total else 0.0)
        for k in ("heme", "solid", "non_oncology", "ambiguous")
    }
    return {
        "n_programs": len(programs),
        "trials": {
            **trial_counts,
            "rates": rates,
            "ambiguous_with_mesh": ambiguous_with_mesh,
            "ambiguous_without_mesh": ambiguous_without_mesh,
        },
        "assets": asset_counts,
        "ambiguous_dominates_trials": bool(
            trial_total and trial_counts["ambiguous"] > trial_total / 2
        ),
    }


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
