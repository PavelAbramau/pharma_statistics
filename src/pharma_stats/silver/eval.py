"""Strict evaluation for the silver auto-labeller against gold.

- split_by_asset: partitions GOLD (gate 3) labels into a few-shot pool and
  a held-out evaluation set, grouped by asset (candidate_id — today the
  same as program_id, since the real five-entity Program model doesn't
  exist yet; grouping by candidate_id is what keeps this correct once it
  does, when one asset splits into several programs) so no asset's
  programs appear in both halves. An eval program must never appear in a
  prompt.
- per_field_accuracy: status / kill_reason / public_confirmation_date
  scored SEPARATELY, since they're expected to differ a lot.
- self_consistency_ceiling: re-exposes labelling.stats.self_consistency so
  "agreement vs the labeller's own ceiling" is the explicit comparison,
  not agreement vs 100%.
"""
from __future__ import annotations

import random
from typing import Optional

from pharma_stats.labelling.stats import self_consistency as self_consistency_ceiling  # noqa: F401
from pharma_stats.labelling.store import fully_labelled_program_ids, latest_by_program

ACCURACY_FIELDS = ("status", "kill_reason", "public_confirmation_date")


def split_by_asset(
    gold_records: list[dict], eval_fraction: float = 0.3, seed: int = 0,
) -> tuple[list[str], list[str]]:
    """(few_shot_program_ids, eval_program_ids) over GOLD gate-3 records
    only — a gate-1/2 triage rejection was never a program (see
    labelling/store.py) and has no status/kill_reason/date to evaluate
    against. Split at the asset (candidate_id) level: every program
    belonging to one asset lands entirely in one half."""
    fully_labelled = fully_labelled_program_ids(gold_records)
    latest = latest_by_program(gold_records)

    by_asset: dict[str, list[str]] = {}
    for pid in fully_labelled:
        r = latest[pid]
        asset_id = r.get("candidate_id") or pid
        by_asset.setdefault(asset_id, []).append(pid)

    asset_ids = sorted(by_asset)
    rng = random.Random(seed)
    rng.shuffle(asset_ids)
    n_eval_assets = round(len(asset_ids) * eval_fraction)
    eval_assets = set(asset_ids[:n_eval_assets])

    few_shot_ids, eval_ids = [], []
    for asset_id in asset_ids:
        target = eval_ids if asset_id in eval_assets else few_shot_ids
        target.extend(by_asset[asset_id])
    return sorted(few_shot_ids), sorted(eval_ids)


def per_field_accuracy(
    predictions: dict[str, dict], gold_records: list[dict], program_ids: list[str],
    fields: tuple = ACCURACY_FIELDS,
) -> dict[str, dict]:
    """predictions: {program_id: {"status":..., "kill_reason":..., "public_confirmation_date":...}}
    — only ever the eval-set program_ids, and only ever compared against
    gold that was never in the prompt (caller's responsibility via
    split_by_asset). A prediction of None/"not_determinable" counts as an
    abstention: excluded from the accuracy denominator, tracked
    separately, never scored as wrong OR right."""
    gold_by_pid = latest_by_program(gold_records)
    out = {}
    for field in fields:
        n_compared = n_correct = n_abstained = 0
        for pid in program_ids:
            gold = gold_by_pid.get(pid)
            pred = predictions.get(pid)
            if gold is None or pred is None:
                continue
            value = pred.get(field)
            if value is None or value == "not_determinable":
                n_abstained += 1
                continue
            n_compared += 1
            if value == gold.get(field):
                n_correct += 1
        out[field] = {
            "n_compared": n_compared,
            "n_correct": n_correct,
            "n_abstained": n_abstained,
            "accuracy": (n_correct / n_compared) if n_compared else None,
        }
    return out


def accuracy_vs_self_consistency_ceiling(accuracy: Optional[float], ceiling: Optional[float]) -> dict:
    """The comparison this project actually cares about: not "how close to
    100%" but "how close to the labeller's own disagreement-with-himself
    rate" — that rate is a hard ceiling on any model trained/evaluated
    against these labels, so beating it is not a real signal."""
    if accuracy is None or ceiling is None:
        return {"comparable": False, "at_or_above_ceiling": None, "gap": None}
    return {
        "comparable": True,
        "at_or_above_ceiling": accuracy >= ceiling,
        "gap": accuracy - ceiling,
    }
