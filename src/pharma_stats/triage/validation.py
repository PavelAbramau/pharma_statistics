"""Blind validation gate for the triage pipeline — strengthened once
Layer 2 started deciding the bulk of the universe (not just Layer 1's
small, purely-deterministic residue).

Sample size: 80 (up from an original 40), stratified across two axes:
  - decision basis: text-grounded (a real quote, from_recall=False) vs
    recall-based (from_recall=True) — equal counts of each, where enough
    of both exist.
  - decision direction: accept (is_adc=yes) vs reject (is_adc=no) — equal
    counts of each.

Mechanics mirror labelling/trial_scope.py's existing draw_validation_sample
/ VALIDATION_SAMPLE_PATH pattern for the heme_only auto-exclusion:
candidates drawn into the sample are withheld from any auto-commit and
served normally, blind, through the labelling app — the reviewer never
sees the triage verdict. Agreement is computed after the fact against
whatever the human independently decided in gold/labels.jsonl, per
stratum, never blended into one number — a classifier that rejects
everything scores well overall and is useless, and stratifying by
text-vs-recall is specifically how a "recall answers are less reliable"
hypothesis gets checked rather than assumed.

Same thresholds as the original 40-sample gate: below 95% on is_adc or
90% on in_scope, the WHOLE run is rejected and returns to manual — see
check_gate(). Nothing here writes to gold/labels.jsonl.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR
from pharma_stats.labelling import store as gold_store

VALIDATION_SAMPLE_SIZE = 80
IS_ADC_AGREEMENT_THRESHOLD = 0.95
IN_SCOPE_AGREEMENT_THRESHOLD = 0.90
VALIDATION_SAMPLE_PATH = DATA_DIR / "triage_validation_sample.json"


def _stratum(decision: dict) -> tuple[str, str]:
    basis = "recall" if decision.get("from_recall") else "text"
    direction = "accept" if decision.get("is_adc") == "yes" else "reject"
    return basis, direction


def draw_stratified_sample(
    decisions: list[dict], *, sample_size: int = VALIDATION_SAMPLE_SIZE, seed: int = 0,
    already_reserved: Optional[set] = None,
) -> list[dict]:
    """decisions: [{"program_id", "is_adc", "from_recall", ...}, ...] —
    every Layer 2/3 decision made this run, before any commit. Draws up
    to sample_size//4 from each of the four (basis x direction) strata,
    keeping any already-reserved ids unconditionally (stable across
    reruns, same pattern as trial_scope.draw_validation_sample) and
    topping up evenly from whichever strata still have candidates."""
    already_reserved = already_reserved or set()
    by_stratum: dict[tuple, list[dict]] = {}
    for d in decisions:
        by_stratum.setdefault(_stratum(d), []).append(d)

    rng = random.Random(seed)
    kept = [d for d in decisions if d["program_id"] in already_reserved]
    kept_ids = {d["program_id"] for d in kept}
    per_stratum_target = sample_size // 4

    for stratum, pool in by_stratum.items():
        rng.shuffle(pool)
        already_in_stratum = sum(1 for d in kept if _stratum(d) == stratum)
        need = max(0, per_stratum_target - already_in_stratum)
        for d in pool:
            if need <= 0:
                break
            if d["program_id"] in kept_ids:
                continue
            kept.append(d)
            kept_ids.add(d["program_id"])
            need -= 1

    return kept


def load_validation_sample(path: Optional[Path] = None) -> list[dict]:
    path = path or VALIDATION_SAMPLE_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_validation_sample(sample: list[dict], path: Optional[Path] = None) -> None:
    path = path or VALIDATION_SAMPLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample, indent=2), encoding="utf-8")


def compute_agreement(sample: list[dict], gold_records: Optional[list[dict]] = None) -> dict:
    """Per-stratum and overall agreement between each sampled triage
    decision and the human's independent gold decision for that same
    program_id. A program never reached by a human yet is excluded from
    agreement entirely — never counted as either an agreement or a
    disagreement."""
    gold_records = gold_records if gold_records is not None else gold_store.load_records()
    gold_by_pid = gold_store.latest_by_program(gold_records)

    by_stratum: dict[tuple, dict] = {}
    is_adc_compared = is_adc_agree = 0
    in_scope_compared = in_scope_agree = 0

    for d in sample:
        gold = gold_by_pid.get(d["program_id"])
        if gold is None:
            continue
        stratum = _stratum(d)
        cell = by_stratum.setdefault(stratum, {"compared": 0, "agree": 0})

        triage_is_adc = "yes" if d.get("is_adc") == "yes" else "no"
        human_is_adc = "yes" if gold.get("is_adc") == "yes" else "no"
        cell["compared"] += 1
        is_adc_compared += 1
        if triage_is_adc == human_is_adc:
            cell["agree"] += 1
            is_adc_agree += 1

        if gold.get("gate_reached", 0) >= 2 and "in_scope" in d:
            in_scope_compared += 1
            if d.get("in_scope") == gold.get("in_scope"):
                in_scope_agree += 1

    return {
        "by_stratum": {
            f"{basis}/{direction}": {**cell, "agreement_rate": (cell["agree"] / cell["compared"]) if cell["compared"] else None}
            for (basis, direction), cell in by_stratum.items()
        },
        "is_adc": {"compared": is_adc_compared, "agree": is_adc_agree,
                   "agreement_rate": (is_adc_agree / is_adc_compared) if is_adc_compared else None},
        "in_scope": {"compared": in_scope_compared, "agree": in_scope_agree,
                     "agreement_rate": (in_scope_agree / in_scope_compared) if in_scope_compared else None},
    }


def check_gate(agreement: dict) -> tuple[bool, str]:
    """(passed, reason). Below threshold on EITHER axis rejects the WHOLE
    run — same all-or-nothing policy as the original 40-sample gate,
    checked against the stratified numbers here."""
    is_adc = agreement["is_adc"]
    in_scope = agreement["in_scope"]

    if is_adc["agreement_rate"] is None or is_adc["compared"] < VALIDATION_SAMPLE_SIZE:
        return False, (f"only {is_adc['compared']}/{VALIDATION_SAMPLE_SIZE} sampled decisions have a "
                        "human gold comparison yet — not enough to gate on")
    if is_adc["agreement_rate"] < IS_ADC_AGREEMENT_THRESHOLD:
        return False, f"is_adc agreement {is_adc['agreement_rate']:.1%} < {IS_ADC_AGREEMENT_THRESHOLD:.0%} threshold"
    if in_scope["agreement_rate"] is not None and in_scope["agreement_rate"] < IN_SCOPE_AGREEMENT_THRESHOLD:
        return False, (f"in_scope agreement {in_scope['agreement_rate']:.1%} < "
                        f"{IN_SCOPE_AGREEMENT_THRESHOLD:.0%} threshold")
    return True, "passed"


def recall_vs_text_gap(agreement: dict) -> Optional[dict]:
    """Explicit accept/reject-pooled comparison of text-grounded vs
    recall-based agreement, the one number the stratified sample exists
    to produce: if recall scores materially worse, Layer 2 should be
    restricted to text-grounded answers only (see layer2.route_to_layer3's
    docstring) — that's a decision for a human to make from this number,
    not something this module decides on its own."""
    by_stratum = agreement["by_stratum"]
    text_cells = [c for k, c in by_stratum.items() if k.startswith("text/") and c["compared"]]
    recall_cells = [c for k, c in by_stratum.items() if k.startswith("recall/") and c["compared"]]
    if not text_cells or not recall_cells:
        return None
    text_compared = sum(c["compared"] for c in text_cells)
    text_agree = sum(c["agree"] for c in text_cells)
    recall_compared = sum(c["compared"] for c in recall_cells)
    recall_agree = sum(c["agree"] for c in recall_cells)
    text_rate = text_agree / text_compared
    recall_rate = recall_agree / recall_compared
    return {
        "text_agreement_rate": text_rate, "text_compared": text_compared,
        "recall_agreement_rate": recall_rate, "recall_compared": recall_compared,
        "gap": text_rate - recall_rate,
    }
