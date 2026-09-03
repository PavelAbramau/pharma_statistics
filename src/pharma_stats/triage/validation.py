"""Blind validation gate for the triage pipeline — strengthened once
Layer 2 started deciding the bulk of the universe (not just Layer 1's
small, purely-deterministic residue).

Design intent: 80 candidates, stratified 2x2 across decision basis
(text-grounded vs recall-grounded) x direction (accept vs reject), ~20
per cell, drawn from Layer 2/3 decisions ONLY — Layer 1 is deterministic
and already in gold, so including it would let the sample tautologically
agree with itself (see assert_no_layer1).

That design does not survive contact with the real data on this run, and
this module says so loudly instead of shipping a degenerate draw again
(the previous version silently returned 40 instead of 80 with an empty
recall cell). Root cause: `from_recall` is written False on every staged
Layer 2/3 record — Layer 2 only ever stages an answer that already
cleared grounding (see grounding.apply_grounding), and Layer 3's own
answers report from_recall=False even when it found nothing (a real bug
upstream, not something this module can retroactively fix on already-
staged data). So "recall-grounded" per the literal flag never occurs.

The real analogue is grounding.evidence_source's "no_usable_evidence":
a yes/no committed with NO supporting quote at all. Checked against the
actual staged data, that bucket has only 13 members, all Layer 3, all
"reject" — 0 "accept". Too few and too lopsided to be a 20-per-cell
stratum on its own. check_stratum_feasibility() detects exactly this and
propose_substitute_stratification() proposes the real substitute: source
(Layer 2 registry-text vs Layer 3 web-search-text) x direction, which has
healthy population on all four cells — plus surfaces the 13
no_usable_evidence decisions as a separate, explicitly-labelled appendix
rather than dropping them.

Same thresholds regardless of which stratification is actually drawn:
below 95% on is_adc or 90% on in_scope, the WHOLE run is rejected and
returns to manual — see check_gate(). Nothing here writes to
gold/labels.jsonl.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR
from pharma_stats.labelling import store as gold_store
from pharma_stats.triage import grounding

VALIDATION_SAMPLE_SIZE = 80
IS_ADC_AGREEMENT_THRESHOLD = 0.95
IN_SCOPE_AGREEMENT_THRESHOLD = 0.90
VALIDATION_SAMPLE_PATH = DATA_DIR / "triage_validation_sample.json"

# The four cells the ORIGINAL design expects. "no_usable_evidence" (see
# module docstring) is deliberately NOT one of them — it doesn't fit the
# text/recall dichotomy the design asked for; it's reported separately.
EXPECTED_STRATA = [("text", "accept"), ("text", "reject"), ("recall", "accept"), ("recall", "reject")]


class StratificationError(RuntimeError):
    pass


def decided_layer23_only(decisions: list[dict]) -> list[dict]:
    """Layer 2/3 decisions only, excluding Layer 1, unsure, and
    manual_overflow — the only records a validation gate can meaningfully
    compare against a human call. Layer 1 is deterministic (no model
    uncertainty) and already in gold; unsure and overflow were never a
    decision to begin with."""
    return [
        d for d in decisions
        if d.get("layer") in (2, 3) and d.get("is_adc") in ("yes", "no") and not d.get("manual_overflow")
    ]


def assert_no_layer1(decisions: list[dict]) -> None:
    """Checked against the RAW input, not just the filtered pool — a
    caller passing Layer 1 records into a validation draw at all is
    itself the bug this asserts against, even though
    decided_layer23_only would silently filter them out."""
    bad = [d["program_id"] for d in decisions if d.get("layer") == 1]
    if bad:
        raise StratificationError(
            f"{len(bad)} Layer 1 program_id(s) in the validation draw pool — Layer 1 is "
            "deterministic and already in gold, so it would tautologically agree with "
            f"itself and inflate agreement without testing anything. First few: {bad[:10]}"
        )


def _stratum(decision: dict) -> tuple[str, str]:
    """(basis, direction) using grounding.evidence_source (the project's
    one canonical definition of "how was this actually grounded" — see
    triage/report.py, which uses the same function) rather than the raw,
    unreliable from_recall flag."""
    basis = grounding.evidence_source(
        decision.get("is_adc"), bool(decision.get("from_recall")), decision.get("quote"),
    )
    direction = "accept" if decision.get("is_adc") == "yes" else "reject"
    return basis, direction


def check_stratum_feasibility(decisions: list[dict], sample_size: int = VALIDATION_SAMPLE_SIZE) -> dict:
    """Per-cell availability against the ORIGINAL text/recall x
    accept/reject design, plus the no_usable_evidence overflow bucket.
    Never raises — draw_stratified_sample does that; this just reports."""
    pool = decided_layer23_only(decisions)
    availability: dict[tuple, int] = {}
    no_usable_evidence: list[dict] = []
    for d in pool:
        s = _stratum(d)
        if s[0] == "no_usable_evidence":
            no_usable_evidence.append(d)
        availability[s] = availability.get(s, 0) + 1

    per_cell_target = sample_size // 4
    shortfalls = {
        cell: per_cell_target - availability.get(cell, 0)
        for cell in EXPECTED_STRATA if availability.get(cell, 0) < per_cell_target
    }
    return {
        "pool_size": len(pool), "per_cell_target": per_cell_target,
        "availability": {f"{b}/{d}": n for (b, d), n in availability.items()},
        "shortfalls": {f"{b}/{d}": n for (b, d), n in shortfalls.items()},
        "feasible": not shortfalls,
        "no_usable_evidence_count": len(no_usable_evidence),
        "no_usable_evidence_ids": [d["program_id"] for d in no_usable_evidence],
    }


def propose_substitute_stratification(decisions: list[dict]) -> dict:
    """Called when the text/recall grid can't be filled. Proposes
    stratifying by evidence SOURCE (Layer 2 registry-text vs Layer 3
    web-search-text) x direction instead — the axis with real,
    well-populated data on this run — and surfaces the no_usable_evidence
    decisions as a separate, explicitly-labelled appendix (too few and
    too one-sided — all "reject" — to be a real quota cell) rather than
    silently dropping them."""
    pool = decided_layer23_only(decisions)

    def source_cell(d: dict) -> tuple[str, str]:
        direction = "accept" if d.get("is_adc") == "yes" else "reject"
        if d.get("layer") == 2:
            return "layer2_registry_text", direction
        if d.get("quote"):
            return "layer3_web_search_text", direction
        return "layer3_no_usable_evidence", direction

    availability: dict[tuple, list[dict]] = {}
    for d in pool:
        availability.setdefault(source_cell(d), []).append(d)

    main_cells = [
        ("layer2_registry_text", "accept"), ("layer2_registry_text", "reject"),
        ("layer3_web_search_text", "accept"), ("layer3_web_search_text", "reject"),
    ]
    per_cell_target = VALIDATION_SAMPLE_SIZE // 4
    main_availability = {f"{s}/{d}": len(availability.get((s, d), [])) for s, d in main_cells}
    main_feasible = all(n >= per_cell_target for n in main_availability.values())

    appendix = availability.get(("layer3_no_usable_evidence", "reject"), []) + \
        availability.get(("layer3_no_usable_evidence", "accept"), [])

    return {
        "proposed_axes": "evidence source (Layer 2 registry-text vs Layer 3 web-search-text) x accept/reject",
        "main_cells": main_cells, "availability": main_availability, "feasible": main_feasible,
        "appendix_label": "no_usable_evidence (committed yes/no with zero supporting quote)",
        "appendix_count": len(appendix),
        "appendix_ids": [d["program_id"] for d in appendix],
        "note": (
            f"Only {sum(len(v) for k, v in availability.items() if k[0]=='layer3_no_usable_evidence')} "
            "decisions are genuinely ungrounded (no quote at all), all from Layer 3, and none are "
            "'accept' — too few and too lopsided to be a real 20-per-cell stratum. Reported here as a "
            "separate appendix, included in the sample but not counted toward any cell's quota or "
            "gate threshold on its own."
        ),
    }


def draw_stratified_sample(
    decisions: list[dict], *, sample_size: int = VALIDATION_SAMPLE_SIZE, seed: int = 0,
    already_reserved: Optional[set] = None, strict: bool = True,
) -> list[dict]:
    """The ORIGINAL text/recall x accept/reject design. Raises
    StratificationError (not a silent under-fill) when a cell can't reach
    sample_size//4 — pass strict=False only to knowingly accept a
    degenerate draw. On this run it WILL raise; see
    draw_substitute_sample for the design that actually works today."""
    assert_no_layer1(decisions)
    pool = decided_layer23_only(decisions)

    feasibility = check_stratum_feasibility(decisions, sample_size)
    if not feasibility["feasible"] and strict:
        substitute = propose_substitute_stratification(decisions)
        raise StratificationError(
            f"Cannot fill the text/recall x accept/reject grid at {feasibility['per_cell_target']}/cell — "
            f"shortfalls: {feasibility['shortfalls']}. Availability: {feasibility['availability']}. "
            f"PROPOSED SUBSTITUTE: {substitute['proposed_axes']}, availability={substitute['availability']} "
            f"(feasible={substitute['feasible']}). {substitute['note']} "
            "Call draw_substitute_sample() to actually draw it, or draw_stratified_sample(strict=False) "
            "to accept an under-filled draw anyway."
        )

    already_reserved = already_reserved or set()
    by_stratum: dict[tuple, list[dict]] = {}
    for d in pool:
        by_stratum.setdefault(_stratum(d), []).append(d)

    rng = random.Random(seed)
    kept = [d for d in pool if d["program_id"] in already_reserved]
    kept_ids = {d["program_id"] for d in kept}
    per_stratum_target = sample_size // 4

    for stratum, cell_pool in by_stratum.items():
        rng.shuffle(cell_pool)
        already_in_stratum = sum(1 for d in kept if _stratum(d) == stratum)
        need = max(0, per_stratum_target - already_in_stratum)
        for d in cell_pool:
            if need <= 0:
                break
            if d["program_id"] in kept_ids:
                continue
            d = dict(d)
            d["stratum"] = f"{stratum[0]}/{stratum[1]}"
            kept.append(d)
            kept_ids.add(d["program_id"])
            need -= 1

    return kept


def draw_substitute_sample(
    decisions: list[dict], *, sample_size: int = VALIDATION_SAMPLE_SIZE, seed: int = 0,
    already_reserved: Optional[set] = None,
) -> tuple[list[dict], dict]:
    """(sample, appendix_info). The substitute design that's actually
    feasible today: source (Layer 2 registry-text vs Layer 3 web-search-
    text) x accept/reject, ~20/cell, PLUS every no_usable_evidence
    decision appended separately (not counted toward any cell's quota) so
    they're reviewed too, not dropped. Raises StratificationError if even
    the substitute can't be filled — a second silent degenerate draw is
    exactly what this whole rewrite exists to prevent."""
    assert_no_layer1(decisions)
    pool = decided_layer23_only(decisions)
    proposal = propose_substitute_stratification(decisions)
    if not proposal["feasible"]:
        raise StratificationError(
            f"Substitute stratification is ALSO infeasible: availability={proposal['availability']}. "
            "No usable stratification exists in the current staged decisions — do not draw a sample yet."
        )

    def source_cell(d: dict) -> tuple[str, str]:
        direction = "accept" if d.get("is_adc") == "yes" else "reject"
        if d.get("layer") == 2:
            return "layer2_registry_text", direction
        if d.get("quote"):
            return "layer3_web_search_text", direction
        return "layer3_no_usable_evidence", direction

    already_reserved = already_reserved or set()
    by_cell: dict[tuple, list[dict]] = {}
    for d in pool:
        by_cell.setdefault(source_cell(d), []).append(d)

    rng = random.Random(seed)
    main_cells = proposal["main_cells"]
    kept = [d for d in pool if d["program_id"] in already_reserved and source_cell(d) in main_cells]
    kept_ids = {d["program_id"] for d in kept}
    per_cell_target = sample_size // 4

    for cell in main_cells:
        cell_pool = list(by_cell.get(cell, []))
        rng.shuffle(cell_pool)
        already_in_cell = sum(1 for d in kept if source_cell(d) == cell)
        need = max(0, per_cell_target - already_in_cell)
        for d in cell_pool:
            if need <= 0:
                break
            if d["program_id"] in kept_ids:
                continue
            d = dict(d)
            d["stratum"] = f"{cell[0]}/{cell[1]}"
            kept.append(d)
            kept_ids.add(d["program_id"])
            need -= 1

    appendix = by_cell.get(("layer3_no_usable_evidence", "reject"), []) + \
        by_cell.get(("layer3_no_usable_evidence", "accept"), [])
    for d in appendix:
        d = dict(d)
        d["stratum"] = "no_usable_evidence/appendix"
        kept.append(d)

    return kept, {
        "main_cells": {f"{s}/{d}": per_cell_target for s, d in main_cells},
        "appendix_count": len(appendix), "appendix_ids": [d["program_id"] for d in appendix],
    }


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
    program_id. Uses each sampled item's own "stratum" tag if present
    (set by draw_stratified_sample/draw_substitute_sample) so this works
    for either stratification scheme without re-deriving it; falls back
    to _stratum() for callers that built the sample another way. A
    program never reached by a human yet is excluded entirely — never
    counted as either an agreement or a disagreement. The
    no_usable_evidence appendix (stratum "no_usable_evidence/appendix")
    is tracked separately and never rolled into the pooled/gate numbers."""
    gold_records = gold_records if gold_records is not None else gold_store.load_records()
    gold_by_pid = gold_store.latest_by_program(gold_records)

    by_stratum: dict[str, dict] = {}
    appendix_cell = {"compared": 0, "agree": 0}
    is_adc_compared = is_adc_agree = 0
    in_scope_compared = in_scope_agree = 0

    for d in sample:
        gold = gold_by_pid.get(d["program_id"])
        if gold is None:
            continue
        stratum_label = d.get("stratum") or "/".join(_stratum(d))
        is_appendix = stratum_label.startswith("no_usable_evidence")

        triage_is_adc = "yes" if d.get("is_adc") == "yes" else "no"
        human_is_adc = "yes" if gold.get("is_adc") == "yes" else "no"
        agree = triage_is_adc == human_is_adc

        if is_appendix:
            appendix_cell["compared"] += 1
            appendix_cell["agree"] += int(agree)
        else:
            cell = by_stratum.setdefault(stratum_label, {"compared": 0, "agree": 0})
            cell["compared"] += 1
            cell["agree"] += int(agree)
            is_adc_compared += 1
            is_adc_agree += int(agree)

            if gold.get("gate_reached", 0) >= 2 and "in_scope" in d:
                in_scope_compared += 1
                if d.get("in_scope") == gold.get("in_scope"):
                    in_scope_agree += 1

    return {
        "by_stratum": {
            k: {**v, "agreement_rate": (v["agree"] / v["compared"]) if v["compared"] else None}
            for k, v in by_stratum.items()
        },
        "appendix": {
            **appendix_cell,
            "agreement_rate": (appendix_cell["agree"] / appendix_cell["compared"]) if appendix_cell["compared"] else None,
        },
        "is_adc": {"compared": is_adc_compared, "agree": is_adc_agree,
                   "agreement_rate": (is_adc_agree / is_adc_compared) if is_adc_compared else None},
        "in_scope": {"compared": in_scope_compared, "agree": in_scope_agree,
                     "agreement_rate": (in_scope_agree / in_scope_compared) if in_scope_compared else None},
    }


def check_gate(agreement: dict) -> tuple[bool, str]:
    """(passed, reason). Below threshold on EITHER axis rejects the WHOLE
    run — same all-or-nothing policy as the original design."""
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
    """Explicit comparison across whichever two-way split the sample
    actually used (text/recall, or the source substitute) — pools any
    stratum starting with "text" vs "recall", or, for the substitute
    scheme, "layer2_registry_text"/"layer3_web_search_text" against the
    no_usable_evidence appendix if it has enough n to be informative."""
    by_stratum = agreement["by_stratum"]
    text_cells = [c for k, c in by_stratum.items() if k.startswith(("text/", "layer2_registry_text/", "layer3_web_search_text/")) and c["compared"]]
    recall_cells = [c for k, c in by_stratum.items() if k.startswith("recall/") and c["compared"]]
    appendix = agreement.get("appendix") or {}
    if not recall_cells and appendix.get("compared"):
        recall_cells = [appendix]
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
