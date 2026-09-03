"""How triage results change what the labelling app serves.

Three outcomes, matching the reviewer's queue contract:

- Auto-reject (is_adc=no or in_scope=no, and the relevant validation gate
  has passed): never enter the queue. Written to gold with decided_by=auto
  by triage.apply_layer1 / a later model-layer accept step — this module
  only *reads* those decisions so a live session drops them.
- Auto-resolved is_adc=yes AND in_scope=yes: enter at Gate 3, with the
  verdict shown as auto-derived context the reviewer can override.
- Genuinely unresolved: full three-gate flow.

Layer 2/3 model verdicts do NOT skip or prefill until
triage.validation.check_gate has passed. Layer 1 (deterministic) denylist
and non_industry rejections do not wait on that gate; heme_only does, via
the same MeSH-coverage / ambiguous-dominance / agreement checks as
scripts/apply_auto_scope_exclusions.py.

Re-opened program ids (data/reopen_for_review.json) always get the full
three-gate card, even if gold already has a label — the previous line
stays; the reviewer writes a new one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pharma_stats.config import DATA_DIR
from pharma_stats.labelling import trial_scope as ts
from pharma_stats.triage import deterministic as det
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval

REOPEN_PATH = DATA_DIR / "reopen_for_review.json"

# Gate-3 label target the hours estimate is against. The reviewer named ~100.
GATE3_TARGET = 100


@dataclass
class ServePlan:
    start_gate: int  # 1, 2, or 3
    skip: bool
    reopened: bool
    context: Optional[dict] = None  # auto-derived verdict shown on the card


def load_reopen_ids(path: Optional[Path] = None) -> list[str]:
    path = path or REOPEN_PATH
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("program_ids") or []
    return [pid for pid in ids if pid]


def consume_reopen_ids(path: Optional[Path] = None) -> list[str]:
    """Read-and-clear. Consume-on-read so a running app picks new ids up
    on the next /api/next without a restart, and so a restart doesn't
    re-serve an id the reviewer already handled this round."""
    path = path or REOPEN_PATH
    ids = load_reopen_ids(path)
    if not ids:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    data["program_ids"] = []
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ids


def ingest_reopens(session: dict, known_ids: set[str], path: Optional[Path] = None) -> list[str]:
    """Move newly requested reopens onto session['reopen_queue'] (front
    of the serve order). Unknown program_ids are dropped, not crashed on."""
    incoming = consume_reopen_ids(path)
    accepted = [pid for pid in incoming if pid in known_ids]
    queue = session.setdefault("reopen_queue", [])
    already = set(queue)
    for pid in accepted:
        if pid in already:
            continue
        queue.append(pid)
        already.add(pid)
        for list_key in ("order", "validation_order"):
            if pid in session.get(list_key, []):
                session[list_key].remove(pid)
    return accepted


def heme_only_auto_exclude_allowed(programs: list[dict], gold_records: list[dict]) -> tuple[bool, str]:
    """Same two gates as apply_auto_scope_exclusions, plus a third: if
    ambiguous classifications dominate the universe, heme_only (which
    requires every trial to classify heme) is not a clean cohort and
    must not filter anything."""
    coverage = ts.mesh_coverage(programs)
    if coverage["coverage_rate"] < ts.MESH_COVERAGE_THRESHOLD:
        return False, (
            f"MeSH coverage {coverage['coverage_rate']:.1%} "
            f"({coverage['covered']}/{coverage['total']}) is below "
            f"{ts.MESH_COVERAGE_THRESHOLD:.0%}"
        )
    dist = ts.scope_distribution(programs)
    if dist["ambiguous_dominates_trials"]:
        n_amb = dist["trials"]["ambiguous"]
        n_tot = dist["trials"]["total"]
        return False, (
            f"ambiguous classifications dominate "
            f"({n_amb}/{n_tot} trials, {dist['trials']['rates']['ambiguous']:.1%}) — "
            "heme_only auto-exclusion would under-count a noisy cohort, not a clean one"
        )
    sample = ts.load_validation_sample()
    agreement = ts.validation_agreement(sample, gold_records)
    if agreement["agreement_rate"] is None:
        if agreement["compared"] == 0 and not sample:
            # Bootstrap case — apply_auto_scope_exclusions allows this, but
            # the reviewer asked to see the MeSH mix BEFORE anything filters.
            # With no sample and no track record we still refuse heme_only
            # here; denylist / non_industry are unaffected.
            return False, "no heme_only validation sample on file yet — refusing to auto-exclude"
        return False, "heme_only validation sample is not reviewed enough to gate on"
    if agreement["agreement_rate"] < ts.AGREEMENT_THRESHOLD:
        return False, (
            f"heme_only agreement {agreement['agreement_rate']:.0%} "
            f"< {ts.AGREEMENT_THRESHOLD:.0%}"
        )
    return True, "passed"


def model_layer_gate_passed(gold_records: Optional[list[dict]] = None) -> tuple[bool, str]:
    sample = tval.load_validation_sample()
    if not sample:
        return False, "no Layer 2/3 validation sample drawn yet"
    agreement = tval.compute_agreement(sample, gold_records)
    return tval.check_gate(agreement)


def _layer1_context(result: det.Layer1Result) -> dict:
    return {
        "auto_derived": True,
        "layer": 1,
        "rule": result.rule,
        "is_adc": result.is_adc,
        "in_scope": result.in_scope,
        "scope_reason": result.scope_reason,
        "quote": None,
        "source_url": None,
        "evidence": f"Deterministic Layer 1 rule: {result.rule}",
    }


def _staged_context(record: dict) -> dict:
    return {
        "auto_derived": True,
        "layer": record.get("layer"),
        "rule": record.get("rule"),
        "is_adc": record.get("is_adc"),
        "in_scope": record.get("in_scope"),
        "scope_reason": record.get("scope_reason"),
        "quote": record.get("quote"),
        "source_url": record.get("source_url"),
        "from_recall": record.get("from_recall"),
        "evidence": record.get("quote") or record.get("rule") or f"Layer {record.get('layer')} staged decision",
    }


def serve_plan(
    program: dict,
    *,
    reopened: bool = False,
    heme_holdout: bool = False,
    triage_holdout: bool = False,
    heme_auto_ok: bool = False,
    model_gate_ok: bool = False,
    staged_record: Optional[dict] = None,
) -> ServePlan:
    """Decide skip / start_gate / context for one candidate.

    holdout flags force a blind three-gate card (validation samples).
    heme_auto_ok / model_gate_ok gate whether a resolved rejection is
    actually skipped vs. left in the queue for a human.
    """
    if reopened:
        return ServePlan(start_gate=1, skip=False, reopened=True, context=None)
    if heme_holdout or triage_holdout:
        return ServePlan(start_gate=1, skip=False, reopened=False, context=None)

    result = det.evaluate(program)

    # Layer 2/3 staged decisions, only after the 80-sample gate passes.
    if model_gate_ok and staged_record and not staged_record.get("manual_overflow"):
        is_adc = staged_record.get("is_adc")
        in_scope = staged_record.get("in_scope")
        ctx = _staged_context(staged_record)
        if is_adc == "no" or in_scope == "no":
            return ServePlan(start_gate=1, skip=True, reopened=False, context=ctx)
        if is_adc == "yes" and in_scope == "yes":
            return ServePlan(start_gate=3, skip=False, reopened=False, context=ctx)
        if is_adc == "yes":
            return ServePlan(start_gate=2, skip=False, reopened=False, context=ctx)

    if result is None:
        return ServePlan(start_gate=1, skip=False, reopened=False, context=None)

    if result.committable:
        if result.scope_reason == "heme_only" and not heme_auto_ok:
            start = 2 if result.is_adc == "yes" else 1
            ctx = _layer1_context(result) if start == 2 else None
            return ServePlan(start_gate=start, skip=False, reopened=False, context=ctx)
        return ServePlan(start_gate=1, skip=True, reopened=False, context=_layer1_context(result))

    ctx = _layer1_context(result)
    if result.is_adc == "yes" and result.in_scope == "yes":
        # Elimination ("no rejection fired") is not a real in_scope=yes when
        # MeSH coverage is low — heme assets with unclassifiable trials
        # would skip Gate 2. Only the named positive rule (every trial
        # solid, every sponsor industry, start ≥ 2012) enters at Gate 3.
        if result.rule and "layer1_positive_in_scope" in result.rule:
            return ServePlan(start_gate=3, skip=False, reopened=False, context=ctx)
        return ServePlan(start_gate=2, skip=False, reopened=False, context=ctx)
    if result.is_adc == "yes":
        return ServePlan(start_gate=2, skip=False, reopened=False, context=ctx)
    # is_adc pending, in_scope=yes (rule 6): still a Gate 1 molecule question
    return ServePlan(start_gate=1, skip=False, reopened=False, context=ctx)


def queue_composition(
    programs: list[dict],
    remaining_ids: list[str],
    *,
    gold_records: list[dict],
    heme_auto_ok: bool,
    model_gate_ok: bool,
    heme_holdout_ids: set[str],
    triage_holdout_ids: set[str],
    reopen_ids: Optional[list[str]] = None,
) -> dict:
    """How the remaining manual queue splits by entry gate, plus the
    hours-to-target figure the reviewer is actually trying to shrink."""
    from pharma_stats.labelling import stats as label_stats
    from pharma_stats.labelling.store import latest_by_program

    by_id = {p["program_id"]: p for p in programs}
    staged = staging.latest_by_program(staging.load_records())
    reopen_set = set(reopen_ids or [])

    n_gate1 = n_gate2 = n_gate3 = n_skip_still_in_order = 0
    for pid in remaining_ids:
        program = by_id.get(pid)
        if program is None:
            continue
        plan = serve_plan(
            program,
            reopened=pid in reopen_set,
            heme_holdout=pid in heme_holdout_ids,
            triage_holdout=pid in triage_holdout_ids,
            heme_auto_ok=heme_auto_ok,
            model_gate_ok=model_gate_ok,
            staged_record=staged.get(pid),
        )
        if plan.skip:
            n_skip_still_in_order += 1
            continue
        if plan.start_gate == 3:
            n_gate3 += 1
        elif plan.start_gate == 2:
            n_gate2 += 1
        else:
            n_gate1 += 1

    latest = latest_by_program(gold_records)
    n_gate3_labelled = sum(1 for r in latest.values() if r.get("gate_reached") == 3)
    remaining_to_target = max(0, GATE3_TARGET - n_gate3_labelled)
    median_s = label_stats.median_seconds_per_label(gold_records)
    hours_left = (remaining_to_target * median_s / 3600.0) if median_s else None

    return {
        "manual_queue": n_gate1 + n_gate2 + n_gate3,
        "enter_gate1": n_gate1,
        "enter_gate2": n_gate2,
        "enter_gate3": n_gate3,
        "skip_still_in_session_order": n_skip_still_in_order,
        "gate3_labelled": n_gate3_labelled,
        "gate3_target": GATE3_TARGET,
        "remaining_to_target": remaining_to_target,
        "median_seconds_per_label": median_s,
        "hours_left_to_target": hours_left,
    }
