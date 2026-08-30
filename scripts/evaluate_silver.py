"""Strict silver-vs-gold evaluation.

    python scripts/evaluate_silver.py

Reads gold/labels.jsonl and silver/labels.jsonl; wires the maths in
silver/eval.py into a report. Order matters here more than usual:

1. Every disagreement, individually, first — at n=29 gate-3 label events
   (21 unique programs), that IS the output; summary statistics are
   secondary.
2. Coverage (non-abstention rate) and accuracy-among-non-abstentions,
   per field, NEVER blended into one number.
3. A status confusion matrix, called out by failure direction: a false
   "dead" call is a worse failure for a kill detector than a missed one.
4. public_confirmation_date is excluded from scoring — no external
   retrieval is wired into evidence.py yet (see its module docstring), so
   that field is expected to be near-unscoreable and would just be noise.
5. Every accuracy is reported with a Wilson score interval (no scipy
   dependency — this script only needs the `silver` extra). With 21 gate-3
   programs and ~6 in the eval half, most intervals will be wide; that is
   reported plainly, not smoothed over.
6. The eval split is FROZEN the first time this runs: split_by_asset is
   called once with a fixed seed and the asset/program assignment is
   persisted to gold/eval_split.json (commit that file). Every later run
   loads the frozen split instead of recomputing it, so a program can
   never retroactively become eval-only after appearing in a few-shot
   prompt. New gate-3 programs that show up after the freeze and aren't
   in either bucket are reported and excluded from this run's eval set,
   not silently folded in.
7. Accuracy is compared against the labeller's OWN self-consistency
   ceiling (from repeat-probe agreement), not against 100% — and that
   ceiling gets its own CI, with an explicit "too few probes to be
   stable" flag, since there are only 8 of them today.

External-search stratification is deliberately dropped: gold's
status_revised_after_external_search field is uniformly False across all
29 gate-3 label events today, so there's nothing to slice on. Revisit once
some labels actually carry a True there (~150 labels, per the plan).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pharma_stats.config import GOLD_DIR
from pharma_stats.labelling import store as gold_store
from pharma_stats.silver import eval as silver_eval
from pharma_stats.silver import store as silver_store

EVAL_SPLIT_PATH = GOLD_DIR / "eval_split.json"
EVAL_FRACTION = 0.3
SPLIT_SEED = 0
MIN_STABLE_N = 20  # below this, "too few to be stable" — not a hard rule, just a legibility flag
Z = 1.96  # 95%


# ---------------------------------------------------------------- split freeze

def _freeze_or_load_split(gold_records: list[dict]) -> tuple[list[str], list[str]]:
    if EVAL_SPLIT_PATH.exists():
        payload = json.loads(EVAL_SPLIT_PATH.read_text())
        few_shot_ids = payload["few_shot_program_ids"]
        eval_ids = payload["eval_program_ids"]

        current_fully_labelled = gold_store.fully_labelled_program_ids(gold_records)
        known = set(few_shot_ids) | set(eval_ids)
        new_ids = sorted(current_fully_labelled - known)
        if new_ids:
            print(f"NOTE: {len(new_ids)} gate-3 program(s) exist that predate the frozen split at "
                  f"{EVAL_SPLIT_PATH} but aren't assigned in it — excluded from this run's eval set "
                  f"(re-freeze deliberately, don't auto-extend, if you want them included): {new_ids}\n")
        return few_shot_ids, eval_ids

    few_shot_ids, eval_ids = silver_eval.split_by_asset(gold_records, eval_fraction=EVAL_FRACTION, seed=SPLIT_SEED)
    EVAL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_SPLIT_PATH.write_text(json.dumps({
        "seed": SPLIT_SEED,
        "eval_fraction": EVAL_FRACTION,
        "few_shot_program_ids": few_shot_ids,
        "eval_program_ids": eval_ids,
        "note": "Frozen by scripts/evaluate_silver.py on first run. Prompts are zero-shot today, so this "
                "freeze exists for the future: once few-shot examples are added, eval_program_ids is "
                "guaranteed to never have appeared in one. Commit this file.",
    }, indent=2, sort_keys=True) + "\n")
    print(f"Froze eval split (seed={SPLIT_SEED}) to {EVAL_SPLIT_PATH} — commit this file.\n")
    return few_shot_ids, eval_ids


# ---------------------------------------------------------------- stats

def wilson_ci(k: int, n: int, z: float = Z) -> "tuple[float, float] | None":
    """Wilson score interval — no scipy dependency, and better-behaved than
    a normal-approximation interval at the tiny n this project has."""
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n=0 (no basis)"
    ci = wilson_ci(k, n)
    lo, hi = ci
    flag = "  <- TOO FEW TO BE STABLE" if n < MIN_STABLE_N else ""
    return f"{k}/{n} = {k / n:.0%}  (95% Wilson CI: {lo:.0%}-{hi:.0%}, n={n}){flag}"


# ---------------------------------------------------------------- data assembly

def _latest_silver_by_program(silver_records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for r in silver_records:
        pid = r["program_id"]
        if pid not in latest or r["timestamp"] > latest[pid]["timestamp"]:
            latest[pid] = r
    return latest


def _question_trail(silver_rec: dict) -> list[str]:
    """One line per question that actually had a bearing on rule_path,
    for the disagreement listing's 'which question drove it' column."""
    answers = silver_rec.get("answers") or {}
    lines = []

    q1 = answers.get("trial_initiated_since") or {}
    if "value" in q1:
        lines.append(f"Q1 trial_initiated_since (since {q1.get('since_date')}) = {q1['value']} [deterministic, no citation]")

    for key, label in (("discontinuation_statement", "Q2 discontinuation_statement"),
                        ("stop_reason", "Q3 stop_reason")):
        log = answers.get(key) or {}
        if not log.get("prompt"):
            continue
        votes = log.get("votes", [])
        disagreement = log.get("disagreement", False)
        verdict = log.get("citation_verdict") or {}
        citation = verdict.get("citation") or {}
        vote_str = f"disagreed on k={log.get('k')} -> abstain" if disagreement else f"k={log.get('k')} votes={votes}"
        line = f"{label}: {vote_str}"
        if citation.get("quote"):
            line += f" | cited [{citation.get('locator')}]: \"{citation['quote']}\""
        elif verdict:
            line += f" | citation gate: {'PASSED' if verdict.get('passed') else 'FAILED (' + str(verdict.get('reason')) + ')'}"
        lines.append(line)

    if silver_rec.get("red_team_objection"):
        obj = (silver_rec["red_team_objection"].get("objection") or {})
        if obj.get("strength"):
            lines.append(f"Red Team: {obj['strength']} — {obj.get('argument', '')}")

    return lines


# ---------------------------------------------------------------- report sections

def render_disagreements(eval_ids: list[str], gold_by_pid: dict, silver_by_pid: dict) -> None:
    print("=" * 78)
    print("1. DISAGREEMENTS (status), individually — this IS the output at n=%d" % len(eval_ids))
    print("=" * 78)

    disagreements = []
    for pid in eval_ids:
        gold = gold_by_pid.get(pid)
        silver = silver_by_pid.get(pid)
        if gold is None or silver is None:
            continue
        silver_status = None if silver.get("abstained") else silver.get("status")
        if silver_status != gold.get("status"):
            disagreements.append((pid, gold, silver))

    if not disagreements:
        n_scored = sum(1 for pid in eval_ids if pid in gold_by_pid and pid in silver_by_pid)
        print(f"(no disagreements among {n_scored} eval program(s) with a silver prediction)\n")
        return

    for pid, gold, silver in disagreements:
        print(f"\n--- {gold.get('proposed_name')} ({pid}) ---")
        print(f"  gold:   status={gold.get('status')!r} kill_reason={gold.get('kill_reason')!r}")
        if silver.get("abstained"):
            print(f"  silver: ABSTAIN — {silver.get('abstain_reason')}")
        else:
            print(f"  silver: status={silver.get('status')!r} kill_reason={silver.get('kill_reason')!r}")
        print(f"  rule_path: {silver.get('rule_path')}")
        for line in _question_trail(silver):
            print(f"    {line}")
    print(f"\n{len(disagreements)} disagreement(s) shown above.\n")


def render_coverage_and_accuracy(eval_ids: list[str], gold_records: list[dict], silver_by_pid: dict) -> None:
    print("=" * 78)
    print("2. COVERAGE AND ACCURACY (never blended)")
    print("=" * 78)
    print("public_confirmation_date is EXCLUDED from scoring — no external retrieval is wired into "
          "evidence.py yet (SEC EDGAR/press/conference sources are stubs), so it is not fairly scoreable.\n")

    predictions = {}
    for pid in eval_ids:
        silver = silver_by_pid.get(pid)
        if silver is None:
            continue
        predictions[pid] = {
            "status": None if silver.get("abstained") else silver.get("status"),
            "kill_reason": None if silver.get("abstained") else silver.get("kill_reason"),
        }

    n_with_prediction = len(predictions)
    print(f"Eval programs: {len(eval_ids)} total, {n_with_prediction} have a silver prediction on file "
          f"(run scripts/run_silver_labelling.py --program-ids <missing ones> to fill in the rest).\n")

    status_acc = silver_eval.per_field_accuracy(predictions, gold_records, eval_ids, fields=("status",))["status"]
    n_compared, n_correct, n_abstained = status_acc["n_compared"], status_acc["n_correct"], status_acc["n_abstained"]
    n_total_scoreable = n_compared + n_abstained
    print(f"status  — coverage (non-abstention): {_fmt_rate(n_compared, n_total_scoreable)}")
    if n_compared:
        print(f"        — accuracy among non-abstentions: {_fmt_rate(n_correct, n_compared)}")
    else:
        print("        — accuracy among non-abstentions: n=0 (every prediction abstained)")

    dead_pids = [pid for pid in eval_ids if (gold_store.latest_by_program(gold_records).get(pid) or {}).get("status") == "dead_confirmed"]
    kr_acc = silver_eval.per_field_accuracy(predictions, gold_records, dead_pids, fields=("kill_reason",))["kill_reason"]
    kr_total = kr_acc["n_compared"] + kr_acc["n_abstained"]
    print(f"\nkill_reason (scored ONLY over eval programs where gold status=dead_confirmed, n={len(dead_pids)}):")
    print(f"        — coverage (non-abstention): {_fmt_rate(kr_acc['n_compared'], kr_total)}")
    if kr_acc["n_compared"]:
        print(f"        — accuracy among non-abstentions: {_fmt_rate(kr_acc['n_correct'], kr_acc['n_compared'])}")
    else:
        print("        — accuracy among non-abstentions: n=0")
    print()


def render_confusion_matrix(eval_ids: list[str], gold_by_pid: dict, silver_by_pid: dict) -> None:
    print("=" * 78)
    print("3. STATUS CONFUSION MATRIX — direction matters more than raw accuracy")
    print("=" * 78)

    rows = []
    for pid in eval_ids:
        gold = gold_by_pid.get(pid)
        silver = silver_by_pid.get(pid)
        if gold is None or silver is None:
            continue
        silver_status = "ABSTAIN" if silver.get("abstained") else (silver.get("status") or "ABSTAIN")
        rows.append((gold.get("proposed_name"), pid, gold.get("status"), silver_status))

    if not rows:
        print("(no eval program has both a gold label and a silver prediction yet)\n")
        return

    gold_statuses = sorted({r[2] for r in rows})
    silver_statuses = sorted({r[3] for r in rows})
    print(f"n={len(rows)} eval programs with a paired gold+silver outcome.\n")
    header = "gold \\ silver".ljust(18) + "".join(s.ljust(16) for s in silver_statuses)
    print(header)
    for gs in gold_statuses:
        line = gs.ljust(18)
        for ss in silver_statuses:
            count = sum(1 for r in rows if r[2] == gs and r[3] == ss)
            line += str(count).ljust(16)
        print(line)

    false_dead = [r for r in rows if r[3] == "dead_confirmed" and r[2] != "dead_confirmed"]
    missed_dead = [r for r in rows if r[2] == "dead_confirmed" and r[3] != "dead_confirmed"]
    print(f"\nFALSE-DEAD calls (silver said dead_confirmed, gold disagrees) — the worse failure mode for a "
          f"kill detector, n={len(false_dead)}:")
    for name, pid, gs, ss in false_dead:
        print(f"  - {name} ({pid}): gold={gs}")
    if not false_dead:
        print("  (none)")

    print(f"\nMISSED deaths (gold=dead_confirmed, silver said something else or abstained) — recall miss, "
          f"less bad but still a gap, n={len(missed_dead)}:")
    for name, pid, gs, ss in missed_dead:
        print(f"  - {name} ({pid}): silver={ss}")
    if not missed_dead:
        print("  (none)")
    print()


def render_external_search_note(gold_records: list[dict]) -> None:
    print("=" * 78)
    print("4. EXTERNAL-SEARCH STRATIFICATION — dropped")
    print("=" * 78)
    gate3 = [r for r in gold_records if r.get("action") == "label" and r.get("gate_reached") == 3]
    n_true = sum(1 for r in gate3 if r.get("status_revised_after_external_search"))
    print(f"status_revised_after_external_search is True for {n_true}/{len(gate3)} gate-3 label events. "
          "Nothing to slice on — not reported as a stratum. Revisit at ~150 labels.\n")


def render_self_consistency_ceiling(gold_records: list[dict], eval_ids: list[str], gold_records_by_pid: dict,
                                     silver_by_pid: dict, eval_status_accuracy: "float | None") -> None:
    print("=" * 78)
    print("5. ACCURACY vs SELF-CONSISTENCY CEILING (not vs 100%)")
    print("=" * 78)
    sc = silver_eval.self_consistency_ceiling(gold_records)
    n = sc["repeats_served"]
    if n == 0:
        print("No repeat probes on file yet — no ceiling to compare against.\n")
        return

    ci = wilson_ci(sc["agreements"], n)
    lo, hi = ci
    flag = " — TOO FEW REPEAT PROBES TO CALL THIS STABLE, treat as directional only" if n < MIN_STABLE_N else ""
    print(f"Labeller's self-consistency ceiling: {sc['agreements']}/{n} = {sc['agreement_rate']:.0%} "
          f"(95% Wilson CI: {lo:.0%}-{hi:.0%}, n={n}){flag}\n")

    if eval_status_accuracy is None:
        print("(no eval-set status accuracy computed — see section 2)\n")
        return
    comparison = silver_eval.accuracy_vs_self_consistency_ceiling(eval_status_accuracy, sc["agreement_rate"])
    if comparison["comparable"]:
        verdict = "AT OR ABOVE" if comparison["at_or_above_ceiling"] else "BELOW"
        print(f"Silver status accuracy ({eval_status_accuracy:.0%}) is {verdict} the ceiling "
              f"(gap: {comparison['gap']:+.0%}). Given the n above, treat this comparison as directional, "
              "not a pass/fail line.\n")
    else:
        print("Not comparable (missing accuracy or ceiling).\n")


def main() -> None:
    gold_records = gold_store.load_records()
    silver_records = silver_store.load_records()

    if not silver_records:
        print(f"No silver records at {silver_store.SILVER_LABELS_PATH} yet — run "
              "scripts/run_silver_labelling.py first (see --only-gold-labelled for the eval-comparable run).")
        return

    few_shot_ids, eval_ids = _freeze_or_load_split(gold_records)
    gold_by_pid = gold_store.latest_by_program(gold_records)
    silver_by_pid = _latest_silver_by_program(silver_records)

    print(f"Frozen split: {len(few_shot_ids)} few-shot program(s), {len(eval_ids)} eval program(s) "
          f"(from {EVAL_SPLIT_PATH}).\n")

    render_disagreements(eval_ids, gold_by_pid, silver_by_pid)
    render_coverage_and_accuracy(eval_ids, gold_records, silver_by_pid)
    render_confusion_matrix(eval_ids, gold_by_pid, silver_by_pid)
    render_external_search_note(gold_records)

    predictions = {
        pid: {"status": None if silver_by_pid[pid].get("abstained") else silver_by_pid[pid].get("status")}
        for pid in eval_ids if pid in silver_by_pid
    }
    status_acc = silver_eval.per_field_accuracy(predictions, gold_records, eval_ids, fields=("status",))["status"]
    eval_status_accuracy = status_acc["accuracy"]
    render_self_consistency_ceiling(gold_records, eval_ids, gold_by_pid, silver_by_pid, eval_status_accuracy)


if __name__ == "__main__":
    main()
