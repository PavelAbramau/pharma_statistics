"""Run the silver auto-labeller end to end.

    python scripts/run_silver_labelling.py --dry-run                       # no API calls, no key needed
    python scripts/run_silver_labelling.py --only-gold-labelled --dry-run  # ditto, scoped to eval-comparable programs
    python scripts/run_silver_labelling.py --program-ids cand_x,cand_y
    python scripts/run_silver_labelling.py --stage 1                       # first 50, real spend, asks to confirm

--limit (default 5) caps how many programs ANY selection actually runs —
including --only-gold-labelled. `--only-gold-labelled` on its own does NOT
select every gold-labelled program; it selects up to --limit of them. To
run all current gate-3 labels: `--only-gold-labelled --limit <N>`, where N
is whatever `python scripts/run_silver_labelling.py --dry-run
--only-gold-labelled --limit 100000` reports as the selected count (the
real number changes as you keep labelling — this script always prints it
fresh rather than a docstring guessing a fixed count).

Requires ANTHROPIC_API_KEY (real API spend — see silver/model_client.py).
Writes ONLY to silver/labels.jsonl, labeller="auto" (silver/store.py
refuses anything else) — never touches gold/labels.jsonl and is never
surfaced by the labelling app. See audit/gold_set.py's "zero auto-sourced
records in gold" check. A per-program failure (timeout, rate limit, any
ModelClientError) is logged to silver/failures.jsonl and the run
continues to the next program — see label_one_program's caller in main().

Every program gets exactly one silver record logging the full prompt, all
k raw responses (k=3, escalated to 5 only on disagreement — see
silver/sampling.py), the parsed answers, the citation-gate verdict per
claim, the deterministic rule path, the Red Team gate outcome (run only
for dead_confirmed/superseded — see silver/red_team.GATE_STATUSES — logged
as skipped with a reason otherwise), and the final abstention decision.
Read them with scripts/silver_review.py.

Cost: --dry-run estimates from a local ~4-chars/token approximation (never
a real API call, per its own contract) using the SAME evidence-trimming
and prompt construction the real run uses; the real run logs the SDK's
actual token usage per call (silver/model_client.Usage) into every record
and prints a running total live. Use --max-spend to cap real spend — the
run finishes whatever program it's mid-way through, then stops before
starting the next one, so a partial run is never lost.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from pharma_stats.config import DATA_DIR
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store as gold_store
from pharma_stats.silver import evidence, model_client, prompts, questions, red_team
from pharma_stats.silver import store as silver_store
from pharma_stats.silver.questions import DecomposedAnswers, TrialInitiatedSinceAnswer

DEFAULT_LIMIT = 5  # deliberately small — this is real API spend, never assumed large
FAILURES_PATH = silver_store.SILVER_LABELS_PATH.parent / "failures.jsonl"
STAGE_STATE_PATH = DATA_DIR / "silver_stage_state.json"
STAGE_SIZES = {1: 50, 2: 150}  # stage 3 = remainder of the selection after stages 1-2

# Local, offline approximation for --dry-run only — never used for real
# accounting (the real run logs exact usage from the API response). ~4
# characters/token is a standard rough estimate for English text; --dry-run
# makes zero API calls by design, so an exact count via
# model_client.count_tokens (which itself hits the network) is not an
# option here.
_CHARS_PER_TOKEN_ESTIMATE = 4
ASSUMED_OUTPUT_TOKENS_PER_CALL = 200  # small structured-JSON answers


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def label_one_program(program: dict, *, since_date: str, model: str) -> dict:
    """Answers all four decomposed questions, applies the deterministic
    rule, runs the (status-gated) Red Team check, and returns the full
    silver payload (including every log field) — but does NOT write it.
    Split out from main() so it's independently testable with
    model_client.complete mocked. Raises model_client.ModelClientError on
    any API failure — the caller (main()) is responsible for catching it,
    logging to failures.jsonl, and continuing."""
    ev = evidence.build_evidence(program)
    program_name = program["proposed_name"]

    # Q1 — deterministic, no model call: we already know this exactly.
    trial_since_value = evidence.trials_initiated_since(program, since_date)
    q1_log = {
        "question": "trial_initiated_since", "prompt": None, "model": None,
        "raw_responses": [], "parsed_samples": [],
        "note": "answered deterministically from warehouse data — no API call",
        "value": trial_since_value, "since_date": since_date,
        "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
    }

    # Q2, Q3 — k=3 self-consistency, escalated to 5 only on disagreement.
    discontinuation_answer, q2_log = prompts.ask_discontinuation_statement(program_name, ev, model=model)
    stop_reason_answer, q3_log = prompts.ask_stop_reason(program_name, ev, model=model)

    # Q4 — no citable evidence source wired in yet; always abstains, no API call.
    successor_answer, q4_log = prompts.ask_successor_asset(program_name, ev)

    answers = DecomposedAnswers(
        trial_initiated_since=TrialInitiatedSinceAnswer(value=trial_since_value, since_date=since_date),
        discontinuation_statement=discontinuation_answer,
        stop_reason=stop_reason_answer,
        successor_asset=successor_answer,
    )
    candidate = questions.apply_deterministic_rules(answers)

    # Red Team: gated on status inside generate_objection itself (see
    # red_team.GATE_STATUSES) — always called when a label exists so the
    # log always states the gate outcome, even when it skips.
    if candidate["abstain"]:
        red_team_log = {
            "skipped": True, "reason": "abstained before a label existed to red-team",
            "prompt": None, "model": None,
            "usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        }
    else:
        label_for_review = {"status": candidate["status"], "kill_reason": candidate["kill_reason"]}
        objection, red_team_log = red_team.generate_objection(label_for_review, ev, model=model)
        if red_team.forces_abstention(objection):
            candidate = {
                "status": None, "kill_reason": None, "public_confirmation_date": None,
                "never_publicly_confirmed": False, "abstain": True,
                "abstain_reason": f"Red Team objection (strong, evidenced): {objection.argument}",
                "rule_path": candidate["rule_path"] + "|red_team_override",
            }

    return {
        "program_id": program["program_id"], "proposed_name": program_name,
        "status": candidate["status"], "kill_reason": candidate["kill_reason"],
        "public_confirmation_date": candidate["public_confirmation_date"],
        "never_publicly_confirmed": candidate["never_publicly_confirmed"],
        "abstained": candidate["abstain"], "abstain_reason": candidate.get("abstain_reason"),
        "rule_path": candidate["rule_path"],
        "answers": {
            "trial_initiated_since": q1_log,
            "discontinuation_statement": q2_log,
            "stop_reason": q3_log,
            "successor_asset": q4_log,
        },
        "self_consistency": {
            "discontinuation_statement": {"k": q2_log["k"], "disagreement": q2_log["disagreement"]},
            "stop_reason": {"k": q3_log["k"], "disagreement": q3_log["disagreement"]},
        },
        "red_team_objection": red_team_log,
    }


def _select_programs(args, programs: list[dict]) -> list[dict]:
    by_id = {p["program_id"]: p for p in programs}

    if args.program_ids:
        ids = [pid.strip() for pid in args.program_ids.split(",") if pid.strip()]
        missing = [pid for pid in ids if pid not in by_id]
        if missing:
            print(f"Warning: {len(missing)} program_id(s) not found in provisional_programs: {missing}")
        selected = [by_id[pid] for pid in ids if pid in by_id]
    else:
        gold_records = gold_store.load_records()
        fully_labelled = gold_store.fully_labelled_program_ids(gold_records)
        if args.only_gold_labelled:
            selected = [by_id[pid] for pid in fully_labelled if pid in by_id]
        else:
            # default: everything WITHOUT a gold gate-3 label — so a plain rerun
            # naturally picks fresh programs instead of re-labelling reference ones
            selected = [p for p in programs if p["program_id"] not in fully_labelled]

    if args.resume:
        already = {r["program_id"] for r in silver_store.load_records()}
        before = len(selected)
        selected = [p for p in selected if p["program_id"] not in already]
        skipped = before - len(selected)
        if skipped:
            print(f"--resume: skipping {skipped} program(s) already in {silver_store.SILVER_LABELS_PATH}")

    return selected[: args.limit]


# ---------------------------------------------------------------- staging --

def _load_stage_state() -> dict:
    if not STAGE_STATE_PATH.exists():
        return {}
    return json.loads(STAGE_STATE_PATH.read_text(encoding="utf-8"))


def _save_stage_state(state: dict) -> None:
    STAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGE_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _apply_stage(stage: int, selected: list[dict]) -> list[dict]:
    """Stage 1 = first 50 of the selection, stage 2 = next 150, stage 3 =
    remainder. Refuses to run stage N until stage N-1 is recorded as
    completed in STAGE_STATE_PATH — "I read the reasoning logs between
    stages" only works if the script can't be raced ahead of that."""
    state = _load_stage_state()
    if stage > 1 and str(stage - 1) not in state:
        raise SystemExit(
            f"Stage {stage} refuses to run: stage {stage - 1} hasn't completed yet "
            f"(no entry in {STAGE_STATE_PATH}). Run --stage {stage - 1} first."
        )
    s1 = STAGE_SIZES[1]
    s2 = STAGE_SIZES[2]
    if stage == 1:
        return selected[:s1]
    if stage == 2:
        return selected[s1:s1 + s2]
    if stage == 3:
        return selected[s1 + s2:]
    raise ValueError(f"invalid stage {stage!r}")


def _mark_stage_complete(stage: int, n_labelled: int, total_cost: float) -> None:
    state = _load_stage_state()
    state[str(stage)] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "programs_labelled": n_labelled, "total_cost_usd": total_cost,
    }
    _save_stage_state(state)


# ---------------------------------------------------------------- dry run --

def _dry_run_estimate(selected: list[dict], *, since_date: str, model: str) -> dict:
    """Builds every prompt exactly as the real run would (same evidence
    trimming, same question construction) and estimates cost from a local
    token approximation — zero API calls, per --dry-run's contract.
    Red Team cost is a range: 0 calls (nothing resolves to a gated status)
    to 1 call/program (every one does), since the real trigger depends on
    Q2/Q3's answer, which dry-run never computes."""
    if model not in model_client.PRICING_PER_MILLION_TOKENS:
        raise model_client.ModelClientError(
            f"no pricing on file for model {model!r} — add it to model_client.PRICING_PER_MILLION_TOKENS"
        )

    per_program = []
    for program in selected:
        ev = evidence.build_evidence(program)
        ev_text = evidence.evidence_text(ev)
        name = program["proposed_name"]

        q2_prompt = prompts._discontinuation_prompt(name, ev_text)
        q3_prompt = prompts._stop_reason_prompt(name, ev_text)
        rt_prompt = red_team._prompt({"status": "dead_confirmed", "kill_reason": "unknown_silent"}, ev_text)

        q2_in = _estimate_tokens(q2_prompt) + _estimate_tokens(prompts._EXTRACTION_SYSTEM)
        q3_in = _estimate_tokens(q3_prompt) + _estimate_tokens(prompts._EXTRACTION_SYSTEM)
        rt_in = _estimate_tokens(rt_prompt) + _estimate_tokens(red_team.RED_TEAM_SYSTEM)

        # typical case: unanimous at initial_k (per observed runs); worst
        # case: every question disagrees and escalates to escalated_k
        typical_calls = prompts.INITIAL_K * 2  # Q2 + Q3
        worst_calls = prompts.ESCALATED_K * 2

        typical_in_tokens = (q2_in + q3_in) * prompts.INITIAL_K
        worst_in_tokens = (q2_in + q3_in) * prompts.ESCALATED_K
        typical_out_tokens = ASSUMED_OUTPUT_TOKENS_PER_CALL * typical_calls
        worst_out_tokens = ASSUMED_OUTPUT_TOKENS_PER_CALL * worst_calls

        typical_cost = model_client.estimate_cost(typical_in_tokens, typical_out_tokens, model)
        worst_cost = model_client.estimate_cost(worst_in_tokens, worst_out_tokens, model)
        red_team_cost = model_client.estimate_cost(rt_in, ASSUMED_OUTPUT_TOKENS_PER_CALL, model)

        per_program.append({
            "program_id": program["program_id"], "proposed_name": name,
            "typical_calls": typical_calls, "worst_calls": worst_calls,
            "typical_cost_usd": typical_cost, "worst_cost_usd": worst_cost + red_team_cost,
            "red_team_cost_usd_if_triggered": red_team_cost,
        })

    # empirical Red Team trigger rate from real gold data, rather than
    # assuming 0% or 100% — how often a gate-3 label actually lands on a
    # GATE_STATUSES status
    gold_records = gold_store.load_records()
    gate3 = [gold_store.latest_by_program(gold_records)[pid] for pid in gold_store.fully_labelled_program_ids(gold_records)]
    red_team_rate = (
        sum(1 for r in gate3 if r.get("status") in red_team.GATE_STATUSES) / len(gate3)
        if gate3 else 0.5  # no gold data yet to estimate from — assume half
    )

    total_typical = sum(p["typical_cost_usd"] for p in per_program) + \
        red_team_rate * sum(p["red_team_cost_usd_if_triggered"] for p in per_program)
    total_worst = sum(p["worst_cost_usd"] for p in per_program)

    return {
        "per_program": per_program, "n": len(selected),
        "red_team_expected_trigger_rate": red_team_rate,
        "total_typical_cost_usd": total_typical, "total_worst_cost_usd": total_worst,
        "model": model,
    }


def _print_dry_run(estimate: dict, selected: list[dict], *, since_date: str) -> None:
    print(f"[dry run] {estimate['n']} program(s) selected, model={estimate['model']}, "
          f"Q1 cutoff since {since_date}. No API calls made.\n")
    for i, program in enumerate(selected):
        trial_since = evidence.trials_initiated_since(program, since_date)
        p = estimate["per_program"][i]
        print(f"[{i + 1}/{estimate['n']}] {program['proposed_name']} ({program['program_id']}) "
              f"— Q1 trial_initiated_since={trial_since} [free] — "
              f"est. ${p['typical_cost_usd']:.4f} typical / ${p['worst_cost_usd']:.4f} worst-case")
    print(
        f"\nEstimated calls/program: {estimate['per_program'][0]['typical_calls'] if estimate['per_program'] else 0} "
        f"typical (k={prompts.INITIAL_K} x 2 questions) to "
        f"{estimate['per_program'][0]['worst_calls'] + 1 if estimate['per_program'] else 0} worst-case "
        f"(k={prompts.ESCALATED_K} x 2 questions + 1 red-team pass)."
    )
    print(f"Red Team empirical trigger rate (from current gold data): {estimate['red_team_expected_trigger_rate']:.1%}")
    print(f"Estimated total cost: ${estimate['total_typical_cost_usd']:.2f} typical "
          f"/ ${estimate['total_worst_cost_usd']:.2f} worst-case (local token approximation — "
          f"~{_CHARS_PER_TOKEN_ESTIMATE} chars/token, not billed truth).")


def _confirm(estimate: dict) -> bool:
    print(f"\nAbout to spend real API budget on {estimate['n']} program(s): "
          f"~${estimate['total_typical_cost_usd']:.2f} typical, up to ${estimate['total_worst_cost_usd']:.2f} worst-case.")
    response = input("Type 'yes' to proceed: ").strip().lower()
    return response == "yes"


# --------------------------------------------------------------- failures --

def _log_failure(program: dict, error: Exception) -> None:
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "program_id": program["program_id"], "proposed_name": program["proposed_name"],
        "error_type": type(error).__name__, "error": str(error),
    }
    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                     help=f"max programs to label THIS RUN (default {DEFAULT_LIMIT} — real API spend). "
                          "Applies on top of every other selector, including --only-gold-labelled: "
                          "that flag alone does NOT select every gold-labelled program, only up to --limit "
                          "of them. Pass a large --limit explicitly if you want them all.")
    ap.add_argument("--program-ids", type=str, default=None,
                     help="comma-separated program_ids — overrides the default selection")
    ap.add_argument("--only-gold-labelled", action="store_true",
                     help="restrict to programs with an existing gold gate-3 label (for review comparison)")
    ap.add_argument("--since-months", type=int, default=24,
                     help="Q1 cutoff: has a trial started in the last N months (default 24)")
    ap.add_argument("--model", default=model_client.DEFAULT_MODEL,
                     help=f"default {model_client.DEFAULT_MODEL} — must be a model that accepts "
                          "explicit temperature (Opus 5 / Sonnet 5 / Fable 5 do not)")
    ap.add_argument("--dry-run", action="store_true",
                     help="show selected programs, build every prompt exactly as the real run would, "
                          "estimate cost from a local token approximation. ZERO API calls, no key needed.")
    ap.add_argument("--max-spend", type=float, default=None,
                     help="abort once REAL accumulated cost (from actual API usage) exceeds this many "
                          "dollars — finishes the in-progress program first so nothing completed is lost.")
    ap.add_argument("--resume", action="store_true",
                     help="skip programs already present in silver/labels.jsonl — for a job spanning hours")
    ap.add_argument("--yes", action="store_true",
                     help="skip the interactive confirmation prompt (selection > 10 programs still shows the estimate)")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                     help=f"stage 1 = first {STAGE_SIZES[1]} of the selection, stage 2 = next {STAGE_SIZES[2]}, "
                          "stage 3 = remainder. Each stage refuses to run until the previous one is recorded "
                          f"complete in {STAGE_STATE_PATH}.")
    args = ap.parse_args()

    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    selected = _select_programs(args, programs)
    if args.stage is not None:
        selected = _apply_stage(args.stage, selected)
    if not selected:
        print("Nothing to label (check --program-ids / --only-gold-labelled / --limit / --stage).")
        return

    since_date = (date.today() - timedelta(days=args.since_months * 30.44)).isoformat()
    estimate = _dry_run_estimate(selected, since_date=since_date, model=args.model)

    if args.dry_run:
        _print_dry_run(estimate, selected, since_date=since_date)
        return

    print(f"Labelling {len(selected)} program(s) with model={args.model}, Q1 cutoff since {since_date}.")
    if len(selected) > 10 and not args.yes:
        _print_dry_run(estimate, selected, since_date=since_date)
        if not _confirm(estimate):
            print("Not confirmed — aborting, nothing spent.")
            return

    session_id = f"silver:{date.today().isoformat()}"
    total_cost = 0.0
    n_labelled = 0
    n_failed = 0
    failures: list[dict] = []

    for i, program in enumerate(selected):
        print(f"[{i + 1}/{len(selected)}] {program['proposed_name']} ({program['program_id']})...", end=" ")
        try:
            payload = label_one_program(program, since_date=since_date, model=args.model)
        except model_client.ModelClientError as e:
            print(f"FAILED ({type(e).__name__}) — logged, continuing")
            _log_failure(program, e)
            failures.append({"program_id": program["program_id"], "proposed_name": program["proposed_name"],
                              "error": str(e)})
            n_failed += 1
            continue

        record = silver_store.build_record(payload, session_id=session_id)
        silver_store.append_record(record)
        n_labelled += 1
        program_cost = record["cost_usd"]
        total_cost += program_cost
        print(f"{'ABSTAIN' if payload['abstained'] else payload['status']} "
              f"(${program_cost:.4f}, running total ${total_cost:.2f})")

        if args.max_spend is not None and total_cost > args.max_spend:
            print(f"\n--max-spend ${args.max_spend:.2f} exceeded (spent ${total_cost:.2f}) — "
                  f"stopping after program {i + 1}/{len(selected)}. "
                  f"{n_labelled} record(s) already written are safe.")
            break

    print(f"\nWrote {n_labelled} silver record(s) to {silver_store.SILVER_LABELS_PATH}"
          + (f"; {n_failed} failure(s) logged to {FAILURES_PATH}" if n_failed else ""))
    if failures:
        print("Failure summary:")
        for f in failures:
            print(f"  - {f['proposed_name']} ({f['program_id']}): {f['error']}")
    print(f"Total real spend this run: ${total_cost:.2f}"
          + (f" (${total_cost / n_labelled:.4f}/label)" if n_labelled else ""))

    if args.stage is not None:
        _mark_stage_complete(args.stage, n_labelled, total_cost)
        print(f"Stage {args.stage} marked complete in {STAGE_STATE_PATH}.")
        if args.stage == 1 and n_labelled:
            print(f"Cost per label after stage 1: ${total_cost / n_labelled:.4f} — "
                  f"review the reasoning logs (scripts/silver_review.py) before deciding on stage 2.")

    print("Review with: python scripts/silver_review.py")


if __name__ == "__main__":
    main()
