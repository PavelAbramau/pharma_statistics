"""Run the silver auto-labeller end to end.

    python scripts/run_silver_labelling.py --limit 5
    python scripts/run_silver_labelling.py --program-ids cand_x,cand_y
    python scripts/run_silver_labelling.py --only-gold-labelled --limit 13
    python scripts/run_silver_labelling.py --only-gold-labelled --dry-run   # no API calls, no key needed

For the "13 gold-labelled + 20 unlabelled" first run:

    python scripts/run_silver_labelling.py --only-gold-labelled --limit 13
    python scripts/run_silver_labelling.py --limit 20

(the default selection, with no flags, already excludes anything with a
gold gate-3 label, so the second command naturally picks 20 fresh ones.)

Requires ANTHROPIC_API_KEY (real API spend — see silver/model_client.py;
roughly 11 model calls per program: k=5 for each of 2 sampled questions,
plus 1 red-team pass). Writes ONLY to silver/labels.jsonl, labeller="auto"
(silver/store.py refuses anything else) — never touches gold/labels.jsonl
and is never surfaced by the labelling app. See audit/gold_set.py's "zero
auto-sourced records in gold" check.

Every program gets exactly one silver record logging the full prompt, all
k raw responses, the parsed answers, the citation-gate verdict per claim,
the deterministic rule path, and the final abstention decision (including
a Red Team override, if one fires) — see silver/store.py.build_record.
The logs are the point of a first run more than the labels are; read them
with scripts/silver_review.py.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store as gold_store
from pharma_stats.silver import evidence, model_client, prompts, questions, red_team
from pharma_stats.silver import store as silver_store
from pharma_stats.silver.questions import DecomposedAnswers, TrialInitiatedSinceAnswer

DEFAULT_LIMIT = 5  # deliberately small — this is real API spend, never assumed large


def label_one_program(program: dict, *, since_date: str, model: str) -> dict:
    """Answers all four decomposed questions, applies the deterministic
    rule, runs the Red Team check, and returns the full silver payload
    (including every log field) — but does NOT write it. Split out from
    main() so it's independently testable with model_client.complete
    mocked."""
    ev = evidence.build_evidence(program)
    program_name = program["proposed_name"]

    # Q1 — deterministic, no model call: we already know this exactly.
    trial_since_value = evidence.trials_initiated_since(program, since_date)
    q1_log = {
        "question": "trial_initiated_since", "prompt": None, "model": None,
        "raw_responses": [], "parsed_samples": [],
        "note": "answered deterministically from warehouse data — no API call",
        "value": trial_since_value, "since_date": since_date,
    }

    # Q2, Q3 — k=5 self-consistency sampled against CT.gov evidence only.
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

    red_team_log = None
    if not candidate["abstain"]:
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
        return [by_id[pid] for pid in ids if pid in by_id][: args.limit]

    gold_records = gold_store.load_records()
    fully_labelled = gold_store.fully_labelled_program_ids(gold_records)

    if args.only_gold_labelled:
        return [by_id[pid] for pid in fully_labelled if pid in by_id][: args.limit]

    # default: everything WITHOUT a gold gate-3 label — so a plain rerun
    # naturally picks fresh programs instead of re-labelling reference ones
    return [p for p in programs if p["program_id"] not in fully_labelled][: args.limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                     help=f"max programs to label this run (default {DEFAULT_LIMIT} — real API spend)")
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
                     help="show which programs would be labelled and the API-call budget, make ZERO "
                          "model calls, write nothing to silver/labels.jsonl. Doesn't need "
                          "ANTHROPIC_API_KEY or the anthropic package installed.")
    args = ap.parse_args()

    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    selected = _select_programs(args, programs)
    if not selected:
        print("Nothing to label (check --program-ids / --only-gold-labelled / --limit).")
        return

    since_date = (date.today() - timedelta(days=args.since_months * 30.44)).isoformat()

    if args.dry_run:
        print(f"[dry run] Would label {len(selected)} program(s) with model={args.model}, "
              f"Q1 cutoff since {since_date}. No model calls made, nothing written.\n")
        for i, program in enumerate(selected):
            trial_since = evidence.trials_initiated_since(program, since_date)
            print(f"[{i + 1}/{len(selected)}] {program['proposed_name']} ({program['program_id']}) "
                  f"— Q1 trial_initiated_since={trial_since} [deterministic, free]")
        print(f"\nEstimated spend: ~11 model calls/program x {len(selected)} = "
              f"~{11 * len(selected)} calls (k=5 x 2 sampled questions + 1 red-team pass; "
              "fewer if Q2/Q3 abstain before red-team runs).")
        return

    print(f"Labelling {len(selected)} program(s) with model={args.model}, Q1 cutoff since {since_date}.")
    print("This spends real API budget: ~11 calls/program (k=5 x 2 sampled questions + 1 red-team pass).")

    session_id = f"silver:{date.today().isoformat()}"
    for i, program in enumerate(selected):
        print(f"[{i + 1}/{len(selected)}] {program['proposed_name']} ({program['program_id']})...", end=" ")
        try:
            payload = label_one_program(program, since_date=since_date, model=args.model)
        except model_client.ModelClientError as e:
            print(f"\nABORTING: {e}")
            sys.exit(1)
        record = silver_store.build_record(payload, session_id=session_id)
        silver_store.append_record(record)
        print("ABSTAIN" if payload["abstained"] else payload["status"])

    print(f"\nWrote {len(selected)} silver record(s) to {silver_store.SILVER_LABELS_PATH}")
    print("Review with: python scripts/silver_review.py")


if __name__ == "__main__":
    main()
