"""End-to-end triage pipeline runner.

    python scripts/run_triage_pipeline.py --dry-run                 # Step A: no completion calls
    python scripts/run_triage_pipeline.py --pilot --limit 100       # Step B: real spend, writes reports/triage_pilot.md
    python scripts/run_triage_pipeline.py --full                    # Step C: real spend, resume-safe, cost-capped
    python scripts/run_triage_pipeline.py --review-page             # Step D: blind HTML review page for the validation sample
    python scripts/run_triage_pipeline.py --report                  # Step E: final report, once the sample is judged

Pool integrity (fix 1): every pool this script builds excludes every
program_id with ANY existing gold record (gate 1, 2, or 3) —
pool.select_candidate_pool asserts this and raises rather than silently
trusting the exclusion; pool.assert_not_reviewed is checked again
immediately before every staged write, since a long run can span hours
during which a human keeps labelling in the app.

Zero-text bypass (fix 2): candidates with no supporting trial text skip
Layer 2 entirely (pipeline.partition_by_text_evidence) — a recall-only
answer on an unnamed dev code is the highest-hallucination-risk case
here, so it goes straight to Layer 3 rather than spending a Layer 2 call
on a guess we'd distrust anyway.

Layer 3 overflow (fix 3): anything beyond layer3.MAX_LAYER3_CANDIDATES is
NEVER auto-decided or dropped — pipeline.stage_manual_overflow flags each
one explicitly in the staging table with manual_overflow=True, and this
script prints the count whenever the cap binds.

--limit (fix 4) caps how many with-text candidates Layer 2 processes in
--pilot or --full — for a deliberately small first spend.

Nothing here writes to gold/labels.jsonl. Every decision lands in
triage/staged_decisions.jsonl (see triage/staging.py) for a human to
bulk-accept or reject after reading the run report.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone

import duckdb

from pharma_stats.config import DATA_DIR, REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.silver import model_client
from pharma_stats.triage import grounding
from pharma_stats.triage import layer2, layer3, pipeline as tpl
from pharma_stats.triage import pool as tpool
from pharma_stats.triage import report as trep
from pharma_stats.triage import staging
from pharma_stats.triage import validation as tval

FAILURES_PATH = staging.STAGING_PATH.parent / "triage_failures.jsonl"


def _log_failure(program_id: str, name: str, error: Exception) -> None:
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "program_id": program_id,
        "proposed_name": name, "error_type": type(error).__name__, "error": str(error),
    }
    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _print_pool_stats(stats: dict) -> None:
    print(f"Pool integrity check passed. {stats['total_materialized']} materialized programs, "
          f"{stats['total_gold_records']} gold records covering {stats['total_reviewed_programs']} "
          f"reviewed programs (gate1={stats['gate1_rejected_count']}, gate2={stats['gate2_rejected_count']}, "
          f"gate3={stats['gate3_labelled_count']}). Overlap with candidate pool: {stats['overlap_count']} "
          f"(must be 0). Pool size: {stats['pool_size']}.")


def cmd_dry_run(args) -> None:
    pool, stats = tpool.select_candidate_pool()
    _print_pool_stats(stats)

    # Step A is explicitly "across the full residue" — --limit only applies
    # to --pilot/--full (fix 4's actual purpose: capping a real-spend run).
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        report = tpl.dry_run_report(pool, con, model=args.model, layer2_limit=None)
    finally:
        con.close()

    l1 = report["layer1"]
    print(f"\n=== Layer 1 (free) ===")
    print(f"resolved: {l1['resolved']}/{l1['total']} ({l1['resolved_rate']:.1%}), "
          f"committable now: {l1['committable']} ({l1['committable_rate']:.1%})")
    print(f"residue -> Layer 2/3: {report['residue_size']}")

    print(f"\n=== Layer 2 ===")
    print(f"candidates with text evidence: {report['layer2_candidates']} in {report['layer2_groups']} group(s) "
          f"of {layer2.BATCH_SIZE}")
    tok_note = "exact (real tokenizer)" if report["exact_tokenizer"] else "approximated (~4 chars/token — no API key/network for the real tokenizer)"
    print(f"token counts: {tok_note}")
    if report["layer2_group_tokens"]:
        avg_tok = sum(report["layer2_group_tokens"]) / len(report["layer2_group_tokens"])
        print(f"avg prompt tokens/group: {avg_tok:.0f}")
    print(f"estimated cost (Batch API, 50% off): typical (k={layer2.INITIAL_K}) "
          f"${report['layer2_typical_cost_usd']:.2f} / worst-case (k={layer2.ESCALATED_K}) "
          f"${report['layer2_worst_cost_usd']:.2f}")

    print(f"\n=== Layer 3 ===")
    print(f"known minimum queue (zero-text bypass, fix 2): {report['layer3_known_minimum_queue']}")
    print(f"cap: {report['layer3_cap']}")
    if report["layer3_overflow_at_minimum"]:
        print(f"⚠ cap already binds on the KNOWN minimum alone: {report['layer3_overflow_at_minimum']} "
              f"would overflow to the manual queue, flagged, even before Layer 2 routes anything more here.")
    print(f"minimum cost (zero-text bypass only): ${report['layer3_minimum_cost_usd']:.2f}")
    print(f"cost IF every Layer 2 candidate also routed here (worst case, capped): "
          f"${report['layer3_cost_if_all_with_text_also_route_usd']:.2f}")
    print("(Actual Layer 3 volume beyond the known minimum depends on Layer 2's real answers — "
          "not knowable without running Layer 2 for real.)")

    total_typical = report["layer2_typical_cost_usd"] + report["layer3_minimum_cost_usd"]
    print(f"\n=== Total (typical, lower-bound on Layer 3) ===\n${total_typical:.2f}")


def _pilot_resume_sets() -> tuple[dict, set[str], set[str], set[str], set[str]]:
    """(staged_latest, already_l2_ids, already_l3_ids, pilot_names, pilot_l3_names)."""
    staged = staging.latest_by_program(staging.load_records())
    already_l2 = {pid for pid, r in staged.items() if r.get("layer") == 2}
    already_l3 = {pid for pid, r in staged.items() if r.get("layer") == 3}
    pilot_names: set[str] = set()
    pilot_l3_names: set[str] = set()
    md = REPORTS_DIR / "triage_pilot.md"
    if md.exists():
        _, rows = trep.parse_pilot_markdown(md)
        for r in rows:
            pilot_names.add(r["name"])
            if r["routed_to_layer3"]:
                pilot_l3_names.add(r["name"])
    return staged, already_l2, already_l3, pilot_names, pilot_l3_names


def _stage_layer2(ev: dict, answer, *, run_id: str, model: str) -> None:
    tpool.assert_not_reviewed(ev["program_id"])
    record = staging.build_record({
        "program_id": ev["program_id"], "proposed_name": ev["name"],
        "is_adc": answer.is_adc, "layer": 2, "rule": None,
        "model": model, "prompt_version": layer2.PROMPT_VERSION,
        "from_recall": answer.from_recall, "quote": answer.quote,
        "evidence_source": answer.evidence_source, "confidence": answer.confidence,
        "grounding_forced_recall": answer.grounding_forced_recall,
    }, run_id=run_id)
    staging.append_record(record)


def _write_pilot_reports(rows: list[dict], *, run_id: str, cost: float) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    header = f"Run `{run_id}` — {len(rows)} candidate(s), real Layer 2 spend ${cost:.4f}."
    md = REPORTS_DIR / "triage_pilot.md"
    lines = [
        "# Triage pilot report", "", header, "",
        "Check before spending anything more: are quotes verbatim from the evidence given, "
        "and is `from_recall` set honestly? Not whether the verdicts look plausible.", "",
        "| name | verdict | confidence | evidence | quote | from_recall | -> Layer 3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        quote = (r.get("quote") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['name']} | {r['verdict']} | {r['confidence']} | {r['evidence_source']} | "
            f"{quote} | {r['from_recall']} | {r['routed_to_layer3']} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_path = REPORTS_DIR / "triage_pilot.html"
    trep.write_pilot_html(rows, html_path, header=header, spend=f"${cost:.4f}")
    print(f"\nWrote {md} and {html_path}. Stop here and review before spending anything more.")


def cmd_pilot_or_full(args) -> None:
    run_id = f"triage:{args.mode}:{uuid.uuid4().hex[:8]}"
    pool, stats = tpool.select_candidate_pool()
    _print_pool_stats(stats)

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        residue, evidences = tpl.build_residue_evidence(pool, con)
    finally:
        con.close()

    all_evidences = list(evidences)
    ev_by_pid = {e["program_id"]: e for e in all_evidences}
    ev_by_name = {e.get("name"): e for e in all_evidences}

    staged_latest, already_l2, already_l3, pilot_names, pilot_l3_names = _pilot_resume_sets()

    if args.mode == "pilot":
        evidences = evidences[: args.limit]
        print(f"[pilot] limiting to {len(evidences)} candidate(s) (--limit {args.limit})")
    else:
        # Resume: do not re-spend Layer 2 on the pilot's 94 (or anything
        # already staged at layer 2). Layer 1 may have since resolved a
        # few of those names; they're already gone from `residue`.
        before = len(evidences)
        evidences = [
            e for e in evidences
            if e["program_id"] not in already_l2 and e.get("name") not in pilot_names
        ]
        print(f"[full] skipping {before - len(evidences)} already-processed Layer 2 candidate(s); "
              f"{len(evidences)} remaining.")

    with_text, no_text = tpl.partition_by_text_evidence(evidences)
    print(f"{len(with_text)} candidate(s) have text evidence -> Layer 2; "
          f"{len(no_text)} have none -> straight to Layer 3 (fix 2)")

    total_cost = 0.0
    pilot_rows: list[dict] = []
    n_staged = n_failed = 0
    n_l2_to_l3 = 0

    layer3_queue: list[dict] = list(no_text)
    seen_l3 = {e["program_id"] for e in layer3_queue}

    def _enqueue_l3(ev: dict) -> None:
        pid = ev["program_id"]
        if pid in already_l3 or pid in seen_l3:
            return
        layer3_queue.append(ev)
        seen_l3.add(pid)

    if args.mode == "full":
        n_pilot_l3 = 0
        for name in pilot_l3_names:
            ev = ev_by_name.get(name)
            if ev:
                before_n = len(seen_l3)
                _enqueue_l3(ev)
                if len(seen_l3) > before_n:
                    n_pilot_l3 += 1
        if n_pilot_l3:
            print(f"[full] queued {n_pilot_l3} pilot-routed candidate(s) for Layer 3.")
        # Re-ground staged Layer 2: a yes/no that claimed text without a
        # probative quote must go to Layer 3, not sit as a trusted L2 no.
        n_reground = 0
        for pid, rec in staged_latest.items():
            if rec.get("layer") != 2 or rec.get("is_adc") not in ("yes", "no"):
                continue
            _fr, forced = grounding.apply_grounding(
                rec.get("is_adc"), bool(rec.get("from_recall")), rec.get("quote"),
            )
            if not (forced or rec.get("from_recall")):
                continue
            ev = ev_by_pid.get(pid)
            if ev is None:
                continue  # already gold or no longer in residue
            before_n = len(seen_l3)
            _enqueue_l3(ev)
            if len(seen_l3) > before_n:
                n_reground += 1
        if n_reground:
            print(f"[full] re-grounded {n_reground} staged Layer 2 decision(s) onto the Layer 3 queue.")

    # ---- Layer 2 ----
    layer2_groups = layer2.group_into_batches(with_text)
    n_l2 = 0
    for group in layer2_groups:
        try:
            results, log = layer2.run_layer2(group, model=args.model)
        except model_client.ModelClientError as e:
            for ev in group:
                _log_failure(ev["program_id"], ev["name"], e)
                n_failed += 1
            continue
        total_cost += log["usage"]["cost_usd"]
        for ev in group:
            answer = results.get(ev["program_id"])
            if answer is None:
                continue
            n_l2 += 1
            routes_to_l3 = layer2.route_to_layer3(answer)
            if routes_to_l3:
                _enqueue_l3(ev)
                n_l2_to_l3 += 1
            else:
                try:
                    _stage_layer2(ev, answer, run_id=run_id, model=args.model)
                    n_staged += 1
                except tpool.PoolIntegrityError as e:
                    _log_failure(ev["program_id"], ev["name"], e)
                    n_failed += 1
                    continue

            pilot_rows.append({
                "program_id": ev["program_id"], "name": ev["name"],
                "verdict": answer.is_adc, "confidence": answer.confidence,
                "evidence_source": answer.evidence_source,
                "quote": answer.quote, "from_recall": answer.from_recall,
                "routed_to_layer3": routes_to_l3,
                "grounding_forced_recall": answer.grounding_forced_recall,
            })

        if args.max_spend is not None and total_cost > args.max_spend:
            print(f"--max-spend ${args.max_spend:.2f} exceeded — stopping Layer 2 early, "
                  f"{n_staged} decision(s) already staged are safe.")
            break

    # ---- Layer 3 (pilot mode: skipped — Layer 2 alone is the pilot's point) ----
    n_l3_run = 0
    if args.mode == "full":
        projected = int(round(0.43 * n_l2)) if n_l2 else 0
        print(f"\nLayer 3 queue: {len(layer3_queue)} "
              f"(this-run Layer 2 routed {n_l2_to_l3}/{n_l2 or 0}; "
              f"43% projection on this-run Layer 2 would be ~{projected}). "
              f"Cap {layer3.MAX_LAYER3_CANDIDATES}.")
        within_cap, overflow = tpl.cap_layer3_queue(layer3_queue)
        if overflow:
            n_flagged = tpl.stage_manual_overflow(overflow, run_id=run_id, reason="layer3 cap exceeded")
            print(f"⚠ Layer 3 queue ({len(layer3_queue)}) exceeds cap ({layer3.MAX_LAYER3_CANDIDATES}) — "
                  f"{n_flagged} candidate(s) flagged manual_overflow=True in the staging table (fix 3), "
                  "never auto-decided or dropped.")
        l3_payload = [{"program_id": c["program_id"], "name": c["name"]} for c in within_cap]
        if l3_payload:
            try:
                answers, log = layer3.run_layer3(l3_payload, model=args.model)
            except model_client.ModelClientError as e:
                for c in l3_payload:
                    _log_failure(c["program_id"], c["name"], e)
                    n_failed += 1
                answers = {}
            else:
                n_l3_run = len(answers)
                total_cost += log.get("usage", {}).get("cost_usd", 0.0)
                for c in l3_payload:
                    answer = answers.get(c["program_id"])
                    if answer is None:
                        continue
                    try:
                        tpool.assert_not_reviewed(c["program_id"])
                    except tpool.PoolIntegrityError as e:
                        _log_failure(c["program_id"], c["name"], e)
                        n_failed += 1
                        continue
                    record = staging.build_record({
                        "program_id": c["program_id"], "proposed_name": c["name"],
                        "is_adc": answer.is_adc, "layer": 3,
                        "model": args.model, "prompt_version": layer3.PROMPT_VERSION,
                        "quote": answer.quote, "from_recall": False,
                        "evidence_source": "text" if answer.quote else "no_usable_evidence",
                    }, run_id=run_id)
                    staging.append_record(record)
                    n_staged += 1

        # Validation sample across ALL staged yes/no (pilot + this run), not
        # just this run_id — otherwise the 80-sample would ignore the pilot.
        latest = staging.latest_by_program(staging.load_records())
        decisions = [
            r for r in latest.values()
            if r.get("is_adc") in ("yes", "no") and not r.get("manual_overflow")
        ]
        sample = tval.draw_stratified_sample(decisions)
        tval.save_validation_sample(sample)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        blind = REPORTS_DIR / "triage_validation_blind.html"
        blind.write_text(trep.render_validation_blind_html(sample), encoding="utf-8")
        print(f"Validation sample drawn: {len(sample)} candidate(s) -> {tval.VALIDATION_SAMPLE_PATH}")
        print(f"Blind review page: {blind}")
        print(f"Layer 3 actually run: {n_l3_run} (queue was {len(layer3_queue)}; "
              f"this-run L2→L3 rate {n_l2_to_l3}/{n_l2 or 0} = "
              f"{(n_l2_to_l3 / n_l2) if n_l2 else 0:.0%} vs 43% pilot projection).")

    print(f"\nStaged {n_staged} decision(s), {n_failed} failure(s) logged to {FAILURES_PATH}.")
    print(f"Total real spend this run: ${total_cost:.2f}")

    if args.mode == "pilot":
        _write_pilot_reports(pilot_rows, run_id=run_id, cost=total_cost)


def cmd_review_page(_args) -> None:
    sample = tval.load_validation_sample()
    if not sample:
        print(f"No validation sample at {tval.VALIDATION_SAMPLE_PATH} — run --full first.")
        sys.exit(1)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "triage_validation_blind.html"
    out.write_text(trep.render_validation_blind_html(sample), encoding="utf-8")
    print(f"Wrote {out} ({len(sample)} candidates, verdicts withheld).")


def cmd_report(_args) -> None:
    sample = tval.load_validation_sample()
    if not sample:
        print("No validation sample yet.")
        sys.exit(1)
    agreement = tval.compute_agreement(sample)
    passed, reason = tval.check_gate(agreement)
    print(f"Validation sample: {len(sample)}")
    print(f"is_adc: {agreement['is_adc']}")
    print(f"in_scope: {agreement['in_scope']}")
    print(f"gate: {'PASSED' if passed else 'CLOSED'} — {reason}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Step A — no completion calls")
    ap.add_argument("--pilot", action="store_true", help="Step B — real Layer 2 spend, writes reports/triage_pilot.md")
    ap.add_argument("--full", action="store_true", help="Step C — real Layer 2+3 spend, resume-safe, cost-capped")
    ap.add_argument("--review-page", action="store_true", help="Step D — blind HTML for the validation sample")
    ap.add_argument("--report", action="store_true", help="Step E — agreement report once the sample is judged")
    ap.add_argument("--limit", type=int, default=100, help="cap on with-text candidates entering Layer 2 (default 100)")
    ap.add_argument("--max-spend", type=float, default=None, help="abort Layer 2 once exceeded; already-staged decisions are kept")
    ap.add_argument("--model", default=model_client.DEFAULT_MODEL)
    args = ap.parse_args()

    modes = [m for m in (args.dry_run, args.pilot, args.full, args.review_page, args.report) if m]
    if len(modes) != 1:
        print("Pass exactly one of --dry-run / --pilot / --full / --review-page / --report.")
        sys.exit(1)

    if args.dry_run:
        cmd_dry_run(args)
        return
    if args.review_page:
        cmd_review_page(args)
        return
    if args.report:
        cmd_report(args)
        return

    args.mode = "pilot" if args.pilot else "full"
    cmd_pilot_or_full(args)


if __name__ == "__main__":
    main()
