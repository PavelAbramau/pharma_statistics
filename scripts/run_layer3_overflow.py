"""Reprocess the manual_overflow candidates through Layer 3, with the
cost-reduction levers from the $32.83 cost investigation applied
(response_inclusion=excluded, a terser prompt — see triage/layer3.py's
module docstring for exactly what's real here and what isn't: there is
NO literal "truncate to N tokens" parameter on web_search).

    python scripts/run_layer3_overflow.py --dry-run
    python scripts/run_layer3_overflow.py --max-spend 5

Processes in small chunks (CHUNK_SIZE) so --max-spend can actually abort
mid-run against REAL accumulated cost, not just refuse to start a next
chunk after the whole thing already ran. Stages results — does not write
gold. Resume-safe: candidates already re-staged by a prior run of this
script are skipped.
"""
from __future__ import annotations

import argparse
import uuid

from pharma_stats.silver import model_client
from pharma_stats.triage import layer3, pool as tpool, staging

CHUNK_SIZE = 20
RERUN_RULE_TAG = "layer3_overflow_rerun"


def _load_overflow_candidates() -> list[dict]:
    staged = staging.load_records()
    latest = staging.latest_by_program(staged)
    return [r for r in latest.values() if r.get("manual_overflow")]


def _already_rerun_ids(staged_records: list[dict]) -> set[str]:
    latest = staging.latest_by_program(staged_records)
    return {pid for pid, r in latest.items() if r.get("rule") == RERUN_RULE_TAG}


def _estimate(n: int, model: str) -> dict:
    """Honest range, not a point estimate — see layer3.py's module
    docstring. Flat search fee is certain ($10/1000, not batch-discounted);
    token cost has a wide range because response_inclusion=excluded's real
    effect on output tokens hasn't been measured yet."""
    flat_fee = (n / 1000) * model_client.WEB_SEARCH_COST_PER_1000
    optimistic_tokens = model_client.estimate_cost(1500 * n, 300 * n, model, batch=True)
    pessimistic_tokens = model_client.estimate_cost(2000 * n, 4000 * n, model, batch=True)
    return {
        "n": n, "flat_fee_usd": flat_fee,
        "optimistic_total_usd": flat_fee + optimistic_tokens,
        "pessimistic_total_usd": flat_fee + pessimistic_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--model", default=model_client.DEFAULT_MODEL)
    args = ap.parse_args()

    candidates = _load_overflow_candidates()
    staged_records = staging.load_records()
    already = _already_rerun_ids(staged_records)
    candidates = [c for c in candidates if c["program_id"] not in already]
    print(f"{len(candidates)} manual_overflow candidate(s) to reprocess "
          f"({len(already)} already re-run and skipped)")

    est = _estimate(len(candidates), args.model)
    print(f"Flat search fee (certain, not batch-discounted): ${est['flat_fee_usd']:.2f}")
    print(f"Estimated total: ${est['optimistic_total_usd']:.2f} optimistic "
          f"to ${est['pessimistic_total_usd']:.2f} pessimistic "
          "(wide range — response_inclusion=excluded's real effect on output tokens is unmeasured; "
          "the flat fee alone already exceeds a $1.50 target for this many candidates).")
    if args.dry_run:
        print("[dry run] no API calls made.")
        return

    if args.max_spend is not None and est["flat_fee_usd"] > args.max_spend:
        print(f"--max-spend ${args.max_spend:.2f} is below the CERTAIN flat search fee "
              f"(${est['flat_fee_usd']:.2f}) for {len(candidates)} candidates — refusing to start at all.")
        return

    run_id = f"triage:layer3_overflow:{uuid.uuid4().hex[:8]}"
    payload = [{"program_id": c["program_id"], "name": c["proposed_name"]} for c in candidates]
    chunks = [payload[i:i + CHUNK_SIZE] for i in range(0, len(payload), CHUNK_SIZE)]

    total_cost = 0.0
    n_staged = 0
    for i, chunk in enumerate(chunks):
        try:
            answers, log = layer3.run_layer3(chunk, model=args.model)
        except model_client.ModelClientError as e:
            print(f"[chunk {i + 1}/{len(chunks)}] FAILED: {e} — logged, continuing")
            continue

        chunk_cost = log["usage"]["cost_usd"]
        total_cost += chunk_cost
        print(f"[chunk {i + 1}/{len(chunks)}] {len(chunk)} candidate(s), "
              f"${chunk_cost:.4f} (search fee ${log['usage']['web_search_flat_fee_usd']:.4f}), "
              f"running total ${total_cost:.2f}")

        for c in chunk:
            answer = answers.get(c["program_id"])
            if answer is None:
                continue
            try:
                tpool.assert_not_reviewed(c["program_id"])
            except tpool.PoolIntegrityError as e:
                print(f"  SKIPPED (pool integrity): {e}")
                continue
            record = staging.build_record({
                "program_id": c["program_id"], "proposed_name": c["name"],
                "is_adc": answer.is_adc, "layer": 3, "rule": RERUN_RULE_TAG,
                "model": args.model, "prompt_version": layer3.PROMPT_VERSION,
                "quote": answer.quote,
            }, run_id=run_id)
            staging.append_record(record)
            n_staged += 1

        if args.max_spend is not None and total_cost > args.max_spend:
            print(f"\n--max-spend ${args.max_spend:.2f} exceeded (spent ${total_cost:.2f}) — "
                  f"stopping after chunk {i + 1}/{len(chunks)}. "
                  f"{n_staged} decision(s) already staged are safe; "
                  f"{len(chunks) - i - 1} chunk(s) not yet run.")
            break

    print(f"\nStaged {n_staged} decision(s), total spend ${total_cost:.2f} (run_id={run_id}).")


if __name__ == "__main__":
    main()
