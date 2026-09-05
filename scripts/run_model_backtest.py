"""End-to-end: build the training table, fit the cause-specific hazards,
run the time-cut backtest for `dead` against the silence-score heuristic,
and print the audit gate's verdict. See docs/decisions/0005-0007 and
models/discrete_time_survival.py / models/backtest.py's module
docstrings for the full design and its honest limitations.

    python scripts/run_model_backtest.py [--cutoff YYYY-MM-DD] [--min-precision 0.5]

Writes reports/model_backtest.md. Prints per-outcome event counts
(dead/approved/superseded) and flags the fitted hazards' coefficients
with cluster-robust (sponsor) SEs for `dead` — the only outcome with
enough events to trust a coefficient at all.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import duckdb

from pharma_stats.config import DATA_DIR, REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.models import backtest as bt
from pharma_stats.models import discrete_time_survival as dts

MODEL_RESULT_PATH = DATA_DIR / "model_backtest_result.json"
FLAG_THRESHOLD = 0.5  # the fixed operating point used to publish model_flag_date for label_sufficiency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2022-01-01")
    ap.add_argument("--min-precision", type=float, default=0.5)
    args = ap.parse_args()
    cutoff = date.fromisoformat(args.cutoff)
    panel_end = date.today()

    programs = pp.load_materialized()
    gold_records = store.load_records()
    gold_latest = store.latest_by_program(gold_records)
    gate3_programs = [p for p in programs if gold_latest.get(p["program_id"], {}).get("gate_reached") == 3]
    print(f"{len(gate3_programs)} gate-3 (fully labelled) programs.")

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        print("Building training table (panel truncated at cutoff)...")
        train_df = dts.build_training_table(gate3_programs, con, panel_end=cutoff)
        print(f"Training table: {len(train_df)} (program, month) rows.")
        for oc in dts.OUTCOME_CLASSES:
            print(f"  event_{oc}: {int(train_df[f'event_{oc}'].sum())} events "
                  f"({train_df['program_id'].nunique()} programs)")

        print("\nFitting cause-specific hazards...")
        hazards = {}
        for oc in dts.OUTCOME_CLASSES:
            hazards[oc] = dts.fit_cause_specific_hazard(train_df, f"event_{oc}")
            h = hazards[oc]
            print(f"  {oc}: {h.n_events} events, covariates={h.covariates or '(intercept-only)'}")
            if h.covariates:
                print(f"    coef={dict(h.result.params.round(4))}")
                print(f"    cluster-robust SE={dict(h.result.bse.round(4))}")

        print(f"\nRunning backtest at cutoff={cutoff} (building each program's full panel once)...")
        panels = bt.build_program_panels(gate3_programs, con, cutoff=cutoff, panel_end=panel_end)
        print(f"Built {len(panels)} program panel(s).")

        # Panel-coverage diagnostic: a truncation fix that's too
        # aggressive can leave post-cutoff deaths with nothing to score
        # against at all, in which case the curve above is silently
        # starved of positives regardless of threshold. Checked directly
        # rather than inferred from the curve's recall numbers.
        post_cutoff_deaths = [p for p in panels if p.true_outcome == "dead"
                               and p.true_event_date is not None and p.true_event_date > cutoff]
        deaths_with_rows = [p for p in post_cutoff_deaths if p.post_cutoff_rows]
        print(f"\n{len(post_cutoff_deaths)} 'dead' program(s) confirmed after {cutoff}; "
              f"{len(deaths_with_rows)} have >=1 post-cutoff panel row "
              f"({len(post_cutoff_deaths) - len(deaths_with_rows)} have none).")

        model_curve = bt.build_curve(gate3_programs, con, hazards["dead"], cutoff=cutoff, panel_end=panel_end,
                                      use_heuristic=False, panels=panels)
        heuristic_curve = bt.build_curve(gate3_programs, con, None, cutoff=cutoff, panel_end=panel_end,
                                          use_heuristic=True, panels=panels)
        model_best = bt.best_precision_point(model_curve)
        heuristic_best = bt.best_precision_point(heuristic_curve)

        # Fresh (non-cutoff, full-data) fit + flag dates at a fixed
        # threshold — this is what audit/label_sufficiency.py's cluster
        # bootstrap consumes as model_flag_date. Separate from the
        # backtest's cutoff-truncated fit (that one exists to test
        # out-of-sample precision honestly); this one is the model's
        # actual best current estimate, using everything on file.
        print("\nFitting the full-data hazard for published flag dates...")
        full_df = dts.build_training_table(gate3_programs, con, panel_end=panel_end)
        full_hazard = dts.fit_cause_specific_hazard(full_df, "event_dead")
        full_panels = bt.build_program_panels(gate3_programs, con, cutoff=date(2000, 1, 1), panel_end=panel_end)
        flags = bt.model_flag_dates_from_panels(full_panels, full_hazard, threshold=FLAG_THRESHOLD)
        flag_date_by_program = {
            pid: r.flag_date.isoformat() for pid, r in flags.items() if r.flag_date is not None
        }
    finally:
        con.close()

    gate = bt.compare_at_matched_precision(model_curve, heuristic_curve, min_precision=args.min_precision)

    lines = ["# Model backtest", "", f"Cutoff: {cutoff}. Panel end: {panel_end}.", "",
              "## Panel coverage (post-cutoff deaths)", "",
              f"{len(post_cutoff_deaths)} 'dead' program(s) confirmed after {cutoff}; "
              f"{len(deaths_with_rows)} have >=1 post-cutoff panel row "
              f"({len(post_cutoff_deaths) - len(deaths_with_rows)} have none — these can never be "
              "flagged at any threshold, by construction).",
              "", "## Training event counts", ""]
    for oc in dts.OUTCOME_CLASSES:
        lines.append(f"- {oc}: {int(train_df[f'event_{oc}'].sum())} events")
    def _row(p) -> str:
        # precision and median_lead_time_days are independently
        # nullable: n_flagged>0 with n_correct==0 gives a real precision
        # of 0.0 (not None) but no lead times to compute a median from.
        prec = f"{p.precision:.0%}" if p.precision is not None else "n/a"
        rec = f"{p.recall:.0%}" if p.recall is not None else "n/a"
        lead = f"{p.median_lead_time_days:.0f}" if p.median_lead_time_days is not None else "n/a"
        return f"| {p.threshold} | {p.n_flagged} | {p.n_correct} | {p.n_true_dead} | {prec} | {rec} | {lead} |"

    def _best_line(label: str, best) -> str:
        if best is None:
            return f"- {label}: no threshold on this curve ever flags anything (precision undefined everywhere)."
        prec = f"{best.precision:.0%}" if best.precision is not None else "n/a"
        rec = f"{best.recall:.0%}" if best.recall is not None else "n/a"
        lead = f"{best.median_lead_time_days:.0f}d" if best.median_lead_time_days is not None else "n/a"
        return (f"- {label}: best precision {prec} at threshold={best.threshold} "
                f"(n_flagged={best.n_flagged}, n_correct={best.n_correct}, recall={rec}, "
                f"median lead time={lead}).")

    lines += ["", "## Model curve (dead, cause-specific hazard)", "",
              "Threshold sweep covers every observed predicted-hazard value post-cutoff "
              "(not a fixed a-priori grid — see backtest.observed_model_thresholds).", "",
              "| threshold | n_flagged | n_correct | n_true_dead | precision | recall | median_lead_days |",
              "|---|---|---|---|---|---|---|"]
    lines += [_row(p) for p in model_curve]
    lines += ["", _best_line("Model best-precision point", model_best)]

    lines += ["", "## Heuristic curve (silence-score band)", "",
              "| band>= | n_flagged | n_correct | n_true_dead | precision | recall | median_lead_days |",
              "|---|---|---|---|---|---|---|"]
    lines += [_row(p) for p in heuristic_curve]
    lines += ["", _best_line("Heuristic best-precision point", heuristic_best)]

    lines += ["", "## Audit gate", "", f"**{'PASS' if gate.passed else 'FAIL'}** — {gate.reason}", ""]

    text = "\n".join(lines)
    print("\n" + text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "model_backtest.md"
    out.write_text(text, encoding="utf-8")
    print(f"\nWrote {out}")

    def _point_dict(p):
        return {"threshold": p.threshold, "n_flagged": p.n_flagged, "n_correct": p.n_correct,
                "n_true_dead": p.n_true_dead, "precision": p.precision, "recall": p.recall,
                "median_lead_time_days": p.median_lead_time_days}

    result = {
        "generated_at": date.today().isoformat(),
        "cutoff": cutoff.isoformat(),
        "min_precision": args.min_precision,
        "gate_passed": gate.passed,
        "gate_reason": gate.reason,
        "training_event_counts": {oc: int(train_df[f"event_{oc}"].sum()) for oc in dts.OUTCOME_CLASSES},
        "post_cutoff_deaths": len(post_cutoff_deaths),
        "post_cutoff_deaths_with_panel_rows": len(deaths_with_rows),
        "model_curve": [_point_dict(p) for p in model_curve],
        "heuristic_curve": [_point_dict(p) for p in heuristic_curve],
        "model_best_precision_point": _point_dict(model_best) if model_best is not None else None,
        "heuristic_best_precision_point": _point_dict(heuristic_best) if heuristic_best is not None else None,
        "flag_threshold": FLAG_THRESHOLD,
        "flag_date_by_program": flag_date_by_program,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {MODEL_RESULT_PATH} ({len(flag_date_by_program)} program flag dates)")
    print(f"\nGATE: {'PASS' if gate.passed else 'FAIL'}")


if __name__ == "__main__":
    main()
