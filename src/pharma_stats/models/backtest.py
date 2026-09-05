"""Time-cut backtest for the `dead` cause-specific hazard: fit on data
truncated at a cutoff date (administrative censoring for anyone whose
real event falls after it), predict forward month-by-month past the
cutoff, and compare the resulting precision/lead-time curve against the
existing silence-score heuristic's own curve at the same cutoff. The
model must beat the heuristic or this FAILs — see audit/model.py.

Only `dead` is evaluated here: it's the only outcome with a real,
ground-truth event date (docs/decisions/0005, 0007) and enough events
(63) to fit and test anything. approved/superseded are reported
separately as descriptive counts, never put through this same curve.

Panels are built ONCE per program and reused across every threshold on
the curve (build_curve does this) — a first draft rebuilt each program's
full panel (already the single most expensive operation in this whole
pipeline — see features/trial_asof.py's module docstring) once per
threshold, ~15x more I/O than necessary for no reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from pharma_stats.features.panel import build_program_month_panel
from pharma_stats.labelling import store
from pharma_stats.models.discrete_time_survival import CauseSpecificHazard, determine_program_outcome

BAND_THRESHOLDS = [0, 1, 2, 3, 4]


@dataclass
class FlagResult:
    program_id: str
    flag_date: Optional[date]  # first month the rule crosses threshold, None if never
    true_outcome: str
    true_event_date: Optional[date]


def _first_crossing(dates: list[date], scores: list[float], threshold: float) -> Optional[date]:
    for d, s in zip(dates, scores):
        if s >= threshold:
            return d
    return None


@dataclass
class ProgramPanel:
    program_id: str
    post_cutoff_rows: list[dict]  # panel rows with as_of >= cutoff, up to panel_end
    true_outcome: str
    true_event_date: Optional[date]


def build_program_panels(
    programs: list[dict], con: duckdb.DuckDBPyConnection, *, cutoff: date, panel_end: date,
) -> list[ProgramPanel]:
    """The expensive pass — one full panel build per program, done ONCE
    regardless of how many thresholds get evaluated afterward.

    Real bug found and fixed (2026-09-04): a program whose true event
    happened BEFORE cutoff still had its panel extended all the way to
    panel_end (today) — the last known pre-event state just carries
    forward for every later month (see features/trial_asof.py's
    carry-forward design). Evaluating "flags" in that carried-forward
    window against a confirmation date that's already in the past
    produces impossible, artificially LATE (positive) lead times: the
    model/heuristic can't possibly flag something before the evaluation
    window even starts. Truncating each panel at min(event_month,
    panel_end) — same as build_training_table — means a program already
    resolved before cutoff correctly contributes NO post-cutoff rows at
    all, rather than a phantom "flagged too late" one."""
    gold_latest = store.latest_by_program(store.load_records())
    out = []
    for program in programs:
        pid = program["program_id"]
        g = gold_latest.get(pid)
        if g is None or g.get("gate_reached") != 3:
            continue
        outcome = determine_program_outcome(program, g, panel_end)
        rows = build_program_month_panel(program, con, end=min(outcome.event_month, panel_end))
        post_cutoff_rows = [r for r in rows if date.fromisoformat(r["as_of"]) >= cutoff]
        out.append(ProgramPanel(pid, post_cutoff_rows, outcome.outcome, outcome.event_month))
    return out


def compute_model_predictions(
    panels: list[ProgramPanel], hazard: CauseSpecificHazard,
) -> dict[str, tuple[list[date], list[float]]]:
    """Predicted hazard per post-cutoff row, computed ONCE per panel and
    reused across every threshold on the curve — same one-pass-then-sweep
    shape as build_program_panels itself. Also the source of the observed
    prediction values that `observed_model_thresholds` sweeps over."""
    out = {}
    for panel in panels:
        if not panel.post_cutoff_rows:
            continue
        df = pd.DataFrame(panel.post_cutoff_rows)
        df["log_cost_index"] = np.log1p(df["cost_index"].fillna(0.0))
        # Same imputation as build_training_table: a month with no
        # resolvable trial state gets silence_score_asof=None, which
        # breaks statsmodels' predict() with a raw TypeError rather than
        # a clean NaN — see discrete_time_survival.build_training_table.
        df["silence_score_asof"] = df["silence_score_asof"].fillna(0.0).astype(float)
        preds = hazard.predict(df)
        dates_ = [date.fromisoformat(r["as_of"]) for r in panel.post_cutoff_rows]
        out[panel.program_id] = (dates_, [float(p) for p in preds])
    return out


def observed_model_thresholds(predictions: dict[str, tuple[list[date], list[float]]]) -> list[float]:
    """Threshold candidates for the model curve: every distinct predicted
    hazard value actually observed post-cutoff, not a fixed a-priori grid.

    A fixed grid (the old THRESHOLDS = arange(0.05, 0.95, 0.05)) silently
    assumes the fitted hazard's range overlaps that grid. It doesn't: a
    rare-event Logit intercept alone can put every prediction under 0.05,
    in which case a fixed grid reports "0 flags at every threshold" —
    true but uninformative, since it never actually samples where the
    model's scores live. Sweeping the observed values is the same fix
    already applied to the heuristic curve, which sweeps BAND_THRESHOLDS
    (the heuristic's own small observed domain) rather than some
    unrelated scale."""
    values = sorted({round(p, 6) for _, preds in predictions.values() for p in preds})
    return values


def model_flag_dates_from_predictions(
    panels: list[ProgramPanel], predictions: dict[str, tuple[list[date], list[float]]], *, threshold: float,
) -> dict[str, FlagResult]:
    out = {}
    for panel in panels:
        pred = predictions.get(panel.program_id)
        if pred is None:
            out[panel.program_id] = FlagResult(panel.program_id, None, panel.true_outcome, panel.true_event_date)
            continue
        dates_, preds = pred
        flag = _first_crossing(dates_, preds, threshold)
        out[panel.program_id] = FlagResult(panel.program_id, flag, panel.true_outcome, panel.true_event_date)
    return out


def model_flag_dates_from_panels(panels: list[ProgramPanel], hazard: CauseSpecificHazard, *, threshold: float) -> dict[str, FlagResult]:
    """Single-threshold convenience wrapper (used to publish
    model_flag_date at one fixed operating point — see
    scripts/run_model_backtest.py). The curve sweep uses
    compute_model_predictions + model_flag_dates_from_predictions
    instead, so predictions aren't recomputed once per threshold."""
    predictions = compute_model_predictions(panels, hazard)
    return model_flag_dates_from_predictions(panels, predictions, threshold=threshold)


def silence_heuristic_flag_dates_from_panels(panels: list[ProgramPanel], *, band_threshold: int) -> dict[str, FlagResult]:
    out = {}
    for panel in panels:
        if not panel.post_cutoff_rows:
            out[panel.program_id] = FlagResult(panel.program_id, None, panel.true_outcome, panel.true_event_date)
            continue
        dates_ = [date.fromisoformat(r["as_of"]) for r in panel.post_cutoff_rows]
        bands = [float(r["band_asof"]) if r["band_asof"] is not None else -1.0 for r in panel.post_cutoff_rows]
        flag = _first_crossing(dates_, bands, float(band_threshold))
        out[panel.program_id] = FlagResult(panel.program_id, flag, panel.true_outcome, panel.true_event_date)
    return out


@dataclass
class CurvePoint:
    threshold: float
    n_flagged: int
    n_correct: int  # flagged AND truly dead
    precision: Optional[float]
    median_lead_time_days: Optional[float]
    n_true_dead: int = 0  # total truly-dead programs in the evaluated population (recall denominator)
    recall: Optional[float] = None
    lead_times_days: list[float] = field(default_factory=list)


def precision_lead_time_curve(flag_results: dict[str, FlagResult]) -> CurvePoint:
    """Summarizes ONE set of flag results (one threshold's worth).

    n_true_dead (and therefore recall) is computed over ALL evaluated
    programs, including ones with no post-cutoff panel rows at all (flag
    date forced to None upstream) — those can never be flagged at any
    threshold, so leaving them in the denominator is what makes recall
    correctly reflect panel coverage rather than silently capping the
    denominator at only the programs the panel-build happened to leave
    something to score against."""
    flagged = [r for r in flag_results.values() if r.flag_date is not None]
    correct = [r for r in flagged if r.true_outcome == "dead"]
    true_dead = [r for r in flag_results.values() if r.true_outcome == "dead"]
    lead_times = [
        (r.flag_date - r.true_event_date).days for r in correct if r.true_event_date is not None
    ]
    precision = len(correct) / len(flagged) if flagged else None
    recall = len(correct) / len(true_dead) if true_dead else None
    median_lt = float(np.median(lead_times)) if lead_times else None
    return CurvePoint(
        threshold=0.0, n_flagged=len(flagged), n_correct=len(correct), n_true_dead=len(true_dead),
        precision=precision, recall=recall, median_lead_time_days=median_lt, lead_times_days=lead_times,
    )


def best_precision_point(curve: list[CurvePoint]) -> Optional[CurvePoint]:
    """The point achieving the highest precision anywhere on the curve
    (ties broken by earliest — most negative — median lead time, then by
    higher recall). None only if every point has n_flagged == 0."""
    usable = [p for p in curve if p.precision is not None]
    if not usable:
        return None
    return max(
        usable,
        key=lambda p: (p.precision, -(p.median_lead_time_days or 0.0), p.recall or 0.0),
    )


def build_curve(
    programs: list[dict], con: duckdb.DuckDBPyConnection, hazard: Optional[CauseSpecificHazard],
    *, cutoff: date, panel_end: date, use_heuristic: bool,
    panels: Optional[list[ProgramPanel]] = None,
) -> list[CurvePoint]:
    """Pass a pre-built `panels` (build_program_panels) when calling this
    more than once (e.g. once for the model curve, once for the
    heuristic curve, over the SAME programs/cutoff) to avoid rebuilding
    every program's panel twice."""
    if panels is None:
        panels = build_program_panels(programs, con, cutoff=cutoff, panel_end=panel_end)

    points = []
    if use_heuristic:
        for band_t in BAND_THRESHOLDS:
            flags = silence_heuristic_flag_dates_from_panels(panels, band_threshold=band_t)
            point = precision_lead_time_curve(flags)
            point.threshold = band_t
            points.append(point)
    else:
        # Sweep every observed predicted-hazard value, not a fixed a
        # priori grid — see observed_model_thresholds's docstring for why
        # a fixed grid can silently miss the range the fitted model
        # actually predicts in.
        predictions = compute_model_predictions(panels, hazard)
        for t in observed_model_thresholds(predictions):
            flags = model_flag_dates_from_predictions(panels, predictions, threshold=t)
            point = precision_lead_time_curve(flags)
            point.threshold = t
            points.append(point)
    return points


@dataclass
class GateResult:
    passed: bool
    reason: str
    model_curve: list[CurvePoint]
    heuristic_curve: list[CurvePoint]


def compare_at_matched_precision(
    model_curve: list[CurvePoint], heuristic_curve: list[CurvePoint], *, min_precision: float = 0.5,
) -> GateResult:
    """The gate: among points on each curve with precision >= min_precision
    and at least one correct flag, does the model's BEST median lead time
    beat the heuristic's BEST median lead time? Loud FAIL (not a silent
    skip) if either curve has no usable point at all — that's itself a
    finding, not a pass by default."""
    def best(curve: list[CurvePoint]) -> Optional[CurvePoint]:
        # lead_time = flag_date - public_confirmation_date (docs/decisions/0005)
        # -- MORE NEGATIVE means the flag came earlier, i.e. better. "Best"
        # is the most negative (earliest) median lead time, so min(), not max().
        usable = [p for p in curve if p.precision is not None and p.precision >= min_precision and p.n_correct > 0]
        if not usable:
            return None
        return min(usable, key=lambda p: p.median_lead_time_days)

    model_best = best(model_curve)
    heuristic_best = best(heuristic_curve)

    if model_best is None and heuristic_best is None:
        return GateResult(False, f"neither the model nor the silence heuristic reaches {min_precision:.0%} "
                                  "precision with any correct flag at this cutoff — no usable comparison",
                           model_curve, heuristic_curve)
    if model_best is None:
        return GateResult(False, f"model never reaches {min_precision:.0%} precision; "
                                  f"heuristic does (median lead {heuristic_best.median_lead_time_days:.0f}d)",
                           model_curve, heuristic_curve)
    if heuristic_best is None:
        return GateResult(True, f"model reaches {min_precision:.0%} precision (median lead "
                                 f"{model_best.median_lead_time_days:.0f}d); heuristic never does",
                           model_curve, heuristic_curve)

    passed = model_best.median_lead_time_days < heuristic_best.median_lead_time_days  # more negative = earlier = better
    reason = (f"model median lead {model_best.median_lead_time_days:.0f}d (precision "
              f"{model_best.precision:.0%}, n={model_best.n_correct}) vs heuristic "
              f"{heuristic_best.median_lead_time_days:.0f}d (precision {heuristic_best.precision:.0%}, "
              f"n={heuristic_best.n_correct})")
    return GateResult(passed, reason, model_curve, heuristic_curve)
