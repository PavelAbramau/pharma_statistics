"""Tests for models/backtest.py's curve summarization and the audit
gate's comparison logic — the part of this whole build that decides
PASS/FAIL, so it gets the most scrutiny."""
from __future__ import annotations

from datetime import date

from pharma_stats.models import backtest as bt
from pharma_stats.models.backtest import FlagResult, compare_at_matched_precision, precision_lead_time_curve


def _flag(pid, flag_date, true_outcome, true_event_date=None):
    return FlagResult(pid, flag_date, true_outcome, true_event_date)


def test_model_flag_dates_from_panels_handles_none_silence_score():
    """Real bug found 2026-09-04: a panel row with no resolvable trial
    state has silence_score_asof=None, which statsmodels' predict()
    chokes on with a raw TypeError rather than a clean NaN error. Must
    be imputed the same way build_training_table already does."""
    from pharma_stats.models.discrete_time_survival import fit_cause_specific_hazard
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 60
    train_df = pd.DataFrame({
        "silence_score_asof": rng.uniform(0, 100, n),
        "log_cost_index": rng.uniform(0, 5, n),
        "cost_index": rng.uniform(0, 100, n),
        "sponsor": rng.integers(0, 10, n),
        "event_dead": [1] * 12 + [0] * (n - 12),
    })
    hazard = fit_cause_specific_hazard(train_df, "event_dead")

    panel = bt.ProgramPanel(
        program_id="p1",
        post_cutoff_rows=[
            {"as_of": "2023-01-01", "silence_score_asof": None, "band_asof": None, "cost_index": None},
            {"as_of": "2023-02-01", "silence_score_asof": 50.0, "band_asof": 2, "cost_index": 10.0},
        ],
        true_outcome="dead", true_event_date=date(2023, 3, 1),
    )
    flags = bt.model_flag_dates_from_panels([panel], hazard, threshold=0.01)
    assert "p1" in flags  # must not raise


def test_precision_lead_time_curve_basic():
    flags = {
        "p1": _flag("p1", date(2021, 1, 1), "dead", date(2021, 3, 1)),  # correct, lead = -59d (flagged BEFORE confirmation... wait sign)
        "p2": _flag("p2", date(2021, 6, 1), "active"),  # flagged, wrong (false positive)
        "p3": _flag("p3", None, "dead", date(2021, 1, 1)),  # never flagged -- doesn't count as flagged at all
    }
    point = precision_lead_time_curve(flags)
    assert point.n_flagged == 2  # p1, p2 (p3 never flagged)
    assert point.n_correct == 1  # only p1 is truly dead
    assert point.precision == 0.5


def test_precision_lead_time_curve_lead_time_sign():
    # flag_date BEFORE the real confirmation date -> negative days -> a
    # REAL positive lead (the model got there first) once redefined as
    # model_flag_date - public_confirmation_date... wait: lead_time =
    # flag_date - confirmation_date. Flagging 60 days before confirmation
    # means flag_date < confirmation_date, so the raw days value is
    # NEGATIVE. That's intentional: this project defines an early flag
    # as advance warning, i.e. flag_date - confirmation_date < 0 IS the
    # good case ("caught it before the world knew").
    flags = {"p1": _flag("p1", date(2021, 1, 1), "dead", date(2021, 3, 1))}
    point = precision_lead_time_curve(flags)
    assert point.lead_times_days == [(date(2021, 1, 1) - date(2021, 3, 1)).days]
    assert point.lead_times_days[0] == -59


def test_precision_lead_time_curve_no_flags_is_none_precision():
    flags = {"p1": _flag("p1", None, "dead", date(2021, 1, 1))}
    point = precision_lead_time_curve(flags)
    assert point.n_flagged == 0
    assert point.precision is None
    assert point.median_lead_time_days is None


def _curve_point(threshold, precision, n_correct, median_lead):
    from pharma_stats.models.backtest import CurvePoint
    return CurvePoint(threshold=threshold, n_flagged=n_correct + 5, n_correct=n_correct,
                       precision=precision, median_lead_time_days=median_lead)


def test_gate_passes_when_model_beats_heuristic_at_matched_precision():
    model_curve = [_curve_point(0.5, 0.6, 10, median_lead=-90)]  # 90 days early
    heuristic_curve = [_curve_point(3, 0.6, 8, median_lead=-30)]  # only 30 days early
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is True


def test_gate_fails_when_heuristic_beats_model():
    model_curve = [_curve_point(0.5, 0.6, 10, median_lead=-20)]
    heuristic_curve = [_curve_point(3, 0.6, 8, median_lead=-90)]
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is False


def test_gate_fails_loudly_when_model_never_reaches_min_precision():
    model_curve = [_curve_point(0.5, 0.3, 2, median_lead=-90)]  # below min_precision
    heuristic_curve = [_curve_point(3, 0.6, 8, median_lead=-30)]
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is False
    assert "never reaches" in result.reason


def test_gate_fails_loudly_when_neither_curve_is_usable():
    model_curve = [_curve_point(0.5, None, 0, median_lead=None)]
    heuristic_curve = [_curve_point(3, None, 0, median_lead=None)]
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is False
    assert "no usable comparison" in result.reason


def test_build_program_panels_truncates_at_event_date_not_panel_end(monkeypatch):
    """Real bug found 2026-09-04: a program whose event already happened
    before cutoff must contribute ZERO post-cutoff rows, never a
    carried-forward "still resolvable" state extended all the way to
    panel_end — that phantom window produced impossible, artificially
    LATE (positive) lead times in the first real backtest run."""
    from pharma_stats.labelling import store

    program = {"program_id": "p1", "proposed_name": "Drug X", "synonyms": [], "trials": []}
    gold = {"p1": {"status": "dead_confirmed", "public_confirmation_date": "2020-06-01",
                   "gate_reached": 3, "action": "label", "is_repeat_probe": False,
                   "program_id": "p1", "timestamp": "2020-06-01T00:00:00Z"}}
    monkeypatch.setattr(store, "load_records", lambda: [gold["p1"]])
    monkeypatch.setattr(store, "latest_by_program", lambda records: gold)

    captured_end = {}

    def fake_panel(program, con, end=None):
        captured_end["end"] = end
        # simulate carry-forward: rows exist all the way to `end`, which
        # is exactly the behaviour that must be truncated at the event
        # date, not left extending to panel_end.
        months = []
        cursor = date(2018, 1, 1)
        while cursor <= end:
            months.append({"as_of": cursor.isoformat(), "band_asof": 3, "cost_index": 0.0})
            cursor = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
        return months

    monkeypatch.setattr(bt, "build_program_month_panel", fake_panel)

    panels = bt.build_program_panels([program], con=None, cutoff=date(2022, 1, 1), panel_end=date(2026, 1, 1))
    assert captured_end["end"] == date(2020, 6, 1)  # truncated at the event date, never panel_end
    assert panels[0].post_cutoff_rows == []  # event was before cutoff -> nothing to evaluate


def test_gate_passes_when_heuristic_never_reaches_precision_but_model_does():
    model_curve = [_curve_point(0.5, 0.6, 10, median_lead=-90)]
    heuristic_curve = [_curve_point(3, 0.2, 1, median_lead=-30)]
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is True
    assert "heuristic never does" in result.reason
