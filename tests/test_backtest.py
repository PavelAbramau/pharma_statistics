"""Tests for models/backtest.py's curve summarization and the audit
gate's comparison logic — the part of this whole build that decides
PASS/FAIL, so it gets the most scrutiny."""
from __future__ import annotations

from datetime import date

from pharma_stats.models.backtest import FlagResult, compare_at_matched_precision, precision_lead_time_curve


def _flag(pid, flag_date, true_outcome, true_event_date=None):
    return FlagResult(pid, flag_date, true_outcome, true_event_date)


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


def test_gate_passes_when_heuristic_never_reaches_precision_but_model_does():
    model_curve = [_curve_point(0.5, 0.6, 10, median_lead=-90)]
    heuristic_curve = [_curve_point(3, 0.2, 1, median_lead=-30)]
    result = compare_at_matched_precision(model_curve, heuristic_curve, min_precision=0.5)
    assert result.passed is True
    assert "heuristic never does" in result.reason
