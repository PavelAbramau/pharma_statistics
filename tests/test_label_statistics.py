"""Tests for pharma_stats.stats.label_statistics — pure functions over
hand-built (programs, records) fixtures, no warehouse or gold file I/O
needed since every function takes both as plain lists of dicts."""
from __future__ import annotations

import pytest

from pharma_stats.stats import label_statistics as ls


def _program(pid, *, band, archetype, sponsor="Acme", trials=None):
    return {
        "program_id": pid,
        "band": band,
        "primary_archetype": archetype,
        "sponsors_over_time": [{"sponsor": sponsor, "last_seen": "2026-01-01"}],
        "trials": trials or [],
    }


def _label(pid, *, band, archetype, status, kill_reason=None, timestamp="2026-01-01T00:00:00+00:00",
           gate_reached=3, is_repeat_probe=False, action="label"):
    return {
        "action": action, "gate_reached": gate_reached, "is_repeat_probe": is_repeat_probe,
        "program_id": pid, "timestamp": timestamp,
        "status": status, "kill_reason": kill_reason,
        "stratum_band": band, "stratum_archetype": archetype,
    }


# --- gate3_labels_by_program ------------------------------------------

def test_gate3_labels_by_program_excludes_triage_rejections_and_repeats():
    records = [
        _label("p1", band=0, archetype="other", status=None, gate_reached=1),
        _label("p2", band=0, archetype="other", status="active", gate_reached=3),
        _label("p2", band=0, archetype="other", status="active", gate_reached=3,
               is_repeat_probe=True, timestamp="2026-01-02T00:00:00+00:00"),
    ]
    out = ls.gate3_labels_by_program(records)
    assert set(out.keys()) == {"p2"}
    assert out["p2"]["is_repeat_probe"] is False


def test_gate3_labels_by_program_keeps_latest_revision():
    records = [
        _label("p1", band=0, archetype="other", status="active", timestamp="2026-01-01T00:00:00+00:00"),
        _label("p1", band=0, archetype="other", status="dead_confirmed",
               kill_reason="futility_efficacy", timestamp="2026-02-01T00:00:00+00:00"),
    ]
    out = ls.gate3_labels_by_program(records)
    assert out["p1"]["status"] == "dead_confirmed"


# --- compute_stratum_weights --------------------------------------------

def _two_stratum_fixture():
    programs = [
        _program("p1", band=0, archetype="other"),
        _program("p2", band=0, archetype="other"),
        _program("p3", band=0, archetype="other"),
        _program("p4", band=0, archetype="other"),
        _program("p5", band=3, archetype="registry_terminated_stated_reason"),
        _program("p6", band=3, archetype="registry_terminated_stated_reason"),
    ]
    records = [
        _label("p1", band=0, archetype="other", status="active"),
        _label("p2", band=0, archetype="other", status="dead_confirmed", kill_reason="futility_efficacy"),
        _label("p5", band=3, archetype="registry_terminated_stated_reason",
               status="dead_confirmed", kill_reason="strategic_portfolio"),
    ]
    return programs, records


def test_compute_stratum_weights():
    programs, records = _two_stratum_fixture()
    weights = ls.compute_stratum_weights(programs, records)
    assert weights[(0, "other")].population == 4
    assert weights[(0, "other")].labelled == 2
    assert weights[(0, "other")].weight == 2.0
    assert weights[(3, "registry_terminated_stated_reason")].population == 2
    assert weights[(3, "registry_terminated_stated_reason")].labelled == 1
    assert weights[(3, "registry_terminated_stated_reason")].weight == 2.0


def test_compute_stratum_weights_refuses_on_unlabelled_stratum():
    programs, records = _two_stratum_fixture()
    programs += [
        _program("p7", band=1, archetype="other"),
        _program("p8", band=1, archetype="other"),
        _program("p9", band=1, archetype="other"),
    ]  # a whole populated stratum with zero labels
    with pytest.raises(ls.InsufficientStratumCoverageError) as exc_info:
        ls.compute_stratum_weights(programs, records)
    assert exc_info.value.missing_strata == [(1, "other", 3)]


def test_compute_stratum_weights_ignores_unscored_population():
    programs, records = _two_stratum_fixture()
    programs.append(_program("p_unscored", band=None, archetype="other"))
    # must not raise — band=None programs have no stratum to be missing from
    weights = ls.compute_stratum_weights(programs, records)
    assert (None, "other") not in weights


# --- weighted distributions ----------------------------------------------

def test_weighted_status_distribution_matches_hand_computed_weights():
    programs, records = _two_stratum_fixture()
    dist = ls.weighted_status_distribution(programs, records)
    by_status = {row["status"]: row for row in dist}
    # weight(stratum A)=2.0 (p1 active), weight(stratum B)=2.0 (p5 dead via B)
    # + p2 dead via A weight 2.0 -> dead_confirmed weighted = 4.0, active = 2.0, total = 6.0
    assert by_status["active"]["raw_labelled_n"] == 1
    assert by_status["dead_confirmed"]["raw_labelled_n"] == 2
    assert abs(by_status["active"]["weighted_share"] - 2 / 6) < 1e-9
    assert abs(by_status["dead_confirmed"]["weighted_share"] - 4 / 6) < 1e-9


def test_weighted_kill_reason_distribution_only_covers_dead_confirmed():
    programs, records = _two_stratum_fixture()
    dist = ls.weighted_kill_reason_distribution(programs, records)
    by_reason = {row["kill_reason"]: row for row in dist}
    assert set(by_reason.keys()) == {"futility_efficacy", "strategic_portfolio"}
    assert abs(by_reason["futility_efficacy"]["weighted_share"] - 0.5) < 1e-9
    assert abs(by_reason["strategic_portfolio"]["weighted_share"] - 0.5) < 1e-9


def test_weighted_status_distribution_refuses_without_full_coverage():
    programs, records = _two_stratum_fixture()
    programs.append(_program("p_gap", band=2, archetype="other"))
    with pytest.raises(ls.InsufficientStratumCoverageError):
        ls.weighted_status_distribution(programs, records)


# --- sponsor-cluster bootstrap ---------------------------------------------

def test_bootstrap_weighted_share_ci_point_estimate_matches_weighted_math():
    programs, records = _two_stratum_fixture()
    # distinct sponsors, so no clustering distortion of the point estimate
    for i, p in enumerate(programs):
        p["sponsors_over_time"] = [{"sponsor": f"Sponsor{i}", "last_seen": "2026-01-01"}]

    result = ls.bootstrap_weighted_share_ci(
        programs, records, lambda r: r["status"] == "dead_confirmed", n_resamples=500, seed=0,
    )
    assert abs(result["point_estimate"] - 4 / 6) < 1e-9
    assert result["n_labelled"] == 3
    assert result["n_sponsors"] == 3
    assert result["ci_lo"] <= result["point_estimate"] <= result["ci_hi"]


def test_bootstrap_widens_under_sponsor_correlation():
    """Same shape as audit/label_sufficiency's cluster-bootstrap test: 20
    programs in one fully-covered stratum (weight=1 each), but only 2
    distinct sponsors, one always dead_confirmed and one always active.
    A per-program bootstrap would report a tight CI around 0.5; the
    sponsor-cluster bootstrap must stay wide, since the real uncertainty
    is "which of two sponsors dominates the draw"."""
    programs, records = [], []
    for i in range(20):
        sponsor = "SponsorA" if i % 2 == 0 else "SponsorB"
        status = "dead_confirmed" if sponsor == "SponsorA" else "active"
        pid = f"p{i}"
        programs.append(_program(pid, band=0, archetype="other", sponsor=sponsor))
        records.append(_label(
            pid, band=0, archetype="other", status=status,
            kill_reason="futility_efficacy" if status == "dead_confirmed" else None,
        ))

    result = ls.bootstrap_weighted_share_ci(
        programs, records, lambda r: r["status"] == "dead_confirmed", n_resamples=1000, seed=0,
    )
    assert result["n_sponsors"] == 2
    assert abs(result["point_estimate"] - 0.5) < 1e-9
    assert (result["ci_hi"] - result["ci_lo"]) > 0.5


# --- stated-vs-true kill reason -------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (None, None),
    ("", None),
    ("N/A", None),  # too short to carry signal
    ("Lack of efficacy observed at interim analysis.", "futility_efficacy"),
    ("Study halted due to safety concerns in Cohort 2.", "toxicity_safety"),
    ("Business decision to discontinue the program for portfolio reasons.", "strategic_portfolio"),
    ("Terminated due to slow patient enrollment at all sites.", "accrual_failure"),
    ("Discontinued due to loss of funding for the program.", "funding_insolvency"),
    ("Study closed early by the sponsor for unspecified reasons.", "unknown_silent"),
])
def test_classify_stated_kill_reason(text, expected):
    assert ls.classify_stated_kill_reason(text) == expected


def test_classify_stated_kill_reason_prefers_specific_over_generic():
    # contains both a strategic-sounding phrase and a specific efficacy
    # claim — the more specific futility_efficacy reading should win
    text = "Business decision following a failed interim analysis showing no clinical benefit."
    assert ls.classify_stated_kill_reason(text) == "futility_efficacy"


def _dead_program_with_why_stopped(pid, why_stopped, *, band=0, archetype="other"):
    return _program(pid, band=band, archetype=archetype, trials=[
        {"status": "TERMINATED", "why_stopped": why_stopped, "last_update_post_date": "2025-01-01"},
    ])


def test_stated_vs_true_kill_reason_sample_rows_and_summary():
    programs = [
        _dead_program_with_why_stopped("p1", "Lack of efficacy at interim analysis."),  # true says safety -> mismatch
        _dead_program_with_why_stopped("p2", "Business decision to deprioritise the program."),  # matches
    ]
    records = [
        _label("p1", band=0, archetype="other", status="dead_confirmed", kill_reason="toxicity_safety"),
        _label("p2", band=0, archetype="other", status="dead_confirmed", kill_reason="strategic_portfolio"),
    ]
    rows = ls.stated_vs_true_kill_reason_sample_rows(programs, records)
    by_pid = {r["program_id"]: r for r in rows}
    assert by_pid["p1"]["stated_kill_reason"] == "futility_efficacy"
    assert by_pid["p1"]["true_kill_reason"] == "toxicity_safety"
    assert by_pid["p2"]["stated_kill_reason"] == "strategic_portfolio"
    assert by_pid["p2"]["true_kill_reason"] == "strategic_portfolio"

    summary = ls.kill_reason_divergence_sample_summary(programs, records)
    assert summary["n"] == 2
    assert abs(summary["agreement_rate"] - 0.5) < 1e-9
    assert summary["true_kill_reason_counts"] == {"toxicity_safety": 1, "strategic_portfolio": 1}
    assert summary["confusion_matrix"]["stated=futility_efficacy|true=toxicity_safety"] == 1
    assert summary["confusion_matrix"]["stated=strategic_portfolio|true=strategic_portfolio"] == 1


def test_weighted_kill_reason_divergence_ci():
    programs = [
        _dead_program_with_why_stopped("p1", "Lack of efficacy at interim analysis."),  # matches
        _dead_program_with_why_stopped("p2", "Lack of efficacy at interim analysis."),  # mismatch: true=safety
    ]
    records = [
        _label("p1", band=0, archetype="other", status="dead_confirmed", kill_reason="futility_efficacy"),
        _label("p2", band=0, archetype="other", status="dead_confirmed", kill_reason="toxicity_safety"),
    ]
    result = ls.weighted_kill_reason_divergence_ci(programs, records, n_resamples=200, seed=0)
    assert abs(result["point_estimate_mismatch_rate"] - 0.5) < 1e-9
    assert result["n_dead_confirmed_labelled"] == 2


def test_weighted_kill_reason_divergence_ci_refuses_on_incomplete_coverage():
    programs, records = _two_stratum_fixture()
    programs.append(_program("p_gap", band=2, archetype="other"))
    with pytest.raises(ls.InsufficientStratumCoverageError):
        ls.weighted_kill_reason_divergence_ci(programs, records)


# --- summary() ---------------------------------------------------------

def test_summary_reports_refusal_instead_of_crashing():
    programs, records = _two_stratum_fixture()
    programs.append(_program("p_gap", band=2, archetype="other"))
    out = ls.summary(programs, records)
    assert "population_estimates_refused" in out
    assert "weighted_status_distribution" not in out
    # the sample-level (unweighted-claim) summary must still be present
    assert "kill_reason_divergence_sample" in out


def test_summary_full_coverage_populates_every_section():
    programs, records = _two_stratum_fixture()
    out = ls.summary(programs, records)
    assert "population_estimates_refused" not in out
    for key in (
        "stratum_coverage", "weighted_status_distribution", "weighted_kill_reason_distribution",
        "weighted_dead_confirmed_share_ci", "weighted_kill_reason_divergence_ci",
        "kill_reason_divergence_sample",
    ):
        assert key in out
