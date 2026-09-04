"""Tests for pharma_stats.stats.corpus_statistics — pure functions over
hand-built provisional_programs-shaped fixtures, no warehouse needed."""
from __future__ import annotations

from pharma_stats.stats import corpus_statistics as cs


def _program(
    pid, *, band, archetype, sponsor="Acme", history_coverage="full",
    latest_status="ACTIVE_NOT_RECRUITING", discovery_strategy="seed",
    trial_count=1, nct_ids=None,
):
    return {
        "program_id": pid,
        "band": band,
        "primary_archetype": archetype,
        "sponsors_over_time": [{"sponsor": sponsor, "last_seen": "2026-01-01"}],
        "history_coverage": history_coverage,
        "latest_status": latest_status,
        "discovery_strategy": discovery_strategy,
        "trial_count": trial_count,
        "nct_ids": nct_ids if nct_ids is not None else [f"NCT{pid}"],
    }


def test_total_programs():
    programs = [_program("p1", band=0, archetype="other"), _program("p2", band=1, archetype="other")]
    assert cs.total_programs(programs) == 2


def test_band_distribution_includes_unscored_bucket():
    programs = [
        _program("p1", band=0, archetype="other"),
        _program("p2", band=0, archetype="other"),
        _program("p3", band=None, archetype="other"),
    ]
    dist = cs.band_distribution(programs)
    by_band = {row["band"]: row for row in dist}
    assert by_band[0]["count"] == 2
    assert by_band[0]["band_label"] == "0-20"
    assert by_band[None]["count"] == 1
    assert by_band[None]["band_label"] == cs.UNSCORED_BAND
    assert abs(by_band[0]["share"] - 2 / 3) < 1e-9


def test_archetype_distribution_shares_sum_to_one():
    programs = [
        _program("p1", band=0, archetype="actively_amended"),
        _program("p2", band=0, archetype="actively_amended"),
        _program("p3", band=0, archetype="other"),
    ]
    dist = cs.archetype_distribution(programs)
    assert sum(row["share"] for row in dist) == 1.0
    by_archetype = {row["archetype"]: row["count"] for row in dist}
    assert by_archetype["actively_amended"] == 2
    assert by_archetype["other"] == 1


def test_stratum_population_counts_excludes_unscored():
    programs = [
        _program("p1", band=0, archetype="other"),
        _program("p2", band=0, archetype="other"),
        _program("p3", band=None, archetype="other"),
        _program("p4", band=2, archetype="actively_amended"),
    ]
    counts = cs.stratum_population_counts(programs)
    assert counts == {(0, "other"): 2, (2, "actively_amended"): 1}
    assert sum(counts.values()) == 3  # the unscored program never counted


def test_history_coverage_distribution_covers_all_levels():
    programs = [
        _program("p1", band=0, archetype="other", history_coverage="full"),
        _program("p2", band=0, archetype="other", history_coverage="none"),
    ]
    dist = cs.history_coverage_distribution(programs)
    levels = {row["history_coverage"]: row["count"] for row in dist}
    assert levels == {"full": 1, "partial": 0, "none": 1}


def test_latest_status_distribution_sorted_descending():
    programs = [
        _program("p1", band=0, archetype="other", latest_status="RECRUITING"),
        _program("p2", band=0, archetype="other", latest_status="RECRUITING"),
        _program("p3", band=0, archetype="other", latest_status="TERMINATED"),
    ]
    dist = cs.latest_status_distribution(programs)
    assert dist[0] == {"latest_status": "RECRUITING", "count": 2, "share": 2 / 3}


def test_sponsor_distribution_rolls_up_tail_into_other():
    programs = [_program(f"p{i}", band=0, archetype="other", sponsor=f"Sponsor{i}") for i in range(25)]
    dist = cs.sponsor_distribution(programs, top_n=20)
    assert len(dist) == 21  # 20 named + 1 rollup row
    assert dist[-1]["sponsor"] == "other (5 sponsors)"
    assert dist[-1]["count"] == 5
    assert sum(row["count"] for row in dist) == 25


def test_sponsor_distribution_uses_most_recently_seen_sponsor():
    programs = [{
        "program_id": "p1", "band": 0, "primary_archetype": "other",
        "sponsors_over_time": [
            {"sponsor": "OldCo", "last_seen": "2020-01-01"},
            {"sponsor": "NewCo", "last_seen": "2025-01-01"},
        ],
        "history_coverage": "full", "latest_status": "ACTIVE_NOT_RECRUITING",
        "discovery_strategy": "seed", "trial_count": 1, "nct_ids": ["NCT1"],
    }]
    dist = cs.sponsor_distribution(programs, top_n=None)
    assert dist == [{"sponsor": "NewCo", "count": 1, "share": 1.0}]


def test_trial_count_stats():
    programs = [
        _program("p1", band=0, archetype="other", trial_count=2, nct_ids=["NCTA", "NCTB"]),
        _program("p2", band=0, archetype="other", trial_count=0, nct_ids=[]),
    ]
    stats = cs.trial_count_stats(programs)
    assert stats["programs"] == 2
    assert stats["total_trial_references"] == 2
    assert stats["distinct_trials"] == 2
    assert stats["zero_trial_programs"] == 1
    assert stats["mean_trials_per_program"] == 1.0


def test_summary_bundles_everything():
    programs = [_program("p1", band=0, archetype="other")]
    out = cs.summary(programs)
    assert out["total_programs"] == 1
    assert "band_distribution" in out
    assert "sponsor_distribution" in out
    assert "trial_count_stats" in out
