"""Descriptive statistics over the whole candidate/provisional-program
corpus — the full population, not a sample of it. Every count here is
reported straight, with no weighting: unlike gold/labels.jsonl (a
stratified sample of this population — see label_statistics.py), this
*is* the population, so a raw count already answers "how many programs
does the corpus have in band X / archetype Y / sponsor Z". Weighting only
enters once you start asking a labels-based question about outcomes this
module doesn't know anything about (status, kill_reason).

Every function takes ``programs: list[dict]`` — provisional_programs rows,
as returned by ``pharma_stats.labelling.provisional_programs.load_materialized()``
— rather than reading the warehouse itself, so it stays testable against
hand-built fixtures without a live DuckDB file. ``load_corpus()`` is the
one place that touches the warehouse, for callers (scripts, the label
statistics module) that want the real thing.
"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Optional

from pharma_stats.labelling import provisional_programs as pp

UNSCORED_BAND = "unscored"  # provisional_programs.py: band is None when a
# candidate has no resolvable trial snapshot at all — never one of the 5
# real score bands. Kept as its own bucket here (never dropped) so the
# corpus total always reconciles to len(programs).


def load_corpus(warehouse_db=None) -> list[dict]:
    return pp.load_materialized(warehouse_db=warehouse_db)


def _band_label(band: Optional[int]) -> str:
    if band is None:
        return UNSCORED_BAND
    lo, hi = pp.SCORE_BANDS[band]
    return f"{lo}-{min(hi, 100)}"


def total_programs(programs: list[dict]) -> int:
    return len(programs)


def band_distribution(programs: list[dict]) -> list[dict]:
    """Population count per silence-score band, including the unscored
    bucket — sorted by band index with unscored last."""
    counts = Counter(p["band"] for p in programs)
    total = len(programs)
    order = list(range(len(pp.SCORE_BANDS))) + [None]
    return [
        {
            "band": b, "band_label": _band_label(b),
            "count": counts.get(b, 0),
            "share": (counts.get(b, 0) / total) if total else 0.0,
        }
        for b in order if counts.get(b, 0) or b is not None
    ]


def archetype_distribution(programs: list[dict]) -> list[dict]:
    # every program's primary_archetype comes from pp.ARCHETYPES (see
    # _primary_archetype, which defaults to "other" — always a member),
    # so iterating that fixed list is a complete, stable ordering.
    counts = Counter(p["primary_archetype"] for p in programs)
    total = len(programs)
    return [
        {"archetype": a, "count": counts.get(a, 0), "share": (counts.get(a, 0) / total) if total else 0.0}
        for a in pp.ARCHETYPES if counts.get(a, 0)
    ]


def stratum_population_counts(programs: list[dict]) -> dict[tuple[int, str], int]:
    """Population count per (band, archetype) stratum — the sampling
    frame the labelling queue draws from (labelling/queue.py) and the
    denominator label_statistics.py's inverse-probability weights need.
    Deliberately excludes band=None (unscored) programs: they were never
    one of the 5 real score bands, so they have no stratum to be a
    population count for — same exclusion labelling/stats.stratum_progress
    already makes for the same reason."""
    counts: dict[tuple[int, str], int] = {}
    for p in programs:
        if p["band"] is None:
            continue
        key = (p["band"], p["primary_archetype"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def history_coverage_distribution(programs: list[dict]) -> list[dict]:
    counts = Counter(p["history_coverage"] for p in programs)
    total = len(programs)
    return [
        {"history_coverage": lvl, "count": counts.get(lvl, 0), "share": (counts.get(lvl, 0) / total) if total else 0.0}
        for lvl in pp.HISTORY_COVERAGE_LEVELS
    ]


def latest_status_distribution(programs: list[dict]) -> list[dict]:
    """Raw CT.gov overallStatus of each program's most-recently-touched
    trial — a registry-status snapshot of the corpus, not the labelled
    program_status ladder (active/dead_confirmed/...), which only exists
    where a human has reviewed the evidence. None means no resolvable
    trial snapshot at all."""
    counts = Counter(p["latest_status"] for p in programs)
    total = len(programs)
    return sorted(
        (
            {"latest_status": s, "count": n, "share": (n / total) if total else 0.0}
            for s, n in counts.items()
        ),
        key=lambda r: -r["count"],
    )


def sponsor_distribution(programs: list[dict], top_n: Optional[int] = 20) -> list[dict]:
    """Program count per lead sponsor (see provisional_programs.lead_sponsor
    for the "most recently seen" convention). top_n=None returns every
    sponsor; otherwise the top_n most common plus a rolled-up "other"
    row so counts still reconcile to the corpus total."""
    counts = Counter(pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs)
    total = len(programs)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if top_n is not None and len(ranked) > top_n:
        head, tail = ranked[:top_n], ranked[top_n:]
        other_n = sum(n for _, n in tail)
        ranked = head + [("other (%d sponsors)" % len(tail), other_n)]
    return [
        {"sponsor": sponsor, "count": n, "share": (n / total) if total else 0.0}
        for sponsor, n in ranked
    ]


def discovery_strategy_distribution(programs: list[dict]) -> list[dict]:
    counts = Counter(p.get("discovery_strategy") for p in programs)
    total = len(programs)
    return sorted(
        (
            {"discovery_strategy": s, "count": n, "share": (n / total) if total else 0.0}
            for s, n in counts.items()
        ),
        key=lambda r: -r["count"],
    )


def trial_count_stats(programs: list[dict]) -> dict:
    """How many trials the corpus references, and how they're distributed
    across programs — a basket-trial-heavy corpus looks very different
    from a one-trial-per-program one, and this is the number that would
    show it."""
    counts = [p["trial_count"] for p in programs]
    total_trial_refs = sum(counts)
    distinct_ncts = {n for p in programs for n in (p.get("nct_ids") or [])}
    return {
        "programs": len(programs),
        "total_trial_references": total_trial_refs,
        "distinct_trials": len(distinct_ncts),
        "mean_trials_per_program": mean(counts) if counts else 0.0,
        "median_trials_per_program": median(counts) if counts else 0.0,
        "max_trials_per_program": max(counts) if counts else 0,
        "zero_trial_programs": sum(1 for c in counts if c == 0),
    }


def summary(programs: Optional[list[dict]] = None, warehouse_db=None) -> dict:
    """Everything above, bundled for a report script or a quick console
    dump. Loads the corpus itself when programs isn't already in hand."""
    programs = programs if programs is not None else load_corpus(warehouse_db=warehouse_db)
    return {
        "total_programs": total_programs(programs),
        "band_distribution": band_distribution(programs),
        "archetype_distribution": archetype_distribution(programs),
        "history_coverage_distribution": history_coverage_distribution(programs),
        "latest_status_distribution": latest_status_distribution(programs),
        "sponsor_distribution": sponsor_distribution(programs),
        "discovery_strategy_distribution": discovery_strategy_distribution(programs),
        "trial_count_stats": trial_count_stats(programs),
    }
