"""Label sufficiency stage: the check that answers "is N labels enough"
by measurement instead of assumption. Bootstrap the headline lead-time
estimate (public_confirmation_date - label_evidence_date, dead_confirmed
labels only) using the first N labels for increasing N, and report how
much the 95% CI width narrows per additional 10 labels. When the
narrowing flattens, more labelling is buying little.

Resamples SPONSORS, not individual programs (cluster bootstrap) — lead
times aren't independent across programs from the same sponsor (a
sponsor's internal disclosure practice is a real, shared driver of how
fast confirmation follows evidence), measured ICC=0.18. A naive
per-observation bootstrap treats correlated observations as independent
and understates the true CI width, making the gold set look more
sufficient than it is. The cluster bootstrap resamples sponsors with
replacement and keeps each drawn sponsor's whole observation set intact
— the number of DISTINCT SPONSORS at a given N, not the raw label
count, is what actually drives precision under this ICC, so it's
reported alongside the CI width.
"""
from __future__ import annotations

import random
from datetime import date

from pharma_stats.audit.types import Check, info, ok
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store

STAGE = "label_sufficiency"

MIN_OBSERVATIONS = 10
BOOTSTRAP_RESAMPLES = 500
STEP = 10
CONVERGED_THRESHOLD_DAYS = 1.0  # marginal narrowing below this = "stopped moving"


def _lead_times_in_label_order(records: list[dict], sponsor_by_program: dict[str, str]) -> list[tuple[int, str]]:
    # gate 3 only: a gate 1/2 triage rejection was never a program, so it
    # must never count as a "label" toward this sufficiency bootstrap.
    non_repeat = sorted(
        (
            r for r in records
            if r["action"] == "label" and r.get("gate_reached") == 3 and not r["is_repeat_probe"]
        ),
        key=lambda r: r["timestamp"],
    )
    out = []
    for r in non_repeat:
        if r["status"] != "dead_confirmed" or r.get("never_publicly_confirmed"):
            continue
        ed, cd = r.get("label_evidence_date"), r.get("public_confirmation_date")
        if not ed or not cd:
            continue
        try:
            days = (date.fromisoformat(cd) - date.fromisoformat(ed)).days
        except ValueError:
            continue
        sponsor = sponsor_by_program.get(r["program_id"], "UNKNOWN")
        out.append((days, sponsor))
    return out


def _cluster_bootstrap_ci_width(
    observations: list[tuple[int, str]], n: int, rng: random.Random,
) -> tuple[float, int]:
    """(ci_width, n_distinct_sponsors) for the first n observations in
    label order. Resamples sponsors with replacement, keeping each
    drawn sponsor's full set of observations together — standard
    cluster bootstrap, not a per-observation resample."""
    sample = observations[:n]
    by_sponsor: dict[str, list[int]] = {}
    for days, sponsor in sample:
        by_sponsor.setdefault(sponsor, []).append(days)
    sponsors = list(by_sponsor.keys())

    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = [rng.choice(sponsors) for _ in range(len(sponsors))]
        values = [d for s in drawn for d in by_sponsor[s]]
        if values:
            means.append(sum(values) / len(values))
    means.sort()
    lo_idx = int(0.025 * len(means))
    hi_idx = min(int(0.975 * len(means)), len(means) - 1)
    return means[hi_idx] - means[lo_idx], len(sponsors)


def run() -> list[Check]:
    records = store.load_records()
    programs = pp.load_materialized()
    sponsor_by_program = {p["program_id"]: pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs}

    observations = _lead_times_in_label_order(records, sponsor_by_program)

    if len(observations) < MIN_OBSERVATIONS:
        return [info(
            STAGE, "cluster-bootstrap 95% CI width vs N labels (is the gold set big enough yet?)",
            expected=f">={MIN_OBSERVATIONS} dead_confirmed labels with both dates to begin",
            actual=f"{len(observations)} usable lead-time observations",
            detail="lead time = public_confirmation_date - label_evidence_date; "
                   "never_publicly_confirmed labels are excluded (no lead time to measure)",
        )]

    rng = random.Random(0)  # deterministic report across reruns on the same data
    points: list[tuple[int, float, int]] = []  # (n, ci_width, n_sponsors)
    for n in range(MIN_OBSERVATIONS, len(observations) + 1, STEP):
        width, n_sponsors = _cluster_bootstrap_ci_width(observations, n, rng)
        points.append((n, width, n_sponsors))
    if points[-1][0] != len(observations):
        width, n_sponsors = _cluster_bootstrap_ci_width(observations, len(observations), rng)
        points.append((len(observations), width, n_sponsors))

    last_n, last_width, last_n_sponsors = points[-1]
    marginal = points[-1][1] - points[-2][1] if len(points) > 1 else None

    converged = marginal is not None and abs(marginal) < CONVERGED_THRESHOLD_DAYS
    level = ok if converged else info
    detail = " | ".join(f"N={n} ({s} sponsors): 95% CI width={w:.1f}d" for n, w, s in points)

    return [level(
        STAGE, f"sponsor-cluster-bootstrap 95% CI width for headline lead time, N=10..{last_n} step {STEP}",
        expected=f"marginal narrowing per +{STEP} labels shrinks toward < {CONVERGED_THRESHOLD_DAYS:.0f}d "
                 "(that's when more labelling stops buying precision)",
        actual=(f"N={last_n} ({last_n_sponsors} distinct sponsors): width={last_width:.1f}d, "
                f"marginal vs N-{STEP}={marginal:+.1f}d") if marginal is not None
               else f"N={last_n} ({last_n_sponsors} distinct sponsors): width={last_width:.1f}d",
        detail=detail,
    )]
