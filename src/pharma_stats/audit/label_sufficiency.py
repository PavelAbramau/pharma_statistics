"""Label sufficiency stage: the check that answers "is N labels enough"
by measurement instead of assumption. Bootstrap the headline lead-time
estimate (public_confirmation_date - label_evidence_date, dead_confirmed
labels only) using the first N labels for increasing N, and report how
much the 95% CI width narrows per additional 10 labels. When the
narrowing flattens, more labelling is buying little."""
from __future__ import annotations

import random
from datetime import date

from pharma_stats.audit.types import Check, info, ok
from pharma_stats.labelling import store

STAGE = "label_sufficiency"

MIN_OBSERVATIONS = 10
BOOTSTRAP_RESAMPLES = 500
STEP = 10
CONVERGED_THRESHOLD_DAYS = 1.0  # marginal narrowing below this = "stopped moving"


def _lead_times_in_label_order(records: list[dict]) -> list[int]:
    non_repeat = sorted(
        (r for r in records if r["action"] == "label" and not r["is_repeat_probe"]),
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
        out.append(days)
    return out


def _bootstrap_ci_width(values: list[int], n: int, rng: random.Random) -> float:
    sample = values[:n]
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [rng.choice(sample) for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = int(0.025 * BOOTSTRAP_RESAMPLES)
    hi_idx = min(int(0.975 * BOOTSTRAP_RESAMPLES), BOOTSTRAP_RESAMPLES - 1)
    return means[hi_idx] - means[lo_idx]


def run() -> list[Check]:
    records = store.load_records()
    lead_times = _lead_times_in_label_order(records)

    if len(lead_times) < MIN_OBSERVATIONS:
        return [info(
            STAGE, "bootstrap CI width vs N labels (is the gold set big enough yet?)",
            expected=f">={MIN_OBSERVATIONS} dead_confirmed labels with both dates to begin",
            actual=f"{len(lead_times)} usable lead-time observations",
            detail="lead time = public_confirmation_date - label_evidence_date; "
                   "never_publicly_confirmed labels are excluded (no lead time to measure)",
        )]

    rng = random.Random(0)  # deterministic report across reruns on the same data
    points: list[tuple[int, float]] = []
    for n in range(MIN_OBSERVATIONS, len(lead_times) + 1, STEP):
        points.append((n, _bootstrap_ci_width(lead_times, n, rng)))
    if points[-1][0] != len(lead_times):
        points.append((len(lead_times), _bootstrap_ci_width(lead_times, len(lead_times), rng)))

    last_n, last_width = points[-1]
    marginal = points[-1][1] - points[-2][1] if len(points) > 1 else None

    converged = marginal is not None and abs(marginal) < CONVERGED_THRESHOLD_DAYS
    level = ok if converged else info
    detail = " | ".join(f"N={n}: 95% CI width={w:.1f}d" for n, w in points)

    return [level(
        STAGE, f"bootstrap 95% CI width for headline lead time, N=10..{last_n} step {STEP}",
        expected=f"marginal narrowing per +{STEP} labels shrinks toward < {CONVERGED_THRESHOLD_DAYS:.0f}d "
                 "(that's when more labelling stops buying precision)",
        actual=f"N={last_n}: width={last_width:.1f}d, marginal vs N-{STEP}="
               f"{marginal:+.1f}d" if marginal is not None else f"N={last_n}: width={last_width:.1f}d",
        detail=detail,
    )]
