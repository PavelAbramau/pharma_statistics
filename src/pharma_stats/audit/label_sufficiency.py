"""Label sufficiency stage: the check that answers "is N labels enough"
by measurement instead of assumption. Bootstrap the headline lead-time
estimate using the first N labels for increasing N, and report how much
the 95% CI width narrows per additional 10 labels. When the narrowing
flattens, more labelling is buying little.

Lead time is (see docs/decisions/0005-lead-time-redefinition.md):
    model_flag_date - public_confirmation_date
NOT public_confirmation_date - label_evidence_date (the original
definition) — checked against the real 63 dead_confirmed labels and
found to be noise (median ~0d, range -1375..+1764d): label_evidence_date
and public_confirmation_date were almost always the same search result,
not two independently-dated events. label_evidence_date is never read
here again. model_flag_date comes from models/ (the discrete-time
competing-risks survival model) — until that produces real flag dates,
this stage reports BLOCKED rather than compute a number from data it no
longer trusts.

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

import json
import random
from datetime import date
from pathlib import Path
from typing import Optional

from pharma_stats.audit.types import Check, info, ok
from pharma_stats.config import DATA_DIR
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store

STAGE = "label_sufficiency"
MODEL_RESULT_PATH = DATA_DIR / "model_backtest_result.json"

MIN_OBSERVATIONS = 10
BOOTSTRAP_RESAMPLES = 500
STEP = 10
CONVERGED_THRESHOLD_DAYS = 1.0  # marginal narrowing below this = "stopped moving"


def _lead_sponsor(sponsors_over_time: list[dict]) -> str:
    """Whichever sponsor has the latest last_seen date — same "most
    current" convention triage/evidence.py uses for its lead_sponsor
    field, reused here so the same program always clusters under the
    same sponsor label everywhere in the project."""
    if not sponsors_over_time:
        return "UNKNOWN"
    dated = [s for s in sponsors_over_time if s.get("last_seen")]
    pool = dated or sponsors_over_time
    return max(pool, key=lambda s: s.get("last_seen") or "")["sponsor"] or "UNKNOWN"


def _lead_times_in_label_order(
    records: list[dict], sponsor_by_program: dict[str, str], flag_date_by_program: dict[str, str],
) -> list[tuple[int, str]]:
    """(days, sponsor) — days = model_flag_date - public_confirmation_date.
    flag_date_by_program (program_id -> ISO date string) comes from
    models/ — a program with no entry there has no lead time to measure
    yet (the model never flagged it), not a zero."""
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
        cd = r.get("public_confirmation_date")
        fd = flag_date_by_program.get(r["program_id"])
        if not cd or not fd:
            continue
        try:
            days = (date.fromisoformat(fd) - date.fromisoformat(cd)).days
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


def _load_default_flag_dates(path: Optional[Path] = None) -> dict[str, str]:
    """scripts/run_model_backtest.py writes this file. Resolved at call
    time (not import time), same convention as every other path constant
    in this project, so tests can point it at an empty tmp path rather
    than accidentally reading real project data."""
    path = path or MODEL_RESULT_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("flag_date_by_program") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def run(flag_date_by_program: Optional[dict[str, str]] = None) -> list[Check]:
    """flag_date_by_program: program_id -> ISO date, from
    models.discrete_time_survival's fitted model. Explicit None (the
    harness's bare call) tries loading scripts/run_model_backtest.py's
    published result file; an explicit {} means "checked, nothing there"
    and skips that load. Either way, no data at all reports BLOCKED,
    never silently computed from label_evidence_date again."""
    if flag_date_by_program is None:
        flag_date_by_program = _load_default_flag_dates()
    if not flag_date_by_program:
        return [info(
            STAGE, "sponsor-cluster-bootstrap 95% CI width for headline lead time",
            expected="model_flag_date from models/ for each dead_confirmed program "
                     "(see docs/decisions/0005-lead-time-redefinition.md)",
            actual="BLOCKED — no model_flag_date supplied yet",
            detail="label_evidence_date is never used for this again — run "
                   "models.discrete_time_survival and pass its flag dates to this stage.",
        )]

    records = store.load_records()
    programs = pp.load_materialized()
    sponsor_by_program = {p["program_id"]: pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs}

    observations = _lead_times_in_label_order(records, sponsor_by_program, flag_date_by_program)

    if len(observations) < MIN_OBSERVATIONS:
        return [info(
            STAGE, "cluster-bootstrap 95% CI width vs N labels (is the gold set big enough yet?)",
            expected=f">={MIN_OBSERVATIONS} dead_confirmed labels with a model flag date to begin",
            actual=f"{len(observations)} usable lead-time observations",
            detail="lead time = model_flag_date - public_confirmation_date; "
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
