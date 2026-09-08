"""Population-level statistics from gold/labels.jsonl.

gold/labels.jsonl is NOT a random sample of the corpus: labelling/queue.py
draws it stratified by (silence-score band x archetype), round-robin
across cells, specifically so early labels wouldn't all come from one end
of the score distribution (see queue.py's module docstring). That is
exactly the design that makes a raw "N% of labelled programs are
dead_confirmed" figure wrong as a population estimate — a program that
looks dead-ish (high band) was oversampled relative to its true share of
the corpus, and a program that looks untouched (low band) was
undersampled. Report the raw percentage as-is and you are describing the
labelling queue's sampling design, not the corpus.

So this module never exposes an unweighted population estimate. Every
population-level number here is inverse-probability-weighted by stratum:
weight(program) = population(stratum) / labelled(stratum), i.e. each
labelled program stands in for "population(stratum) / labelled(stratum)"
programs from its own (band, archetype) cell — standard post-
stratification. If some non-empty population stratum has zero labels,
there is no weight to assign its share of the corpus, and computing a
population estimate anyway would silently understate it (or drop it) —
so compute_stratum_weights refuses outright (InsufficientStratumCoverageError)
rather than emit a plausible-looking number quietly built on incomplete
coverage. Callers that only want to describe the labelled sample itself
(never a population claim) use the *_sample_* functions, which are named
to make that scope explicit.

Confidence intervals resample SPONSORS, not programs (cluster bootstrap):
audit/label_sufficiency.py measured ICC=0.18 for lead time correlating
within-sponsor, and there's no reason to expect kill-reason or status
judgements are any less correlated by a sponsor's own disclosure habits.
A per-program bootstrap would treat correlated observations as
independent and understate CI width, same failure mode label_sufficiency
exists to avoid — reused here via provisional_programs.lead_sponsor, the
same "most-recently-seen sponsor" clustering key used everywhere else in
the project a program needs one sponsor label.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.stats.corpus_statistics import stratum_population_counts

BOOTSTRAP_RESAMPLES = 1000

# Deterministic (no model call) keyword classifier of CT.gov's own
# why_stopped free text into the project's kill-reason taxonomy — the
# "stated" side of the stated-vs-true comparison. Order matters: checked
# top to bottom, first match wins. Safety/efficacy phrasing is checked
# ahead of the catch-all business/strategic phrasing so a text like
# "business decision following a failed interim efficacy analysis" reads
# as futility_efficacy, the more specific claim. This is deliberately a
# small, reviewable, hand-authored list (CLAUDE.md: don't fuzzy-match a
# controlled vocabulary at runtime) — it exists only to characterise how
# much a text-only reading of the registry would get wrong next to a
# human reviewer's full-evidence judgement, not to drive any pipeline
# decision.
_STATED_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("toxicity_safety", (
        "safety", "toxicity", "adverse event", "adverse effect", "dose-limiting",
        "dose limiting", "tolerability", "died", "death", "fatal",
    )),
    ("futility_efficacy", (
        "futility", "lack of efficacy", "insufficient efficacy", "no clinical benefit",
        "did not meet", "failed to meet", "interim analysis", "efficacy",
        "did not demonstrate", "no significant difference",
    )),
    ("accrual_failure", (
        "enrollment", "enrolment", "accrual", "recruit", "slow recruitment",
    )),
    ("funding_insolvency", (
        "funding", "financial", "insolven", "bankrupt", "budget",
    )),
    ("competitive_landscape", (
        "competitive landscape", "competitor", "competing product", "market landscape",
    )),
    ("ip_legal", (
        "patent", "intellectual property", "litigation", "legal dispute",
    )),
    ("strategic_portfolio", (
        "business decision", "business reason", "strategic", "portfolio",
        "sponsor decision", "sponsor's decision", "pipeline priorit", "reprioriti",
        "program discontinuation", "no longer a priority",
    )),
]

# Below this, why_stopped is empty/too-short to carry any classifiable
# signal — mirrors provisional_programs._is_vague's own length floor, so
# "vague" means the same thing everywhere in the project.
_MIN_STATED_TEXT_LEN = 15


class InsufficientStratumCoverageError(Exception):
    """Raised when a population-level estimate is requested but at least
    one non-empty (band, archetype) stratum has zero labelled programs —
    there is no weight to assign that stratum's share of the corpus, so
    refuse rather than emit a number that silently omits or misrepresents
    it. .missing_strata carries the (band, archetype, population) rows
    responsible, so a caller can say exactly what to go label next."""

    def __init__(self, missing_strata: list[tuple[int, str, int]]):
        self.missing_strata = missing_strata
        detail = ", ".join(f"(band={b}, {a}): {n} unlabelled" for b, a, n in missing_strata)
        super().__init__(
            f"{len(missing_strata)} population stratum/strata have zero labels — "
            f"refusing to emit a population estimate that would silently omit them: {detail}"
        )


@dataclass
class StratumWeight:
    band: int
    archetype: str
    population: int
    labelled: int
    weight: float  # population / labelled — each labelled program's stand-in count


def gate3_labels_by_program(records: list[dict]) -> dict[str, dict]:
    """Latest gate-3 ("real label") record per program — the gold set
    proper, excluding gate-1/2 triage rejections and repeat probes (see
    labelling/store.py: those were never a program, or never a new
    observation, to begin with)."""
    fully_labelled = store.fully_labelled_program_ids(records)
    latest = store.latest_by_program(records)
    return {pid: r for pid, r in latest.items() if pid in fully_labelled}


def compute_stratum_weights(
    programs: list[dict], records: list[dict],
) -> dict[tuple[int, str], StratumWeight]:
    """Population stratum counts (corpus_statistics.stratum_population_counts)
    against label counts per stratum, keyed off each label's OWN
    stratum_band/stratum_archetype (stamped at serve time — the actual
    sampling-design fact, not whatever the current corpus happens to
    rescore the program as). Raises InsufficientStratumCoverageError if
    any non-empty population stratum was never sampled at all."""
    population = stratum_population_counts(programs)
    labelled = gate3_labels_by_program(records)

    labelled_counts: Counter[tuple[int, str]] = Counter()
    for r in labelled.values():
        band, archetype = r.get("stratum_band"), r.get("stratum_archetype")
        if band is None:
            continue  # served from the "unscored" bucket — not one of the 5 real strata
        labelled_counts[(band, archetype)] += 1

    missing = [
        (b, a, n) for (b, a), n in population.items() if n > 0 and labelled_counts.get((b, a), 0) == 0
    ]
    if missing:
        raise InsufficientStratumCoverageError(sorted(missing))

    return {
        (b, a): StratumWeight(band=b, archetype=a, population=n, labelled=labelled_counts[(b, a)],
                               weight=n / labelled_counts[(b, a)])
        for (b, a), n in population.items()
    }


def weighted_status_distribution(programs: list[dict], records: list[dict]) -> list[dict]:
    """Population estimate of the program_status ladder's mix across the
    WHOLE corpus (not just the labelled subset), by inverse-probability
    weighting each labelled program by population(stratum)/labelled(stratum).
    Raises InsufficientStratumCoverageError if coverage doesn't support it."""
    weights = compute_stratum_weights(programs, records)
    labelled = gate3_labels_by_program(records)

    weight_by_status: dict[Optional[str], float] = {}
    raw_n_by_status: dict[Optional[str], int] = {}
    total_weight = 0.0
    for pid, r in labelled.items():
        band, archetype = r.get("stratum_band"), r.get("stratum_archetype")
        if band is None:
            continue
        w = weights[(band, archetype)].weight
        status = r.get("status")
        weight_by_status[status] = weight_by_status.get(status, 0.0) + w
        raw_n_by_status[status] = raw_n_by_status.get(status, 0) + 1
        total_weight += w

    return sorted(
        (
            {
                "status": status,
                "weighted_share": (w / total_weight) if total_weight else 0.0,
                "weighted_n_equivalent": w,
                "raw_labelled_n": raw_n_by_status[status],
            }
            for status, w in weight_by_status.items()
        ),
        key=lambda r: -r["weighted_share"],
    )


def weighted_kill_reason_distribution(programs: list[dict], records: list[dict]) -> list[dict]:
    """Population estimate of the kill-reason mix WITHIN dead_confirmed
    programs (not the whole corpus) — same IPW machinery, restricted to
    labelled programs whose status is dead_confirmed."""
    weights = compute_stratum_weights(programs, records)
    labelled = {
        pid: r for pid, r in gate3_labels_by_program(records).items() if r.get("status") == "dead_confirmed"
    }

    weight_by_reason: dict[Optional[str], float] = {}
    raw_n_by_reason: dict[Optional[str], int] = {}
    total_weight = 0.0
    for pid, r in labelled.items():
        band, archetype = r.get("stratum_band"), r.get("stratum_archetype")
        if band is None:
            continue
        w = weights[(band, archetype)].weight
        reason = r.get("kill_reason")
        weight_by_reason[reason] = weight_by_reason.get(reason, 0.0) + w
        raw_n_by_reason[reason] = raw_n_by_reason.get(reason, 0) + 1
        total_weight += w

    return sorted(
        (
            {
                "kill_reason": reason,
                "weighted_share": (w / total_weight) if total_weight else 0.0,
                "weighted_n_equivalent": w,
                "raw_labelled_n": raw_n_by_reason[reason],
            }
            for reason, w in weight_by_reason.items()
        ),
        key=lambda r: -r["weighted_share"],
    )


def _cluster_bootstrap_share(
    observations: list[tuple[str, float, bool]], rng: random.Random, n_resamples: int,
) -> list[float]:
    """observations: (sponsor, weight, in_category) per labelled program.
    Each resample draws len(distinct sponsors) sponsors WITH replacement,
    pools every observation belonging to a drawn sponsor (a sponsor drawn
    twice contributes its whole observation set twice — standard cluster
    bootstrap, same shape as audit/label_sufficiency._cluster_bootstrap_ci_width),
    and recomputes the IPW-weighted share for that resample. Per-program
    weights travel with the observation and are unaffected by resampling."""
    by_sponsor: dict[str, list[tuple[float, bool]]] = {}
    for sponsor, w, in_cat in observations:
        by_sponsor.setdefault(sponsor, []).append((w, in_cat))
    sponsors = list(by_sponsor.keys())
    if not sponsors:
        return []

    shares = []
    for _ in range(n_resamples):
        drawn = [rng.choice(sponsors) for _ in range(len(sponsors))]
        pooled = [obs for s in drawn for obs in by_sponsor[s]]
        total_w = sum(w for w, _ in pooled)
        if total_w:
            shares.append(sum(w for w, in_cat in pooled if in_cat) / total_w)
    return shares


def _percentile_ci(values: list[float], lo=0.025, hi=0.975) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    s = sorted(values)
    lo_idx = int(lo * len(s))
    hi_idx = min(int(hi * len(s)), len(s) - 1)
    return s[lo_idx], s[hi_idx]


def bootstrap_weighted_share_ci(
    programs: list[dict], records: list[dict], category: Callable[[dict], bool],
    *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
) -> dict:
    """Population-level IPW share of labelled programs matching `category`
    (applied to each gate-3 label record), with a sponsor-cluster-
    bootstrap 95% CI. `category` runs over the SAME label records
    weighted_status_distribution etc. do — e.g. `lambda r: r["status"] ==
    "dead_confirmed"`. Raises InsufficientStratumCoverageError, same as
    every other population-level function here, if coverage is short."""
    weights = compute_stratum_weights(programs, records)
    labelled = gate3_labels_by_program(records)
    sponsor_by_pid = {p["program_id"]: pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs}

    observations = []
    point_num, point_den = 0.0, 0.0
    for pid, r in labelled.items():
        band, archetype = r.get("stratum_band"), r.get("stratum_archetype")
        if band is None:
            continue
        w = weights[(band, archetype)].weight
        in_cat = bool(category(r))
        sponsor = sponsor_by_pid.get(pid, "UNKNOWN")
        observations.append((sponsor, w, in_cat))
        point_den += w
        if in_cat:
            point_num += w

    rng = random.Random(seed)
    shares = _cluster_bootstrap_share(observations, rng, n_resamples)
    lo, hi = _percentile_ci(shares)
    n_sponsors = len({s for s, _, _ in observations})
    return {
        "point_estimate": (point_num / point_den) if point_den else None,
        "ci_lo": lo, "ci_hi": hi,
        "n_labelled": len(observations), "n_sponsors": n_sponsors,
        "n_resamples": n_resamples,
    }


def classify_stated_kill_reason(why_stopped: Optional[str]) -> Optional[str]:
    """Map CT.gov's own why_stopped free text to the kill-reason
    taxonomy by keyword — the "stated" side of the stated-vs-true
    comparison. None means no usable text at all (blank or too short to
    carry signal — mirrors provisional_programs._is_vague's floor);
    "unknown_silent" means text is present but doesn't match any known
    phrasing (a real, distinct outcome from "nothing was said")."""
    if not why_stopped or len(why_stopped.strip()) < _MIN_STATED_TEXT_LEN:
        return None
    lowered = why_stopped.lower()
    for reason, keywords in _STATED_KEYWORDS:
        if any(k in lowered for k in keywords):
            return reason
    return "unknown_silent"


def _stated_why_stopped_text(program: Optional[dict]) -> Optional[str]:
    """The why_stopped text of whichever of a program's trials most
    recently stopped — the registry statement a reader comparing "what
    CT.gov says" against "what actually happened" would see. None if the
    program is unknown or has no terminal-status trial on file."""
    if not program:
        return None
    terminal = [
        t for t in (program.get("trials") or [])
        if t.get("status") in pp.TERMINAL_STOP_STATUSES and t.get("why_stopped")
    ]
    if not terminal:
        return None
    latest = max(terminal, key=lambda t: t.get("last_update_post_date") or "")
    return latest["why_stopped"]


def stated_vs_true_kill_reason_sample_rows(programs: list[dict], records: list[dict]) -> list[dict]:
    """One row per gate-3 dead_confirmed label: the reviewer's true
    kill_reason next to the registry's stated one (via
    classify_stated_kill_reason). Describes the LABELLED SAMPLE only —
    never treat this as a population claim; see
    weighted_kill_reason_divergence_ci for that."""
    programs_by_id = {p["program_id"]: p for p in programs}
    rows = []
    for pid, r in gate3_labels_by_program(records).items():
        if r.get("status") != "dead_confirmed" or not r.get("kill_reason"):
            continue
        why_text = _stated_why_stopped_text(programs_by_id.get(pid))
        rows.append({
            "program_id": pid,
            "true_kill_reason": r["kill_reason"],
            "stated_kill_reason": classify_stated_kill_reason(why_text),
            "why_stopped_text": why_text,
        })
    return rows


def kill_reason_divergence_sample_summary(programs: list[dict], records: list[dict]) -> dict:
    """Sample-level (not population-weighted) description of how often
    the registry's stated reason agrees with the reviewer's true
    kill_reason: counts, a confusion matrix, and the raw agreement rate
    among the programs actually labelled. This is a fact about the gold
    set as sampled, not a claim about the corpus — see
    weighted_kill_reason_divergence_ci for the IPW population estimate of
    the same question."""
    rows = stated_vs_true_kill_reason_sample_rows(programs, records)
    n = len(rows)
    true_counts = Counter(r["true_kill_reason"] for r in rows)
    stated_counts = Counter(r["stated_kill_reason"] or "not_stated" for r in rows)
    confusion: Counter[tuple[str, str]] = Counter(
        (r["stated_kill_reason"] or "not_stated", r["true_kill_reason"]) for r in rows
    )
    agree = sum(1 for r in rows if r["stated_kill_reason"] == r["true_kill_reason"])
    return {
        "n": n,
        "agreement_rate": (agree / n) if n else None,
        "true_kill_reason_counts": dict(sorted(true_counts.items(), key=lambda kv: -kv[1])),
        "stated_kill_reason_counts": dict(sorted(stated_counts.items(), key=lambda kv: -kv[1])),
        "confusion_matrix": {f"stated={s}|true={t}": n for (s, t), n in confusion.items()},
    }


def agreement_rate_sample_ci(
    programs: list[dict], records: list[dict], *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
) -> dict:
    """Sponsor-cluster-bootstrap 95% CI on the RAW (unweighted) stated-
    vs-true kill-reason agreement rate among labelled dead_confirmed
    programs — describes the gold set as sampled, same scope as
    kill_reason_divergence_sample_summary's point estimate, just with an
    interval around it. Never requires stratum coverage (unlike
    weighted_kill_reason_divergence_ci): every observation gets equal
    weight 1.0, since this doesn't claim to speak for the corpus.

    Same cluster-bootstrap machinery as bootstrap_weighted_share_ci
    (_cluster_bootstrap_share resamples sponsors, not programs — ICC=0.18
    reasoning is identical, see module docstring), just unweighted."""
    rows = stated_vs_true_kill_reason_sample_rows(programs, records)
    sponsor_by_pid = {p["program_id"]: pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs}

    observations = [
        (sponsor_by_pid.get(r["program_id"], "UNKNOWN"), 1.0, r["stated_kill_reason"] == r["true_kill_reason"])
        for r in rows
    ]
    rng = random.Random(seed)
    shares = _cluster_bootstrap_share(observations, rng, n_resamples)
    lo, hi = _percentile_ci(shares)
    n_sponsors = len({s for s, _, _ in observations})
    n_agree = sum(1 for _, _, in_cat in observations if in_cat)
    return {
        "point_estimate": (n_agree / len(observations)) if observations else None,
        "ci_lo": lo, "ci_hi": hi,
        "n": len(observations), "n_sponsors": n_sponsors, "n_resamples": n_resamples,
    }


def weighted_kill_reason_divergence_ci(
    programs: list[dict], records: list[dict], *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
) -> dict:
    """Population estimate (IPW + sponsor-cluster bootstrap) of how often
    the registry's stated kill reason would MISMATCH the true, evidence-
    reviewed kill reason, among dead_confirmed programs across the whole
    corpus — not just the labelled sample (see
    kill_reason_divergence_sample_summary for that). Refuses
    (InsufficientStratumCoverageError) under the same coverage rule as
    every other population estimate here.

    Doesn't reuse bootstrap_weighted_share_ci: "mismatch" depends on the
    associated program's why_stopped text (via classify_stated_kill_reason),
    not on the label record alone, so this needs its own pass over
    programs_by_id rather than a record-only category predicate."""
    weights = compute_stratum_weights(programs, records)  # raises if coverage is short
    programs_by_id = {p["program_id"]: p for p in programs}
    sponsor_by_pid = {p["program_id"]: pp.lead_sponsor(p.get("sponsors_over_time") or []) for p in programs}

    observations = []
    point_num, point_den = 0.0, 0.0
    for pid, r in gate3_labels_by_program(records).items():
        if r.get("status") != "dead_confirmed" or not r.get("kill_reason"):
            continue
        band, archetype = r.get("stratum_band"), r.get("stratum_archetype")
        if band is None:
            continue
        w = weights[(band, archetype)].weight
        stated = classify_stated_kill_reason(_stated_why_stopped_text(programs_by_id.get(pid)))
        mismatch = stated != r["kill_reason"]
        observations.append((sponsor_by_pid.get(pid, "UNKNOWN"), w, mismatch))
        point_den += w
        if mismatch:
            point_num += w

    rng = random.Random(seed)
    shares = _cluster_bootstrap_share(observations, rng, n_resamples)
    lo, hi = _percentile_ci(shares)
    n_sponsors = len({s for s, _, _ in observations})
    return {
        "point_estimate_mismatch_rate": (point_num / point_den) if point_den else None,
        "ci_lo": lo, "ci_hi": hi,
        "n_dead_confirmed_labelled": len(observations), "n_sponsors": n_sponsors,
        "n_resamples": n_resamples,
    }


def summary(programs: list[dict], records: list[dict]) -> dict:
    """Everything above, bundled for a report script. Population-level
    figures are computed inside a try/except keyed on
    InsufficientStratumCoverageError — that's the refusal path working as
    designed, not a bug to swallow, so the missing-coverage detail is
    reported in place of the number rather than the whole summary
    failing outright."""
    out: dict = {
        "n_gate3_labels": len(gate3_labels_by_program(records)),
        "kill_reason_divergence_sample": kill_reason_divergence_sample_summary(programs, records),
        "kill_reason_divergence_sample_rows": stated_vs_true_kill_reason_sample_rows(programs, records),
        "agreement_rate_sample_ci": agreement_rate_sample_ci(programs, records),
    }
    try:
        weights = compute_stratum_weights(programs, records)
        out["stratum_coverage"] = [
            {"band": w.band, "archetype": w.archetype, "population": w.population,
             "labelled": w.labelled, "weight": w.weight}
            for w in sorted(weights.values(), key=lambda w: (w.band, w.archetype))
        ]
        out["weighted_status_distribution"] = weighted_status_distribution(programs, records)
        out["weighted_kill_reason_distribution"] = weighted_kill_reason_distribution(programs, records)
        out["weighted_dead_confirmed_share_ci"] = bootstrap_weighted_share_ci(
            programs, records, lambda r: r.get("status") == "dead_confirmed",
        )
        out["weighted_kill_reason_divergence_ci"] = weighted_kill_reason_divergence_ci(programs, records)
    except InsufficientStratumCoverageError as e:
        out["population_estimates_refused"] = str(e)
    return out
