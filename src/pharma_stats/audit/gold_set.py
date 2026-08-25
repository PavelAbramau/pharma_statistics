"""Gold set stage: stratum coverage, the dead_confirmed date invariant
re-verified independently of the app's own validator, self-consistency
on the silently-repeated probes, timing, and blind/unblinded mix."""
from __future__ import annotations

from pharma_stats.audit.types import Check, fail, info, ok, warn
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import stats as label_stats
from pharma_stats.labelling import store

STAGE = "gold_set"

# below this many total labels, an empty stratum cell is just "haven't
# gotten there yet", not evidence of a stratification bug
MIN_LABELS_BEFORE_FLAGGING_EMPTY_CELLS = 20


def run() -> list[Check]:
    records = store.load_records()
    label_events = [r for r in records if r["action"] == "label" and not r["is_repeat_probe"]]

    if not records:
        return [info(
            STAGE, "gold label records",
            expected=">=1 record in gold/labels.jsonl",
            actual="0 — no labelling has happened yet",
            detail="run `python scripts/run_labelling_app.py` to start",
        )]

    checks: list[Check] = []
    checks += _dead_confirmed_invariant(records)
    checks += _stratum_coverage(records, label_events)
    checks += _timing_and_blind(records, label_events)
    checks += _self_consistency(records)
    return checks


def _dead_confirmed_invariant(records: list[dict]) -> list[Check]:
    dead = [r for r in records if r["action"] == "label" and r["status"] == "dead_confirmed"]
    missing_kill_reason = [r for r in dead if not r.get("kill_reason")]
    missing_evidence_date = [r for r in dead if not r.get("label_evidence_date")]
    missing_confirmation = [
        r for r in dead
        if not r.get("public_confirmation_date") and not r.get("never_publicly_confirmed")
    ]

    def result(name, bad):
        return (fail if bad else ok)(
            STAGE, name, expected="0 violations among dead_confirmed records",
            actual=f"{len(bad)} / {len(dead)} violate it",
            detail=", ".join(r["event_id"] for r in bad[:10]),
        )

    return [
        result("every dead_confirmed record has a kill_reason", missing_kill_reason),
        result("every dead_confirmed record has a label_evidence_date", missing_evidence_date),
        result(
            "every dead_confirmed record has public_confirmation_date OR never_publicly_confirmed",
            missing_confirmation,
        ),
    ]


def _stratum_coverage(records: list[dict], label_events: list[dict]) -> list[Check]:
    programs = pp.load_materialized()
    if not programs:
        return [info(
            STAGE, "stratum coverage",
            expected="provisional_programs materialized", actual="not materialized yet",
            detail="run the labelling app once, or pharma_stats.labelling.provisional_programs.materialize()",
        )]

    labelled_ids = store.labelled_program_ids(records)
    progress = label_stats.stratum_progress(programs, labelled_ids)
    empty_cells = [p for p in progress if p["total"] > 0 and p["labelled"] == 0]

    lines = [f"band {p['band']}/{p['archetype']}: {p['labelled']}/{p['total']}" for p in progress]
    detail = " | ".join(lines)

    if len(labelled_ids) < MIN_LABELS_BEFORE_FLAGGING_EMPTY_CELLS:
        return [info(
            STAGE, "label counts per (band, archetype) stratum",
            expected=f"non-zero coverage across strata once >= {MIN_LABELS_BEFORE_FLAGGING_EMPTY_CELLS} labels exist",
            actual=f"{len(labelled_ids)} labels so far — too early to judge",
            detail=detail,
        )]

    return [(warn if empty_cells else ok)(
        STAGE, "label counts per (band, archetype) stratum",
        expected="every non-empty stratum has >=1 label",
        actual=f"{len(empty_cells)} / {len(progress)} strata still unsampled",
        detail=detail,
    )]


def _timing_and_blind(records: list[dict], label_events: list[dict]) -> list[Check]:
    median_s = label_stats.median_seconds_per_label(records)
    blind = label_stats.blind_counts(records)
    total = blind["blind_label_count"] + blind["unblinded_label_count"]
    unblinded_share = blind["unblinded_label_count"] / total if total else 0.0

    checks = [info(
        STAGE, "median seconds per label",
        expected="~180-240s (the stated 3-4 minute target)",
        actual=f"{median_s:.0f}s" if median_s is not None else "no timed labels yet",
        detail="",
    )]
    checks.append((warn if unblinded_share > 0.2 else info)(
        STAGE, "blind vs unblinded label mix",
        expected="mostly blind; unblinded labels are weaker evidence",
        actual=f"{blind['blind_label_count']} blind / {blind['unblinded_label_count']} unblinded "
               f"({unblinded_share:.0%} unblinded)",
        detail="",
    ))
    return checks


def _self_consistency(records: list[dict]) -> list[Check]:
    sc = label_stats.self_consistency(records)
    if sc["repeats_served"] == 0:
        return [info(
            STAGE, "self-consistency (agreement on silently repeated programs)",
            expected=">=1 repeat probe compared", actual="0 so far",
            detail="repeats fire ~every 10th serve; keep labelling",
        )]
    rate = sc["agreement_rate"]
    return [(warn if rate is not None and rate < 0.8 else ok)(
        STAGE, "self-consistency (agreement on silently repeated programs)",
        expected=">=80% agreement with your own prior label",
        actual=f"{rate:.0%} ({sc['agreements']} / {sc['repeats_served']})" if rate is not None else "n/a",
        detail="this rate is a ceiling on any model trained against these labels",
    )]
