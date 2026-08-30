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
    # gate 3 only: gates 1-2 are triage rejections (never a program), and
    # status_revision_rate below is specifically about the status
    # judgement, which only exists at gate 3.
    label_events = [
        r for r in records
        if r["action"] == "label" and r.get("gate_reached") == 3 and not r["is_repeat_probe"]
    ]

    # Runs unconditionally, even on an empty gold set — this is a
    # structural safety check, not something that waits for labelling to
    # start. See silver/store.py: the gold set is the evaluation set for
    # the silver auto-labeller and must stay independent of it.
    silver_isolation_check = _no_auto_sourced_records_in_gold(records)

    if not records:
        return silver_isolation_check + [info(
            STAGE, "gold label records",
            expected=">=1 record in gold/labels.jsonl",
            actual="0 — no labelling has happened yet",
            detail="run `python scripts/run_labelling_app.py` to start",
        )]

    checks: list[Check] = list(silver_isolation_check)
    checks += _dead_confirmed_invariant(records)
    checks += _history_coverage_invariant(records)
    checks += _gate_breakdown(records)
    checks += _stratum_coverage(records, label_events)
    checks += _timing_and_blind(records, label_events)
    checks += _self_consistency(records)
    checks += _status_revision_rate(label_events)
    return checks


def _no_auto_sourced_records_in_gold(records: list[dict]) -> list[Check]:
    """Absolute constraint: the gold set is the evaluation set for the
    silver auto-labeller (silver/labels.jsonl) and must stay independent
    of it. silver/store.py stamps labeller="auto" on every record it
    writes and refuses to write anywhere but silver/labels.jsonl; this is
    the independent check, against gold/labels.jsonl itself, that nothing
    upstream ever violated that."""
    auto = [r for r in records if r.get("labeller") == "auto"]
    return [(fail if auto else ok)(
        STAGE, "zero auto-sourced (labeller='auto') records in gold/labels.jsonl",
        expected="0", actual=f"{len(auto)} / {len(records)}",
        detail=", ".join(r.get("event_id", "?") for r in auto[:10]),
    )]


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


def _history_coverage_invariant(records: list[dict]) -> list[Check]:
    """The hard guard, re-verified against what was actually served and
    saved: a served/saved program's evidence must have been complete
    (history_coverage_at_serve_time == 'full') at the moment it was
    served. /api/next already refuses to serve anything less, and
    validate_label_payload already refuses to save anything less — this
    is the third, independent check, against the append-only record
    itself rather than trusting either of those held. An empty/partial
    event timeline is otherwise indistinguishable on screen from a
    program that was genuinely never amended, so a violation here means
    a label may have been made against evidence that looked like silence
    but was actually just missing data."""
    labels = [r for r in records if r["action"] in ("label", "skip")]
    bad = [r for r in labels if r.get("history_coverage_at_serve_time") != "full"]
    return [(fail if bad else ok)(
        STAGE, "zero served programs had less-than-full history_coverage at serve time",
        expected="0 violations", actual=f"{len(bad)} / {len(labels)} violate it",
        detail=", ".join(
            f"{r['event_id']} ({r['program_id']}: {r.get('history_coverage_at_serve_time')!r})"
            for r in bad[:10]
        ),
    )]


def _gate_breakdown(records: list[dict]) -> list[Check]:
    """How much of the review effort is real labelling (gate 3) vs triage
    (gates 1-2)? Surfaced here so the audit report and the live app's
    /api/session agree on the same counts — see label_stats.gate_counts.
    Also breaks gate-1 rejections down by matching pattern: a pile-up on
    one weak literal term is a patterns.py tuning signal; an even spread
    means the noise is inherent to sponsor expansion."""
    counts = label_stats.gate_counts(records)
    total = sum(counts.values())
    pattern_counts = label_stats.gate1_rejection_pattern_counts(records)
    lines = []
    for p in pattern_counts:
        term_suffix = f":{p['matched_term']!r}" if p["matched_term"] else ""
        lines.append(f"{p['discovery_strategy']}/{p['match_strength']}{term_suffix}={p['count']}")
    pattern_detail = " | ".join(lines)
    return [info(
        STAGE, "review effort by gate reached",
        expected="n/a — informational, this is what patterns.py tuning reads",
        actual=f"gate1={counts['gate1_rejected_count']}, gate2={counts['gate2_rejected_count']}, "
               f"gate3={counts['gate3_labelled_count']} (of {total} reviewed)",
        detail=f"gate-1 rejections by matched pattern: {pattern_detail or '(none yet)'}",
    )]


def _stratum_coverage(records: list[dict], label_events: list[dict]) -> list[Check]:
    programs = pp.load_materialized()
    if not programs:
        return [info(
            STAGE, "stratum coverage",
            expected="provisional_programs materialized", actual="not materialized yet",
            detail="run the labelling app once, or pharma_stats.labelling.provisional_programs.materialize()",
        )]

    # gate 3 only — a gate 1/2 triage rejection was never a program, so it
    # must never count toward stratum coverage or the label-count target
    labelled_ids = store.fully_labelled_program_ids(records)
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


def _status_revision_rate(label_events: list[dict]) -> list[Check]:
    """How often does checking external evidence (PubMed / web search /
    CT.gov, opened from the review screen) actually change the status the
    labeller had formed from in-app evidence alone? A high rate says the
    in-app review screen is missing signal the external sources have; a
    near-zero rate says the outbound links are mostly confirmatory."""
    if not label_events:
        return [info(
            STAGE, "status revised after external search",
            expected="n/a", actual="no labels yet", detail="",
        )]
    revised = [r for r in label_events if r.get("status_revised_after_external_search")]
    rate = len(revised) / len(label_events)
    return [info(
        STAGE, "status revised after external search",
        expected="tracked, not a pass/fail bar",
        actual=f"{rate:.0%} ({len(revised)} / {len(label_events)})",
        detail="high: the review screen is missing signal external sources have; "
               "near-zero: the outbound links are mostly confirmatory",
    )]


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
