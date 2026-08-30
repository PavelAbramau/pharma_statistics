"""Gold-set statistics shared by the live app (/api/session) and the
audit harness's gold stage — one implementation, checked in two places.

Every function that measures "labelling progress" (stratum_progress,
median_seconds_per_label, blind_counts, self_consistency) counts only
gate_reached=3 records — a Gate 1/2 triage rejection was never a program,
so it must never inflate the label-count target or the timing/blind QA
signals those functions feed. See store.fully_labelled_program_ids."""
from __future__ import annotations

from statistics import median
from typing import Optional


def stratum_progress(programs: list[dict], labelled_ids: set[str]) -> list[dict]:
    cells: dict[tuple[int, str], dict] = {}
    for p in programs:
        key = (p["band"], p["primary_archetype"])
        cells.setdefault(key, {"band": p["band"], "archetype": p["primary_archetype"], "total": 0, "labelled": 0})
        cells[key]["total"] += 1
        if p["program_id"] in labelled_ids:
            cells[key]["labelled"] += 1
    return sorted(cells.values(), key=lambda c: (c["band"], c["archetype"]))


def _gate3_label_events(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if r["action"] == "label" and r.get("gate_reached") == 3 and not r["is_repeat_probe"]
    ]


def blind_counts(records: list[dict]) -> dict:
    label_events = _gate3_label_events(records)
    blind_count = sum(1 for r in label_events if r["blind"])
    return {
        "blind_label_count": blind_count,
        "unblinded_label_count": len(label_events) - blind_count,
    }


def median_seconds_per_label(records: list[dict]) -> Optional[float]:
    seconds = [r["seconds_spent"] for r in _gate3_label_events(records) if r.get("seconds_spent")]
    return median(seconds) if seconds else None


def gate_counts(records: list[dict]) -> dict:
    """How many programs are currently sitting at each gate outcome — the
    triage-vs-real-label split the queue and the sufficiency checks must
    respect. Counted from each program's LATEST label record (a program
    re-decided after an earlier gate-1 reject would move buckets)."""
    from pharma_stats.labelling.store import latest_by_program

    counts = {1: 0, 2: 0, 3: 0}
    for r in latest_by_program(records).values():
        gate = r.get("gate_reached")
        if gate in counts:
            counts[gate] += 1
    return {
        "gate1_rejected_count": counts[1],
        "gate2_rejected_count": counts[2],
        "gate3_labelled_count": counts[3],
    }


def gate1_rejection_pattern_counts(records: list[dict]) -> list[dict]:
    """Running distribution of Gate 1 rejections (is_adc != yes) by the
    discovery evidence that put the candidate in front of a reviewer at
    all — strategy + match_strength + matched_term. This is the tuning
    signal for discovery.patterns: a pile-up on one weak literal term (e.g.
    "conjugate") says tighten or drop it; an even spread across strategies
    says the noise is inherent to sponsor expansion, not a pattern bug.
    Every gate-1 reject counts here (not just each program's latest), so a
    since-reconsidered candidate's earlier miss still shows up as data."""
    counts: dict[tuple, int] = {}
    for r in records:
        if r["action"] != "label" or r.get("gate_reached") != 1 or r.get("is_repeat_probe"):
            continue
        key = (r.get("discovery_strategy"), r.get("match_strength"), r.get("matched_term"))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"discovery_strategy": s, "match_strength": ms, "matched_term": mt, "count": n}
        for (s, ms, mt), n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def self_consistency(records: list[dict]) -> dict:
    """Agreement rate between each silently-repeated label and the primary
    (non-repeat) label for the same program — the labeller's own
    disagreement-with-himself rate, which caps any model trained against
    him. Repeats are only ever drawn from fully-labelled (gate 3) programs
    (see queue.py), so this stays a comparison of real status/kill_reason
    judgements, not gate-1/2 triage."""
    from pharma_stats.labelling.store import latest_by_program

    primary_by_pid = latest_by_program(records)
    repeats = [r for r in records if r["action"] == "label" and r["is_repeat_probe"]]
    agree, total_compared = 0, 0
    for r in repeats:
        primary = primary_by_pid.get(r["program_id"])
        if not primary:
            continue
        total_compared += 1
        if primary["status"] == r["status"] and primary.get("kill_reason") == r.get("kill_reason"):
            agree += 1
    return {
        "repeats_served": total_compared,
        "agreements": agree,
        "agreement_rate": (agree / total_compared) if total_compared else None,
    }
