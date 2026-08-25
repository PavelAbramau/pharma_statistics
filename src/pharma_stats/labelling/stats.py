"""Gold-set statistics shared by the live app (/api/session) and the
audit harness's gold stage — one implementation, checked in two places."""
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


def blind_counts(records: list[dict]) -> dict:
    label_events = [r for r in records if r["action"] == "label" and not r["is_repeat_probe"]]
    blind_count = sum(1 for r in label_events if r["blind"])
    return {
        "blind_label_count": blind_count,
        "unblinded_label_count": len(label_events) - blind_count,
    }


def median_seconds_per_label(records: list[dict]) -> Optional[float]:
    seconds = [
        r["seconds_spent"] for r in records
        if r["action"] == "label" and not r["is_repeat_probe"] and r.get("seconds_spent")
    ]
    return median(seconds) if seconds else None


def self_consistency(records: list[dict]) -> dict:
    """Agreement rate between each silently-repeated label and the primary
    (non-repeat) label for the same program — the labeller's own
    disagreement-with-himself rate, which caps any model trained against
    him."""
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
