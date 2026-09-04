"""Money-layer feature panel: the first real consumer of financial_events
(154k+ rows written by scripts/build_financial_layer_cost_index.py,
nothing downstream read them until this module). Turns the raw monthly
events into two per-program, per-month features —
``conviction_ratio`` and ``estimated_cumulative_spend`` — both registered
in audit/leakage.md with a knowability-date contract, and checked there
by pharma_stats.audit.features.

Both features are read from financial_events, never recomputed from
finance.cost_model / finance.conviction directly: those modules already
wrote their output there, and this module's job is only to shape it into
a feature panel, not to re-derive it.

- ``conviction_ratio``: financial_events.conviction_ratio_monthly, as-is.
  Its own event_date is already its knowability date — the peer
  comparison at build time used only that same month's peer spend (see
  finance/conviction.py), never a later one.
- ``estimated_cumulative_spend``: NOT a raw event type. A running sum of
  financial_events.synthetic_cost_index_monthly, ordered by event_date,
  one point per program. Summing only points whose own event_date is
  <= the row's as_of can never pull in a later-knowable value, so the
  cumulative row's knowability date is still just its own as_of — see
  audit/leakage.md for the full argument.

Per docs/decisions/0004, neither feature is a quality signal — see that
decision before reading either into a causal claim anywhere downstream.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Optional

from pharma_stats.finance import store as fstore

COST_EVENT_TYPE = "synthetic_cost_index_monthly"
CONVICTION_EVENT_TYPE = "conviction_ratio_monthly"

# Every feature this module produces and registers in audit/leakage.md —
# checked directly against that register by pharma_stats.audit.features,
# not just asserted here.
FEATURE_NAMES = ("conviction_ratio", "estimated_cumulative_spend")


def build_money_layer_panel(warehouse_db: Optional[Any] = None) -> list[dict]:
    """One row per (program_id, as_of) that has a financial_events point
    that month — a cost-index event, a conviction-ratio event, or both.
    Never synthesizes a row for a month with no event at all: "no
    financial_events row this month" and "spend of exactly 0 this month"
    are different facts and must never be conflated.

    Each row: {program_id, as_of (ISO date string), conviction_ratio
    (float|None — only set for months with their own conviction event),
    estimated_cumulative_spend (float|None — forward-filled from the
    latest cost-index point at or before as_of), knowability_date (always
    == as_of — the leakage contract this panel guarantees)}.

    Sorted by (program_id, as_of) ascending, so callers get each
    program's rows already in the order value_as_of requires.
    """
    records = fstore.load_records(warehouse_db)

    cost_points: dict[str, list[tuple[date, float]]] = defaultdict(list)
    conviction_by_month: dict[tuple[str, date], float] = {}
    months_by_program: dict[str, set] = defaultdict(set)

    for r in records:
        if r["value"] is None:
            continue
        pid, d = r["subject_id"], r["event_date"]
        if r["event_type"] == COST_EVENT_TYPE:
            cost_points[pid].append((d, r["value"]))
            months_by_program[pid].add(d)
        elif r["event_type"] == CONVICTION_EVENT_TYPE:
            conviction_by_month[(pid, d)] = r["value"]
            months_by_program[pid].add(d)

    panel: list[dict] = []
    for pid, months in months_by_program.items():
        points_sorted = sorted(cost_points.get(pid, []))
        cumulative_by_month: dict[date, float] = {}
        running = 0.0
        for d, value in points_sorted:
            running += value
            cumulative_by_month[d] = running

        last_seen: Optional[float] = None
        for d in sorted(months):
            if d in cumulative_by_month:
                last_seen = cumulative_by_month[d]
            panel.append({
                "program_id": pid,
                "as_of": d.isoformat(),
                "conviction_ratio": conviction_by_month.get((pid, d)),
                "estimated_cumulative_spend": last_seen,
                "knowability_date": d.isoformat(),
            })

    panel.sort(key=lambda row: (row["program_id"], row["as_of"]))
    return panel


def index_by_program(panel: list[dict]) -> dict[str, list[dict]]:
    """Group panel rows by program_id, preserving build_money_layer_panel's
    ascending as_of order within each group — the contract value_as_of
    depends on."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for row in panel:
        idx[row["program_id"]].append(row)
    return idx


def value_as_of(rows_for_program: list[dict], as_of: date, field: str) -> Optional[float]:
    """Latest non-None `field` value at or before `as_of`, from one
    program's rows (must already be ascending by as_of — index_by_program's
    contract). Same as-of-resolution pattern as
    cost_model._state_as_of_from_history, applied to the feature panel
    instead of raw trial history. None if the program has no such value
    yet as of that date — never guessed as 0."""
    best: Optional[float] = None
    for row in rows_for_program:
        row_date = date.fromisoformat(row["as_of"])
        if row_date > as_of:
            break
        if row.get(field) is not None:
            best = row[field]
    return best
