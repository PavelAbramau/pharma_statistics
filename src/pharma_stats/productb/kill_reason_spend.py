"""B1: kill reason vs. spend — the core input to Product B's sourcing
screener (diagrams/diagram.md: "Product B — sourcing screener: kill-reason
· recoverability · crowding").

Per docs/decisions/0004 (spend and survival are jointly determined; spend
is a conviction signal, never a quality one), this must NEVER be read as
"programs that spent more failed for a better reason." It is descriptive:
a program the sponsor was still resourcing when it died on efficacy
grounds is a different sourcing proposition than one dropped cheaply in a
portfolio cut, regardless of which reason is "better." Reported per
docs/decisions/0004 for exactly that reason — this module exists to
surface the distinction honestly, not to rank kill reasons by merit.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date
from typing import Optional

from pharma_stats.finance import panel as money_panel
from pharma_stats.labelling import store


@dataclass
class KillReasonSpendRow:
    program_id: str
    proposed_name: Optional[str]
    kill_reason: str
    label_evidence_date: str
    estimated_cumulative_spend: Optional[float]
    conviction_ratio: Optional[float]


def dead_confirmed_records(gold_records: list[dict]) -> list[dict]:
    """Latest label per program (store.latest_by_program already excludes
    repeat-probe re-serves and non-"label" actions), filtered to
    dead_confirmed records that actually carry a kill_reason — the gold
    store's own validator (labelling/store.validate_label_payload)
    guarantees every dead_confirmed record has one, but a record written
    before that invariant existed is still possible in principle, so this
    filters rather than assumes."""
    latest = store.latest_by_program(gold_records)
    return [r for r in latest.values() if r.get("status") == "dead_confirmed" and r.get("kill_reason")]


def spend_at_death_rows(dead_records: list[dict], panel: list[dict]) -> list[KillReasonSpendRow]:
    """One row per dead_confirmed record with a label_evidence_date,
    resolving both money-layer features as of THAT date (never a later
    one, and never the program's final/most-recent value if that postdates
    the kill label) via pharma_stats.finance.panel.value_as_of."""
    by_program = money_panel.index_by_program(panel)
    rows = []
    for r in dead_records:
        evidence_date = r.get("label_evidence_date")
        if not evidence_date:
            continue
        as_of = date.fromisoformat(evidence_date)
        pid = r["program_id"]
        program_rows = by_program.get(pid, [])
        rows.append(KillReasonSpendRow(
            program_id=pid,
            proposed_name=r.get("proposed_name"),
            kill_reason=r["kill_reason"],
            label_evidence_date=evidence_date,
            estimated_cumulative_spend=money_panel.value_as_of(program_rows, as_of, "estimated_cumulative_spend"),
            conviction_ratio=money_panel.value_as_of(program_rows, as_of, "conviction_ratio"),
        ))
    return rows


def summarize_by_kill_reason(rows: list[KillReasonSpendRow]) -> dict[str, dict]:
    """Per kill_reason: n (all dead_confirmed programs with that reason),
    n_with_spend_data (the subset with a resolvable estimated_cumulative_
    spend), and median/mean/min/max spend among that subset. Small-N
    summaries are reported as-is, never smoothed or hidden — recall over
    precision, same as everywhere else in this project (CLAUDE.md)."""
    counts: dict[str, int] = {}
    spends_by_reason: dict[str, list[float]] = {}
    for row in rows:
        counts[row.kill_reason] = counts.get(row.kill_reason, 0) + 1
        if row.estimated_cumulative_spend is not None:
            spends_by_reason.setdefault(row.kill_reason, []).append(row.estimated_cumulative_spend)

    out = {}
    for reason, n in counts.items():
        spends = spends_by_reason.get(reason, [])
        out[reason] = {
            "n": n,
            "n_with_spend_data": len(spends),
            "median_spend": statistics.median(spends) if spends else None,
            "mean_spend": statistics.mean(spends) if spends else None,
            "min_spend": min(spends) if spends else None,
            "max_spend": max(spends) if spends else None,
        }
    return out
