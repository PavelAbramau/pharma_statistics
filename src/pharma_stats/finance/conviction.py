"""Conviction ratio: a program's synthetic cost index vs. the median for
its peers at the same phase (and, ideally, indication) — a signal about
sponsor conviction (are they spending unusually much/little relative to
comparable programs), not about the asset's underlying quality.

Peer grouping caveat: indication_code is still the provisional_programs.py
placeholder ("UNSPECIFIED") — the real OncoTree normalisation hasn't been
built (README.md, CLAUDE.md). Peer groups here use (highest phase across
the program's trials, heme/solid scope_category) as the best available
substitute for "phase and indication" until a real indication axis
exists — coarser than intended, and this coarseness is exactly why a
program's conviction ratio should be read as approximate, same spirit as
the cost index itself.
"""
from __future__ import annotations

import statistics
from typing import Optional

from pharma_stats.finance import cost_model as cm


def peer_group_key(program: dict) -> tuple[Optional[str], str]:
    """(highest_phase, scope_category) — see module docstring for why
    this substitutes for "phase and indication" today."""
    phases: list[str] = []
    for t in program.get("trials") or []:
        p = cm.highest_phase(t.get("phases"))
        if p:
            phases.append(p)
    phase = cm.highest_phase(phases)
    scope_category = program.get("scope_category") or "unknown"
    return phase, scope_category


def conviction_ratio(program_spend: float, peer_spends: list[float]) -> Optional[float]:
    """program_spend / median(peer_spends). None if there's no usable
    peer denominator (median is 0, or fewer than 2 peers — a ratio
    against a single-member "peer group" of itself is not a comparison)."""
    usable_peers = [s for s in peer_spends if s > 0]
    if len(usable_peers) < 2:
        return None
    median = statistics.median(usable_peers)
    if median <= 0:
        return None
    return program_spend / median


def compute_conviction_ratios(programs: list[dict], spend_by_program: dict[str, float]) -> dict[str, dict]:
    """{program_id: {"peer_group": (phase, scope_category), "spend":
    float, "peer_median": float|None, "conviction_ratio": float|None,
    "n_peers": int}}. spend_by_program is a program_id -> current cost
    index (see cost_model.program_cost_index_snapshot)."""
    groups: dict[tuple, list[str]] = {}
    for p in programs:
        pid = p["program_id"]
        if pid not in spend_by_program:
            continue
        groups.setdefault(peer_group_key(p), []).append(pid)

    out: dict[str, dict] = {}
    for p in programs:
        pid = p["program_id"]
        if pid not in spend_by_program:
            continue
        key = peer_group_key(p)
        peer_ids = [x for x in groups[key] if x != pid]
        peer_spends = [spend_by_program[x] for x in peer_ids]
        spend = spend_by_program[pid]
        ratio = conviction_ratio(spend, peer_spends)
        median = statistics.median([s for s in peer_spends if s > 0]) if len([s for s in peer_spends if s > 0]) >= 2 else None
        out[pid] = {
            "peer_group": key, "spend": spend, "peer_median": median,
            "conviction_ratio": ratio, "n_peers": len(peer_ids),
        }
    return out
