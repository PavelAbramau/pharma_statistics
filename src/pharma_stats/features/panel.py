"""Program x month feature panel: one row per (program, month) from a
program's earliest resolvable trial state to an end date, every
time-varying column resolved as-of that month only. This is the input
models/discrete_time_survival.py trains on.

Static columns (target, payload_chemotype, indication_mesh_term) are
attached once per program, same value on every row — see
attributes/target.py, attributes/payload.py, attributes/indication.py
and docs/decisions/0001 for why those are safe to read once rather than
per-month.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import duckdb

from pharma_stats.attributes import indication as ind
from pharma_stats.attributes import payload as pay
from pharma_stats.attributes import target as tgt
from pharma_stats.features import trial_asof
from pharma_stats.features.knowability import assert_all_columns_registered
from pharma_stats.finance import cost_model as cm
from pharma_stats.labelling.provisional_programs import compute_silence_score, _band_for_score


def _month_range(start: date, end: date) -> list[date]:
    months = []
    cursor = start.replace(day=1)
    while cursor <= end:
        months.append(cursor)
        year, month = cursor.year, cursor.month
        cursor = date(year + (month // 12), (month % 12) + 1, 1)
    return months


def build_program_month_panel(
    program: dict, con: duckdb.DuckDBPyConnection, *, end: Optional[date] = None,
) -> list[dict]:
    """[{program_id, as_of, silence_score_asof, band_asof, cost_index,
    contacts_locations_amendment_cadence_asof, target, payload_chemotype,
    indication_mesh_term}]. Empty if no trial on this program has any
    indexed history at all."""
    end = end or date.today()
    trials_meta = program.get("trials") or []
    nct_ids = [t["nct_id"] for t in trials_meta]

    # Expensive I/O (one snapshot read per fetched version) done ONCE per
    # trial here, never per month — a first draft called
    # trial_asof.resolve_trial_summary_as_of (full cache rebuild) inside
    # the month loop below and didn't finish a 10-program backtest in two
    # minutes. See features/trial_asof.py's module docstring.
    histories = {nct_id: cm.trial_version_history(nct_id, con) for nct_id in nct_ids}
    trial_caches = {nct_id: trial_asof.build_trial_cache(nct_id, con) for nct_id in nct_ids}
    earliest = min((h[0]["posted_date"] for h in histories.values() if h), default=None)
    if earliest is None:
        return []

    name = program.get("proposed_name")
    synonyms = program.get("synonyms") or []
    target, _source = tgt.derive_target(name, synonyms, text_snippets=None)
    payload_chemotype = pay.derive_payload_chemotype(name, synonyms)
    indication_mesh_term = ind.program_indication_mesh_term(program)

    rows = []
    for month in _month_range(earliest, end):
        trial_summaries = [
            s for s in (trial_asof.resolve_from_cache(trial_caches[nct_id], month) for nct_id in nct_ids)
            if s is not None
        ]
        silence_score, _breakdown = compute_silence_score(trial_summaries, as_of=month)
        band = _band_for_score(silence_score)

        cost_index = sum(
            cm._cost_from_entry(cm._state_as_of_from_history(histories[nct_id], month), month)
            for nct_id in nct_ids if histories.get(nct_id)
        )

        cadence_count = 0
        for s in trial_summaries:
            for h in s.history:
                if "Contacts/Locations" in (h.get("changed_modules") or []):
                    cadence_count += 1
        elapsed_years = max((month - earliest).days / 365.25, 1 / 365.25)
        cadence = cadence_count / elapsed_years

        row = {
            "program_id": program["program_id"],
            "as_of": month.isoformat(),
            "silence_score_asof": silence_score,
            "band_asof": band,
            "cost_index": cost_index,
            "contacts_locations_amendment_cadence_asof": cadence,
            "target": target or "undisclosed",
            "payload_chemotype": payload_chemotype,
            "indication_mesh_term": indication_mesh_term or "unknown",
        }
        rows.append(row)

    if rows:
        assert_all_columns_registered([k for k in rows[0] if k not in ("program_id", "as_of")])
    return rows
