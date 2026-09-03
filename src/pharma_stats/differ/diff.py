"""Core version-pair differ.

Three constraints, non-negotiable:

1. Never diff a date/enrollment field across an ESTIMATED -> ACTUAL
   boundary as if it were a plan change. The sponsor's estimate becoming
   a confirmed fact is not evidence of anything happening to the trial —
   it's the trial reaching that point. Emit a *_finalized event instead,
   with no "direction" (there's nothing to compare).
2. event_date is always the *to_version*'s posted_date (when the change
   became publicly knowable), never submitted_date. A backtest run as-of
   any date must never be able to see a change before CT.gov actually
   posted it.
3. Enrollment and date changes carry a direction ("increased"/
   "decreased", "pushed_later"/"pulled_earlier") — a bare "changed" loses
   the one bit of information that actually matters for silent-kill
   detection (a quietly *shrunk* enrollment target is the signal; a
   typo-fixed +1 is not, but at least direction lets a human tell them
   apart alongside magnitude).
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pharma_stats.differ.events import EvidenceEvent


def _get(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def _parse_month_or_day(raw: Optional[str]) -> Optional[date]:
    if not raw:
        return None
    parts = raw.split("-")
    try:
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _diff_scalar(nct_id, fv, tv, event_date, event_type, field_name, prev, curr, detail_fmt):
    if prev == curr or prev is None or curr is None:
        return None
    return EvidenceEvent(
        nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
        event_type=event_type, field=field_name, direction=None,
        from_value=prev, to_value=curr, detail=detail_fmt.format(prev=prev, curr=curr),
    )


def _diff_dated_field(
    nct_id: str, fv: int, tv: int, event_date: date, field_name: str,
    prev_struct: Optional[dict], curr_struct: Optional[dict],
) -> list[EvidenceEvent]:
    """Applies rule 1 (ESTIMATED/ACTUAL boundary) and rule 3 (direction)
    to a {"date": "...", "type": "ESTIMATED"|"ACTUAL"} field."""
    if not prev_struct or not curr_struct:
        return []
    prev_date, prev_type = _parse_month_or_day(prev_struct.get("date")), prev_struct.get("type")
    curr_date, curr_type = _parse_month_or_day(curr_struct.get("date")), curr_struct.get("type")
    if prev_date is None or curr_date is None:
        return []

    if prev_type != curr_type:
        if curr_type == "ACTUAL":
            # the estimate became fact — not a plan change, a finalization
            return [EvidenceEvent(
                nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
                event_type=f"{field_name}_finalized", field=field_name, direction="finalized",
                from_value=prev_struct.get("date"), to_value=curr_struct.get("date"),
                detail=f"{field_name} finalized: estimated {prev_struct.get('date')} -> "
                       f"actual {curr_struct.get('date')}",
            )]
        # ACTUAL -> ESTIMATED (or other non-forward transition): the trial
        # un-finalized a date it had already reported as fact — a real
        # positive liveness signal (the trial reopened/resumed), not a
        # plan-change direction to claim either way.
        return [EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="trial_reopened", field=field_name, direction=None,
            from_value=prev_struct.get("date"), to_value=curr_struct.get("date"),
            detail=f"{field_name} reverted from actual to estimated: {prev_type} {prev_struct.get('date')} -> "
                   f"{curr_type} {curr_struct.get('date')} — trial likely reopened/resumed",
        )]

    if prev_date == curr_date:
        return []
    direction = "pushed_later" if curr_date > prev_date else "pulled_earlier"
    delta_days = (curr_date - prev_date).days
    return [EvidenceEvent(
        nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
        event_type=f"{field_name}_pushed", field=field_name, direction=direction,
        from_value=prev_struct.get("date"), to_value=curr_struct.get("date"),
        detail=f"{field_name} {direction.replace('_', ' ')} by {abs(delta_days)}d "
               f"({prev_struct.get('date')} -> {curr_struct.get('date')}, both {curr_type})",
    )]


def _diff_enrollment(nct_id, fv, tv, event_date, prev: Optional[dict], curr: Optional[dict]) -> list[EvidenceEvent]:
    if not prev or not curr:
        return []
    prev_count, prev_type = prev.get("count"), prev.get("type")
    curr_count, curr_type = curr.get("count"), curr.get("type")
    if prev_count is None or curr_count is None:
        return []

    if prev_type != curr_type:
        if curr_type == "ACTUAL":
            return [EvidenceEvent(
                nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
                event_type="enrollment_finalized", field="enrollment", direction="finalized",
                from_value=prev_count, to_value=curr_count,
                detail=f"enrollment finalized: estimated {prev_count} -> actual {curr_count}",
            )]
        return [EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="trial_reopened", field="enrollment", direction=None,
            from_value=prev_count, to_value=curr_count,
            detail=f"enrollment reverted from actual to estimated: {prev_type} {prev_count} -> "
                   f"{curr_type} {curr_count} — trial likely reopened/resumed",
        )]

    if prev_count == curr_count:
        return []
    direction = "increased" if curr_count > prev_count else "decreased"
    return [EvidenceEvent(
        nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
        event_type="enrollment_target_changed", field="enrollment", direction=direction,
        from_value=prev_count, to_value=curr_count,
        detail=f"enrollment target {direction} from {prev_count} to {curr_count} (both {curr_type})",
    )]


def _arm_labels(study: dict) -> set:
    arms = _get(study, "protocolSection", "armsInterventionsModule", "armGroups", default=[]) or []
    return {a.get("label") for a in arms if a.get("label")}


def _diff_arms(nct_id, fv, tv, event_date, prev: dict, curr: dict) -> list[EvidenceEvent]:
    prev_arms, curr_arms = _arm_labels(prev), _arm_labels(curr)
    events = []
    for removed in sorted(prev_arms - curr_arms):
        events.append(EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="arm_removed", field="armGroups", direction=None,
            from_value=removed, to_value=None, detail=f'arm/cohort removed: "{removed}"',
        ))
    for added in sorted(curr_arms - prev_arms):
        events.append(EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="arm_added", field="armGroups", direction=None,
            from_value=None, to_value=added, detail=f'arm/cohort added: "{added}"',
        ))
    return events


def _primary_outcomes(study: dict) -> dict:
    outcomes = _get(study, "protocolSection", "outcomesModule", "primaryOutcomes", default=[]) or []
    return {o.get("measure"): o.get("timeFrame") for o in outcomes if o.get("measure")}


def _diff_outcomes(nct_id, fv, tv, event_date, prev: dict, curr: dict) -> list[EvidenceEvent]:
    """Structural change only — whether a change is a "downgrade" is a
    judgement call this differ deliberately does not make; it reports
    what changed and leaves interpretation to the labeller/reviewer."""
    prev_o, curr_o = _primary_outcomes(prev), _primary_outcomes(curr)
    events = []
    for measure in sorted(set(prev_o) - set(curr_o)):
        events.append(EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="primary_outcome_removed", field="primaryOutcomes", direction=None,
            from_value=measure, to_value=None, detail=f'primary outcome removed: "{measure}"',
        ))
    for measure in sorted(set(curr_o) - set(prev_o)):
        events.append(EvidenceEvent(
            nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
            event_type="primary_outcome_added", field="primaryOutcomes", direction=None,
            from_value=None, to_value=measure, detail=f'primary outcome added: "{measure}"',
        ))
    for measure in sorted(set(prev_o) & set(curr_o)):
        if prev_o[measure] != curr_o[measure]:
            events.append(EvidenceEvent(
                nct_id=nct_id, from_version=fv, to_version=tv, event_date=event_date,
                event_type="primary_outcome_changed", field="primaryOutcomes", direction=None,
                from_value=f"{measure} ({prev_o[measure]})", to_value=f"{measure} ({curr_o[measure]})",
                detail=f'primary outcome timeframe changed for "{measure}": '
                       f"{prev_o[measure]} -> {curr_o[measure]}",
            ))
    return events


def diff_versions(
    nct_id: str, from_version: int, to_version: int,
    prev_study: dict, curr_study: dict, event_date: date,
) -> list[EvidenceEvent]:
    """Diff two adjacent (or any two ordered) study-body snapshots.
    `event_date` must be the *to_version*'s posted_date — the caller
    (extract.py) is responsible for sourcing that from history_index,
    never from a submitted-date field."""
    events: list[EvidenceEvent] = []

    prev_status = _get(prev_study, "protocolSection", "statusModule", "overallStatus")
    curr_status = _get(curr_study, "protocolSection", "statusModule", "overallStatus")
    e = _diff_scalar(nct_id, from_version, to_version, event_date, "status_changed", "overallStatus",
                      prev_status, curr_status, "status changed from {prev} to {curr}")
    if e:
        events.append(e)

    prev_phase = tuple(sorted(_get(prev_study, "protocolSection", "designModule", "phases", default=[]) or []))
    curr_phase = tuple(sorted(_get(curr_study, "protocolSection", "designModule", "phases", default=[]) or []))
    e = _diff_scalar(nct_id, from_version, to_version, event_date, "phase_changed", "phases",
                      prev_phase, curr_phase, "phase changed from {prev} to {curr}")
    if e:
        events.append(e)

    prev_sponsor = _get(prev_study, "protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "name")
    curr_sponsor = _get(curr_study, "protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "name")
    e = _diff_scalar(nct_id, from_version, to_version, event_date, "sponsor_changed", "leadSponsor",
                      prev_sponsor, curr_sponsor, "lead sponsor changed from {prev} to {curr}")
    if e:
        events.append(e)

    events += _diff_enrollment(
        nct_id, from_version, to_version, event_date,
        _get(prev_study, "protocolSection", "designModule", "enrollmentInfo"),
        _get(curr_study, "protocolSection", "designModule", "enrollmentInfo"),
    )
    events += _diff_dated_field(
        nct_id, from_version, to_version, event_date, "primary_completion_date",
        _get(prev_study, "protocolSection", "statusModule", "primaryCompletionDateStruct"),
        _get(curr_study, "protocolSection", "statusModule", "primaryCompletionDateStruct"),
    )
    events += _diff_dated_field(
        nct_id, from_version, to_version, event_date, "completion_date",
        _get(prev_study, "protocolSection", "statusModule", "completionDateStruct"),
        _get(curr_study, "protocolSection", "statusModule", "completionDateStruct"),
    )
    events += _diff_arms(nct_id, from_version, to_version, event_date, prev_study, curr_study)
    events += _diff_outcomes(nct_id, from_version, to_version, event_date, prev_study, curr_study)

    return events
