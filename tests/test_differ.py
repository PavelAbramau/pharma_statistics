"""Tests for the local EvidenceEvent differ.

No network — everything here operates on in-memory study dicts shaped
like the real CT.gov v2 protocolSection, matching what
provisional_programs.py and extract.py both parse.
"""
from __future__ import annotations

from datetime import date

from pharma_stats.differ.diff import diff_versions
from pharma_stats.differ.events import EVENT_TYPES


def _study(
    status="RECRUITING", phases=None, sponsor="Acme Oncology",
    enrollment_count=100, enrollment_type="ESTIMATED",
    primary_completion=("2025-01-01", "ESTIMATED"), completion=("2025-06-01", "ESTIMATED"),
    arms=None, primary_outcomes=None,
):
    return {
        "protocolSection": {
            "statusModule": {
                "overallStatus": status,
                "primaryCompletionDateStruct": {"date": primary_completion[0], "type": primary_completion[1]},
                "completionDateStruct": {"date": completion[0], "type": completion[1]},
            },
            "designModule": {
                "phases": phases or ["PHASE2"],
                "enrollmentInfo": {"count": enrollment_count, "type": enrollment_type},
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": sponsor}},
            "armsInterventionsModule": {"armGroups": arms or [{"label": "Arm A"}, {"label": "Arm B"}]},
            "outcomesModule": {"primaryOutcomes": primary_outcomes or [{"measure": "ORR", "timeFrame": "12mo"}]},
        }
    }


EVENT_DATE = date(2024, 6, 1)


def test_identical_studies_produce_no_events():
    s = _study()
    events = diff_versions("NCT1", 1, 2, s, s, EVENT_DATE)
    assert events == []


def test_status_change_detected():
    prev, curr = _study(status="RECRUITING"), _study(status="TERMINATED")
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {e.event_type for e in events}
    assert "status_changed" in types
    e = next(e for e in events if e.event_type == "status_changed")
    assert e.from_value == "RECRUITING" and e.to_value == "TERMINATED"
    assert e.event_date == EVENT_DATE


def test_phase_and_sponsor_change_detected():
    prev = _study(phases=["PHASE1"], sponsor="Old Sponsor Inc")
    curr = _study(phases=["PHASE2"], sponsor="New Sponsor Inc")
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {e.event_type for e in events}
    assert "phase_changed" in types
    assert "sponsor_changed" in types


def test_enrollment_decrease_has_direction():
    prev = _study(enrollment_count=300, enrollment_type="ESTIMATED")
    curr = _study(enrollment_count=150, enrollment_type="ESTIMATED")
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    e = next(e for e in events if e.event_type == "enrollment_target_changed")
    assert e.direction == "decreased"
    assert e.from_value == 300 and e.to_value == 150


def test_enrollment_increase_has_direction():
    prev = _study(enrollment_count=100, enrollment_type="ESTIMATED")
    curr = _study(enrollment_count=120, enrollment_type="ESTIMATED")
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    e = next(e for e in events if e.event_type == "enrollment_target_changed")
    assert e.direction == "increased"


def test_enrollment_estimated_to_actual_is_finalized_not_changed():
    """The invariant: an ESTIMATED->ACTUAL transition must never be
    reported as enrollment_target_changed, even when the numbers differ —
    that's the estimate becoming fact, not a silent cut."""
    prev = _study(enrollment_count=300, enrollment_type="ESTIMATED")
    curr = _study(enrollment_count=214, enrollment_type="ACTUAL")
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {e.event_type for e in events}
    assert "enrollment_target_changed" not in types
    assert "enrollment_finalized" in types
    e = next(e for e in events if e.event_type == "enrollment_finalized")
    assert e.direction == "finalized"
    assert e.from_value == 300 and e.to_value == 214


def test_completion_date_pushed_has_direction_and_magnitude():
    prev = _study(completion=("2025-01-01", "ESTIMATED"))
    curr = _study(completion=("2025-07-01", "ESTIMATED"))
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    e = next(e for e in events if e.event_type == "completion_date_pushed")
    assert e.direction == "pushed_later"

    prev2 = _study(completion=("2025-07-01", "ESTIMATED"))
    curr2 = _study(completion=("2025-01-01", "ESTIMATED"))
    events2 = diff_versions("NCT1", 1, 2, prev2, curr2, EVENT_DATE)
    e2 = next(e for e in events2 if e.event_type == "completion_date_pushed")
    assert e2.direction == "pulled_earlier"


def test_completion_date_estimated_to_actual_is_finalized_not_pushed():
    """Same invariant as enrollment, for both date fields. This is the
    check the user says has already caught one real error — it must stay
    impossible to regress."""
    prev = _study(
        primary_completion=("2024-03-01", "ESTIMATED"),
        completion=("2024-06-01", "ESTIMATED"),
    )
    curr = _study(
        primary_completion=("2024-05-15", "ACTUAL"),  # actual date differs from the estimate
        completion=("2024-09-20", "ACTUAL"),
    )
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {e.event_type for e in events}
    assert "primary_completion_date_pushed" not in types
    assert "completion_date_pushed" not in types
    assert "primary_completion_date_finalized" in types
    assert "completion_date_finalized" in types


def test_no_pushed_or_changed_event_ever_spans_an_estimated_actual_boundary():
    """Property-style guard: across a battery of ESTIMATED/ACTUAL
    transition combinations (including the reverse, ACTUAL->ESTIMATED),
    no enrollment_target_changed / *_date_pushed event is ever produced
    when the type actually changed — regardless of by how much the
    underlying value also moved."""
    transitions = [
        ("ESTIMATED", "ACTUAL"),
        ("ACTUAL", "ESTIMATED"),
    ]
    for enrollment_from_type, enrollment_to_type in transitions:
        for pc_from_type, pc_to_type in transitions:
            prev = _study(
                enrollment_count=500, enrollment_type=enrollment_from_type,
                primary_completion=("2024-01-01", pc_from_type),
                completion=("2024-01-01", pc_from_type),
            )
            curr = _study(
                enrollment_count=1, enrollment_type=enrollment_to_type,  # extreme delta on purpose
                primary_completion=("2030-01-01", pc_to_type),
                completion=("2030-01-01", pc_to_type),
            )
            events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
            forbidden = {"enrollment_target_changed", "primary_completion_date_pushed", "completion_date_pushed"}
            fired = {e.event_type for e in events}
            assert not (fired & forbidden), (
                f"boundary leak for enrollment {enrollment_from_type}->{enrollment_to_type}, "
                f"dates {pc_from_type}->{pc_to_type}: fired {fired & forbidden}"
            )


def test_event_date_is_always_the_passed_in_knowability_date():
    """diff_versions must never invent its own date from the study bodies
    (e.g. lastUpdateSubmitDate) — event_date is whatever the caller
    passes, which extract.py sources from history_index.posted_date."""
    prev, curr = _study(status="RECRUITING"), _study(status="ACTIVE_NOT_RECRUITING")
    knowability_date = date(2019, 11, 4)
    events = diff_versions("NCT1", 1, 2, prev, curr, knowability_date)
    assert all(e.event_date == knowability_date for e in events)


def test_arm_added_and_removed():
    prev = _study(arms=[{"label": "Arm A"}, {"label": "Arm B"}])
    curr = _study(arms=[{"label": "Arm A"}, {"label": "Arm C"}])
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {(e.event_type, e.from_value or e.to_value) for e in events}
    assert ("arm_removed", "Arm B") in types
    assert ("arm_added", "Arm C") in types


def test_primary_outcome_added_removed_changed():
    prev = _study(primary_outcomes=[
        {"measure": "ORR", "timeFrame": "12mo"},
        {"measure": "OS", "timeFrame": "24mo"},
    ])
    curr = _study(primary_outcomes=[
        {"measure": "ORR", "timeFrame": "18mo"},  # timeframe changed
        {"measure": "PFS", "timeFrame": "12mo"},  # OS removed, PFS added
    ])
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    types = {e.event_type for e in events}
    assert "primary_outcome_changed" in types
    assert "primary_outcome_removed" in types
    assert "primary_outcome_added" in types


def test_all_event_types_used_in_tests_are_in_the_vocab():
    prev = _study(
        status="RECRUITING", phases=["PHASE1"], sponsor="A", enrollment_count=100,
        enrollment_type="ESTIMATED", primary_completion=("2024-01-01", "ESTIMATED"),
        completion=("2024-01-01", "ESTIMATED"), arms=[{"label": "X"}],
        primary_outcomes=[{"measure": "M1", "timeFrame": "1mo"}],
    )
    curr = _study(
        status="TERMINATED", phases=["PHASE2"], sponsor="B", enrollment_count=1,
        enrollment_type="ACTUAL", primary_completion=("2024-02-01", "ACTUAL"),
        completion=("2024-02-01", "ACTUAL"), arms=[{"label": "Y"}],
        primary_outcomes=[{"measure": "M2", "timeFrame": "2mo"}],
    )
    events = diff_versions("NCT1", 1, 2, prev, curr, EVENT_DATE)
    for e in events:
        assert e.event_type in EVENT_TYPES
