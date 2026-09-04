"""Tests for the main/validation dual-queue mechanism in labelling/queue.py
— switch_queue, and pop_next/requeue operating on whichever queue is
active rather than always "order"."""
from __future__ import annotations

from pharma_stats.labelling import queue as q


def test_new_session_has_validation_order_and_defaults_to_main():
    session = q.new_session([], exclude_ids=set())
    assert session["validation_order"] == []
    assert session["active_queue"] == "main"


def test_switch_queue_changes_active_queue():
    session = q.new_session([], exclude_ids=set())
    q.switch_queue(session, "validation")
    assert session["active_queue"] == "validation"
    q.switch_queue(session, "main")
    assert session["active_queue"] == "main"


def test_switch_queue_rejects_unknown_name():
    session = q.new_session([], exclude_ids=set())
    try:
        q.switch_queue(session, "bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pop_next_uses_main_order_by_default():
    session = q.new_session([], exclude_ids=set())
    session["order"] = ["p1", "p2"]
    session["validation_order"] = ["v1"]
    pid, is_repeat = q.pop_next(session, labelled_ids=set())
    assert pid == "p1"
    assert session["order"] == ["p2"]
    assert session["validation_order"] == ["v1"]  # untouched


def test_pop_next_uses_validation_order_when_active():
    session = q.new_session([], exclude_ids=set())
    session["order"] = ["p1"]
    session["validation_order"] = ["v1", "v2"]
    q.switch_queue(session, "validation")
    pid, is_repeat = q.pop_next(session, labelled_ids=set())
    assert pid == "v1"
    assert session["validation_order"] == ["v2"]
    assert session["order"] == ["p1"]  # untouched


def test_repeat_probe_never_fires_on_validation_queue():
    session = q.new_session([], exclude_ids=set())
    session["validation_order"] = [f"v{i}" for i in range(30)]
    q.switch_queue(session, "validation")
    session["total_served"] = 8  # next call would be total+1=9 -> not a multiple of 10 anyway
    session["total_served"] = 9  # next call would be total+1=10 -> WOULD be a repeat probe on main
    pid, is_repeat = q.pop_next(session, labelled_ids={"already1", "already2"})
    assert is_repeat is False
    assert pid == "v0"


def test_requeue_with_explicit_queue_name_ignores_current_active_queue():
    """The TTL-stale-sweep case: a card served from auto_review must
    requeue back into auto_review even if the reviewer has since
    switched the active queue to main."""
    session = q.new_session([], exclude_ids=set())
    session["auto_review_order"] = []
    session["order"] = []
    q.switch_queue(session, "auto_review")
    # ... time passes, reviewer switches back to main ...
    q.switch_queue(session, "main")
    q.requeue(session, "a1", queue_name="auto_review")
    assert session["auto_review_order"] == ["a1"]
    assert session["order"] == []


def test_make_serve_token_records_origin_queue():
    session = q.new_session([], exclude_ids=set())
    q.switch_queue(session, "auto_review")
    program = {"program_id": "p1", "band": 1, "primary_archetype": "other",
               "silence_score": 10, "history_coverage": "full"}
    token = q.make_serve_token(session, program, is_repeat=False)
    assert session["pending_serve"][token]["origin_queue"] == "auto_review"


def test_requeue_appends_to_active_queue():
    session = q.new_session([], exclude_ids=set())
    session["order"] = []
    session["validation_order"] = []
    q.requeue(session, "p1")
    assert session["order"] == ["p1"]
    assert session["validation_order"] == []

    q.switch_queue(session, "validation")
    q.requeue(session, "v1")
    assert session["validation_order"] == ["v1"]
    assert session["order"] == ["p1"]
