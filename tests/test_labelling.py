"""Tests for the labelling app: provisional program scoring, the
stratified queue, gold JSONL validation, and the FastAPI routes end to
end (against real fixture snapshots, no mocking of the scoring logic
itself)."""
from __future__ import annotations

import functools
import json
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.history.orchestrator import BACKFILL_QUEUE_SCHEMA
from pharma_stats.labelling import migration
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import queue as q
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope


def _study(overall_status, why_stopped=None, enrollment_type="ACTUAL",
           enrollment_count=100, has_results=False, completion_date="2020-01-01",
           last_update="2020-06-01", status_verified="2020-06"):
    return {
        "protocolSection": {
            "statusModule": {
                "overallStatus": overall_status,
                "whyStopped": why_stopped,
                "completionDateStruct": {"date": completion_date, "type": "ACTUAL"},
                "primaryCompletionDateStruct": {"date": completion_date, "type": "ACTUAL"},
                "lastUpdatePostDateStruct": {"date": last_update, "type": "ACTUAL"},
                "statusVerifiedDate": status_verified,
                "startDateStruct": {"date": "2018-01-01"},
            },
            "designModule": {
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": enrollment_count, "type": enrollment_type},
            },
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Acme Oncology"}},
            "conditionsModule": {"conditions": ["Breast Cancer"]},
        },
        "hasResults": has_results,
    }


@pytest.fixture()
def warehouse(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"
    db_path = tmp_path / "warehouse.duckdb"

    def save(nct_id, study):
        snap.save_snapshot("ctgov", nct_id, url="https://x", body=json.dumps(study),
                            fetched_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
                            raw_dir=raw_dir, manifest_db=manifest_db)

    save("NCT00000001", _study("TERMINATED", why_stopped="Business decision."))
    save("NCT00000002", _study(
        "TERMINATED",
        why_stopped="The independent data monitoring committee recommended stopping "
                    "the trial after an interim analysis showed the experimental arm "
                    "did not meet the pre-specified efficacy threshold.",
    ))
    save("NCT00000003", _study("UNKNOWN"))
    save("NCT00000004", _study("RECRUITING", last_update="2020-01-01", status_verified="2019-01"))
    save("NCT00000005", _study("COMPLETED", has_results=False, completion_date="2018-01-01"))

    # provisional_programs.py resolves raw bodies via snap.latest(...) with
    # its default manifest_db; pin that default to this test's manifest so
    # nothing touches the real project's data/manifest.duckdb.
    monkeypatch.setattr(snap, "latest", functools.partial(snap.latest, manifest_db=manifest_db))

    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE asset_candidates (
            candidate_id VARCHAR, proposed_name VARCHAR, synonyms VARCHAR[],
            sponsors_over_time JSON, trial_count INTEGER, nct_ids VARCHAR[],
            first_trial_start_date VARCHAR, last_trial_start_date VARCHAR,
            strategies VARCHAR[], ambiguous BOOLEAN, dev_code_only BOOLEAN,
            discovery_strategy VARCHAR, match_strength VARCHAR, matched_term VARCHAR,
            review_status VARCHAR
        )
    """)
    con.execute(HISTORY_INDEX_SCHEMA)
    con.execute(BACKFILL_QUEUE_SCHEMA)
    rows = [
        ("cand_vague", "VagueMab", [], "[]", 1, ["NCT00000001"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
        ("cand_stated", "StatedMab", [], "[]", 1, ["NCT00000002"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
        ("cand_unknown", "UnknownMab", [], "[]", 1, ["NCT00000003"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
        ("cand_stale", "StaleMab", [], "[]", 1, ["NCT00000004"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
        ("cand_noresults", "NoResultsMab", [], "[]", 1, ["NCT00000005"], None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"),
    ]
    con.executemany(
        "INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    # Each fixture trial is a single-version registration — full,
    # complete history_coverage (never amended is a known fact, not a
    # gap) so existing tests don't need to think about the coverage guard.
    now = datetime(2026, 8, 19, tzinfo=timezone.utc)
    for nct_id in ["NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004", "NCT00000005"]:
        con.execute(
            """
            INSERT INTO history_index (nct_id, version, posted_date, submitted_date, status,
                study_type, changed_modules, schema_hash, indexed_at)
            VALUES (?, 0, ?, ?, 'RECRUITING', 'INTERVENTIONAL', [], 'testhash', ?)
            """,
            [nct_id, date(2019, 1, 1), date(2019, 1, 1), now],
        )
        con.execute(
            """
            INSERT INTO backfill_queue (nct_id, priority_tier, priority_key, status,
                latest_version_indexed, bodies_fetched_through_version, updated_at)
            VALUES (?, 1, 0.0, 'done', 0, -1, ?)
            """,
            [nct_id, now],
        )
    con.close()

    return db_path


def test_trial_history_coverage_classification(warehouse):
    con = duckdb.connect(str(warehouse))
    # NCT00000001: single version, done -> full (never amended is a fact)
    # NCT00000002: single version, but last refresh errored -> partial
    con.execute("UPDATE backfill_queue SET status='error' WHERE nct_id='NCT00000002'")
    # NCT00000003: multi-version, a signal-relevant version never fetched -> partial
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, status, changed_modules, indexed_at)
           VALUES ('NCT00000003', 1, ?, 'TERMINATED', ['Study Status'], ?)""",
        [date(2020, 1, 1), datetime(2020, 1, 1, tzinfo=timezone.utc)],
    )
    # NCT00000004: multi-version, that signal version IS fetched -> full
    con.execute(
        """INSERT INTO history_index (nct_id, version, posted_date, status, changed_modules, indexed_at)
           VALUES ('NCT00000004', 1, ?, 'RECRUITING', ['Study Status'], ?)""",
        [date(2020, 1, 1), datetime(2020, 1, 1, tzinfo=timezone.utc)],
    )
    con.execute("UPDATE backfill_queue SET bodies_fetched_through_version=1 WHERE nct_id='NCT00000004'")
    con.close()

    con = duckdb.connect(str(warehouse), read_only=True)
    assert pp._trial_history_coverage("NCT00000001", con) == "full"
    assert pp._trial_history_coverage("NCT00000002", con) == "partial"
    assert pp._trial_history_coverage("NCT00000003", con) == "partial"
    assert pp._trial_history_coverage("NCT00000004", con) == "full"
    assert pp._trial_history_coverage("NCT_NEVER_INDEXED", con) == "none"
    con.close()

    assert pp._program_history_coverage(["full", "full"]) == "full"
    assert pp._program_history_coverage(["full", "partial"]) == "partial"
    assert pp._program_history_coverage(["none", "none"]) == "none"
    assert pp._program_history_coverage([]) == "none"


def test_scoring_and_archetypes(warehouse):
    con = duckdb.connect(str(warehouse), read_only=True)
    programs = pp.build_all_programs(con, as_of=date(2026, 8, 19))
    con.close()
    by_id = {p["candidate_id"]: p for p in programs}

    assert by_id["cand_vague"]["primary_archetype"] == "registry_terminated_vague_reason"
    assert by_id["cand_stated"]["primary_archetype"] == "registry_terminated_stated_reason"
    assert by_id["cand_unknown"]["primary_archetype"] == "unknown_status"
    assert by_id["cand_noresults"]["primary_archetype"] == "completed_no_results"

    # a still-"RECRUITING" trial silently stale for years should score higher
    # than an openly, concretely terminated one
    assert by_id["cand_stale"]["silence_score"] > by_id["cand_stated"]["silence_score"]
    assert 0 <= by_id["cand_vague"]["silence_score"] <= 100


def _trial(status, **overrides) -> pp.TrialSummary:
    defaults = dict(
        nct_id="NCT_X", status=status, phases=["PHASE2"], why_stopped=None,
        enrollment_count=100, enrollment_type="ACTUAL", conditions=["Breast Cancer"],
        sponsor="Acme Oncology", start_date=date(2018, 1, 1),
        primary_completion_date=date(2020, 1, 1), primary_completion_type="ACTUAL",
        completion_date=date(2020, 1, 1), completion_type="ACTUAL",
        last_update_post_date=date(2020, 6, 1), status_verified_date=date(2020, 6, 1),
        has_results=False, source_snapshot="latest",
    )
    defaults.update(overrides)
    return pp.TrialSummary(**defaults)


def test_compute_silence_score_no_trials_returns_none_not_a_midpoint():
    score, breakdown = pp.compute_silence_score([], as_of=date(2026, 8, 19))
    assert score is None
    assert pp._band_for_score(score) is None


def test_compute_silence_score_completed_trial_is_not_a_dead_zone():
    # the bug: COMPLETED is in neither ACTIVE_LIKE_STATUSES nor
    # TERMINAL_STOP_STATUSES, so staleness/verification_lapse both
    # hard-returned 0.0 regardless of how stale the record actually was.
    fresh = _trial("COMPLETED", last_update_post_date=date(2026, 8, 1))
    stale = _trial("COMPLETED", last_update_post_date=date(2019, 1, 1))
    fresh_score, fresh_bd = pp.compute_silence_score([fresh], as_of=date(2026, 8, 19))
    stale_score, stale_bd = pp.compute_silence_score([stale], as_of=date(2026, 8, 19))
    assert stale_bd["staleness"] > 0
    assert stale_score > fresh_score


def test_compute_silence_score_terminated_with_actual_enrollment_is_not_all_zero():
    # the bug: a TERMINATED trial with a stated reason AND ACTUAL enrolment
    # scored exactly 0 on all four components — no path to score anything.
    t = _trial(
        "TERMINATED",
        why_stopped="The independent data monitoring committee recommended stopping the trial "
                    "after an interim analysis showed the experimental arm did not meet the "
                    "pre-specified efficacy threshold.",
        enrollment_type="ACTUAL", last_update_post_date=date(2019, 1, 1),
    )
    score, breakdown = pp.compute_silence_score([t], as_of=date(2026, 8, 19))
    assert score > 0
    assert breakdown["staleness"] > 0


def test_compute_silence_score_explained_terminal_discounted_vs_unexplained():
    same_date = date(2019, 1, 1)
    explained = _trial(
        "TERMINATED", why_stopped="Interim analysis showed no meaningful difference in overall "
                                  "survival between the two treatment arms of this study.",
        last_update_post_date=same_date,
    )
    unexplained = _trial("TERMINATED", why_stopped=None, last_update_post_date=same_date)
    explained_score, _ = pp.compute_silence_score([explained], as_of=date(2026, 8, 19))
    unexplained_score, _ = pp.compute_silence_score([unexplained], as_of=date(2026, 8, 19))
    assert explained_score < unexplained_score


def test_compute_silence_score_status_ambiguity_aggregates_across_all_trials():
    # a multi-trial asset's second trial (UNKNOWN) must contribute even
    # though it isn't the most-recently-touched one
    clean = _trial("RECRUITING", last_update_post_date=date(2026, 8, 1))
    unknown = _trial("UNKNOWN", nct_id="NCT_Y", last_update_post_date=date(2018, 1, 1))
    score, breakdown = pp.compute_silence_score([clean, unknown], as_of=date(2026, 8, 19))
    assert breakdown["status_ambiguity"] == pp.STATUS_AMBIGUITY_WEIGHT  # max subscore (UNKNOWN) = 1.0


def test_genuine_combo_trial_excluded_from_both_assets_not_assigned_to_either(warehouse):
    con = duckdb.connect(str(warehouse))
    # two independently-verified real assets sharing one trial — the
    # genuine-combination case, which must not be silently attributed to
    # either asset until many-to-many trial<->program linking exists
    con.execute(
        "INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ["cand_a", "DrugA", [], "[]", 2, ["NCT00000005", "NCT00000099"],
         None, None, ["pattern_match"], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"],
    )
    con.execute(
        "INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ["cand_b", "DrugB", [], "[]", 1, ["NCT00000099"],
         None, None, ["seed_expansion"], False, False,
         "seed_expansion", "seed", "DrugB", "unreviewed"],
    )
    con.close()

    con = duckdb.connect(str(warehouse), read_only=True)
    programs = pp.build_all_programs(con, as_of=date(2026, 8, 19))
    con.close()
    by_id = {p["candidate_id"]: p for p in programs}

    a, b = by_id["cand_a"], by_id["cand_b"]
    assert "NCT00000099" not in a["nct_ids"]
    assert "NCT00000099" not in b["nct_ids"]
    assert "NCT00000005" in a["nct_ids"]  # the non-shared trial stays

    a_excluded = {e["nct_id"]: e["shared_with"] for e in a["excluded_shared_trials"]}
    b_excluded = {e["nct_id"]: e["shared_with"] for e in b["excluded_shared_trials"]}
    assert a_excluded == {"NCT00000099": ["DrugB"]}
    assert b_excluded == {"NCT00000099": ["DrugA"]}
    assert not any(t["nct_id"] == "NCT00000099" for t in a["trials"])
    assert not any(e["nct_id"] == "NCT00000099" for e in a["timeline"])


def test_typed_evidence_events_used_in_timeline_when_available(warehouse):
    con = duckdb.connect(str(warehouse))
    from pharma_stats.differ.extract import EVIDENCE_EVENTS_SCHEMA
    con.execute(EVIDENCE_EVENTS_SCHEMA)
    con.execute(
        """
        INSERT INTO evidence_events VALUES
        (1, 'NCT00000004', 1, 2, '2020-03-01', 'enrollment_target_changed', 'enrollment',
         'decreased', '300', '150', 'enrollment target decreased from 300 to 150', '0.1.0', now())
        """
    )
    con.close()

    con = duckdb.connect(str(warehouse), read_only=True)
    programs = pp.build_all_programs(con, as_of=date(2026, 8, 19))
    con.close()
    by_id = {p["candidate_id"]: p for p in programs}

    timeline = by_id["cand_stale"]["timeline"]
    typed = [e for e in timeline if e.get("event_type") == "enrollment_target_changed"]
    assert len(typed) == 1
    assert typed[0]["direction"] == "decreased"
    assert typed[0]["label"] == "enrollment target decreased from 300 to 150"
    # untyped fallback must not also appear for a trial that has typed events
    assert not any(e["label"].startswith("amendment (untyped") for e in timeline if e["nct_id"] == "NCT00000004")


def test_materialize_and_load_roundtrip(warehouse):
    n = pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    assert n == 5
    loaded = pp.load_materialized(warehouse_db=warehouse)
    assert len(loaded) == 5
    assert all(p["provisional"] is True for p in loaded)
    assert all(p["indication_code"] == "UNSPECIFIED" for p in loaded)


def test_stratified_order_interleaves_bands(warehouse):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    programs = pp.load_materialized(warehouse_db=warehouse)
    order = q.build_stratified_order(programs, exclude_ids=set())
    assert set(order) == {p["program_id"] for p in programs}
    # not sorted by score (that's the whole point)
    scores = {p["program_id"]: p["silence_score"] for p in programs}
    assert [scores[pid] for pid in order] != sorted(scores[pid] for pid in order)[::-1] or len(order) < 2


def test_validate_label_payload_requires_dead_confirmed_fields():
    base = {"action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes"}
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            **base, "status": "dead_confirmed", "confidence": "high",
            "history_coverage_at_serve_time": "full",
        })

    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            **base, "status": "dead_confirmed", "confidence": "high",
            "kill_reason": "futility_efficacy", "label_evidence_date": "2024-01-01",
            "history_coverage_at_serve_time": "full",
        })  # missing confirmation date / never_publicly_confirmed

    store.validate_label_payload({
        **base, "status": "dead_confirmed", "confidence": "high",
        "kill_reason": "futility_efficacy", "label_evidence_date": "2024-01-01",
        "never_publicly_confirmed": True, "history_coverage_at_serve_time": "full",
    })  # should not raise

    store.validate_label_payload({"action": "skip"})  # should not raise


def test_validate_label_payload_requires_confirmation_evidence_type_with_confirmation_date():
    base = {
        "action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "status": "dead_confirmed", "confidence": "high",
        "kill_reason": "futility_efficacy", "label_evidence_date": "2024-01-01",
        "history_coverage_at_serve_time": "full", "public_confirmation_date": "2024-03-01",
    }
    with pytest.raises(store.ValidationError):
        store.validate_label_payload(base)  # missing confirmation_evidence_type

    with pytest.raises(store.ValidationError):
        store.validate_label_payload({**base, "confirmation_evidence_type": "not_a_real_type"})

    store.validate_label_payload({**base, "confirmation_evidence_type": "press_release"})  # should not raise
    # the ambiguous case is a legitimate value, not an error
    store.validate_label_payload({**base, "confirmation_evidence_type": "pipeline_page_removal"})

    # never_publicly_confirmed (no confirmation date at all) needs no evidence type
    never_confirmed = {**base, "public_confirmation_date": None, "never_publicly_confirmed": True}
    store.validate_label_payload(never_confirmed)


def test_validate_label_payload_third_party_fields_must_be_paired():
    base = {"action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes", "status": "active",
            "confidence": "high", "history_coverage_at_serve_time": "full"}
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({**base, "third_party_first_noted_date": "2024-01-01"})  # no source
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({**base, "third_party_source": "Citeline"})  # no date

    store.validate_label_payload(base)  # neither given — fine
    store.validate_label_payload({
        **base, "third_party_first_noted_date": "2024-01-01", "third_party_source": "Citeline",
    })  # both given — fine


def test_validate_label_payload_gate1_rejection_is_terminal():
    """Gate 1: is_adc != yes saves and stops right there — no in_scope, no
    status, no coverage requirement, no proceeding further."""
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({"action": "label", "gate_reached": 1})  # missing is_adc

    with pytest.raises(store.ValidationError):
        store.validate_label_payload({"action": "label", "gate_reached": 1, "is_adc": "bogus"})

    with pytest.raises(store.ValidationError):
        # is_adc=yes at gate 1 must proceed to gate 2, not save here
        store.validate_label_payload({"action": "label", "gate_reached": 1, "is_adc": "yes"})

    store.validate_label_payload({"action": "label", "gate_reached": 1, "is_adc": "no"})  # should not raise
    store.validate_label_payload({"action": "label", "gate_reached": 1, "is_adc": "unsure"})  # should not raise
    store.validate_label_payload({
        "action": "label", "gate_reached": 1, "is_adc": "no",
        "in_scope": "no", "scope_reason": "not_an_adc",
    })
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            "action": "label", "gate_reached": 1, "is_adc": "no", "in_scope": "yes",
        })


def test_validate_label_payload_gate2_requires_scope_reason_when_out_of_scope():
    """Gate 2 is only reachable once is_adc=yes; in_scope=no requires a
    reason and is terminal (no status, no coverage requirement)."""
    with pytest.raises(store.ValidationError):
        # gate 2 requires is_adc=yes (gate 1 must have passed)
        store.validate_label_payload({"action": "label", "gate_reached": 2, "is_adc": "no", "in_scope": "no"})

    with pytest.raises(store.ValidationError):
        store.validate_label_payload({"action": "label", "gate_reached": 2, "is_adc": "yes"})  # missing in_scope

    with pytest.raises(store.ValidationError):
        # in_scope=no requires a reason from the enum
        store.validate_label_payload({"action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "no"})

    with pytest.raises(store.ValidationError):
        # in_scope=yes at gate 2 must proceed to gate 3, not save here
        store.validate_label_payload({"action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "yes"})

    # the canonical "haematology ADC" case: a real ADC, still out of scope
    store.validate_label_payload({
        "action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only",
    })  # should not raise

    # not_an_adc is a scope_reason only for is_adc=no (gate 1). Gate 2 is
    # only reachable once is_adc is already yes.
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            "action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "no", "scope_reason": "not_an_adc",
        })


def test_fully_labelled_and_reviewed_program_ids_track_latest_gate():
    r1 = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 2, "is_adc": "yes", "in_scope": "no",
         "scope_reason": "heme_only"},
        session_id="s1", served_stratum={},
    )
    r2 = store.build_record(
        {"action": "label", "program_id": "p2", "gate_reached": 1, "is_adc": "no"},
        session_id="s1", served_stratum={},
    )
    r3 = store.build_record(
        {"action": "label", "program_id": "p3", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
         "status": "active", "confidence": "high"},
        session_id="s1", served_stratum={},
    )
    records = [r1, r2, r3]
    # gate 1/2 rejections are reviewed (never re-served) but NOT fully labelled
    assert store.reviewed_program_ids(records) == {"p1", "p2", "p3"}
    assert store.fully_labelled_program_ids(records) == {"p3"}

    # a later re-review supersedes the earlier gate for that program
    r4 = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
         "status": "active", "confidence": "high"},
        session_id="s1", served_stratum={},
    )
    r4["timestamp"] = "9999-01-01T00:00:00+00:00"  # force it to sort as latest
    records.append(r4)
    assert store.fully_labelled_program_ids(records) == {"p1", "p3"}


def test_validate_label_payload_refuses_incomplete_coverage():
    """The hard guard: even an otherwise-perfectly-valid label must be
    rejected if the program's evidence wasn't fully covered when served —
    this is the save-time half of the coverage guard, defense in depth
    behind /api/next's serve-time refusal. Only gate 3 needs this — gates
    1-2 judge molecule identity/scope, not the silence signal."""
    base = {
        "action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "status": "active", "confidence": "high",
    }
    for bad_coverage in ("partial", "none", None):
        with pytest.raises(store.ValidationError):
            store.validate_label_payload({**base, "history_coverage_at_serve_time": bad_coverage})
    store.validate_label_payload({**base, "history_coverage_at_serve_time": "full"})  # should not raise

    # gate 1/2 rejections are NOT blocked on coverage
    store.validate_label_payload({"action": "label", "gate_reached": 1, "is_adc": "no"})
    store.validate_label_payload({
        "action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "no", "scope_reason": "pre_2012",
    })


def test_append_only_gold_store(tmp_path):
    path = tmp_path / "labels.jsonl"
    r1 = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
         "status": "active", "confidence": "high"},
        session_id="s1", served_stratum={"band": 0, "archetype": "other", "silence_score": 5},
    )
    store.append_record(r1, path=path)
    r2 = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
         "status": "dormant_suspected", "confidence": "medium"},
        session_id="s1", served_stratum={"band": 0, "archetype": "other", "silence_score": 5},
    )
    store.append_record(r2, path=path)

    records = store.load_records(path)
    assert len(records) == 2  # revision is a new line, not an edit
    latest = store.latest_by_program(records)
    assert latest["p1"]["status"] == "dormant_suspected"


def test_repeat_probe_fires_around_ten_percent():
    session = {"order": [f"p{i}" for i in range(50)], "total_served": 0,
               "pending_serve": {}, "served_log": []}
    labelled = {"p_already_labelled"}
    repeats = 0
    for _ in range(30):
        pid, is_repeat = q.pop_next(session, labelled)
        if is_repeat:
            repeats += 1
    assert repeats == 3  # every 10th serve, deterministic


def test_app_end_to_end(warehouse, monkeypatch, tmp_path):
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))

    gold_path = tmp_path / "labels.jsonl"
    session_path = tmp_path / "session.json"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)

    import pharma_stats.labelling.app as appmod
    appmod._state.clear()
    appmod._init_state()

    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)

    r = client.get("/api/next?blind=true")
    body = r.json()
    assert body["done"] is False
    assert "silence_score" not in body["program"]  # blind: hidden pre-judgement
    assert body["program"]["history_coverage"] == "full"  # visible regardless of blind mode
    token = body["serve_token"]

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "status": "dead_confirmed", "confidence": "high",
    })
    assert r.status_code == 422  # missing kill_reason / dates

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "status": "active", "confidence": "low",
        "blind": True, "seconds_spent": 12.5,
    })
    assert r.status_code == 200
    assert r.json()["reveal"]["silence_score"] is not None  # revealed after save

    records = store.load_records(gold_path)
    assert len(records) == 1
    assert records[0]["blind"] is True
    assert records[0]["app_version"]
    assert records[0]["history_coverage_at_serve_time"] == "full"
    assert records[0]["status_revised_after_external_search"] is False
    assert records[0]["discovery_strategy"] == "pattern_match"  # stamped server-side, not client-supplied
    assert records[0]["matched_term"] == "vedotin"

    r = client.get("/api/session")
    stats = r.json()
    assert stats["labelled_count"] == 1


def test_app_gate1_rejection_end_to_end(warehouse, monkeypatch, tmp_path):
    """Gate 1: is_adc != yes is terminal — excluded from the queue, not
    requeued, and does NOT count toward labelled_count (gate 3 only)."""
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))

    gold_path = tmp_path / "labels.jsonl"
    session_path = tmp_path / "session.json"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)

    import pharma_stats.labelling.app as appmod
    appmod._state.clear()
    appmod._init_state()

    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)

    r = client.get("/api/next?blind=true")
    body = r.json()
    program_id = body["program"]["program_id"]
    token = body["serve_token"]

    r = client.post("/api/labels", json={"serve_token": token, "action": "label", "gate_reached": 1})
    assert r.status_code == 422  # is_adc required

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "gate_reached": 1, "is_adc": "no",
        "evidence_note": "generic mAb, no ADC suffix",
    })
    assert r.status_code == 200

    records = store.load_records(gold_path)
    assert records[-1]["gate_reached"] == 1
    assert records[-1]["is_adc"] == "no"
    assert records[-1]["in_scope"] == "no"
    assert records[-1]["scope_reason"] == "not_an_adc"
    assert records[-1]["discovery_strategy"] == "pattern_match"  # stamped from the served candidate

    session = q.load_session(session_path)
    assert program_id not in session["order"]  # excluded, not requeued
    assert store.reviewed_program_ids(records) == {program_id}
    assert store.fully_labelled_program_ids(records) == set()

    stats = client.get("/api/session").json()
    assert stats["labelled_count"] == 0  # a gate-1 reject is not a label
    assert stats["gate1_rejected_count"] == 1
    assert stats["gate1_rejection_pattern_counts"] == [
        {"discovery_strategy": "pattern_match", "match_strength": "suffix", "matched_term": "vedotin", "count": 1},
    ]


def test_app_gate2_rejection_end_to_end(warehouse, monkeypatch, tmp_path):
    """Gate 2: an ADC (is_adc=yes) that's out of scope is also terminal,
    but counts separately from gate 1 and still doesn't reach labelled_count."""
    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))

    gold_path = tmp_path / "labels.jsonl"
    session_path = tmp_path / "session.json"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)

    import pharma_stats.labelling.app as appmod
    appmod._state.clear()
    appmod._init_state()

    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)

    r = client.get("/api/next?blind=true")
    body = r.json()
    program_id = body["program"]["program_id"]
    token = body["serve_token"]

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "gate_reached": 2, "is_adc": "yes", "in_scope": "no",
    })
    assert r.status_code == 422  # scope_reason required

    r = client.post("/api/labels", json={
        "serve_token": token, "action": "label", "gate_reached": 2,
        "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only",
    })
    assert r.status_code == 200

    records = store.load_records(gold_path)
    assert records[-1]["gate_reached"] == 2
    assert records[-1]["in_scope"] == "no"
    assert records[-1]["scope_reason"] == "heme_only"

    session = q.load_session(session_path)
    assert program_id not in session["order"]  # excluded, not requeued
    assert store.reviewed_program_ids(records) == {program_id}
    assert store.fully_labelled_program_ids(records) == set()

    stats = client.get("/api/session").json()
    assert stats["labelled_count"] == 0
    assert stats["gate2_rejected_count"] == 1
    # gate 2 rejections don't pollute the gate-1 pattern-tuning counter
    assert stats["gate1_rejection_pattern_counts"] == []


def test_app_refuses_to_serve_incomplete_coverage_and_requeues_it(warehouse, monkeypatch, tmp_path):
    """The hard guard end to end: a program whose only trial has zero
    history_index coverage must never come back from /api/next, even
    though it's sitting right there in the queue — it should be silently
    skipped (and requeued, not dropped) in favour of a fully-covered one."""
    con = duckdb.connect(str(warehouse))
    con.execute(
        "INSERT INTO asset_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ["cand_nocoverage", "NoCoverageMab", [], "[]", 1, ["NCT_NEVER_INDEXED"],
         None, None, [], False, False,
         "pattern_match", "suffix", "vedotin", "unreviewed"],
    )
    con.close()

    pp.materialize(warehouse_db=warehouse, as_of=date(2026, 8, 19))
    programs = pp.load_materialized(warehouse_db=warehouse)
    by_id = {p["candidate_id"]: p for p in programs}
    assert by_id["cand_nocoverage"]["history_coverage"] == "none"

    gold_path = tmp_path / "labels.jsonl"
    session_path = tmp_path / "session.json"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(q, "SESSION_PATH", session_path)
    monkeypatch.setattr(pp, "WAREHOUSE_DB", warehouse)

    import pharma_stats.labelling.app as appmod
    appmod._state.clear()
    appmod._init_state()

    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)

    seen_ids = set()
    for _ in range(len(programs)):
        r = client.get("/api/next?blind=true")
        body = r.json()
        if body["done"]:
            break
        assert body["program"]["candidate_id"] != "cand_nocoverage"
        assert body["program"]["history_coverage"] == "full"
        seen_ids.add(body["program"]["candidate_id"])

    # the no-coverage program was requeued, not dropped — it's still in
    # the session's fresh queue, just never served
    session = q.load_session(session_path)
    assert "cand_nocoverage" in session["order"]


LYMPHOMA_MESH = [{"id": "D008228", "term": "Lymphoma, Non-Hodgkin"}]
LEUKEMIA_MESH = [{"id": "D015470", "term": "Leukemia, Myeloid, Acute"}]
BREAST_MESH = [{"id": "D001943", "term": "Breast Neoplasms"}]
LUNG_MESH = [{"id": "D002289", "term": "Carcinoma, Non-Small-Cell Lung"}]
GENERIC_MESH = [{"id": "D009369", "term": "Neoplasms"}]


def test_classify_trial_uses_mesh_ids_not_condition_text():
    assert trial_scope.classify_trial(LEUKEMIA_MESH, [], ["Acute Myeloid Leukemia"]) == "heme"
    assert trial_scope.classify_trial(LUNG_MESH, [], ["NSCLC"]) == "solid"
    # no MeSH data at all — never guess from text, even when the text is obvious
    assert trial_scope.classify_trial([], [], ["Acute Myeloid Leukemia"]) == "ambiguous"
    # MeSH present but not yet in our dictionary — don't guess either
    assert trial_scope.classify_trial([{"id": "D999999", "term": "Unmapped Thing"}], [], []) == "ambiguous"
    # heme + solid signal together — ambiguous, not "both"
    assert trial_scope.classify_trial(LEUKEMIA_MESH + BREAST_MESH, [], []) == "ambiguous"
    # only generic root terms — the all-comers/basket signature
    assert trial_scope.classify_trial(GENERIC_MESH, [], []) == "ambiguous"


def test_classify_trial_named_ambiguous_overrides_beat_mesh_signal():
    # a pure lymphoma MeSH hit would otherwise say "heme" — the named
    # override must still win
    assert trial_scope.classify_trial(LYMPHOMA_MESH, [], ["Primary CNS Lymphoma"]) == "ambiguous"
    assert trial_scope.classify_trial(LYMPHOMA_MESH, [], ["Cutaneous T-Cell Lymphoma"]) == "ambiguous"
    assert trial_scope.classify_trial([], [], ["Multiple Myeloma with Bone Involvement"]) == "ambiguous"
    assert trial_scope.classify_trial(GENERIC_MESH, [], ["All Solid Tumors, all comers"]) == "ambiguous"


def test_classify_asset_and_spans_heme_and_solid():
    assert trial_scope.classify_asset(["heme", "heme"]) == "heme_only"
    assert trial_scope.classify_asset(["heme", "solid"]) == "needs_review"
    assert trial_scope.classify_asset(["heme", "ambiguous"]) == "needs_review"
    assert trial_scope.classify_asset([]) == "needs_review"

    assert trial_scope.spans_heme_and_solid(["heme", "solid"]) is True
    assert trial_scope.spans_heme_and_solid(["heme", "heme"]) is False


def test_auto_scope_decision_only_for_heme_only_and_never_sets_is_adc():
    decision = trial_scope.auto_scope_decision("heme_only")
    assert decision == {"in_scope": "no", "scope_reason": "heme_only", "decided_by": "auto"}
    assert "is_adc" not in decision  # deliberately not asserted — see module docstring
    assert trial_scope.auto_scope_decision("needs_review") is None


def test_is_non_oncology_and_non_industry_hints():
    assert trial_scope.is_non_oncology_asset(["non_oncology", "non_oncology"]) is True
    assert trial_scope.is_non_oncology_asset(["non_oncology", "heme"]) is False
    assert trial_scope.is_non_oncology_asset([]) is False

    assert trial_scope.is_non_industry_sponsor([{"sponsor": "Acme", "class": "INDUSTRY"}]) is False
    assert trial_scope.is_non_industry_sponsor([{"sponsor": "NIH", "class": "NIH"}]) is True
    assert trial_scope.is_non_industry_sponsor(
        [{"sponsor": "Acme", "class": "INDUSTRY"}, {"sponsor": "NIH", "class": "NIH"}]
    ) is True


def test_has_mesh_data():
    assert trial_scope.has_mesh_data([{"id": "D008228", "term": "x"}], []) is True
    assert trial_scope.has_mesh_data([], [{"id": "D009369", "term": "x"}]) is True
    assert trial_scope.has_mesh_data([], []) is False


def test_text_hint_category_only_ever_returns_heme_or_none():
    assert trial_scope.text_hint_category(["Acute Myeloid Leukemia"]) == "heme"
    assert trial_scope.text_hint_category(["Diffuse Large B-Cell Lymphoma"]) == "heme"
    assert trial_scope.text_hint_category(["Non-Small Cell Lung Cancer"]) is None
    assert trial_scope.text_hint_category(["All Solid Tumors"]) is None  # not the ALL acronym
    assert trial_scope.text_hint_category([]) is None


def test_mesh_coverage_aggregates_across_programs():
    programs = [
        {"trial_has_mesh": {"NCT1": True, "NCT2": False}},
        {"trial_has_mesh": {"NCT3": True}},
    ]
    result = trial_scope.mesh_coverage(programs)
    assert result == {"covered": 2, "total": 3, "coverage_rate": pytest.approx(2 / 3)}


def test_mesh_coverage_empty_universe():
    assert trial_scope.mesh_coverage([]) == {"covered": 0, "total": 0, "coverage_rate": 0.0}


def test_draw_validation_sample_keeps_prior_reservations_and_tops_up():
    sample = trial_scope.draw_validation_sample(
        candidate_ids=[f"p{i}" for i in range(50)], already_reserved=set(), target_size=5, seed=0,
    )
    assert len(sample) == 5

    # a previously-reserved id must survive even if it's since dropped out
    # of candidate_ids (e.g. it got reviewed and is no longer "qualifying")
    kept = trial_scope.draw_validation_sample(
        candidate_ids=["p1", "p2"], already_reserved={"already_reserved_but_gone"}, target_size=3, seed=0,
    )
    assert "already_reserved_but_gone" in kept
    assert len(kept) == 3


def test_validation_agreement_only_counts_scope_decided_records():
    sample = [
        {"program_id": "p1", "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"},
        {"program_id": "p2", "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"},
        {"program_id": "p3", "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"},
    ]
    records = [
        # agrees: human independently reached the same scope call
        store.build_record(
            {"action": "label", "program_id": "p1", "gate_reached": 2,
             "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only"},
            session_id="s1", served_stratum={},
        ),
        # disagrees: human decided it WAS in scope
        store.build_record(
            {"action": "label", "program_id": "p2", "gate_reached": 3,
             "is_adc": "yes", "in_scope": "yes", "status": "active", "confidence": "high"},
            session_id="s1", served_stratum={},
        ),
        # p3 never reached gate 2 — not comparable yet, must not count as a miss
        store.build_record(
            {"action": "label", "program_id": "p3", "gate_reached": 1, "is_adc": "no"},
            session_id="s1", served_stratum={},
        ),
    ]
    result = trial_scope.validation_agreement(sample, records)
    assert result == {"compared": 2, "agreements": 1, "agreement_rate": 0.5}


def test_validate_label_payload_auto_decided_scope_rejection():
    """decided_by=auto is the only path that can save a scope decision
    without is_adc — see trial_scope.auto_scope_decision."""
    with pytest.raises(store.ValidationError):
        # every decided_by=auto record must say which triage layer decided it
        store.validate_label_payload({
            "action": "label", "gate_reached": 1, "decided_by": "auto", "is_adc": "no",
        })
    with pytest.raises(store.ValidationError):
        # auto records must be in_scope=no
        store.validate_label_payload({
            "action": "label", "gate_reached": 2, "decided_by": "auto", "in_scope": "yes",
            "triage_layer": 1,
        })
    with pytest.raises(store.ValidationError):
        store.validate_label_payload({
            "action": "label", "gate_reached": 2, "decided_by": "auto", "in_scope": "no",
            "triage_layer": 1,
        })  # missing scope_reason
    with pytest.raises(store.ValidationError):
        # decided_by=auto is never valid at gate 3
        store.validate_label_payload({
            "action": "label", "gate_reached": 3, "decided_by": "auto", "triage_layer": 1,
        })

    store.validate_label_payload({
        "action": "label", "gate_reached": 2, "decided_by": "auto", "triage_layer": 1,
        "in_scope": "no", "scope_reason": "heme_only",
    })  # should not raise, is_adc absent entirely

    record = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 2, "decided_by": "auto",
         "triage_layer": 1, "in_scope": "no", "scope_reason": "heme_only"},
        session_id="s1", served_stratum={},
    )
    assert record["decided_by"] == "auto"
    assert record["is_adc"] is None


def test_validate_label_payload_auto_decided_is_adc_rejection():
    """decided_by=auto at gate 1: only a confident is_adc=no verdict is a
    valid terminal record — see pharma_stats.triage."""
    with pytest.raises(store.ValidationError):
        # is_adc=yes is never terminal at gate 1, auto or human
        store.validate_label_payload({
            "action": "label", "gate_reached": 1, "decided_by": "auto",
            "triage_layer": 1, "is_adc": "yes",
        })
    with pytest.raises(store.ValidationError):
        # "unsure" is never committed — it escalates to the next layer or
        # the human queue, never saved as a decision
        store.validate_label_payload({
            "action": "label", "gate_reached": 1, "decided_by": "auto",
            "triage_layer": 2, "is_adc": "unsure",
        })

    store.validate_label_payload({
        "action": "label", "gate_reached": 1, "decided_by": "auto",
        "triage_layer": 1, "is_adc": "no",
    })  # should not raise

    record = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 1, "decided_by": "auto",
         "triage_layer": 1, "triage_rule": "layer1_denylist", "is_adc": "no"},
        session_id="s1", served_stratum={},
    )
    assert record["decided_by"] == "auto"
    assert record["triage_layer"] == 1
    assert record["triage_rule"] == "layer1_denylist"


def test_has_adc_naming_suffix():
    assert migration.has_adc_naming_suffix("Trastuzumab deruxtecan") == "deruxtecan"
    assert migration.has_adc_naming_suffix("Enfortumab vedotin") == "vedotin"
    assert migration.has_adc_naming_suffix("Pembrolizumab") is None


def test_build_record_fills_in_scope_on_is_adc_no():
    record = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no"},
        session_id="s1", served_stratum={},
    )
    assert record["in_scope"] == "no"
    assert record["scope_reason"] == "not_an_adc"


def test_scope_backfill_skips_already_consistent_rows_and_catches_inconsistent():
    consistent = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
         "in_scope": "no", "scope_reason": "not_an_adc", "proposed_name": "Pembrolizumab"},
        session_id="s1", served_stratum={},
    )
    missing = {
        "action": "label", "program_id": "p2", "gate_reached": 1, "is_adc": "no",
        "in_scope": None, "scope_reason": None, "proposed_name": "AZD4045",
        "event_id": "old-p2", "timestamp": "2026-08-27T00:00:00+00:00",
        "decided_by": "human", "blind": True,
    }
    inconsistent = {
        "action": "label", "program_id": "p3", "gate_reached": 1, "is_adc": "no",
        "in_scope": "yes", "scope_reason": None, "proposed_name": "ETBX-051",
        "event_id": "old-p3", "timestamp": "2026-08-30T00:00:00+00:00",
        "decided_by": "human", "blind": True,
    }
    need = migration.rows_needing_scope_backfill([consistent, missing, inconsistent])
    assert {r["program_id"] for r in need} == {"p2", "p3"}

    filled = migration.build_scope_backfill_record(inconsistent)
    store.validate_label_payload({
        "action": "label", "gate_reached": 1, "is_adc": filled["is_adc"],
        "in_scope": filled["in_scope"], "scope_reason": filled["scope_reason"],
    })
    assert filled["is_adc"] == "no"
    assert filled["in_scope"] == "no"
    assert filled["scope_reason"] == "not_an_adc"
    assert filled["session_id"] == migration.BACKFILL_SESSION_ID
    assert filled["program_id"] == "p3"


def test_not_an_adc_suffix_candidates_uses_latest_and_synonyms():
    no_suffix = store.build_record(
        {"action": "label", "program_id": "p1", "gate_reached": 1, "is_adc": "no",
         "proposed_name": "Pembrolizumab"},
        session_id="s1", served_stratum={},
    )
    suffix_on_name = store.build_record(
        {"action": "label", "program_id": "p2", "gate_reached": 1, "is_adc": "no",
         "proposed_name": "Sacituzumab govitecan"},
        session_id="s1", served_stratum={},
    )
    suffix_on_synonym_only = store.build_record(
        {"action": "label", "program_id": "p3", "gate_reached": 1, "is_adc": "no",
         "proposed_name": "SGN-35"},
        session_id="s1", served_stratum={},
    )
    later_flip = store.build_record(
        {"action": "label", "program_id": "p2", "gate_reached": 2, "is_adc": "yes",
         "in_scope": "no", "scope_reason": "heme_only", "proposed_name": "Sacituzumab govitecan"},
        session_id="s1", served_stratum={},
    )
    later_flip["timestamp"] = "9999-01-01T00:00:00+00:00"

    result = migration.not_an_adc_suffix_candidates(
        [no_suffix, suffix_on_name, suffix_on_synonym_only, later_flip],
        {"p3": {"synonyms": ["Brentuximab vedotin"]}},
    )
    assert {r["program_id"] for r in result} == {"p3"}
    assert result[0]["matched_suffix"] == "vedotin"
    assert result[0]["matched_on"] == "Brentuximab vedotin"


def test_flag_invalid_migration_candidates_finds_pre_schema_rows_with_adc_suffix():
    old_row_with_suffix = store.build_record(
        {"action": "flag_invalid", "program_id": "p1", "proposed_name": "Sacituzumab govitecan"},
        session_id="s1", served_stratum={},
    )
    old_row_without_suffix = store.build_record(
        {"action": "flag_invalid", "program_id": "p2", "proposed_name": "Some Chemo Backbone"},
        session_id="s1", served_stratum={},
    )
    already_migrated_row = store.build_record(
        {"action": "label", "program_id": "p3", "proposed_name": "Trastuzumab deruxtecan",
         "gate_reached": 2, "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only"},
        session_id="s1", served_stratum={},
    )
    records = [old_row_with_suffix, old_row_without_suffix, already_migrated_row]

    result = migration.flag_invalid_migration_candidates(records)
    assert {r["program_id"] for r in result} == {"p1"}
    assert result[0]["matched_suffix"] == "govitecan"
