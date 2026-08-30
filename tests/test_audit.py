"""Tests for the audit harness: a check that can't fail is decoration,
so these exercise both the passing and the failing path of each real
(non-stub) stage."""
from __future__ import annotations

import functools
import json
from datetime import date, datetime, timezone

import duckdb
import pytest

from pharma_stats import snapshot as snap
from pharma_stats.audit import __main__ as audit_main
from pharma_stats.audit import differ as differ_stage
from pharma_stats.audit import gold_set, label_sufficiency, provenance, report, universe
from pharma_stats.audit.types import Check
from pharma_stats.differ.extract import EVIDENCE_EVENTS_SCHEMA
from pharma_stats.history.index import HISTORY_INDEX_SCHEMA
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts


def test_report_renders_and_counts_levels():
    checks = [
        Check("s1", "a", "FAIL", "0", "1"),
        Check("s1", "b", "WARN", "0", "1"),
        Check("s2", "c", "INFO", "-", "-"),
        Check("s2", "d", "PASS", "0", "0"),
    ]
    text = report.render(checks, stages_run=["s1", "s2"])
    assert "1 FAIL / 1 WARN / 1 INFO / 1 PASS" in text
    assert "## s1" in text and "## s2" in text
    assert "FAIL — stop and look at these first" in text


def test_report_write_creates_timestamped_file(tmp_path):
    checks = [Check("s1", "a", "PASS", "0", "0")]
    path = report.write(checks, stages_run=["s1"], out_dir=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path


@pytest.fixture()
def raw_and_manifest(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    manifest_db = tmp_path / "manifest.duckdb"

    def save(source, id_, body, fetched_at):
        return snap.save_snapshot(
            source, id_, url="https://x", body=body, fetched_at=fetched_at,
            raw_dir=raw_dir, manifest_db=manifest_db,
        )

    save("ctgov", "NCT111", json.dumps({"a": 1}), datetime(2026, 1, 1, tzinfo=timezone.utc))
    save("ctgov", "NCT111", json.dumps({"a": 2}), datetime(2026, 2, 1, tzinfo=timezone.utc))
    save("ctgov", "NCT222", json.dumps({"b": 1}), datetime(2026, 1, 15, tzinfo=timezone.utc))

    monkeypatch.setattr(provenance, "RAW_DIR", raw_dir)
    monkeypatch.setattr(provenance, "MANIFEST_DB", manifest_db)
    monkeypatch.setattr(snap, "get_as_of", functools.partial(snap.get_as_of, manifest_db=manifest_db))

    return raw_dir, manifest_db


def test_provenance_passes_on_clean_store(raw_and_manifest):
    checks = provenance.run()
    assert not any(c.level == "FAIL" for c in checks)
    probe = next(c for c in checks if "get_as_of returns" in c.name)
    assert probe.level == "PASS"
    assert probe.expected.startswith("3 ")  # 3 probes for the one (source,id) pair with 2 dates


def test_provenance_detects_manifest_sha_tamper(raw_and_manifest):
    raw_dir, manifest_db = raw_and_manifest
    con = duckdb.connect(str(manifest_db))
    con.execute("UPDATE snapshots SET sha256 = 'deadbeef' WHERE id = 'NCT222'")
    con.close()

    checks = provenance.run()
    mismatch = next(c for c in checks if "manifest sha256 matches" in c.name)
    assert mismatch.level == "FAIL"
    assert "1 / " in mismatch.actual


def test_provenance_detects_get_as_of_regression(raw_and_manifest, monkeypatch):
    # simulate a regressed get_as_of that always returns the latest snapshot,
    # ignoring as_of entirely — the exact bug this probe exists to catch
    _, manifest_db = raw_and_manifest
    original_get_as_of = snap.get_as_of

    def broken_latest_always(source, id_, as_of, *, manifest_db=manifest_db):
        return original_get_as_of(source, id_, date.today(), manifest_db=manifest_db)

    monkeypatch.setattr(snap, "get_as_of", broken_latest_always)
    checks = provenance.run()
    probe = next(c for c in checks if "get_as_of returns" in c.name)
    assert probe.level == "FAIL"
    assert "future" in probe.detail  # our own commentary on why this matters


def test_gold_set_flags_missing_dead_confirmed_fields(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    # hand-crafted record that bypasses the app's own validator entirely —
    # exactly the scenario this independent re-check exists for
    bad = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "program_id": "p1", "status": "dead_confirmed", "kill_reason": None,
        "confidence": "high", "evidence_note": "", "label_evidence_date": None,
        "public_confirmation_date": None, "never_publicly_confirmed": False,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
    }
    store.append_record(bad, path=gold_path)

    checks = gold_set.run()
    invariant_checks = [c for c in checks if "dead_confirmed record has" in c.name]
    assert len(invariant_checks) == 3
    assert all(c.level == "FAIL" for c in invariant_checks)


def test_gold_set_passes_clean_dead_confirmed_record(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    good = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "program_id": "p1", "status": "dead_confirmed", "kill_reason": "futility_efficacy",
        "confidence": "high", "evidence_note": "", "label_evidence_date": "2024-01-01",
        "public_confirmation_date": None, "never_publicly_confirmed": True,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
    }
    store.append_record(good, path=gold_path)

    checks = gold_set.run()
    invariant_checks = [c for c in checks if "dead_confirmed record has" in c.name]
    assert all(c.level == "PASS" for c in invariant_checks)


def test_gold_set_catches_a_served_program_with_incomplete_coverage(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    # this should be structurally impossible (both /api/next and
    # validate_label_payload refuse it) — this check exists to catch it
    # anyway, straight from the append-only record
    bad = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "program_id": "p1", "status": "active", "kill_reason": None,
        "confidence": "high", "evidence_note": "", "label_evidence_date": None,
        "public_confirmation_date": None, "never_publicly_confirmed": False,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
        "history_coverage_at_serve_time": "partial",
    }
    store.append_record(bad, path=gold_path)

    checks = gold_set.run()
    invariant = next(c for c in checks if "less-than-full history_coverage" in c.name)
    assert invariant.level == "FAIL"
    assert "p1" in invariant.detail


def test_gold_set_passes_when_all_served_programs_had_full_coverage(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    good = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
        "program_id": "p1", "status": "active", "kill_reason": None,
        "confidence": "high", "evidence_note": "", "label_evidence_date": None,
        "public_confirmation_date": None, "never_publicly_confirmed": False,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
        "history_coverage_at_serve_time": "full",
    }
    store.append_record(good, path=gold_path)

    checks = gold_set.run()
    invariant = next(c for c in checks if "less-than-full history_coverage" in c.name)
    assert invariant.level == "PASS"


def test_gold_set_reports_status_revision_rate(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())

    for i, revised in enumerate([True, False, False, False]):
        rec = {
            "event_id": f"e{i}", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
            "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
            "program_id": f"p{i}", "status": "active", "kill_reason": None,
            "confidence": "high", "evidence_note": "", "label_evidence_date": None,
            "public_confirmation_date": None, "never_publicly_confirmed": False,
            "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
            "history_coverage_at_serve_time": "full",
            "status_revised_after_external_search": revised,
        }
        store.append_record(rec, path=gold_path)

    checks = gold_set.run()
    revision = next(c for c in checks if "revised after external search" in c.name)
    assert revision.actual == "25% (1 / 4)"


def test_label_sufficiency_reports_insufficient_data(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LABELS_PATH", tmp_path / "labels.jsonl")
    checks = label_sufficiency.run()
    assert checks[0].level == "INFO"
    assert "0 usable" in checks[0].actual


def test_label_sufficiency_ignores_gate1_and_gate2_rejections(tmp_path, monkeypatch):
    """A pile of Gate 1/2 triage rejections must never look like progress
    toward the bootstrap — they were never programs, let alone dead_confirmed
    labels with dates."""
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)

    for i in range(15):
        rec = {
            "event_id": f"reject{i}", "timestamp": f"2026-01-{i+1:02d}T00:00:00+00:00", "action": "label",
            "gate_reached": 1, "is_adc": "no",
            "program_id": f"reject{i}", "status": None, "kill_reason": None,
            "confidence": None, "evidence_note": "", "label_evidence_date": None,
            "public_confirmation_date": None, "never_publicly_confirmed": False,
            "blind": True, "is_repeat_probe": False, "seconds_spent": 5,
        }
        store.append_record(rec, path=gold_path)

    checks = label_sufficiency.run()
    assert checks[0].level == "INFO"
    assert "0 usable" in checks[0].actual  # 15 gate-1 rejections, 0 real labels


def test_label_sufficiency_bootstraps_with_enough_data(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)

    for i in range(15):
        rec = {
            "event_id": f"e{i}", "timestamp": f"2026-01-{i+1:02d}T00:00:00+00:00", "action": "label",
            "gate_reached": 3, "is_adc": "yes", "in_scope": "yes",
            "program_id": f"p{i}", "status": "dead_confirmed", "kill_reason": "futility_efficacy",
            "confidence": "high", "evidence_note": "", "label_evidence_date": "2024-01-01",
            "public_confirmation_date": "2024-06-01", "never_publicly_confirmed": False,
            "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
        }
        store.append_record(rec, path=gold_path)

    checks = label_sufficiency.run()
    assert len(checks) == 1
    assert checks[0].level in ("INFO", "PASS")
    assert "N=15" in checks[0].actual


def test_cli_exits_nonzero_on_fail(tmp_path, monkeypatch, capsys):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    monkeypatch.setattr(gold_set, "pp", type("_", (), {"load_materialized": staticmethod(lambda: [])})())
    bad = {
        "event_id": "e1", "timestamp": "2026-01-01T00:00:00+00:00", "action": "label",
        "program_id": "p1", "status": "dead_confirmed", "kill_reason": None,
        "confidence": "high", "evidence_note": "", "label_evidence_date": None,
        "public_confirmation_date": None, "never_publicly_confirmed": False,
        "blind": True, "is_repeat_probe": False, "seconds_spent": 30,
    }
    store.append_record(bad, path=gold_path)

    from pharma_stats.audit import report as report_mod
    monkeypatch.setattr(report_mod, "AUDIT_DIR", tmp_path / "audit_out")

    exit_code = audit_main.main(["--stage", "gold_set"])
    assert exit_code == 1
    assert (tmp_path / "audit_out").exists()


def test_differ_stage_reports_not_materialized(tmp_path, monkeypatch):
    db_path = tmp_path / "warehouse.duckdb"
    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(differ_stage, "WAREHOUSE_DB", db_path)
    checks = differ_stage.run()
    assert len(checks) == 1
    assert checks[0].level == "INFO"
    assert "not materialized" in checks[0].actual


def test_differ_stage_catches_boundary_invariant_violation(tmp_path, monkeypatch):
    db_path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(EVIDENCE_EVENTS_SCHEMA)
    con.execute(HISTORY_INDEX_SCHEMA)
    # a hand-corrupted row: enrollment_target_changed with no direction —
    # exactly what a regressed differ that forgot the ESTIMATED/ACTUAL
    # guard would write
    con.execute(
        """
        INSERT INTO evidence_events VALUES
        (1, 'NCT1', 1, 2, '2024-01-01', 'enrollment_target_changed', 'enrollment',
         NULL, '300', '1', 'bad row', '0.1.0', now())
        """
    )
    con.close()
    monkeypatch.setattr(differ_stage, "WAREHOUSE_DB", db_path)
    checks = differ_stage.run()
    invariant = next(c for c in checks if "ESTIMATED/ACTUAL boundary" in c.name)
    assert invariant.level == "FAIL"


def test_differ_stage_passes_on_clean_events(tmp_path, monkeypatch):
    db_path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(EVIDENCE_EVENTS_SCHEMA)
    con.execute(HISTORY_INDEX_SCHEMA)
    con.execute(
        """
        INSERT INTO evidence_events VALUES
        (1, 'NCT1', 1, 2, '2024-01-01', 'enrollment_target_changed', 'enrollment',
         'decreased', '300', '150', 'ok row', '0.1.0', now()),
        (2, 'NCT1', 2, 3, '2024-02-01', 'status_changed', 'overallStatus',
         NULL, 'RECRUITING', 'TERMINATED', 'ok row', '0.1.0', now())
        """
    )
    con.close()
    monkeypatch.setattr(differ_stage, "WAREHOUSE_DB", db_path)
    checks = differ_stage.run()
    invariant = next(c for c in checks if "ESTIMATED/ACTUAL boundary" in c.name)
    assert invariant.level == "PASS"


def test_recall_probe_matches_across_a_trailing_parenthetical(tmp_path, monkeypatch):
    """Regression for the 2026-08-26 false-"missing" bug: a candidate
    whose proposed_name carries a trailing brand parenthetical (as raw
    CT.gov intervention strings often do) must still count as a match for
    the bare known-drug name — exact-set equality alone wrongly reported
    genuinely-discovered assets as missing."""
    known_path = tmp_path / "known_adcs.txt"
    known_path.write_text("enapotamab vedotin | AXL-107-MMAE\ntotally-novel-compound\n")
    monkeypatch.setattr(universe, "KNOWN_ADCS_PATH", known_path)

    candidates = [
        {"candidate_id": "a", "proposed_name": "Enapotamab vedotin (HuMax-AXL-ADC)", "synonyms": []},
    ]
    checks = universe._recall_probe(candidates)
    recall = next(c for c in checks if c.name.startswith("recall probe:"))
    assert recall.actual == "1 / 2 found"
    assert "totally-novel-compound" in recall.detail
    assert "enapotamab" not in recall.detail.lower()


def test_recall_probe_reports_all_found(tmp_path, monkeypatch):
    known_path = tmp_path / "known_adcs.txt"
    known_path.write_text("drug a\ndrug b\n")
    monkeypatch.setattr(universe, "KNOWN_ADCS_PATH", known_path)
    candidates = [
        {"candidate_id": "a", "proposed_name": "Drug A", "synonyms": []},
        {"candidate_id": "b", "proposed_name": "Drug B", "synonyms": []},
    ]
    checks = universe._recall_probe(candidates)
    recall = next(c for c in checks if c.name.startswith("recall probe:"))
    assert recall.level == "PASS"
    assert recall.actual == "2 / 2 found"


def test_clustering_check_splits_genuine_combo_from_noise():
    candidates = [
        {"candidate_id": "a", "proposed_name": "DrugA", "nct_ids": ["NCT1"],
         "strategies": ["pattern_match"], "ambiguous": False, "review_status": "reviewed"},
        {"candidate_id": "b", "proposed_name": "DrugB", "nct_ids": ["NCT1"],
         "strategies": ["seed_expansion"], "ambiguous": False, "review_status": "reviewed"},
        {"candidate_id": "c", "proposed_name": "DrugC", "nct_ids": ["NCT2"],
         "strategies": ["pattern_match"], "ambiguous": False, "review_status": "reviewed"},
        {"candidate_id": "d", "proposed_name": "DrugD", "nct_ids": ["NCT2"],
         "strategies": ["sponsor_expansion"], "ambiguous": False, "review_status": "reviewed"},
    ]
    checks = universe._clustering_and_backlog(candidates)
    genuine = next(c for c in checks if "independently-verified" in c.name)
    noise = next(c for c in checks if "at least one claimant is unverified" in c.name)
    assert genuine.actual == "1 trials"
    assert "NCT1" in genuine.detail
    assert noise.level == "WARN"
    assert "NCT2" in noise.detail


def test_unreviewed_candidates_fails_the_gate():
    candidates = [
        {"candidate_id": "a", "proposed_name": "DrugA", "nct_ids": ["NCT1"],
         "strategies": [], "ambiguous": False, "review_status": "unreviewed"},
    ]
    checks = universe._clustering_and_backlog(candidates)
    gate = next(c for c in checks if "human review queue" in c.name)
    assert gate.level == "FAIL"


def test_heme_solid_span_report_counts_mesh_classified_assets(monkeypatch):
    programs = [
        {"program_id": "p1", "proposed_name": "MixedMab", "spans_heme_and_solid": True,
         "trial_scope": {"NCT1": "heme", "NCT2": "solid"}},
        {"program_id": "p2", "proposed_name": "HemeOnlyMab", "spans_heme_and_solid": False,
         "trial_scope": {"NCT3": "heme"}},
    ]
    monkeypatch.setattr(pp, "load_materialized", lambda: programs)
    checks = universe._heme_solid_span_report()
    check = checks[0]
    assert check.actual == "1 / 2 assets"
    assert "MixedMab" in check.detail
    assert "HemeOnlyMab" not in check.detail


def test_heme_solid_span_report_handles_unmaterialized(monkeypatch):
    monkeypatch.setattr(pp, "load_materialized", lambda: [])
    checks = universe._heme_solid_span_report()
    assert checks[0].level == "INFO"
    assert "not materialized" in checks[0].actual


def test_heme_only_auto_exclusion_agreement_no_sample_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "load_validation_sample", lambda: [])
    checks = universe._heme_only_auto_exclusion_agreement()
    assert checks[0].level == "INFO"
    assert "no validation sample" in checks[0].actual


def test_heme_only_auto_exclusion_agreement_passes_above_threshold(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    sample = [
        {"program_id": f"p{i}", "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"}
        for i in range(20)
    ]
    monkeypatch.setattr(ts, "load_validation_sample", lambda: sample)

    # 19/20 agree — 95%, right at the threshold
    for i in range(19):
        store.append_record(store.build_record(
            {"action": "label", "program_id": f"p{i}", "gate_reached": 2,
             "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only"},
            session_id="s1", served_stratum={},
        ), path=gold_path)
    store.append_record(store.build_record(
        {"action": "label", "program_id": "p19", "gate_reached": 3,
         "is_adc": "yes", "in_scope": "yes", "status": "active", "confidence": "high"},
        session_id="s1", served_stratum={},
    ), path=gold_path)

    checks = universe._heme_only_auto_exclusion_agreement()
    check = checks[0]
    assert check.level == "PASS"
    assert "19 / 20" in check.actual


def test_heme_only_auto_exclusion_agreement_fails_below_threshold(tmp_path, monkeypatch):
    gold_path = tmp_path / "labels.jsonl"
    monkeypatch.setattr(store, "LABELS_PATH", gold_path)
    sample = [
        {"program_id": f"p{i}", "predicted_in_scope": "no", "predicted_scope_reason": "heme_only"}
        for i in range(20)
    ]
    monkeypatch.setattr(ts, "load_validation_sample", lambda: sample)

    for i in range(15):
        store.append_record(store.build_record(
            {"action": "label", "program_id": f"p{i}", "gate_reached": 2,
             "is_adc": "yes", "in_scope": "no", "scope_reason": "heme_only"},
            session_id="s1", served_stratum={},
        ), path=gold_path)
    for i in range(15, 20):
        store.append_record(store.build_record(
            {"action": "label", "program_id": f"p{i}", "gate_reached": 3,
             "is_adc": "yes", "in_scope": "yes", "status": "active", "confidence": "high"},
            session_id="s1", served_stratum={},
        ), path=gold_path)

    checks = universe._heme_only_auto_exclusion_agreement()
    check = checks[0]
    assert check.level == "FAIL"
    assert "15 / 20" in check.actual


def test_mesh_coverage_gate_warns_below_threshold(monkeypatch):
    programs = [
        {"trial_has_mesh": {"NCT1": True, "NCT2": False}},
        {"trial_has_mesh": {"NCT3": False, "NCT4": False}},
    ]
    monkeypatch.setattr(pp, "load_materialized", lambda: programs)
    checks = universe._mesh_coverage_gate()
    check = checks[0]
    assert check.level == "WARN"
    assert "1 / 4" in check.actual


def test_mesh_coverage_gate_passes_above_threshold(monkeypatch):
    programs = [{"trial_has_mesh": {f"NCT{i}": True for i in range(9)} | {"NCT9": False}}]
    monkeypatch.setattr(pp, "load_materialized", lambda: programs)
    checks = universe._mesh_coverage_gate()
    assert checks[0].level == "PASS"
    assert "9 / 10" in checks[0].actual


def test_mesh_coverage_gate_handles_unmaterialized(monkeypatch):
    monkeypatch.setattr(pp, "load_materialized", lambda: [])
    checks = universe._mesh_coverage_gate()
    assert checks[0].level == "INFO"
    assert "not materialized" in checks[0].actual


def test_current_state_read_boundary_fails_on_unwhitelisted_call(monkeypatch, tmp_path):
    bad_module = tmp_path / "bad_module.py"
    bad_module.write_text(
        "def _condition_browse_data():\n"
        "    return snap.latest('ctgov', 'x')\n"
        "\n"
        "def compute_silence_score():\n"
        "    return snap.latest('ctgov', 'y')\n"
    )
    monkeypatch.setattr(universe.inspect, "getsourcefile", lambda mod: str(bad_module))
    monkeypatch.setattr(pp, "CURRENT_STATE_READ_WHITELIST", {"_condition_browse_data"})

    checks = universe._current_state_read_boundary()
    check = checks[0]
    assert check.level == "FAIL"
    assert "compute_silence_score" in check.detail
    assert "_condition_browse_data" not in check.detail


def test_current_state_read_boundary_passes_when_only_whitelisted_calls(monkeypatch, tmp_path):
    clean_module = tmp_path / "clean_module.py"
    clean_module.write_text(
        "def _condition_browse_data():\n"
        "    return snap.latest('ctgov', 'x')\n"
        "\n"
        "def compute_silence_score():\n"
        "    return 42\n"
    )
    monkeypatch.setattr("inspect.getsourcefile", lambda mod: str(clean_module))
    monkeypatch.setattr(pp, "CURRENT_STATE_READ_WHITELIST", {"_condition_browse_data"})

    checks = universe._current_state_read_boundary()
    assert checks[0].level == "PASS"
