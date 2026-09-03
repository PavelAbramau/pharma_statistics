"""Tests for labelling/queue_order.py — disposition-ordered residue queue
for the full coverage pass (likely_reject -> ambiguous -> confirmed_adc),
distinct from queue.build_stratified_order's band/archetype interleave."""
from __future__ import annotations

from pharma_stats.labelling import queue_order as qo
from pharma_stats.labelling import triage_serve
from pharma_stats.triage import evidence as tev


def _program(pid, name="Drug X"):
    return {"program_id": pid, "proposed_name": name}


def test_disposition_bucket_skip_returns_none(monkeypatch):
    plan = triage_serve.ServePlan(start_gate=1, skip=True, reopened=False, context=None)
    bucket, snippet = qo.disposition_bucket(_program("p1"), plan, con=None)
    assert bucket is None
    assert snippet is None


def test_disposition_bucket_gate3_is_confirmed_adc():
    plan = triage_serve.ServePlan(start_gate=3, skip=False, reopened=False,
                                   context={"is_adc": "yes", "in_scope": "yes"})
    bucket, snippet = qo.disposition_bucket(_program("p1"), plan, con=None)
    assert bucket == "confirmed_adc"
    assert snippet is None


def test_disposition_bucket_genuine_adc_pending_scope_is_ambiguous():
    plan = triage_serve.ServePlan(start_gate=2, skip=False, reopened=False, context={"is_adc": "yes"})
    bucket, snippet = qo.disposition_bucket(_program("p1"), plan, con=None)
    assert bucket == "ambiguous"
    assert snippet is None


def test_disposition_bucket_small_molecule_signal_is_likely_reject(monkeypatch):
    monkeypatch.setattr(
        tev, "build_layer2_evidence",
        lambda program, con: {"text_snippets": ["administered orally as a tablet"]},
    )
    plan = triage_serve.ServePlan(start_gate=1, skip=False, reopened=False, context=None)
    bucket, snippet = qo.disposition_bucket(_program("p1"), plan, con=object())
    assert bucket == "likely_reject"
    assert "orally" in snippet


def test_disposition_bucket_no_signal_is_ambiguous(monkeypatch):
    monkeypatch.setattr(tev, "build_layer2_evidence", lambda program, con: {"text_snippets": ["a study of X in patients"]})
    plan = triage_serve.ServePlan(start_gate=1, skip=False, reopened=False, context=None)
    bucket, snippet = qo.disposition_bucket(_program("p1"), plan, con=object())
    assert bucket == "ambiguous"
    assert snippet is None


def test_build_disposition_order_groups_and_orders_buckets(monkeypatch):
    programs = [_program("reject1"), _program("ambig1"), _program("confirmed1")]

    def fake_serve_plan(program, **kwargs):
        pid = program["program_id"]
        if pid == "confirmed1":
            return triage_serve.ServePlan(start_gate=3, skip=False, reopened=False,
                                           context={"is_adc": "yes", "in_scope": "yes"})
        if pid == "ambig1":
            return triage_serve.ServePlan(start_gate=2, skip=False, reopened=False, context={"is_adc": "yes"})
        return triage_serve.ServePlan(start_gate=1, skip=False, reopened=False, context=None)

    monkeypatch.setattr(triage_serve, "serve_plan", fake_serve_plan)
    monkeypatch.setattr(tev, "build_layer2_evidence", lambda program, con: {"text_snippets": ["oral tablet"]})

    ordered, counts = qo.build_disposition_order(
        programs, ["reject1", "ambig1", "confirmed1"],
        gold_records=[], heme_auto_ok=True, model_gate_ok=True,
        heme_holdout_ids=set(), triage_holdout_ids=set(), staged_by_program={}, con=object(),
    )
    assert ordered == ["reject1", "ambig1", "confirmed1"]
    assert counts == {"likely_reject": 1, "ambiguous": 1, "confirmed_adc": 1}


def test_build_disposition_order_excludes_skipped(monkeypatch):
    programs = [_program("skipped1")]
    monkeypatch.setattr(
        triage_serve, "serve_plan",
        lambda program, **kwargs: triage_serve.ServePlan(start_gate=1, skip=True, reopened=False, context=None),
    )
    ordered, counts = qo.build_disposition_order(
        programs, ["skipped1"],
        gold_records=[], heme_auto_ok=True, model_gate_ok=True,
        heme_holdout_ids=set(), triage_holdout_ids=set(), staged_by_program={}, con=None,
    )
    assert ordered == []
    assert counts == {"likely_reject": 0, "ambiguous": 0, "confirmed_adc": 0}


def test_likely_reject_candidates_excludes_holdouts(monkeypatch):
    programs = [_program("holdout1"), _program("real1")]
    monkeypatch.setattr(
        triage_serve, "serve_plan",
        lambda program, **kwargs: triage_serve.ServePlan(start_gate=1, skip=False, reopened=False, context=None),
    )
    monkeypatch.setattr(tev, "build_layer2_evidence", lambda program, con: {"text_snippets": ["oral tablet"]})

    out = qo.likely_reject_candidates(
        programs, ["holdout1", "real1"],
        heme_auto_ok=True, model_gate_ok=True,
        heme_holdout_ids={"holdout1"}, triage_holdout_ids=set(),
        staged_by_program={}, con=object(), limit=10,
    )
    assert [c["program_id"] for c in out] == ["real1"]
    assert "oral" in out[0]["signal_snippet"]


def test_likely_reject_candidates_respects_limit(monkeypatch):
    programs = [_program(f"p{i}") for i in range(5)]
    monkeypatch.setattr(
        triage_serve, "serve_plan",
        lambda program, **kwargs: triage_serve.ServePlan(start_gate=1, skip=False, reopened=False, context=None),
    )
    monkeypatch.setattr(tev, "build_layer2_evidence", lambda program, con: {"text_snippets": ["oral tablet"]})

    out = qo.likely_reject_candidates(
        programs, [p["program_id"] for p in programs],
        heme_auto_ok=True, model_gate_ok=True,
        heme_holdout_ids=set(), triage_holdout_ids=set(),
        staged_by_program={}, con=object(), limit=2,
    )
    assert len(out) == 2
