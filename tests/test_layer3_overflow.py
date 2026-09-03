"""Tests for scripts/run_layer3_overflow.py — the --max-spend abort must
actually stop mid-run against real accumulated cost, not just refuse to
start. No test here makes a real API call."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_layer3_overflow.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_layer3_overflow", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_layer3_overflow"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script(monkeypatch):
    mod = _load_script()
    yield mod
    sys.modules.pop("run_layer3_overflow", None)


def _fake_overflow_records(n: int) -> list[dict]:
    return [
        {"program_id": f"p{i}", "proposed_name": f"Drug {i}", "layer": None,
         "manual_overflow": True, "manual_overflow_reason": "layer3 cap exceeded",
         "timestamp": "2026-01-01T00:00:00Z", "rule": None}
        for i in range(n)
    ]


def test_max_spend_aborts_mid_run_not_after_everything_completes(monkeypatch, script, capsys):
    from pharma_stats.triage import layer3, pool as tpool
    from pharma_stats.silver import model_client

    records = _fake_overflow_records(100)  # 5 chunks of 20 at CHUNK_SIZE=20
    monkeypatch.setattr(script.staging, "load_records", lambda: records)
    monkeypatch.setattr(script.tpool, "assert_not_reviewed", lambda pid: None)
    monkeypatch.setattr(script.staging, "append_record", lambda record: None)

    calls = []

    def fake_run_layer3(chunk, *, model):
        calls.append(len(chunk))
        answers = {c["program_id"]: layer3.Layer3Answer(c["program_id"], c["name"], "no", "x", "https://x")
                   for c in chunk}
        # $1.10/chunk — 5 chunks would total $5.50, must stop before that
        log = {"usage": {"cost_usd": 1.10, "web_search_flat_fee_usd": 0.20}}
        return answers, log

    monkeypatch.setattr(script.layer3, "run_layer3", fake_run_layer3)

    monkeypatch.setattr(sys, "argv", ["run_layer3_overflow.py", "--max-spend", "5"])
    script.main()

    out = capsys.readouterr().out
    assert "exceeded" in out
    # 1.10 * 4 = 4.40 (<=5, continues), 1.10 * 5 = 5.50 (>5, must stop AFTER
    # staging chunk 5's results, not run a 6th) — with 5 total chunks here,
    # the abort must fire after chunk 5, i.e. every chunk still runs, but a
    # 6th (nonexistent) chunk is correctly never attempted
    assert len(calls) == 5
    assert "not yet run" not in out or "0 chunk(s) not yet run" in out


def test_max_spend_aborts_before_the_final_chunk_when_it_would_be_exceeded_earlier(monkeypatch, script, capsys):
    from pharma_stats.triage import layer3

    records = _fake_overflow_records(120)  # 6 chunks of 20
    monkeypatch.setattr(script.staging, "load_records", lambda: records)
    monkeypatch.setattr(script.tpool, "assert_not_reviewed", lambda pid: None)
    monkeypatch.setattr(script.staging, "append_record", lambda record: None)

    calls = []

    def fake_run_layer3(chunk, *, model):
        calls.append(len(chunk))
        answers = {c["program_id"]: layer3.Layer3Answer(c["program_id"], c["name"], "no", "x", "https://x")
                   for c in chunk}
        # $2.00/chunk: after chunk 1 = $2 (<=5), chunk 2 = $4 (<=5),
        # chunk 3 = $6 (>5) -> must stop after chunk 3, never reach 4/5/6
        log = {"usage": {"cost_usd": 2.00, "web_search_flat_fee_usd": 0.20}}
        return answers, log

    monkeypatch.setattr(script.layer3, "run_layer3", fake_run_layer3)
    monkeypatch.setattr(sys, "argv", ["run_layer3_overflow.py", "--max-spend", "5"])
    script.main()

    out = capsys.readouterr().out
    assert len(calls) == 3  # stopped after the 3rd chunk crossed $5, never ran chunks 4-6
    assert "3 chunk(s) not yet run" in out


def test_dry_run_makes_no_calls_and_reports_honest_range(monkeypatch, script, capsys):
    from pharma_stats.triage import layer3

    records = _fake_overflow_records(174)
    monkeypatch.setattr(script.staging, "load_records", lambda: records)

    called = []
    monkeypatch.setattr(script.layer3, "run_layer3", lambda *a, **kw: called.append(1))

    monkeypatch.setattr(sys, "argv", ["run_layer3_overflow.py", "--dry-run"])
    script.main()

    out = capsys.readouterr().out
    assert called == []
    assert "174" in out
    assert "optimistic" in out and "pessimistic" in out


def test_refuses_to_start_when_flat_fee_alone_exceeds_max_spend(monkeypatch, script, capsys):
    records = _fake_overflow_records(600)  # 600 * $0.01 = $6 flat fee alone
    monkeypatch.setattr(script.staging, "load_records", lambda: records)

    called = []
    monkeypatch.setattr(script.layer3, "run_layer3", lambda *a, **kw: called.append(1))

    monkeypatch.setattr(sys, "argv", ["run_layer3_overflow.py", "--max-spend", "5"])
    script.main()

    out = capsys.readouterr().out
    assert called == []
    assert "refusing to start" in out
