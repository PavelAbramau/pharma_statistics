import duckdb

from pharma_stats.history.orchestrator import run_backfill


class FakeCtgovClient:
    """Stands in for CtgovClient in orchestrator tests — no network, no
    throttling, no snapshot store. Just enough surface for index_trial
    and run_backfill's body-fetch loop."""

    def __init__(self, histories: dict[str, list[dict]]):
        self._histories = histories
        self.body_fetch_calls: list[tuple[str, int]] = []

    def get_history(self, nct_id: str) -> list[dict]:
        return self._histories[nct_id]

    def get_study_version(self, nct_id: str, version: int) -> dict:
        self.body_fetch_calls.append((nct_id, version))
        return {"protocolSection": {}}


def _entry(version, modules, status="RECRUITING", d="2020-01-01"):
    return {
        "version": version, "date": d, "status": status, "studyType": "INTERVENTIONAL",
        "moduleLabels": modules, "lastUpdateSubmitQcDate": d,
    }


def test_index_only_run_then_wider_filter_run_does_not_strand_pending_work():
    """Regression test for the bug found 2026-08-20: an index-only run
    (signal_labels=frozenset()) must not mark a trial 'done' in a way that
    causes a later run with real signal_labels to skip its pending body
    fetches."""
    histories = {
        "NCT001": [_entry(0, []), _entry(1, ["Study Design"]), _entry(2, ["Contacts/Locations"])],
    }
    con = duckdb.connect(":memory:")
    client = FakeCtgovClient(histories)

    # Pass 1: index-only (mirrors the buggy `--signal-labels none` run).
    result1 = run_backfill(client, con, {"NCT001"}, signal_labels=frozenset())
    assert result1["bodies_fetched"] == 0
    assert client.body_fetch_calls == []

    # Pass 2: real signal labels — version 1 ("Study Design") should now
    # be fetched. Before the fix, NCT001 was already status='done' from
    # pass 1 and got silently skipped here.
    result2 = run_backfill(client, con, {"NCT001"}, signal_labels=frozenset({"Study Design"}))
    assert result2["bodies_fetched"] == 1
    assert client.body_fetch_calls == [("NCT001", 1)]


def test_rerun_with_same_filter_does_not_refetch_bodies():
    histories = {"NCT001": [_entry(0, []), _entry(1, ["Study Design"])]}
    con = duckdb.connect(":memory:")
    client = FakeCtgovClient(histories)
    labels = frozenset({"Study Design"})

    run_backfill(client, con, {"NCT001"}, signal_labels=labels)
    assert len(client.body_fetch_calls) == 1

    run_backfill(client, con, {"NCT001"}, signal_labels=labels)
    assert len(client.body_fetch_calls) == 1  # unchanged — nothing new to fetch


def test_new_version_appearing_later_is_picked_up_on_rerun():
    histories = {"NCT001": [_entry(0, []), _entry(1, ["Study Design"])]}
    con = duckdb.connect(":memory:")
    client = FakeCtgovClient(histories)
    labels = frozenset({"Study Design"})

    run_backfill(client, con, {"NCT001"}, signal_labels=labels)
    assert len(client.body_fetch_calls) == 1

    # Simulate a new amendment appearing (e.g. next month's run).
    histories["NCT001"].append(_entry(2, ["Outcome Measures"]))
    run_backfill(client, con, {"NCT001"}, signal_labels=frozenset({"Study Design", "Outcome Measures"}))
    assert client.body_fetch_calls == [("NCT001", 1), ("NCT001", 2)]


def test_killed_mid_run_resumes_cleanly():
    histories = {
        "NCT001": [_entry(0, []), _entry(1, ["Study Design"])],
        "NCT002": [_entry(0, []), _entry(1, ["Study Status"])],
    }
    con = duckdb.connect(":memory:")
    client = FakeCtgovClient(histories)
    labels = frozenset({"Study Design", "Study Status"})

    # "Kill" after only one trial by bounding max_trials.
    r1 = run_backfill(client, con, {"NCT001", "NCT002"}, signal_labels=labels, max_trials=1)
    assert r1["trials_done"] == 1
    assert r1["remaining_in_queue"] == 1

    # Resume: must finish the other trial, must not redo the first one's fetch.
    calls_before = len(client.body_fetch_calls)
    r2 = run_backfill(client, con, {"NCT001", "NCT002"}, signal_labels=labels)
    assert r2["remaining_in_queue"] == 0
    assert len(client.body_fetch_calls) == calls_before + 1  # only the second trial's body


def test_priority_queue_puts_evidenced_candidates_first():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE asset_candidates (nct_ids VARCHAR[], dev_code_only BOOLEAN, ambiguous BOOLEAN)"
    )
    con.execute(
        "INSERT INTO asset_candidates VALUES (['NCT_GOLD'], false, false), (['NCT_NOISE'], true, false)"
    )
    histories = {
        "NCT_GOLD": [_entry(0, [])],
        "NCT_NOISE": [_entry(0, [])],
    }
    client = FakeCtgovClient(histories)
    run_backfill(client, con, {"NCT_GOLD", "NCT_NOISE"}, signal_labels=frozenset(), max_trials=0)
    tiers = dict(con.execute("SELECT nct_id, priority_tier FROM backfill_queue").fetchall())
    assert tiers["NCT_GOLD"] == 0
    assert tiers["NCT_NOISE"] == 2
