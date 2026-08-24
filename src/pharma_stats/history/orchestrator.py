"""Resumable, priority-ordered backfill orchestrator.

Single-threaded, 1.05s/request (matches CT.gov robots.txt's Crawl-delay:1
via CtgovClient's default throttle — do not parallelise around it). Runs
per trial, in priority order:

  1. (re)fetch that trial's version-history index (cheap, one request)
  2. selectively fetch new version bodies whose changed modules intersect
     `signal_labels` (the expensive part)

then checkpoints that trial as done before moving to the next. Killed at
any point, the in-flight trial is simply redone on the next run — both
steps are idempotent (history_index upserts by (nct_id, version) primary
key; body fetches are cached/immutable-checked snapshots) — so a kill
mid-trial costs at most a few redundant requests for that one trial, never
corrupts state.

Priority queue: gold-set trials first, then trials with the longest
amendment silence (days since their most recently *known* version's
posted_date — only computable once that trial has an index), then the
remainder. Recomputed at the start of every run so it reflects the latest
index/gold data.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import GOLD_DIR
from pharma_stats.history.index import ensure_schema, index_trial

BACKFILL_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS backfill_queue (
    nct_id VARCHAR PRIMARY KEY,
    priority_tier INTEGER,
    priority_key DOUBLE,
    status VARCHAR DEFAULT 'pending',   -- pending | done | error
    latest_version_indexed INTEGER,
    bodies_fetched_through_version INTEGER DEFAULT -1,
    last_error VARCHAR,
    updated_at TIMESTAMP
)
"""


def _gold_labelled_trial_ids(gold_dir: Path = GOLD_DIR) -> set[str]:
    """Trials referenced by hand-authored labels, if any exist yet. The
    labelling tool writes append-only JSONL to gold/; forward-compatible
    here even though gold/ is empty until that tool is used."""
    import json

    ids: set[str] = set()
    if not gold_dir.exists():
        return ids
    for path in gold_dir.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("nct_id", "nct_ids"):
                v = rec.get(key)
                if isinstance(v, str):
                    ids.add(v)
                elif isinstance(v, list):
                    ids.update(v)
    return ids


def _evidenced_candidate_trial_ids(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Trials belonging to candidates with real naming/seed evidence
    they're an ADC (not dev-code-only, not ambiguous) — the closest thing
    to a "gold set" that exists before any hand labelling has happened."""
    try:
        rows = con.execute(
            "SELECT nct_ids FROM asset_candidates WHERE NOT dev_code_only AND NOT ambiguous"
        ).fetchall()
    except duckdb.CatalogException:
        return set()
    ids: set[str] = set()
    for (nct_ids,) in rows:
        ids.update(nct_ids)
    return ids


def build_priority_queue(
    con: duckdb.DuckDBPyConnection, trial_universe: set[str], *, gold_dir: Path = GOLD_DIR
) -> None:
    """(Re)compute priority tier/key for every trial in the universe and
    upsert into backfill_queue, preserving existing progress state."""
    con.execute(BACKFILL_QUEUE_SCHEMA)

    gold_ids = _gold_labelled_trial_ids(gold_dir) | _evidenced_candidate_trial_ids(con)

    silence_rows = con.execute(
        """
        SELECT nct_id, max(posted_date) AS last_posted
        FROM history_index
        GROUP BY nct_id
        """
    ).fetchall()
    last_posted = {nct_id: d for nct_id, d in silence_rows if d is not None}
    today = date.today()

    now = datetime.now(timezone.utc)
    rows = []
    for nct_id in trial_universe:
        if nct_id in gold_ids:
            tier, key = 0, 0.0
        elif nct_id in last_posted:
            tier, key = 1, -(today - last_posted[nct_id]).days  # more silence -> sorts first (ASC)
        else:
            tier, key = 2, 0.0  # remainder: no index yet, nothing to rank by
        rows.append((nct_id, tier, key, now))

    con.executemany(
        """
        INSERT INTO backfill_queue (nct_id, priority_tier, priority_key, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (nct_id) DO UPDATE SET
            priority_tier = excluded.priority_tier,
            priority_key = excluded.priority_key
        """,
        rows,
    )


def _pending_queue(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Every trial, in priority order — NOT filtered by status.

    `status` is diagnostic only (last outcome: done/error), not a
    completion gate: "done" from a run with a narrower (or empty)
    `signal_labels` doesn't mean a later run with a wider set has nothing
    left to fetch for that trial. The real completion check is
    `bodies_fetched_through_version` vs. the version each row's own
    `to_fetch` query resolves against the *current* signal_labels — that
    happens per-trial inside run_backfill, not here. A trial with nothing
    new to do is cheap (cached index refresh, empty to_fetch), so
    including it here costs ~nothing; excluding it risked silently
    stranding real backlog behind a stale flag, which is exactly the bug
    this replaced.
    """
    rows = con.execute(
        """
        SELECT nct_id FROM backfill_queue
        ORDER BY priority_tier ASC, priority_key ASC, nct_id ASC
        """
    ).fetchall()
    return [r[0] for r in rows]


def run_backfill(
    client: CtgovClient,
    con: duckdb.DuckDBPyConnection,
    trial_universe: set[str],
    *,
    signal_labels: frozenset[str],
    max_seconds: Optional[float] = None,
    max_trials: Optional[int] = None,
    gold_dir: Path = GOLD_DIR,
    progress_every: int = 25,
) -> dict:
    """Run the orchestrator until the queue is empty or a time/trial
    budget is hit. Safe to kill (SIGKILL, SIGTERM, process crash) and
    re-invoke — it resumes from backfill_queue's checkpointed state."""
    ensure_schema(con)
    build_priority_queue(con, trial_universe, gold_dir=gold_dir)

    queue = _pending_queue(con)
    start = time.monotonic()
    trials_done = 0
    versions_indexed = 0
    bodies_fetched = 0
    errors = 0

    for i, nct_id in enumerate(queue):
        if max_seconds is not None and time.monotonic() - start > max_seconds:
            break
        if max_trials is not None and trials_done >= max_trials:
            break

        try:
            n_versions = index_trial(client, con, nct_id)
            versions_indexed += n_versions

            checkpoint = con.execute(
                "SELECT bodies_fetched_through_version FROM backfill_queue WHERE nct_id = ?",
                [nct_id],
            ).fetchone()
            already_through = checkpoint[0] if checkpoint and checkpoint[0] is not None else -1

            to_fetch = con.execute(
                """
                SELECT version FROM history_index
                WHERE nct_id = ? AND version > ? AND version > 0
                  AND len(list_intersect(changed_modules, ?)) > 0
                ORDER BY version ASC
                """,
                [nct_id, already_through, list(signal_labels)],
            ).fetchall()

            max_version_fetched = already_through
            for (v,) in to_fetch:
                client.get_study_version(nct_id, v)  # cached + schema-guarded + immutable
                bodies_fetched += 1
                max_version_fetched = max(max_version_fetched, v)

            latest_version = con.execute(
                "SELECT max(version) FROM history_index WHERE nct_id = ?", [nct_id]
            ).fetchone()[0]

            con.execute(
                """
                UPDATE backfill_queue SET
                    status = 'done', latest_version_indexed = ?,
                    bodies_fetched_through_version = ?, last_error = NULL,
                    updated_at = ?
                WHERE nct_id = ?
                """,
                [latest_version, max_version_fetched, datetime.now(timezone.utc), nct_id],
            )
            trials_done += 1

        except Exception as e:  # noqa: BLE001 — checkpoint the failure and keep going
            errors += 1
            con.execute(
                """
                UPDATE backfill_queue SET status = 'error', last_error = ?, updated_at = ?
                WHERE nct_id = ?
                """,
                [str(e)[:2000], datetime.now(timezone.utc), nct_id],
            )

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.monotonic() - start
            print(
                f"  [{i + 1}/{len(queue)}] {trials_done} done, {errors} errors, "
                f"{bodies_fetched} bodies fetched, {elapsed:.0f}s elapsed",
                flush=True,
            )

    remaining = con.execute("SELECT count(*) FROM backfill_queue WHERE status != 'done'").fetchone()[0]
    return {
        "trials_attempted": trials_done + errors,
        "trials_done": trials_done,
        "errors": errors,
        "versions_indexed": versions_indexed,
        "bodies_fetched": bodies_fetched,
        "remaining_in_queue": remaining,
        "elapsed_seconds": time.monotonic() - start,
    }
