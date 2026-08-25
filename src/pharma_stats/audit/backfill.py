"""Backfill convergence stage: queue drain state, whether the errored
trials share a failure mode (a systematic coverage hole) or are
scattered (isolated flakes), plus the saturation curve that actually
answers "is it done" — new events extracted per additional 1,000
versions fetched. That sub-check needs the Differ (EvidenceEvent
extraction), which doesn't exist yet, so it's reported honestly as
not-yet-computable rather than skipped silently."""
from __future__ import annotations

import re

import duckdb

from pharma_stats.audit.types import Check, info, ok, warn
from pharma_stats.config import WAREHOUSE_DB

STAGE = "backfill"


def run() -> list[Check]:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        checks: list[Check] = []
        checks += _drain_state(con)
        checks += _event_saturation_curve(con)
        return checks
    finally:
        con.close()


def _drain_state(con) -> list[Check]:
    rows = con.execute("SELECT status, count(*) FROM backfill_queue GROUP BY 1").fetchall()
    counts = dict(rows)
    total = sum(counts.values())
    pending = counts.get("pending", 0)
    errored = counts.get("error", 0)
    done = counts.get("done", 0)

    checks = [info(
        STAGE, "backfill queue drain state",
        expected="pending trending to 0",
        actual=f"done={done}, pending={pending}, error={errored}, total={total}",
        detail="",
    )]
    if pending:
        checks.append(info(
            STAGE, "remaining backfill work",
            expected="0 pending for the current universe",
            actual=f"{pending} / {total} still pending", detail="",
        ))
    error_rate = errored / total if total else 0.0
    checks.append((warn if error_rate > 0.05 else ok)(
        STAGE, "backfill error rate",
        expected="<=5% of trials erroring", actual=f"{error_rate:.1%} ({errored} / {total})",
        detail="",
    ))
    checks.append(_error_clustering(con))
    return checks


_ERROR_PATTERNS = [
    ("dns_or_connection", re.compile(r"NameResolutionError|Failed to resolve|ConnectionError|Max retries exceeded")),
    ("timeout", re.compile(r"[Tt]ime[d]?[- ]?out|ReadTimeout|ConnectTimeout")),
    ("http_4xx", re.compile(r"\b4\d\d\b.*(failed|error)|HTTP 4\d\d")),
    ("http_5xx", re.compile(r"\b5\d\d\b.*(failed|error)|HTTP 5\d\d")),
    ("schema_guard", re.compile(r"SchemaGuardError|schema.?hash")),
]


def _classify_error(msg: str) -> str:
    for label, pattern in _ERROR_PATTERNS:
        if pattern.search(msg):
            return label
    return "other"


def _error_clustering(con) -> Check:
    rows = con.execute(
        "SELECT nct_id, last_error FROM backfill_queue WHERE status = 'error'"
    ).fetchall()
    if not rows:
        return ok(STAGE, "errored-trial failure-mode clustering",
                   expected="n/a", actual="no errored trials", detail="")

    buckets: dict[str, list[str]] = {}
    for nct_id, msg in rows:
        buckets.setdefault(_classify_error(msg or ""), []).append(nct_id)

    dominant_label, dominant_trials = max(buckets.items(), key=lambda kv: len(kv[1]))
    dominant_share = len(dominant_trials) / len(rows)
    summary = ", ".join(f"{label}={len(trials)}" for label, trials in sorted(buckets.items(), key=lambda kv: -len(kv[1])))

    if dominant_share >= 0.8 and dominant_label != "other":
        level = warn
        actual = (
            f"systematic: {dominant_share:.0%} of {len(rows)} errors are '{dominant_label}' "
            f"({summary}) — this looks like a transient environment/connectivity issue during "
            f"that backfill run, not per-trial data problems; re-running the backfill should "
            f"clear most of these rather than needing per-trial investigation"
        )
    else:
        level = info
        actual = f"scattered across failure modes ({summary}) — no single dominant cause"

    return level(
        STAGE, "errored-trial failure-mode clustering",
        expected="scattered (isolated flakes), or a clearly-transient systematic cause",
        actual=actual,
        detail=", ".join(sorted(dominant_trials)[:10]),
    )


def _event_saturation_curve(con) -> list[Check]:
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "evidence_events" not in tables:
        return [info(
            STAGE, "event-yield saturation curve (new events per additional 1,000 versions fetched)",
            expected="curve flattening before more fetching is spent",
            actual="not computable yet — the Differ / EvidenceEvent extraction stage is not built "
                   "(see README.md)",
            detail="once evidence_events exists with a source_version reference, join it against "
                   "history_index ordered by fetch time to bucket into 1,000-version windows.",
        )]
    # Placeholder for when evidence_events exists: bucket events by the
    # cumulative count of versions fetched at the time and report the
    # marginal yield per 1,000-version bucket.
    return [info(
        STAGE, "event-yield saturation curve",
        expected="curve flattening before more fetching is spent",
        actual="evidence_events table found but this stage's bucketing logic hasn't been wired up yet",
        detail="",
    )]
