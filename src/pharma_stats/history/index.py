"""History index: one row per (trial, version) with the version's dates
and which modules changed — cheap to build (one /history request per
trial), independently valuable, and the input the backfill orchestrator's
selective body-fetch and priority queue are both built on.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.history import schema_guard

HISTORY_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS history_index (
    nct_id VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    posted_date DATE,
    submitted_date DATE,
    status VARCHAR,
    study_type VARCHAR,
    changed_modules VARCHAR[],
    schema_hash VARCHAR,
    indexed_at TIMESTAMP,
    PRIMARY KEY (nct_id, version)
)
"""

# The user's literal 2026-08-19 request: "Study Design, Outcome Measures,
# Recruitment Status, Study Status, Sponsor/Collaborators". Checked against
# the real CT.gov label vocabulary (sampled 2026-08-19): there is no
# "Recruitment Status" label distinct from "Study Status" — overallStatus
# transitions (including recruitment status) live inside "Study Status".
# Folded in below rather than silently dropped or silently invented.
USER_REQUESTED_SIGNAL_LABELS = frozenset({
    "Study Design", "Outcome Measures", "Study Status", "Sponsor/Collaborators",
})

# Gap found during investigation: "Arms and Interventions" is required to
# detect arm_removed/cohort_dropped, two of the event types in your own
# step-4 spec, but wasn't in the requested list. Recommended, not silently
# applied — the orchestrator takes an explicit `signal_labels` set so you
# can pick either.
RECOMMENDED_SIGNAL_LABELS = USER_REQUESTED_SIGNAL_LABELS | {"Arms and Interventions"}

# Explicitly out of scope regardless of which set is used: pure contact/
# location/reference/results-reporting modules.
COSMETIC_OR_OUT_OF_SCOPE_LABELS = frozenset({
    "Contacts/Locations", "Study Description", "Conditions", "Oversight",
    "Study Identification", "Eligibility",  # sponsor lives in Sponsor/Collaborators, not here
    "Adverse Events", "Baseline Characteristics", "Document Section", "IPDSharing",
    "Limitations and Caveats", "More Information", "Outcome Measures (Results)",
    "Participant Flow", "References",
})


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(HISTORY_INDEX_SCHEMA)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def index_trial(client: CtgovClient, con: duckdb.DuckDBPyConnection, nct_id: str) -> int:
    """Fetch (cache-first) and upsert the version history for one trial.
    Returns the number of versions indexed."""
    changes = client.get_history(nct_id)  # raises schema_guard.SchemaGuardError on shape drift
    schema_hash = schema_guard.check_history_list({
        "changes": changes,
        # the two variable-shaped fields aren't validated on this recomputation
        # path (client already validated them against the raw payload); pass
        # placeholders so the signature function's shape checks still run.
        "lastUpdateVersions": {}, "originalData": {}, "outcomesUpdateCount": 0,
    }, nct_id=nct_id)

    now = datetime.now(timezone.utc)
    rows = [
        (
            nct_id, c["version"], _parse_date(c.get("date")),
            _parse_date(c.get("lastUpdateSubmitQcDate")), c.get("status"),
            c.get("studyType"), list(c.get("moduleLabels") or []), schema_hash, now,
        )
        for c in changes
    ]
    if rows:
        con.executemany(
            """
            INSERT OR REPLACE INTO history_index
                (nct_id, version, posted_date, submitted_date, status, study_type,
                 changed_modules, schema_hash, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def module_filter_stats(
    con: duckdb.DuckDBPyConnection, signal_labels: frozenset[str]
) -> dict:
    """Real numbers from the populated index: total versions, versions
    whose changed_modules intersects signal_labels, and version 0 (initial
    submission, not an amendment) excluded from the "signal" count since
    it has no prior version to diff against."""
    total_versions = con.execute("SELECT count(*) FROM history_index").fetchone()[0]
    total_trials = con.execute("SELECT count(DISTINCT nct_id) FROM history_index").fetchone()[0]
    non_baseline = con.execute("SELECT count(*) FROM history_index WHERE version > 0").fetchone()[0]

    labels_list = list(signal_labels)
    signal_versions = con.execute(
        """
        SELECT count(*) FROM history_index
        WHERE version > 0
          AND len(list_intersect(changed_modules, ?)) > 0
        """,
        [labels_list],
    ).fetchone()[0]

    return {
        "total_trials": total_trials,
        "total_versions": total_versions,
        "non_baseline_versions": non_baseline,
        "signal_versions": signal_versions,
        "reduction_ratio": 1 - (signal_versions / non_baseline) if non_baseline else 0.0,
    }
