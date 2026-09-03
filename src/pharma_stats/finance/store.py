"""financial_events — the EvidenceEvent-shaped sink for every signal the
financial layer produces (synthetic cost, conviction ratio, and, as
they're built, SEC-filing-derived distress/impairment signals).

Deliberately a SEPARATE table from differ.extract's evidence_events, not
a schema change to it: that table's grain is trial-version diffs
(nct_id, from_version, to_version — non-nullable, tightly bound to
registry versioning) and is already live, real, production data other
code reads (provisional_programs.py, silver/evidence.py, the audit
harness). Forcing financial events (sponsor-level, program-level, no
registry version at all) into that shape would mean fabricating
nct_id/version values or making core columns nullable on a table other
code depends on being populated. Same spirit instead — typed, dated by
knowability, with a pointer back to source — generalized to
subject_type/subject_id so a row can point at a program, a sponsor, or
(for impairments a human hasn't adjudicated yet) nothing specific.
"differ" reads its own table; nothing downstream needs to know which
table an event came from as long as it queries through this module's
load_records(), not a raw SQL string.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb

from pharma_stats.config import WAREHOUSE_DB

FINANCIAL_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS financial_events (
    event_id VARCHAR,
    subject_type VARCHAR,   -- "program" | "sponsor" | "asset"
    subject_id VARCHAR,
    event_date DATE,        -- knowability date
    event_type VARCHAR,
    value DOUBLE,
    value_text VARCHAR,      -- for non-numeric payloads (e.g. distress signal detail)
    detail VARCHAR,
    source VARCHAR,          -- e.g. "sertkaya_2016_synthetic", "sec_xbrl", "sec_8k"
    source_url VARCHAR,
    extracted_at TIMESTAMP
)
"""

EVENT_TYPES = [
    "synthetic_cost_index_monthly",
    "conviction_ratio_monthly",
    "rd_expense_quarterly",       # SEC XBRL ResearchAndDevelopmentExpense
    "distress_signal",            # 8-K listing deficiency / delisting / going concern / bankruptcy
    "market_cap_snapshot",
    "ev_share_estimate",
    "iprd_impairment_charge",     # unattributed until a human adjudicates the review queue
]


@dataclass
class FinancialEvent:
    subject_type: str
    subject_id: str
    event_date: date
    event_type: str
    detail: str
    source: str
    value: Optional[float] = None
    value_text: Optional[str] = None
    source_url: Optional[str] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_row(self) -> dict:
        d = asdict(self)
        d["event_date"] = self.event_date.isoformat() if self.event_date else None
        d["extracted_at"] = self.extracted_at.isoformat()
        return d


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(FINANCIAL_EVENTS_SCHEMA)


def append_records(records: list[FinancialEvent], warehouse_db: Optional[Any] = None) -> int:
    if not records:
        return 0
    con = duckdb.connect(str(warehouse_db or WAREHOUSE_DB))
    try:
        ensure_schema(con)
        rows = [tuple(r.to_row().values()) for r in records]
        cols = list(records[0].to_row().keys())
        placeholders = ", ".join(["?"] * len(cols))
        con.executemany(f"INSERT INTO financial_events ({', '.join(cols)}) VALUES ({placeholders})", rows)
        return len(rows)
    finally:
        con.close()


def load_records(warehouse_db: Optional[Any] = None, *, event_type: Optional[str] = None) -> list[dict]:
    db_path = Path(warehouse_db or WAREHOUSE_DB)
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'financial_events'"
        ).fetchone()[0]
        if not exists:
            return []
        if event_type:
            rows = con.execute("SELECT * FROM financial_events WHERE event_type = ?", [event_type]).fetchall()
        else:
            rows = con.execute("SELECT * FROM financial_events").fetchall()
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()
