"""EvidenceEvent vocabulary and record shape.

Every event is dated by when it became *knowable* (the posted_date of
the version that introduced it), never by when the sponsor submitted the
change — get_as_of-style backtests depend on that distinction. See
diff.py for the extraction rules, especially the ESTIMATED/ACTUAL
boundary rule this vocabulary is built around.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional

DIFFER_VERSION = "0.1.0"

EVENT_TYPES = [
    "status_changed",
    "phase_changed",
    "sponsor_changed",
    "enrollment_target_changed",
    "enrollment_finalized",
    "primary_completion_date_pushed",
    "primary_completion_date_finalized",
    "completion_date_pushed",
    "completion_date_finalized",
    "arm_added",
    "arm_removed",
    "primary_outcome_added",
    "primary_outcome_removed",
    "primary_outcome_changed",
]

DIRECTIONS = {"increased", "decreased", "pushed_later", "pulled_earlier", "finalized", None}


@dataclass
class EvidenceEvent:
    nct_id: str
    from_version: int
    to_version: int
    event_date: date  # knowability date == to_version's posted_date, never submitted_date
    event_type: str
    field: str
    direction: Optional[str]
    from_value: Any
    to_value: Any
    detail: str
    differ_version: str = DIFFER_VERSION
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_row(self) -> dict:
        d = asdict(self)
        d["event_date"] = self.event_date.isoformat() if self.event_date else None
        d["extracted_at"] = self.extracted_at.isoformat()
        for k in ("from_value", "to_value"):
            if not isinstance(d[k], (str, int, float, bool, type(None))):
                d[k] = str(d[k])
        return d
