"""Structural schema guard for the undocumented /api/int/ history endpoints.

CT.gov's internal API carries no stability guarantee (see
pharma_stats.clients.ctgov's module docstring). Rather than silently
parsing a response whose shape has changed underneath us, every
history-list and history-version response is checked against the fields
this codebase actually relies on; a mismatch raises loudly with the
offending shape attached, instead of producing quietly-wrong data.

Design note: this is deliberately NOT a blanket recursive structural hash
of the whole response. Two fields on the history-list response —
`lastUpdateVersions` and `originalData` — are legitimately
variable-shaped per trial (their keys depend on which fields that trial
happens to have), and the history-version response's `study` object has
per-trial-optional modules. A full-depth hash would false-positive on
every one of those, which is worse than not guarding at all. Instead this
asserts the specific keys/types the rest of the codebase depends on
(`issubset`, not `==`, so CT.gov *adding* a field doesn't trip it — only
removing/renaming one of ours does), and hashes just that stable subset.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


class SchemaGuardError(RuntimeError):
    pass


HISTORY_LIST_TOP_KEYS = {"changes", "lastUpdateVersions", "originalData", "outcomesUpdateCount"}
HISTORY_LIST_ENTRY_KEYS = {
    "version", "date", "status", "studyType", "moduleLabels", "lastUpdateSubmitQcDate",
}

HISTORY_VERSION_TOP_KEYS = {"studyVersion", "study"}


def _hash(signature: dict) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def check_history_list(obj: Any, *, nct_id: str) -> str:
    """Validate a GET /studies/{nctId}/history response. Returns a hash of
    the stable envelope signature (for logging alongside the snapshot)."""
    if not isinstance(obj, dict):
        raise SchemaGuardError(f"[{nct_id}] /history response is not an object: {type(obj)}")

    top_keys = set(obj.keys())
    if not HISTORY_LIST_TOP_KEYS.issubset(top_keys):
        raise SchemaGuardError(
            f"[{nct_id}] /history top-level keys changed.\n"
            f"expected at least: {sorted(HISTORY_LIST_TOP_KEYS)}\n"
            f"got: {sorted(top_keys)}"
        )

    changes = obj.get("changes")
    if not isinstance(changes, list):
        raise SchemaGuardError(f"[{nct_id}] /history 'changes' is not a list: {type(changes)}")

    for entry in changes:
        if not isinstance(entry, dict):
            raise SchemaGuardError(f"[{nct_id}] /history change entry is not an object: {entry!r}")
        entry_keys = set(entry.keys())
        if not HISTORY_LIST_ENTRY_KEYS.issubset(entry_keys):
            raise SchemaGuardError(
                f"[{nct_id}] /history change entry missing expected keys.\n"
                f"expected at least: {sorted(HISTORY_LIST_ENTRY_KEYS)}\n"
                f"got: {sorted(entry_keys)}\n"
                f"entry: {json.dumps(entry, indent=2)}"
            )
        if not isinstance(entry.get("version"), int):
            raise SchemaGuardError(f"[{nct_id}] /history entry 'version' is not an int: {entry!r}")
        if not isinstance(entry.get("moduleLabels"), list):
            raise SchemaGuardError(
                f"[{nct_id}] /history entry 'moduleLabels' is not a list: {entry!r}"
            )

    entry_key_union = sorted(set().union(*(set(e.keys()) for e in changes))) if changes else []
    signature = {"top_keys": sorted(top_keys), "entry_key_union": entry_key_union}
    return _hash(signature)


def check_history_version(obj: Any, *, nct_id: str, version: int) -> str:
    """Validate a GET /studies/{nctId}/history/{version} response. Returns
    a hash of the stable envelope signature."""
    if not isinstance(obj, dict):
        raise SchemaGuardError(
            f"[{nct_id} v{version}] /history/{{version}} response is not an object: {type(obj)}"
        )

    top_keys = set(obj.keys())
    if not HISTORY_VERSION_TOP_KEYS.issubset(top_keys):
        raise SchemaGuardError(
            f"[{nct_id} v{version}] /history/{{version}} top-level keys changed.\n"
            f"expected at least: {sorted(HISTORY_VERSION_TOP_KEYS)}\n"
            f"got: {sorted(top_keys)}"
        )
    if not isinstance(obj.get("studyVersion"), int):
        raise SchemaGuardError(
            f"[{nct_id} v{version}] 'studyVersion' is not an int: {obj.get('studyVersion')!r}"
        )
    study = obj.get("study")
    if not isinstance(study, dict) or not isinstance(study.get("protocolSection"), dict):
        raise SchemaGuardError(
            f"[{nct_id} v{version}] 'study.protocolSection' missing or not an object.\n"
            f"study top-level keys: {sorted(study.keys()) if isinstance(study, dict) else study!r}"
        )

    signature = {"top_keys": sorted(top_keys), "study_has_protocol_section": True}
    return _hash(signature)
