"""Honest placeholders for pipeline stages that don't exist yet.

Per README.md's "What is built" table, the Differ, controlled-vocab
normalisation, feature panel, and model stages are all "not started".
Faking checks for tables that don't exist would either always pass
(decoration) or crash — neither is useful. These stubs make the gap
visible in every report instead of silently omitting the stage.
"""
from __future__ import annotations

from pharma_stats.audit.types import Check, info


def not_built(stage: str, why: str) -> list[Check]:
    return [info(
        stage, f"'{stage}' pipeline stage",
        expected="implemented, with real checks below",
        actual="not built yet", detail=why,
    )]
