"""Normalisation stage: unmapped-indication burn-down, every program
resolving to a valid asset/OncoTree code/line bucket, zero orphan
events, non-overlapping full-lifetime ownership intervals. None of this
is computable until controlled-vocab normalisation and the Program /
Organization entities exist."""
from __future__ import annotations

from pharma_stats.audit.stubs import not_built
from pharma_stats.audit.types import Check

STAGE = "normalisation"


def run() -> list[Check]:
    return not_built(
        STAGE,
        "Controlled-vocab normalisation and the five-entity warehouse (Program / Organization) "
        "are not implemented (README.md). The labelling app's provisional_programs view stands in "
        "today with program == asset and indication_code='UNSPECIFIED' for every row — see "
        "pharma_stats/labelling/provisional_programs.py. Once real normalisation exists, wire up: "
        "unmapped-indication-string burn-down over runs (trend to zero), every program resolving "
        "to a valid asset + OncoTree code + line bucket, zero orphan events with no program, and "
        "ownership intervals that are non-overlapping and cover each asset's full lifetime.",
    )
