"""Pipeline audit harness. Entry point: python -m pharma_stats.audit.

Every check states what it expects and what it found, so it can actually
fail — see pharma_stats.audit.types.Check. Levels: FAIL (data is wrong,
stop), WARN (suspicious, look at it), INFO (context, doesn't block).
"""
from __future__ import annotations

from pharma_stats.audit import (
    backfill,
    differ,
    features,
    gold_set,
    history,
    label_sufficiency,
    model,
    normalisation,
    provenance,
    universe,
)

STAGE_REGISTRY = {
    "provenance": provenance.run,
    "universe": universe.run,
    "history": history.run,
    "backfill": backfill.run,
    "differ": differ.run,
    "normalisation": normalisation.run,
    "gold_set": gold_set.run,
    "label_sufficiency": label_sufficiency.run,
    "features": features.run,
    "model": model.run,
}

# run order matches the pipeline's own data dependency order.
# label_sufficiency now runs AFTER features/model (moved 2026-09-04,
# docs/decisions/0005): its lead-time bootstrap consumes model_flag_date
# from the model stage's published result, not label_evidence_date.
STAGE_ORDER = [
    "provenance", "universe", "history", "backfill", "differ",
    "normalisation", "gold_set", "features", "model", "label_sufficiency",
]

# A FAIL in one of these stages halts `--stage all` before running anything
# downstream of it, rather than just contributing to the final exit code.
# universe's "unreviewed candidates" check is Phase 0's exit condition —
# every later stage analyses candidates a human hasn't signed off on yet,
# so it should block rather than whisper.
GATING_STAGES = {"universe"}
