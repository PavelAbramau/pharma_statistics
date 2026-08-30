"""Controlled vocabularies for the label record itself (not the asset/
indication/line vocab — that normalisation step doesn't exist yet).
Mirrors CLAUDE.md exactly; keep in sync by hand."""

PROGRAM_STATUSES = [
    "active", "dormant_suspected", "dead_confirmed", "approved",
    "superseded", "unknown",
]

KILL_REASONS = [
    "futility_efficacy", "toxicity_safety", "strategic_portfolio",
    "accrual_failure", "funding_insolvency", "competitive_landscape",
    "ip_legal", "unknown_silent",
]

CONFIDENCE_LEVELS = ["high", "medium", "low"]

# What kind of evidence public_confirmation_date rests on. pipeline_page_removal
# is deliberately included even though it's the ambiguous case — the sponsor
# acted publicly (pulled the asset from a pipeline page) but stated nothing —
# so it gets tagged rather than silently decided one way or the other here;
# downstream analysis runs both "strict" (statements only: press_release,
# sec_filing, earnings_call, regulatory) and "broad" (also includes
# pipeline_page_removal) and reports both, never conflated into one number.
CONFIRMATION_EVIDENCE_TYPES = [
    "press_release", "sec_filing", "earnings_call", "pipeline_page_removal",
    "regulatory", "other",
]

# The review screen is three sequential gates, each only reachable if the
# previous one passed (see store.validate_label_payload):
#   Gate 1 — is_adc: a judgement about the molecule only. yes proceeds to
#            gate 2; no/unsure is terminal (this was never a program).
#   Gate 2 — in_scope: per-program-eligibility for this project's locked
#            scope (ADCs, solid tumours, industry sponsor, 2012-present).
#            Only reachable when is_adc=yes. yes proceeds to gate 3; no
#            (with a reason) is terminal. "not_an_adc" is deliberately not
#            a scope_reason — that's Gate 1's job, and Gate 2 is never
#            reached unless is_adc is already yes.
#   Gate 3 — the full status label. Only reachable when in_scope=yes.
# gate_reached on the saved record (1/2/3) records how far a given review
# actually got, so triage rejections (gates 1-2) can be told apart from
# real labels (gate 3) everywhere they're counted.
IS_ADC_VALUES = ["yes", "no", "unsure"]
IN_SCOPE_VALUES = ["yes", "no"]
SCOPE_OUT_REASONS = ["heme_only", "pre_2012", "non_industry", "non_oncology"]
GATE_VALUES = [1, 2, 3]

# Who made this gate decision. "auto" applies to gate_reached=1 (is_adc)
# and gate_reached=2 (in_scope) records written by an auto-triage source —
# scripts/apply_auto_scope_exclusions.py (heme_only) or the
# pharma_stats.triage pipeline (INN suffix / denylist / generic-class-label
# / sponsor-class / heme-only / model-layer / web-search decisions) —
# never gate_reached=3, and never written by the review app itself. A
# human decision is always a new, later record (append-only) — see
# store.validate_label_payload and store.latest_by_program.
DECIDED_BY_VALUES = ["human", "auto"]

# Which triage layer produced a decided_by=auto record — required whenever
# decided_by=="auto", so "how was this decided" is always answerable from
# the record alone, not from which script happened to write it.
#   1 = deterministic (no model call — pharma_stats.triage.deterministic)
#   2 = batched model call (pharma_stats.triage.layer2)
#   3 = single web-search query (pharma_stats.triage.layer3)
# scripts/apply_auto_scope_exclusions.py's heme_only auto-exclusions also
# count as layer 1 (deterministic, MeSH-based, no model call).
TRIAGE_LAYERS = [1, 2, 3]

APP_VERSION = "0.3.0"
