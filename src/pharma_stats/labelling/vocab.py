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

# Who made this gate decision. "auto" only ever applies to gate_reached=2,
# in_scope=no records written by scripts/apply_auto_scope_exclusions.py
# from a confident MeSH-based heme_only verdict (labelling/trial_scope.py)
# — never written by the review app itself, and never overriding a
# decision a human already made. See store.validate_label_payload.
DECIDED_BY_VALUES = ["human", "auto"]

APP_VERSION = "0.3.0"
