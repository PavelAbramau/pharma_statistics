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

APP_VERSION = "0.1.0"
