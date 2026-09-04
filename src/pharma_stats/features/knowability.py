"""Machine-checkable knowability registry — every column the panel
(features/panel.py) can produce must be registered here with what makes
it safe to use at a given historical month, mirroring (and backing)
audit/leakage.md's prose entries. A panel column not in this registry is
a hard error, not a warning — see assert_all_columns_registered, called
by build_program_month_panel itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureKnowability:
    name: str
    time_varying: bool
    knowability_rule: str
    source: str


REGISTRY: dict[str, FeatureKnowability] = {
    "silence_score_asof": FeatureKnowability(
        "silence_score_asof", True,
        "the panel month itself — every input trial field is resolved via "
        "features.trial_asof.resolve_trial_summary_as_of, which only reads versioned-history "
        "bodies with posted_date <= that month.",
        "features.panel.build_program_month_panel",
    ),
    "band_asof": FeatureKnowability(
        "band_asof", True, "same as silence_score_asof (derived from it).",
        "features.panel.build_program_month_panel",
    ),
    "cost_index": FeatureKnowability(
        "cost_index", True,
        "the panel month itself — see docs/decisions/0003 and finance/cost_model.py's module "
        "docstring. Deliberately excludes site count (current-state-only, not knowable as-of "
        "a historical month).",
        "finance.cost_model.monthly_cost_index_series",
    ),
    "conviction_ratio": FeatureKnowability(
        "conviction_ratio", True,
        "the panel month itself — computed against peers' SAME-MONTH cost_index, never a peer's "
        "current/later value. See audit/leakage.md's conviction_ratio_monthly entry.",
        "finance.conviction.compute_conviction_ratios",
    ),
    "contacts_locations_amendment_cadence_asof": FeatureKnowability(
        "contacts_locations_amendment_cadence_asof", True,
        "the panel month itself — counts only history rows with posted_date <= that month "
        "(features.trial_asof truncates history the same way as every other field).",
        "features.panel.build_program_month_panel",
    ),
    "target": FeatureKnowability(
        "target", False,
        "static asset property, resolved once from antibody-stem dictionary / trial text / name "
        "— never re-resolved per month, never guessed (attributes/target.py).",
        "attributes.target.derive_target",
    ),
    "payload_chemotype": FeatureKnowability(
        "payload_chemotype", False,
        "static asset property, resolved once from the INN suffix (attributes/payload.py).",
        "attributes.payload.derive_payload_chemotype",
    ),
    "indication_mesh_term": FeatureKnowability(
        "indication_mesh_term", False,
        "static universe-membership property under docs/decisions/0001's existing exception "
        "(disease category doesn't change over a program's life) — current-state fetch only, "
        "never resolved historically.",
        "attributes.indication.program_indication_mesh_term",
    ),
}


def assert_all_columns_registered(columns) -> None:
    unregistered = [c for c in columns if c not in REGISTRY]
    if unregistered:
        raise ValueError(
            f"unregistered feature column(s), not in features.knowability.REGISTRY: {unregistered} — "
            "every panel column needs a knowability_rule before it can be used in a model, "
            "per audit/leakage.md's whole purpose."
        )
