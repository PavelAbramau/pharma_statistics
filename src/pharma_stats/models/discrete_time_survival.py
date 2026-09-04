"""Discrete-time competing-risks survival model: dead / approved /
superseded, on the features/ program x month panel.

Implemented as CAUSE-SPECIFIC binary hazards (one Logit per outcome on
the person-period panel), not one joint multinomial model — a standard,
well-precedented way to do discrete-time competing risks, chosen here
specifically because the three outcomes have wildly different event
counts on real data (63 dead / 15 approved / 3 superseded). A joint
model forces the same covariate set on all three; separate cause-specific
hazards let each carry only as many covariates as its event count can
support, rather than reporting spurious coefficients for a 3-event class.
Cluster-robust (sponsor) SEs throughout — see audit/label_sufficiency.py's
ICC=0.18 finding for why sponsor, not program, is the right cluster unit.

Known, real limitation (see docs/decisions/0007): there is no approval or
supersession date field anywhere in this project's data. `dead`'s event
time is the real, hand-verified public_confirmation_date (63 ground-truth
dates). `approved`/`superseded` event times are a proxy — the earliest
ACTUAL (not ESTIMATED) primary_completion_date across the program's
trials — chosen over the gold label's own timestamp (which reflects only
when a human clicked save during this labelling sprint, not when the
program actually resolved). This proxy is a real weakness, disclosed
here and in every report this module produces; the `dead` outcome is
this model's only well-supported one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pharma_stats.features.panel import build_program_month_panel
from pharma_stats.labelling import store

OUTCOME_CLASSES = ("dead", "approved", "superseded")
COVARIATES = ["silence_score_asof", "log_cost_index"]  # kept deliberately small — see module docstring
MIN_EVENTS_FOR_COVARIATES = 10  # below this, fit intercept-only rather than report spurious coefficients


@dataclass
class ProgramOutcome:
    program_id: str
    outcome: str  # "dead" | "approved" | "superseded" | "censored"
    event_month: Optional[date]
    basis: str


def _lead_sponsor(sponsors_over_time: list[dict]) -> str:
    if not sponsors_over_time:
        return "UNKNOWN"
    dated = [s for s in sponsors_over_time if s.get("last_seen")]
    pool = dated or sponsors_over_time
    return max(pool, key=lambda s: s.get("last_seen") or "")["sponsor"] or "UNKNOWN"


def determine_program_outcome(program: dict, gold_record: dict, panel_end: date) -> ProgramOutcome:
    pid = program["program_id"]
    status = gold_record.get("status")

    if status == "dead_confirmed":
        cd = gold_record.get("public_confirmation_date")
        if cd:
            return ProgramOutcome(pid, "dead", date.fromisoformat(cd), "public_confirmation_date")
        # never_publicly_confirmed=True: real ground truth, but genuinely
        # no date to anchor on. label_evidence_date is never used for
        # this (docs/decisions/0005) — right-censor at panel end instead
        # of guessing a date, and flag it as such.
        return ProgramOutcome(pid, "censored", panel_end, "dead_confirmed_but_undated_excluded_from_hazard")

    if status in ("approved", "superseded"):
        candidates = []
        for t in program.get("trials") or []:
            if t.get("primary_completion_type") == "ACTUAL" and t.get("primary_completion_date"):
                candidates.append(date.fromisoformat(t["primary_completion_date"]))
        if candidates:
            return ProgramOutcome(pid, status, min(candidates), "earliest_actual_primary_completion_proxy")
        return ProgramOutcome(pid, "censored", panel_end, f"{status}_but_no_completion_date_available")

    return ProgramOutcome(pid, "censored", panel_end, f"status={status}")


def build_training_table(
    programs: list[dict], con: duckdb.DuckDBPyConnection, *, panel_end: Optional[date] = None,
) -> pd.DataFrame:
    """One row per (program, month) up to and including the program's
    event/censoring month, with event_dead/event_approved/event_superseded
    (1 only on the terminal row of a non-censored program), sponsor
    (cluster unit), and the covariate columns."""
    panel_end = panel_end or date.today()
    gold_latest = store.latest_by_program(store.load_records())

    rows = []
    for program in programs:
        pid = program["program_id"]
        g = gold_latest.get(pid)
        if g is None or g.get("gate_reached") != 3:
            continue
        outcome = determine_program_outcome(program, g, panel_end)
        panel_rows = build_program_month_panel(program, con, end=min(outcome.event_month, panel_end))
        if not panel_rows:
            continue
        sponsor = _lead_sponsor(program.get("sponsors_over_time") or [])
        for i, row in enumerate(panel_rows):
            is_terminal = i == len(panel_rows) - 1
            rows.append({
                **row,
                "sponsor": sponsor,
                "event_dead": int(is_terminal and outcome.outcome == "dead"),
                "event_approved": int(is_terminal and outcome.outcome == "approved"),
                "event_superseded": int(is_terminal and outcome.outcome == "superseded"),
                "outcome_basis": outcome.basis,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["log_cost_index"] = np.log1p(df["cost_index"].fillna(0.0))
    # silence_score_asof is None on a (rare) month where no trial state
    # was resolvable at all (e.g. before any version's body was ever
    # fetched — see features/trial_asof.py's module docstring). Filled
    # with 0 ("no evidence of silence yet") rather than dropping the row
    # — dropping could silently discard a program's only terminal-event
    # row if it happened to land on such a month.
    n_missing = df["silence_score_asof"].isna().sum()
    if n_missing:
        print(f"  (imputing {n_missing} row(s) with no resolvable silence_score_asof to 0.0)")
    df["silence_score_asof"] = df["silence_score_asof"].fillna(0.0).astype(float)
    return df


@dataclass
class CauseSpecificHazard:
    outcome: str
    n_events: int
    covariates: list[str]  # empty list = intercept-only (too few events for covariates)
    result: object  # statsmodels DiscreteResults

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.covariates:
            X = sm.add_constant(pd.DataFrame(index=df.index), has_constant="add")
        else:
            X = sm.add_constant(df[self.covariates], has_constant="add")
        return np.asarray(self.result.predict(X))


def fit_cause_specific_hazard(df: pd.DataFrame, outcome_col: str) -> CauseSpecificHazard:
    """Logit(event ~ covariates), cluster-robust SEs by sponsor. Falls
    back to an intercept-only fit when there are too few events to
    support covariates (MIN_EVENTS_FOR_COVARIATES) — see module
    docstring on why superseded (3 events) can't support any covariate."""
    outcome_name = outcome_col.replace("event_", "")
    n_events = int(df[outcome_col].sum())
    use_covariates = n_events >= MIN_EVENTS_FOR_COVARIATES

    covariates = list(COVARIATES) if use_covariates else []
    if covariates:
        X = sm.add_constant(df[covariates])
    else:
        X = sm.add_constant(pd.DataFrame(index=df.index))  # intercept-only
    y = df[outcome_col]
    model = sm.Logit(y, X)
    result = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["sponsor"]})
    return CauseSpecificHazard(outcome=outcome_name, n_events=n_events, covariates=covariates, result=result)


def fit_all_hazards(df: pd.DataFrame) -> dict[str, CauseSpecificHazard]:
    return {oc: fit_cause_specific_hazard(df, f"event_{oc}") for oc in OUTCOME_CLASSES}
