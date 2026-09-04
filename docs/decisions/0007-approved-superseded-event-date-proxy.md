# 0007 — No real event date for approved/superseded; proxy and its limits

Date: 2026-09-04

## Context

The competing-risks survival model (`models/discrete_time_survival.py`)
needs an event date for every terminal outcome, not just `dead`. Checked
the gold schema (`labelling/store.py`): `label_evidence_date`,
`public_confirmation_date`, and `confirmation_evidence_type` are only
ever required/populated for `dead_confirmed`. No date field exists for
`approved` or `superseded` at all.

The first thing tried — the gold record's own `timestamp` (when the
reviewer saved the label) — was checked against real data and rejected
immediately: all 15 `approved` labels have timestamps between
2026-08-27 and 2026-09-03, a one-week window, because that's when this
project's labelling sprint happened to reach them. Using it as an event
date would place every approval "event" in the same single week
regardless of when the drug was actually approved (some of these are
long-approved ADCs from years earlier), which is not a date, it's a
labelling-queue artifact — the same category of error 0005 removed
`label_evidence_date` for.

The next candidate — the latest `completion_date` across a program's
trials — was also checked and rejected: multi-trial programs (e.g. an
approved ADC with dozens of long-term follow-up studies) carry
`completion_date` values projected decades into the future (real example
found: entries through 2038), because most are `ESTIMATED`, not
`ACTUAL`. Taking the max would place the "event" in the future.

## Decision

For `approved`/`superseded`, event date = the **earliest `ACTUAL` (not
ESTIMATED) `primary_completion_date`** across the program's trials — the
first time a trial on this program actually finished its primary
endpoint, a real, historically-anchored CT.gov date. This is a proxy,
not a measurement: the real regulatory approval date is typically later
than primary completion, by an unknown and variable amount. It is,
however, anchored in real historical data rather than in when a human
happened to review a queue, which is the property that matters for a
model meant to generalize across the whole 2012+ dataset instead of
projecting recent labelling activity onto historical events.

Programs where no trial has an `ACTUAL` primary completion date are
right-censored at the panel end instead of guessing further.

## Consequences

- **`dead` is this model's only well-supported outcome.** Its event date
  (`public_confirmation_date`) is real, hand-verified ground truth for
  63 programs. `approved` (15) and `superseded` (3) both carry proxy
  dates and, in the case of `superseded`, too few events for any
  covariate to be fit at all (see `MIN_EVENTS_FOR_COVARIATES` in
  `discrete_time_survival.py` — those two classes fall back to an
  intercept-only hazard).
- Any report this model produces must disclose which outcome a number
  belongs to and, for `approved`/`superseded`, that the event timing is
  a proxy — never presented with the same confidence as the `dead`
  numbers.
- If a real approval/supersession date field is ever added to the gold
  schema, this proxy should be replaced, not layered on top of.
