# 0005 — Lead-time redefinition: drop label_evidence_date

Date: 2026-09-04

## Context

The project's headline metric is lead time: how much earlier a program's
silent death could be flagged than when it becomes public knowledge. The
original computation (`audit/label_sufficiency.py`) used
`public_confirmation_date - label_evidence_date`, both hand-entered at
Gate 3.

Checked against the real 63 `dead_confirmed` labels on file: median lead
time was 5 days (~0 months), range −1375 to +1764 days. That spread is
not a real signal — it's an artifact of the labelling process.
`label_evidence_date` was entered as "the date of the evidence I found,"
which in practice was almost always the same search result as
`public_confirmation_date` — the reviewer finds the announcement (a press
release, a filing) and records the registry evidence sitting right next
to it, dated the same week. Two records have the exact same value for
both fields; the rest cluster near zero with a long, symmetric-looking
tail in both directions — the signature of noise, not a measured
distance between two independently-dated events.

## Decision

`label_evidence_date` is dropped from every benchmark/lead-time
computation, effective immediately. It remains on the gold schema and
in `gold/labels.jsonl` (append-only — nothing is deleted or edited) as a
provenance field (which registry snapshot the reviewer was looking at),
but it is never again read as one side of a lead-time measurement.

Lead time is redefined as:

    lead_time = model_flag_date - public_confirmation_date

`public_confirmation_date` (63 real, hand-verified ground-truth dates)
stays the trusted anchor. `model_flag_date` is a genuine, independent
second date — the first period a model (not a human search) assigns this
program a hazard/probability of `dead` above a chosen threshold, computed
strictly from information knowable as of that period (see
`features/`'s knowability discipline and `audit/leakage.md`). A positive
lead time now means something real: the model raised the flag before the
world found out. Never `label_evidence_date` again for this purpose —
it was never a second event, just a second write of the same one.

## Consequences

- `audit/label_sufficiency.py`'s bootstrap (sponsor-cluster CI on lead
  time) needs its input redefined the same way once `models/` produces
  `model_flag_date` — tracked as follow-up work, not done in this
  decision.
- Any earlier report, chart, or claim built on the old lead-time
  definition (including this project's own prior session output) should
  be treated as void, not merely imprecise — the old number measured
  reviewer behaviour, not registry silence.
- `label_evidence_date` is not deleted from history (append-only) and
  remains useful for exactly what it always was — an audit trail of what
  the reviewer looked at — just never again a timestamp compared against
  `public_confirmation_date`.
