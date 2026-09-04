# 0006 — Selective body-fetch means as-of resolution must carry forward

Date: 2026-09-04

## Context

Building `features/trial_asof.py` (a general as-of TrialSummary resolver
for the feature panel), a first draft looked up the exact CT.gov version
implied by `history_index` metadata (`max(version) WHERE posted_date <=
as_of`) and fetched that version's body directly. Run against real data,
this failed 50/50 on a probe checking "resolving a trial as of its
earliest version's date returns that version, not a later one" — every
single case returned `None` instead.

Root cause, confirmed at scale: `history/orchestrator.py`'s backfill only
fetches a version's body when (a) `version > 0` (version 0 — the original
registration — is never separately fetched by policy) and (b) its
`changed_modules` intersects the signal labels (`Study Design`, `Outcome
Measures`, `Study Status`, `Sponsor/Collaborators`, `Arms and
Interventions`) — a deliberate, documented cost-saving design (the
history index itself is cheap; a version whose only changes are cosmetic
carries no signal, so its body is never worth fetching). Sampled 500
`version > 0` rows: **283 (56.6%) have no fetched body at all.**

An as-of resolver that requires the *exact* metadata-implied version to
have a body will return `None` for the majority of dates, even when a
perfectly valid earlier body is on disk and — by the fetch policy's own
logic — still accurately describes the trial's state (nothing tracked
changed in between).

## Decision

As-of resolution must use a two-step **sparse carry-forward** pattern,
not a direct version lookup:

1. Build the list of every version that actually has a fetched body,
   sorted by `posted_date`.
2. For a query date, pick the *latest entry at or before* that date —
   never require an exact version match.

`finance/cost_model.py`'s `trial_version_history` already did this
correctly (confirmed by re-reading it during this investigation — the
already-shipped financial layer output, 154,375 `financial_events` rows,
is unaffected). `features/trial_asof.py`'s `resolve_trial_summary_as_of`
was rewritten to the same pattern (`_fetched_version_states` +
latest-at-or-before selection) before it was ever used for anything.

`features/as_of_probe.py` was corrected to match: it must test
resolution at a trial's earliest **fetched** version's date, not its
earliest **indexed** version's date — asking about a date before any
body was ever fetched has no answer (correctly `None`, an honest gap,
not a leak) and isn't the property the probe exists to check.

## Consequences

- Any future as-of resolver in this project must use the sparse
  carry-forward pattern, never a direct `nct_id:v{N}` lookup keyed
  purely on `history_index` metadata.
- A trial's coverage before its first fetched-body version is genuinely
  unknown (contributes `None`/0 to any feature, not a guessed value) —
  this under-counts rather than leaks, the same safe-failure direction
  `history_coverage` already treats as acceptable elsewhere in this
  project.
- `audit/features.py`'s as-of probe now runs this check against real
  data on every audit pass (50-trial sample), not just a synthetic
  fixture — a regression here will show up as a real FAIL, not just a
  passing unit test on made-up data.
