# 0003 — Synthetic program cost index: benchmark and construction

Date: 2026-09-02

## Context

The financial layer's first component (`pharma_stats.finance.cost_model`)
needs a cost estimate for every program, not just the small minority with
a real, disclosed budget. Enrollment, phase, and trial start date are
already resolvable per-trial through the time-cut versioned-history
path; site count and duration require additional handling (see below).
No per-patient-per-month rate exists in the published literature broken
out by phase and therapeutic area — the closest real, citable source
publishes only average *total study cost* by phase, for oncology
specifically.

## Benchmark

Sertkaya A, Wong HH, Jessup A, Beleche T. "Key cost drivers of
pharmaceutical clinical trials in the United States." *Clinical Trials*.
2016;13(2):117-126. Average total oncology study cost by phase:

| Phase | Average cost |
|---|---|
| 1 | $4.5M |
| 2 | $11.2M |
| 3 | $22.1M |

Verified via live web search + fetch on 2026-09-02, not recalled from
training. A related, earlier ASPE-commissioned report by an overlapping
author group (Sertkaya et al., "Examination of Clinical Trial Costs and
Barriers for Drug Development," July 2014) covers similar ground but was
checked directly and does **not** publish itemized per-patient figures
by therapeutic area — its Phase 3 oncology total ($22.1M) matches the
2016 figure, consistent with a shared underlying dataset, but per-patient
breakdowns aren't in either public report. A single blended (not
phase-stratified) oncology per-patient figure of $59,500 is also
reported (PhRMA, via the same search), and a general (not oncology-
specific) per-pivotal-trial median of ~$19M is reported in Moore et al.,
JAMA Internal Medicine, 2020 (138 FDA pivotal trials, 2015-2016
approvals) — both used only as informal cross-checks, not inputs to the
formula below.

## Decision

**No per-patient rate is backed out of these totals.** Doing so would
require assuming an average enrollment and duration for "the" typical
oncology trial per phase — a second, unbenchmarked estimate stacked on
top of the first, and exactly the kind of compounding assumption that
turns a relative index into something that looks more precise than it
is. Instead, the Sertkaya phase totals are used as **relative weights
only** (`cost_model.PHASE_COST_WEIGHT` — the three numbers above,
directly, not derived): a Phase 3 program is weighted ~4.9x a Phase 1
program of otherwise-identical enrollment and duration, matching the
benchmark's own relative cost structure, without claiming the weight
itself is a real per-patient dollar rate.

The full formula: `phase_weight[phase] x enrollment x elapsed_months`,
summed across a program's trials. `enrollment`, `phase`, and trial start
date (for `elapsed_months`) come directly from this project's own
time-cut versioned-history data — real observations, not benchmarked
proxies. Phase 4 has no published oncology figure in this source and is
treated as phase-3-equivalent (post-marketing oncology studies are
typically large registries) — an explicit, flagged assumption, not a
benchmark.

**Site count is excluded from the time-varying series.** Confirmed by
inspecting real snapshots on disk: `contactsLocationsModule` (CT.gov's
location list) exists only on the current-state fetch
(`raw/ctgov/2026-09-01/NCT04717414.json` has 185 locations;
`raw/ctgov/2026-08-22/NCT04717414:v48.json`, a versioned-history body,
has none at all). Per `docs/decisions/0001`, a current-state field is
only safe to read when it's a genuinely static, universe-membership
property — site count isn't; it grows as a trial recruits, so reading
today's count into a historical month's index would leak forward-looking
information into every past month, exactly the failure mode 0001 exists
to prevent. Site count is therefore exposed only as a **static per-
program multiplier** (`cost_model.site_count_factor`, applied in
`program_cost_index_snapshot`) for present-day cross-program ranking —
never inside `monthly_cost_index_series`, which is the time-cut-safe
feed registered in `audit/leakage.md`. The site-count scaling itself
(`n_sites ** 0.5`, sublinear — a fixed per-site setup/monitoring
overhead, not a duplicated trial) is a modeling convention, not a
benchmarked figure; it is documented as such in `cost_model.py` so it's
never mistaken for a cited number.

## Consequences

- The index is explicitly a relative ranking signal, never a dollar
  estimate — CLAUDE.md's controlled-vocabulary discipline and this
  project's whole "ranking, not the absolute figure" framing both apply.
- If the real five-entity Program model and OncoTree indication
  normalisation land later, the conviction ratio's peer grouping
  (`finance.conviction.peer_group_key`, currently phase + coarse
  heme/solid `scope_category`) should be revisited — it's a documented
  stand-in for "phase and indication," not the real thing.
- Any future contributor adding a new current-state-derived factor to
  this module must show it's static (like 0001's three fields) or keep
  it out of the monthly series, the same rule 0001 already established.
