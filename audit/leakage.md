# Leakage register

Every time-varying feature that reaches the feature panel must be
registered here with a knowability-date contract before
`pharma_stats.audit.features` will pass it — see CLAUDE.md and
`src/pharma_stats/audit/features.py`. A feature not listed here is a FAIL,
not an oversight to fix later.

Each entry states: what the feature is, where it's computed, what
`knowability_date` means for it, and why that date can never be later
than the `as_of` a caller asks for. This file is hand-authored and
committed (it is the one exception to `audit/*.md` in `.gitignore`, which
otherwise covers the harness's own timestamped, generated reports).

## `synthetic_cost_index_monthly` (raw event, `financial_events` table)

- **Computed by:** `pharma_stats.finance.cost_model.monthly_cost_index_series`,
  written by `scripts/build_financial_layer_cost_index.py`.
- **Formula:** `phase_weight[phase] x enrollment x elapsed_months`, summed
  across a program's trials, for one calendar month.
- **Knowability date:** `event_date` = the month itself. Every factor
  (enrollment, phase, trial start date) is resolved through the
  time-cut versioned-history path only (`resolve_trial_state_as_of`),
  using the highest indexed trial version with `posted_date <= that
  month` — never a current-state read. Site count is deliberately
  excluded from this series (see `docs/decisions/0003`); it is a
  current-state-only field and would leak today's final count into every
  past month.

## `conviction_ratio_monthly` (raw event, `financial_events` table)

- **Computed by:** `pharma_stats.finance.conviction.compute_conviction_ratios`,
  written by `scripts/build_financial_layer_cost_index.py`.
- **Formula:** a program's `synthetic_cost_index_monthly` for month M,
  divided by the median of its peer group's (`highest_phase`,
  `scope_category`) `synthetic_cost_index_monthly` for the SAME month M.
  `None` (not written) when fewer than 2 usable peers exist that month.
- **Knowability date:** `event_date` = month M. The peer comparison uses
  only same-month peer values, each itself knowable as of M by the
  contract above — never a peer's later or current-day spend.

## `conviction_ratio` (feature panel)

- **Computed by:** `pharma_stats.finance.panel.build_money_layer_panel`.
- **Formula:** `conviction_ratio_monthly` exposed as-is for the months it
  exists; the feature panel adds no aggregation on top of it.
- **Knowability date:** `knowability_date == as_of` == the underlying
  event's `event_date`. No leakage risk beyond the raw event's own
  contract above.
- **Missingness:** `None` for any (program, month) with no usable peer
  denominator that month (see above) — never imputed as 1.0 or 0.

## `estimated_cumulative_spend` (feature panel)

- **Computed by:** `pharma_stats.finance.panel.build_money_layer_panel`,
  as a running sum of `synthetic_cost_index_monthly` ordered by
  `event_date`, one program at a time.
- **Formula:** for a row at month M, `sum(value for (d, value) in the
  program's cost points if d <= M)`.
- **Knowability date:** `knowability_date == as_of == M`. The sum only
  ever includes points whose own `event_date <= M`, and each of those
  points is itself knowable as of its own `event_date` (see
  `synthetic_cost_index_monthly` above) — so no term in the sum is
  knowable later than M, and neither is the sum. This holds regardless of
  how many points are summed; it is not an approximation.
- **Missingness:** `None` before a program's first indexed cost point.
  `pharma_stats.finance.panel.value_as_of` forward-fills the latest known
  cumulative value onto any later `as_of` a caller resolves against — a
  legitimate "still true as of this later date" read, never a leak,
  since it never uses a point later than the value it fills forward.
- **Never a dollar estimate.** Same caveat as the underlying cost index
  (`docs/decisions/0003`): relative ranking / trend signal only.

## Downstream consumer: kill reason vs. spend (`productb/kill_reason_spend.py`)

Resolves both features `value_as_of` each dead_confirmed program's own
`label_evidence_date` — never a later date, and never the program's final
(most-recent-known) value when that postdates the kill label. Per
`docs/decisions/0004`, reads the result descriptively (spend at death,
broken out by stated kill reason), never causally.

## Not yet registered (blocks nothing above, listed for visibility)

- Sponsor-level distress/impairment signals (`rd_expense_quarterly`,
  `distress_signal`, `market_cap_snapshot`, `ev_share_estimate`,
  `iprd_impairment_charge` — see `finance/store.EVENT_TYPES`) are defined
  in the schema but not yet populated or consumed by any feature; they'll
  need their own entries here before any feature panel may use them.
- Everything gated on the five-entity warehouse / controlled-vocab
  normalisation (structural silence features, staleness monotonicity,
  Differ-derived features) — see `src/pharma_stats/audit/features.py`.
