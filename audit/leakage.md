# Leakage registry

Every time-varying feature that will ever be fed into a time-cut
backtest gets one entry here, stating exactly what it's allowed to know
as of the date it describes. This is the enforcement ledger CLAUDE.md's
`snapshot.get_as_of` discipline and `docs/decisions/0001` argue for —
before a feature is used in a backtest, its knowability-date contract
must be written down here, not just implied by the code that computes
it. New entries go at the top.

---

## `synthetic_cost_index_monthly`

- **Module**: `pharma_stats.finance.cost_model.monthly_cost_index_series`
- **Store**: `financial_events` (`event_type="synthetic_cost_index_monthly"`)
- **Knowability date**: the month the point describes (`event_date` ==
  the `as_of` month). Every input (enrollment, phase, trial start date)
  is resolved through `resolve_trial_state_as_of`, which only reads
  versioned-history snapshots with `posted_date <= as_of` — never a
  later or current-state snapshot. Verified by test:
  `tests/test_cost_model.py::test_resolve_trial_state_as_of_never_reads_current_state_fetch`.
- **Known exclusion, not a violation**: site count is deliberately NOT a
  factor in this series (see `docs/decisions/0003`) — it only exists via
  a current-state fetch, which has no "as of." A separate,
  non-time-varying `program_cost_index_snapshot` includes it for
  present-day ranking only; that function must never be called with a
  historical `as_of` date and fed into a backtest.
- **Do not use before**: the program's earliest indexed trial version's
  `posted_date` — the series doesn't extend before that, by construction
  (`monthly_cost_index_series` starts at `min(posted_date)` across a
  program's trials).

## `conviction_ratio_monthly`

- **Module**: `pharma_stats.finance.conviction.compute_conviction_ratios`,
  called per-month in `scripts/build_financial_layer_cost_index.py` against
  that same month's peer cost indices (not each peer's current value).
- **Store**: `financial_events` (`event_type="conviction_ratio_monthly"`)
- **Knowability date**: the month it describes, same as the cost index
  it's built from — every peer's value going into that month's median is
  itself resolved as-of that same month, so the ratio doesn't leak a
  peer's later (or current) spend into an earlier month.
- **Peer-grouping caveat**: peer group is (highest phase, coarse
  heme/solid `scope_category`) — a stand-in for "phase and indication"
  until the real OncoTree normalisation exists (see `docs/decisions/0003`).
  This doesn't create a leakage risk (the grouping key itself is
  time-invariant per program in the current implementation), but it does
  mean two programs in different real indications can share a peer group
  today — a precision limitation, not a knowability one.
- **Do not use** for a program-month with `n_peers < 2` — `None` in that
  case, filtered out before being written (see `conviction_ratio`'s own
  refusal rule), so a ratio's mere presence in `financial_events` already
  implies a real, sized peer group.

## `conviction_ratio` (feature panel)

- **Module**: `pharma_stats.finance.panel.build_money_layer_panel`
- **Formula**: `conviction_ratio_monthly` exposed as-is for the months it
  exists; the feature panel adds no aggregation on top of it.
- **Knowability date**: `knowability_date == as_of` == the underlying
  event's `event_date`. No leakage risk beyond the raw event's own
  contract above.
- **Missingness**: `None` for any (program, month) with no usable peer
  denominator that month (see above) — never imputed as 1.0 or 0.

## `estimated_cumulative_spend` (feature panel)

- **Module**: `pharma_stats.finance.panel.build_money_layer_panel`, as a
  running sum of `synthetic_cost_index_monthly` ordered by `event_date`,
  one program at a time.
- **Formula**: for a row at month M, `sum(value for (d, value) in the
  program's cost points if d <= M)`.
- **Knowability date**: `knowability_date == as_of == M`. The sum only
  ever includes points whose own `event_date <= M`, and each of those
  points is itself knowable as of its own `event_date` (see
  `synthetic_cost_index_monthly` above) — so no term in the sum is
  knowable later than M, and neither is the sum.
- **Missingness**: `None` before a program's first indexed cost point.
  `finance.panel.value_as_of` forward-fills the latest known cumulative
  value onto any later `as_of` a caller resolves against — a legitimate
  "still true as of this later date" read, never a leak.
- **Never a dollar estimate.** Same caveat as the underlying cost index
  (`docs/decisions/0003`): relative ranking / trend signal only.

## Program x month panel (`features/panel.py`, `features/knowability.py`)

Every column `features.panel.build_program_month_panel` can emit is also
registered machine-checkably in `features.knowability.REGISTRY` —
`assert_all_columns_registered` is called before the panel returns a row,
so a new column can't reach a model without an entry in both places.

- **`silence_score_asof`** (time-varying) — the panel month itself; every
  input trial field is resolved via
  `features.trial_asof.resolve_trial_summary_as_of`, which only reads
  versioned-history bodies with `posted_date <= that month` (never a
  current-state read, and never a direct `nct_id:v{N}` lookup — see
  `docs/decisions/0006`'s sparse carry-forward requirement).
- **`band_asof`** (time-varying) — same as `silence_score_asof`, since
  it's derived from it.
- **`cost_index`** (time-varying) — the panel month itself; see
  `docs/decisions/0003` and `finance/cost_model.py`. Deliberately
  excludes site count (current-state-only, not knowable as of a
  historical month).
- **`contacts_locations_amendment_cadence_asof`** (time-varying) — the
  panel month itself; counts only history rows with `posted_date <= that
  month`, truncated the same way as every other field in
  `features/trial_asof.py`.
- **`target`** (static) — resolved once from the antibody-stem
  dictionary / trial text / name, never re-resolved per month, never
  guessed ahead of the normalisation step (`attributes/target.py`).
- **`payload_chemotype`** (static) — resolved once from the INN suffix
  (`attributes/payload.py`).
- **`indication_mesh_term`** (static) — a static universe-membership
  property under `docs/decisions/0001`'s existing exception (disease
  category doesn't change over a program's life); current-state fetch
  only, never resolved historically.

## Leakage probe (as-of, real data)

`features.as_of_probe.run_asof_probe` — not a synthetic fixture check.
Samples real trials with >=2 FETCHED versions from the live
`history_index`, resolves each one as of its own earliest fetched
version's date, and asserts that resolution never returns a later
version's state. Run automatically as part of `audit.features`; verified
50/50 on production data as of 2026-09-04.

## See also

`docs/decisions/0001-current-state-fetch-scope.md` — the general current-
state-vs-versioned-history discipline this registry's entries all lean
on. `docs/decisions/0003-synthetic-cost-benchmark.md` — the cost index's
full construction and the site-count exclusion this file's first entry
summarizes. `docs/decisions/0004-spend-is-not-quality.md` — why neither
feature here should be read as a causal claim about asset quality even
though both are genuinely leakage-safe as predictive features.
