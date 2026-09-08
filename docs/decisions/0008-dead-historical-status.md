# 0008 — Matrix-only universe extension and the `dead_historical` status

Date: 2026-09-08

## Context

The B5 opportunity matrix's coarse (payload chemotype x tumour system)
view is thin by construction (docs/decisions/0005): ~230-350 in-scope
programs spread across ~80 cells. Two real populations sit just outside
CLAUDE.md's locked scope and would deepen those cells without widening
the detector's own remit:

- ADC trials that started before the 2012-01-01 scope floor.
- ADC trials in haematological indications (leukemia/lymphoma/myeloma),
  excluded by the heme/solid split (CLAUDE.md: "solid tumours only").

Both populations are real, public, industry-sponsored ADC programs — not
a different modality and not preclinical data (both stay explicitly
excluded). Extending the matrix's *density tables* backward in time and
sideways in tumour type, without extending the *detector's* scope,
requires a population that never touches gold or the backtest.

## Decision

**A separate, matrix-only cohort, discovered independently
(discovery/matrix_extension.py) and never written to
`gold/labels.jsonl`, `provisional_programs`, or `asset_candidates`.**
Locked-scope dimensions this cohort keeps: ADCs only (Layer-1
`evaluate_is_adc`), industry sponsors only. Dimensions relaxed: start
date (2000-01-01 floor instead of 2012-01-01) and tumour type (heme
included). Discovery reuses only the pattern-matching strategy from
`discovery/candidates.py` (not seed/sponsor expansion) — a bounded,
lower-cost pass appropriate to a lower-priority, matrix-only population.

**No manual labelling, ever, for this cohort.** A deterministic rule
(`labelling/dead_historical.py`) assigns `dead_historical` instead of a
human gate-3 label:

1. No trial started in the last 10 years.
2. Not on the small, hand-curated `discovery/known_approved_adcs.py` list.
3. No successor: no other program from the same sponsor, on the same
   resolved target, with a later first-trial-start-date (checked against
   both this cohort and the main materialized universe).

A program failing any leg is simply left **not classified** — this rule
never guesses a different status (live, approved, superseded) for a
program it can't confidently call historical.

**`dead_historical` is a distinct, lower-confidence status, never merged
with `dead_confirmed`.** `attributes/matrix.py`'s `Cell` carries it in its
own list (`dead_historical_programs` / `n_dead_historical`), separate from
`dead_programs` / `n_dead`; every rendered table shows them as separate
columns, and `n_live`/`n_dead`/quadrant classification are computed
exactly as before — merging this cohort in can only ADD cells or ADD a
count to an existing cell's `dead_historical` column, never change an
existing `dead`/`live` count.

**Tumour-system/heme classification is a text heuristic, not MeSH.** The
main matrix's `attributes/tumour_system.py` needs a current-state CT.gov
fetch per trial (`conditionBrowseModule`), which this bounded, lower-
priority pass does not do. `discovery/heme_scope_text.py` classifies by
keyword match on the raw `conditionsModule.conditions` strings the basic
search API already returns — lower confidence, by design, and never used
for the main gold-quality population.

**The fine (target x specific MeSH indication) matrix view is NOT
extended.** That axis has no non-MeSH proxy this pass can supply
cheaply; extending it would require the same current-state-fetch cost
the coarse view's heme/date relaxation was specifically designed to
avoid. `scripts/build_adc_indication_table.py` still regenerates both
views on every run; only the coarse view gains the extension cohort.

## Consequences

- Every report surfacing this cohort must say, explicitly, that its
  outcomes are inferred by a deterministic rule, never labelled — the
  same "never causal, always descriptive" discipline docs/decisions/0004
  established for the money layer applies here to "never certain."
- `discovery/known_approved_adcs.py` is a small, hand-curated fact list,
  not a live regulatory feed — extend it by hand; an omission there is a
  real risk of over-classifying a successful compound as dead_historical.
- The successor check's target resolution (`matrix_extension.
  build_population_index`) uses only `attributes.target`'s offline
  dictionary tiers against the main universe, not the fine matrix view's
  full trial-text resolution — a real target can go unresolved and
  silently drop that main-universe program out of the successor index,
  biasing the rule toward MISSING a real successor (dead_historical
  over-firing) rather than under-firing.
- If OncoTree/MeSH normalisation ever gets cheap enough to run at this
  cohort's scale, the fine view could be extended the same way and this
  decision revisited — not planned now.
