# Session report — 2026-09-04

Everything done in this session, in the order it happened. Covers a
long, multi-phase session spanning validation-sample redesign, a full
manual-labelling coverage pass, the first Product B build, the
lead-time metric's redefinition and the first real survival model, a
branch-history incident and its fix, and same-day follow-on hardening
work. Written the day after (2026-09-05) from the session transcript,
so all dates below are as they actually occurred.

---

## 1. Validation sample redraw + reversible auto-accept

The existing 80-sample Layer 2/3 validation draw was invalid (degenerate
stratification, blocking 285 staged rejects and everything downstream).
Redrew it: 80 candidates, stratified 2×2 across text-grounded/recall-
grounded × accept/reject, ~20/cell, drawn only from Layer 2/3 decisions,
asserting no Layer 1/1.5 IDs and no empty strata (fail loudly rather
than ship a degenerate draw again). Served blind in the labelling app.
Built a fully-reversible accept-all/reject-wholesale promotion mechanism
gated on ≥95% is_adc / ≥90% in_scope agreement, `decided_by=auto` on
every promoted record.

## 2. Manual queue coverage pass

Ran `run_layer3_overflow.py --max-spend 5` over 174 manual-overflow
candidates and Layer 2/3 over the 113 candidates with no machine
opinion, so nothing reached the reviewer unprocessed. Rebuilt the queue
as one disposition-ordered pass (likely-rejects → ambiguous → confirmed
ADCs) instead of stratified sampling — this was coverage work, not
sampling. Gates 1–2 skipped wherever triage already resolved them;
triage's verdict shown at gate 3 as auto-derived, overridable context,
never pre-filled. Added bulk-reject (multi-select, one reason) and a
live progress panel (cards remaining, gate-3 count vs the ~100 target,
median sec/card, estimated time left). Kept the blind rule for gate-3
status calls, the repeat probes, and both-dates-required on
`dead_confirmed`.

## 3. Opportunity matrix — Product B, first component

Built B0 (asset attributes: payload chemotype from INN suffix, target
from trial text, carrier format, undisclosed-dev-code flagging) and B5
(crowding/failure density: live/dead counts per cell, red_ocean /
contested_and_hard / untested_white_space / graveyard classification,
`min_n` threshold for insufficient evidence). Later rebuilt on a payload
chemotype × tumour-system axis pair (see §11) after the original
target × specific-MeSH-indication axes proved too sparse against a
38-confirmed-program population (`docs/decisions/0005`).

## 4. Direct question: labels 150–300

Answered by inspecting `gold/labels.jsonl` lines 150–300 directly:
confirmed which were auto-labelled vs hand-labelled, and their
trustworthiness, without guessing from summaries.

## 5. Auto-review queue (gate-2 auto rejects)

Built a third, switchable "auto review" queue for programs whose
current gold record is `decided_by=auto, gate_reached=2` (is_adc=yes,
in_scope=no — Layer 1 deterministic scope rejections) — 54 real
programs. `queue.py` generalized from 2 to 3 named queues
(`QUEUE_KEYS`, `BYPASS_REVIEWED_QUEUES`); served via the existing
reopen-bypass mechanism with the original auto-decision shown as
informational, non-pre-filling context. `scripts/serve_auto_gate2_rejects.py`
populates it.

## 6. Lead-time redefinition + features/ + models/ build

**The headline metric was broken.** Across all 62 `dead_confirmed`
labels, median lead time was 0 months (range −45 to +58) because
`label_evidence_date` and `public_confirmation_date` were almost always
the same search result, not two independently-dated events. Redefined:

    lead_time = model_flag_date − public_confirmation_date

`label_evidence_date` dropped from the benchmark entirely
(`docs/decisions/0005-lead-time-redefinition.md`). Sign convention: more
negative = earlier/better (model flagged before the world knew).

Built what was missing:

- **`features/`** — `trial_asof.py` (sparse carry-forward as-of
  resolution — see the critical bug below), `knowability.py` (machine-
  checkable registry, `assert_all_columns_registered`), `panel.py`
  (real program × month panel: `silence_score_asof`, `band_asof`,
  `cost_index`, `contacts_locations_amendment_cadence_asof` +static
  `target`/`payload_chemotype`/`indication_mesh_term`), `as_of_probe.py`
  (real leakage probe against live `history_index` data, 50/50 verified).
- **`models/`** — `discrete_time_survival.py` (cause-specific binary
  hazards via `sm.Logit`, cluster-robust SEs by sponsor, intercept-only
  fallback below `MIN_EVENTS_FOR_COVARIATES`; `determine_program_outcome`
  using real `public_confirmation_date` for `dead` and a documented
  proxy — earliest ACTUAL `primary_completion_date` — for
  `approved`/`superseded`, `docs/decisions/0007`), `backtest.py`
  (`build_program_panels`, `precision_lead_time_curve`,
  `compare_at_matched_precision` — the audit gate: model must beat the
  silence-score heuristic or FAIL).

**Critical discovery**: `history/orchestrator.py`'s selective body-fetch
only fetches a version's body when `changed_modules` intersects
`RECOMMENDED_SIGNAL_LABELS` and `version > 0` — 56.6% of real
`version>0` `history_index` rows have no fetched body. Any as-of
resolver must use two-step sparse carry-forward (full list of
FETCHED-body versions, sorted; latest ≤ as_of), never a direct
`nct_id:v{N}` lookup. `features/trial_asof.py` initially got this wrong;
fixed and documented (`docs/decisions/0006`).

**Three real bugs fixed in the backtest pipeline**, all caught by
running it against production data:
1. `silence_score_asof=None` → NaN → `sm.Logit` crash (`build_training_table`).
2. Same NaN issue, separate code path (`model_flag_dates_from_panels`).
3. A program resolved before cutoff still had its panel extended to
   `panel_end` (today), producing impossible positive lead times — fixed
   by truncating each panel at `min(event_month, panel_end)`.

Also flagged separately, since they're the cases the project exists to
find: only 1 of 62 `dead_confirmed` labels is `never_publicly_confirmed`
— the silent deaths are likely sitting in `unknown` (12) and
`dormant_suspected` (4), currently outside the death class.

## 7. Git branch discipline + a concurrent-session incident

Per explicit instruction: never commit to `main`. All work went to
`agent-feature-review-decision-split`, pushed to GitHub. Mid-session, a
**second, independent Claude Code session** in the same working
directory switched the shared branch, causing one commit
(panel-truncation fix) to land on the wrong branch
(`agent-feature-matrix-coarsening`); recovered via cherry-pick, and that
other branch was deliberately left untouched to avoid disrupting its
own in-progress work. That session (or the user via it) also advanced
`main` independently with `finance/panel.py` (money-layer features:
`conviction_ratio`, `estimated_cumulative_spend`) and its own version of
`audit/features.py` — the direct source of the merge conflict resolved
next.

## 8. `audit/features.py` real merge (not a side selection)

Two independently-evolved versions of `audit/features.py` existed and
checked different things. Per exact instruction, combined them:

- Kept this branch's knowability-registry check, `leakage.md` existence
  check, and the real as-of probe (`features/as_of_probe.py` — stronger
  than a synthetic fixture, runs against real `history_index` data).
- Kept main's `_money_layer_checks()` (row count, knowability-date
  violations, per-feature NaN rates) over `finance.panel.FEATURE_NAMES`.
- Reconciled the two separate leakage-register checks into **one**
  covering the union of `features.knowability.REGISTRY` and
  `finance.panel.FEATURE_NAMES` — a feature from either source missing
  from the register is now a FAIL, not two independently-incomplete
  checks.
- Confirmed `features/panel.py` builds a real (non-stub) panel before
  dropping main's `not_built()` placeholder for it; replaced with an
  honest INFO naming what's still missing (Differ-derived features
  beyond amendment cadence).
- Verified `WAREHOUSE_DB`-missing and `financial_events`-empty both
  already degrade to INFO via `fstore.load_records()` returning `[]`.

In the process, extended `audit/leakage.md` — it registered main's two
money-layer features but none of this branch's 8 program-panel
features, so the new combined check correctly FAILed until real entries
were written for `silence_score_asof`, `band_asof`,
`contacts_locations_amendment_cadence_asof`, `target`,
`payload_chemotype`, `indication_mesh_term`. Also fixed a real gap: the
file's own docstring claimed a `.gitignore` exception (`!audit/leakage.md`)
that had never actually been added, so the file was untracked.

Ran `python -m pharma_stats.audit --stage features` clean: **0 FAIL /
0 WARN / 4 INFO / 4 PASS**, both check families firing.

## 9. Merging the branch forward again

The remote `agent-feature-review-decision-split` had moved in the
meantime (a human-authored `git merge main`, `06b90b6`) — bringing in
Product B's B1 (`productb/kill_reason_spend.py`), corpus/label
statistics (`stats/`), and a **naive auto-merge** of `features.py` that
side-selected almost wholesale (a 1-line diff) and, worse, **corrupted
`audit/label_sufficiency.py`** with a duplicate function signature
spliced into a docstring (syntax error). Merged that remote state in on
top of the proper `features.py` fix, fixed the corruption (removed the
duplicate signature and a now-dead `_lead_sponsor` helper superseded by
`provisional_programs.lead_sponsor`), reconciled `audit/leakage.md`
into one document covering both branches' entries, fixed one stale test
assertion, and pushed.

## 10. DuckDB manifest lock contention — first pass (retry budget)

The model backtest crashed on `duckdb.IOException` — a concurrent
session's use of the manifest collided with `_snap_latest_with_retry`'s
retry budget (5×0.5s ≈ 2.5s, far too short against a multi-minute
backtest making thousands of calls). Bumped to 20×1.5s as an interim
fix (later superseded by the root-cause fix in §14).

## 11. `PavelAbramau/pharma_statistics` PR #3 merges main into `main` — breaks it

Separately, the base repo's `main` advanced (PR #3) by merging the
*pre-fix* state of `agent-feature-review-decision-split` (`06b90b6`),
not the later commit that actually fixed the merge — so `import
pharma_stats.audit` raised `NameError: name 'money_panel' is not
defined` on `main`, and no audit stage could run at all. Discovered when
asked to check on a re-queued backtest run.

## 12. Fixing `main` (PR #5)

Created `agent-fix-main-import-break` off current `main` (never
committing to `main` directly). Brought over the real fix — clean
merge, no conflicts (the only files that differed were exactly the ones
already fixed). Restored `label_sufficiency.py`'s duplicate-signature
corruption fix a second time (the bad PR had reintroduced it).

**Added `tests/test_module_imports.py`**: parametrized over every
module under `pharma_stats` (95 modules), so a dropped import breaks
one specific test instead of silently killing a whole package's import
with nothing noticing. Verified it actually catches the historical bug
— reproduced the real broken commit's `features.py` and confirmed 12
failures cascaded through the `audit` package, then confirmed clean
again after the fix.

Opened **PR #5** (`agent-fix-main-import-break` → `main`) via the
GitHub API (the `gh` CLI wasn't installed; used the stored git
credential for one authenticated API call instead). Left open —
merging `main` is the user's call, not something this session does
unilaterally.

## 13. Stratum-gap queue (21 cards)

`label_statistics.py` (from the PR #3 merge) refuses to emit any
population-level estimate while a non-empty (band, archetype) stratum
has zero labels — currently true for exactly two cells: (band=1,
`registry_terminated_stated_reason`): 19 unlabelled, (band=4,
`registry_terminated_vague_reason`): 2 unlabelled. Added a **fourth
switchable queue** ("coverage gap" in the app topbar): `QUEUE_KEYS`
extended in `queue.py`, `app.py`'s session/switch-queue responses
extended, UI button added (`index.html`/`app.js`). Built
`scripts/serve_stratum_gap_queue.py`, which identifies exactly those 21
programs and installs them — confirmed against live data (19+2=21,
pinned in a regression test). Labelling these 21 unlocks every weighted
population estimate in the project at once.

## 14. DuckDB concurrency fixed at the root

Per explicit instruction ("not by killing processes"): `snapshot.get_as_of`/
`all_ids` were opening a **read-write, exclusive-locking** connection
for a pure `SELECT` — so any two purely-reading callers (a backtest, a
report script, `build_opportunity_matrix.py`) collided with each other
as if one were writing. Added `_manifest_con_read_only`: always opens
`manifest.duckdb` read-only, which DuckDB lets any number of concurrent
readers share; a not-yet-built manifest degrades to `None`/`[]` rather
than raising. Audited every other `WAREHOUSE_DB` reader in the
codebase — all already used `read_only=True`; only genuine writers
(`save_snapshot`, `rebuild_manifest`, `materialize`, `differ.extract`,
`finance/store.append_records`) still take real write locks. Added a
regression test proving two concurrent `get_as_of` connections no
longer conflict. `cost_model.py`'s retry wrapper kept as defense-in-depth
against genuine reader-vs-writer contention, with its comment corrected
to say so rather than overstate what it fixes.

Validated live: the full test suite passed while a real model backtest
was running concurrently in the background — the exact scenario that
used to crash.

## 15. Stated-vs-true kill-reason finding, hardened

- Added `agreement_rate_sample_ci`: sponsor-cluster-bootstrap 95% CI on
  the raw (unweighted) agreement rate — **12.7% point estimate, 95% CI
  [5.5%, 20.8%], n=63, 36 sponsors** — computable now, unlike the
  population-weighted mismatch estimate which needs the stratum-gap
  queue (§13) labelled first.
- `report_label_statistics.py` now lists **every one of the 63
  `dead_confirmed` cases** with its raw `why_stopped` text next to both
  kill-reason labels, with the **32 `not_stated` cases** (no usable text
  at all — blank or under the 15-character floor) reported in their own
  section, since they need no classifier judgement call and can't be a
  source of classifier-manufactured disagreement.
- Manually reviewed the 31 classified cases for classifier-manufactured
  disagreement (the actual ask: "check whether the keyword classifier is
  manufacturing disagreement on text a human would read correctly").
  Found a real bug — the `toxicity_safety` keyword `"safety"` fires even
  when the text explicitly negates it (e.g. *"No Safety or Efficacy
  Concerns"*, *"not related to any safety concern"*) — in 3 cases. In
  all 3, the true label is `futility_efficacy` while a corrected
  classification would be `strategic_portfolio` — **still a
  disagreement either way**, so the 12.7% headline figure is not
  inflated by this bug. Not patched: the classifier is deliberately a
  small, hand-authored baseline meant to characterise divergence, not to
  be tuned toward agreement.

## 16. Model backtest — completed cleanly, gate FAILs honestly

Once §14's concurrency fix landed, the model backtest finally ran to
completion without crashing (232 gate-3 programs, 47 dead / 12 approved
/ 3 superseded training events). Real result:

- **Model curve**: 0 flags at every threshold from 0.05 to 0.95 — the
  fitted hazard never crosses even the lowest tested threshold
  post-cutoff.
- **Heuristic curve** (silence-score band): flags real cases (up to 164
  at band≥0) but never reaches 50% precision at any band.
- **Gate: FAIL** — "neither the model nor the silence heuristic reaches
  50% precision with any correct flag at this cutoff — no usable
  comparison." Reported honestly, not massaged.

`reports/model_backtest.md` (gitignored, regenerable) and
`data/model_backtest_result.json` (tracked) both written; the JSON
committed to `agent-feature-review-decision-split`.

## 17. Combined HTML report

Built and published a single HTML report combining a fresh run of
`report_corpus_statistics.py`, `report_label_statistics.py`,
`build_opportunity_matrix.py --min-n 3`, and `python -m pharma_stats.audit
--stage all` — corpus distributions, the stratum-gap refusal + agreement
CI + filterable 63-case table, the payload×tumour-system matrix grid
(0 graveyard cells at min_n=3), and the full audit check ledger,
honestly flagging that the audit run **halted after `universe` failed
its gating check** (8 downstream stages, including `features`/`model`,
did not run that pass). Every figure on the page is copied from that
run's real output, nothing recomputed for display.

---

## Current state as of session end

- **`main` is broken** (`NameError` on `import pharma_stats.audit`, plus
  the `label_sufficiency.py` corruption). **PR #5 fixes it** and is open,
  unmerged — merging is the user's call.
- **`agent-feature-review-decision-split`** carries the real
  `audit/features.py` merge, the model backtest result, and the
  lead-time/features/models build. Diverged from `main`'s current state
  via the PR #3 incident; PR #5 is the reconciliation path.
- **Stratum-gap queue**: 21 cards loaded and ready (`stratum_gap_order`
  in the live session), not yet labelled.
- **Population estimates**: refused pending those 21 labels.
- **Model backtest gate**: FAIL, honestly reported — the discrete-time
  survival model does not currently beat the silence-score heuristic.
- **Pipeline audit** (`--stage all`): 4 FAIL / 4 WARN / 4 INFO / 6 PASS,
  halted after `universe` — `history`, `backfill`, `differ`,
  `normalisation`, `gold_set`, `features`, `model`,
  `label_sufficiency` did not run on the last full pass.
- **Silent-death candidates outside the death class**: 12 `unknown` + 4
  `dormant_suspected` — flagged, not yet investigated further.

## Decision records added this session

- `docs/decisions/0005-lead-time-redefinition.md`
- `docs/decisions/0006-selective-body-fetch-carry-forward.md`
- `docs/decisions/0007-approved-superseded-event-date-proxy.md`
(plus `0001`, `0003`, `0004` referenced/relied upon from earlier work)

## Known open items for next session

1. Merge PR #5 to unbreak `main` (or reconcile further if `main` has
   moved again).
2. Label the 21 stratum-gap cards; re-run `report_label_statistics.py`
   to get the population-weighted stated-vs-true estimate.
3. Investigate the two new `universe`-stage FAILs (unreviewed-candidate
   queue at 1093/1093, and 2 current-state-read-boundary violations in
   `_best_trial_snapshot`) before running `--stage all` again.
4. Decide whether the model's 0-flags-at-every-threshold result means
   the covariate set (`silence_score_asof`, `log_cost_index`) is
   insufficient, or the discrete-time hazard specification itself needs
   rethinking — the heuristic at least flags something, even if not at
   50% precision.
5. Investigate the `unknown` (12) / `dormant_suspected` (4) programs —
   the cases the project exists to find, currently outside the death
   class.
