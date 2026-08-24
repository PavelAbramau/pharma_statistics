# pharma_statistics

## Project goal

Detect silently-discontinued antibody-drug conjugate (ADC) oncology programs
from public data. Most ADC program deaths are never announced; they show up
only as a status field that stops updating, a registry record that goes
stale, or an enrollment target quietly cut in a trial amendment. The project
builds a provenance-tracked evidence base to detect that pattern and, later,
to measure the gap between when a program actually died and when (if ever)
that became public.

## Locked scope

- **ADCs only.** No other modalities.
- **Solid tumours only.**
- **Industry sponsors only.**
- **2012–present.**
- Expected universe: ~350–450 assets, ~1,500–2,500 trials, on the order of
  25,000 trial-version records. This is small data — it runs on a laptop.
  Do not design for a bigger universe than this.

Do not expand scope (modality, tumour type, sponsor class, date range)
without the user explicitly widening it.

## Stack constraints

DuckDB over Parquet, with raw JSON on disk as the source of truth. No
Postgres, no cloud, no ORM. Keep it that way — the data volume never
justifies more infrastructure than this.

## The five-entity model

- **Asset** — a drug candidate. Payload chemotype, linker chemistry, DAR,
  target antigen (HGNC symbol), carrier format, originator, current owner,
  first-in-human date. Owner is a dated interval, not a scalar field —
  assets change hands (licensing, M&A) and the right sponsor must be
  attributable at any point in time.
- **Program — the unit of analysis.** Keyed as **asset × OncoTree indication
  code × line-of-therapy bucket**. Never analyse at the trial level; always
  roll up to program. A single trial can span multiple programs (a basket
  trial covering several indications splits into one program per
  indication); a single program is typically covered by multiple trials
  over time.
- **Trial** — a registry record (CT.gov study). Evidence for one or more
  programs, never itself the analysis unit.
- **EvidenceEvent** — a typed, dated, signed row describing something that
  changed: a trial-record amendment (enrollment cut, status change,
  endpoint downgrade, arm/cohort removal, completion date pushed, sponsor
  change, verification lapse, phase change), a hand-applied label, or any
  later external-source signal. Every event carries a pointer back to the
  raw snapshot(s) it was derived from — full traceability to raw is
  non-negotiable.
- **Organization** — a sponsor/company, with ownership history over assets
  modeled as dated intervals.

### Controlled vocabularies (fixed, not inferred at runtime)

Vocabularies live as reviewable dictionary files in the repo, not as
fuzzy-matching logic. Unmapped free-text values get emitted to a review
queue, never silently dropped or guessed.

- **Payload chemotype:** `camptothecin_topo1`, `auristatin`, `maytansinoid`,
  `calicheamicin`, `pbd_dimer`, `duocarmycin`, `amanitin`, `tubulysin`,
  `eribulin`, `immune_agonist`, `radioisotope`, `other`, `undisclosed`.
- **Target antigen:** HGNC symbols.
- **Indication:** OncoTree codes.
- **Line of therapy bucket:** `1L`, `2L`, `3L+`, `perioperative`,
  `unspecified`.
- **Program status ladder:** `active`, `dormant_suspected`,
  `dead_confirmed`, `approved`, `superseded` (asset continued but this
  indication was dropped), `unknown`.
- **Kill-reason taxonomy** (for `dead_confirmed` programs):
  `futility_efficacy`, `toxicity_safety`, `strategic_portfolio`,
  `accrual_failure`, `funding_insolvency`, `competitive_landscape`,
  `ip_legal`, `unknown_silent`.

## Immutability rule

`raw/{source}/{YYYY-MM-DD}/{id}.json` holds one immutable snapshot per
fetch: verbatim response body, request URL, ISO-8601 fetch timestamp, and a
SHA-256 of the body. **Nothing downstream may ever mutate a file under
`raw/`.** If a fetch would change the content already on disk for the same
source/id/day, that's an error, not a silent overwrite
(`pharma_stats.snapshot.ImmutabilityError`).

The DuckDB manifest that indexes snapshots is a derived index, not a source
of truth — it can always be rebuilt from `raw/` via
`pharma_stats.snapshot.rebuild_manifest()`. `raw/` is the ground truth.

`pharma_stats.snapshot.get_as_of(source, id, date)` — the most recent
snapshot at or before a given date — is the primitive the time-cut backtest
depends on. Every downstream table that claims to know something "as of"
a date must resolve it through this function, not through whatever the
latest snapshot happens to say.

## Repo layout

- `raw/` — immutable snapshots (see above). Never edit by hand.
- `gold/` — append-only, hand-authored JSONL labels (program outcome
  labels). Never edit or delete a line; corrections are new lines with a
  later timestamp.
- `data/` — derived DuckDB databases (manifest + warehouse). Fully
  rebuildable from `raw/` and `gold/`; safe to delete and regenerate.
- `reports/` — generated audit/coverage reports for human review.
- `src/pharma_stats/` — library code (`snapshot.py`, `clients/`,
  `discovery/`, `labelling/`).
- `scripts/` — one-shot / periodically-rerun entry points.

## Working conventions

- Read the actual current API docs / inspect live responses before writing
  a client against an external source — don't assume endpoint shapes from
  memory or training data.
- Favour over-inclusion plus a human review step over clever filtering when
  identifying candidates (assets, indication mappings, etc.) — recall
  matters more than precision here, since missed assets silently bias every
  base rate downstream. Surface ambiguity to the user rather than resolving
  it silently.
- Don't guess controlled-vocabulary values (payload, target, indication,
  line) ahead of the dedicated normalisation step — early guesses
  contaminate ground truth.
