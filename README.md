# pharma_statistics

Detect silently-discontinued antibody-drug conjugate (ADC) oncology programs
from public data. Scope is locked: **ADCs, solid tumours, industry sponsors,
2012–present.** DuckDB over Parquet/JSON on a laptop; no Postgres, no cloud.

Project conventions live in [`CLAUDE.md`](CLAUDE.md). The five-entity model
(asset, program, trial, evidence event, organization) is specified there.

## What is built

| Layer | Status |
|---|---|
| Immutable snapshot store (`raw/` + DuckDB manifest, `get_as_of`) | done |
| ClinicalTrials.gov v2 client + undocumented `/api/int` history | done |
| ADC candidate discovery (pattern / seed / sponsor union) | done |
| History index + resumable, priority-ordered version backfill | done |
| Schema guard on the undocumented history API | done |
| Controlled-vocab normalisation (payload, target, indication, line) | **not started** |
| Five-entity warehouse (Asset / Program / Trial / EvidenceEvent / Org) | **not started** |
| EvidenceEvent extraction from trial amendments | **done, not yet wired into the labelling app** — see below |
| Labelling UI + gold labels | **provisional v0** — programs = candidate assets (no indication/line split yet), silence score is a hand-built heuristic, event timeline is untyped amendment history. See below. |
| Program-status detection / silent-kill backtest | **not started** |

The ingest → discover → backfill spine works. The analysis pipeline that
would actually flag silent discontinuations does not exist yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests (no network, mock data)

```bash
pytest
```

That runs the snapshot store, CT.gov client (mocked HTTP), schema guard,
history index, backfill orchestrator, discovery clustering, and an
end-to-end mock pipeline (snapshots → candidates → warehouse → backfill).

## Live crawl (optional, hits ClinicalTrials.gov)

```bash
# smoke: a handful of studies through the snapshot store
python scripts/fetch_ctgov_sample.py 10

# candidate universe → data/warehouse.duckdb + reports/
python scripts/build_candidate_universe.py

# resumable history backfill (Ctrl-C is safe; rerun to resume)
python scripts/run_backfill.py --signal-labels recommended --max-seconds 3600
```

`raw/` and `data/` are local-only (see `.gitignore`). The manifest is
derived: `python -c "from pharma_stats.snapshot import rebuild_manifest; print(rebuild_manifest())"`.

Set `CTGOV_CONTACT` to put a contact address in the crawler User-Agent.

## Labelling app (provisional v0)

```bash
python scripts/run_labelling_app.py            # opens http://127.0.0.1:8420
python scripts/run_labelling_app.py --rebuild  # recompute the provisional program view first
```

The real five-entity warehouse and controlled-vocab normalisation don't
exist yet, so this reads a **provisional** program view built straight
from `asset_candidates` + raw CT.gov snapshots: one provisional program
per candidate asset (`indication_code="UNSPECIFIED"`, no line-of-therapy
split), scored with a hand-built silence heuristic — not the project's
eventual model. See `pharma_stats/labelling/provisional_programs.py` for
exactly what the score does and doesn't account for.

Labels are append-only JSONL at `gold/labels.jsonl`. Session/queue state
(stratified by score band × archetype, ~10% silently re-served for
self-consistency) persists to `data/labelling_session.json` and survives
restarts; losing that file costs queue position, never a label.

## Differ (EvidenceEvent extraction)

```bash
python scripts/run_differ.py   # data/warehouse.duckdb::evidence_events + reports/differ_noise_floor.md
```

Local computation only — diffs every pair of adjacent fetched version
bodies (`raw/` + `history_index`), no network. Rules, non-negotiable:

- Never diffs a date/enrollment field across an ESTIMATED -> ACTUAL
  boundary as a plan change — emits a `*_finalized` event instead. An
  estimate becoming a confirmed fact isn't evidence of anything; a
  regression test batteries every ESTIMATED/ACTUAL transition combination
  to keep this impossible to regress (`tests/test_differ.py`).
- `event_date` is always the *to-version*'s `posted_date` (when CT.gov
  actually made the change public) — never `submitted_date`. This is the
  knowability date `label_evidence_date` and any future backtest depend on.
- Enrollment and completion-date changes carry a `direction`
  ("increased"/"decreased", "pushed_later"/"pulled_earlier").

Run `run_differ.py` before trusting anything downstream — it prints (and
writes to `reports/differ_noise_floor.md`) the negative control (a
version diffed against an exact copy of itself; must be zero events) and
the per-event-type firing frequency, so an implausibly-noisy event type
gets caught before it ever reaches a label. Not yet wired into the
labelling app's timeline (which still shows untyped amendment history) —
that's the next step once the noise floor looks right.

## Audit harness

```bash
python -m pharma_stats.audit --stage all       # every stage, timestamped report to audit/
python -m pharma_stats.audit --stage gold_set  # just one stage
```

Verifies each built pipeline stage actually ran, populated correctly,
and converged: raw/manifest provenance (including a probe that
`get_as_of` returns the historically-correct snapshot, not the latest —
the check that protects the time-cut backtest), discovery universe
coverage, history-index integrity, backfill drain state (including
whether errored trials share one failure mode or are scattered), the
differ's noise floor / negative control / ESTIMATED-ACTUAL invariant,
and the gold label set (stratum coverage, the `dead_confirmed` date
invariant re-verified independently of the labelling app's own
validator, self-consistency on silently-repeated labels, and a bootstrap
check for whether the gold set is big enough yet). Levels are FAIL
(stop), WARN (look at it), INFO (context) — exit code is non-zero on any
FAIL.

`universe`'s "unreviewed candidates" check is a **gate**: a FAIL there
halts `--stage all` before running any downstream stage, since they'd
all be analysing a candidate universe no human has signed off on yet.

Stages for pipeline components that don't exist yet (normalisation,
features, model) report an honest "not built" INFO rather than being
silently omitted, so the report always reflects what's actually true of
the codebase.
