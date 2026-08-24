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
| EvidenceEvent extraction from trial amendments | **not started** |
| Labelling UI + gold labels | **not started** |
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
