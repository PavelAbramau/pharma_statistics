"""Current-state fetch pass: GET /api/v2/studies/{nctId} for every
in-universe trial, storing the full record (protocolSection +
derivedSection) through the normal snapshot store under the bare NCT id
— the canonical snapshot pharma_stats.snapshot.get_as_of/.latest resolve,
and the only place derivedSection.conditionBrowseModule (the MeSH data
labelling/trial_scope.py classifies on) ever exists.

    python scripts/fetch_current_state.py [--max-trials N] [--max-seconds S]

~50 minutes for the full ~2845-trial universe at CT.gov's 1.05s/request
crawl delay. Safe to kill and rerun: CtgovClient.get_study() is cache-first
per (source, id, day) — a trial already fetched TODAY is a local cache hit,
not a network call, so resuming later the same day costs nothing for
trials already done. Resuming on a LATER day does re-fetch everything not
yet done that day (the snapshot store is a per-day cache, by design — see
snapshot.py) — acceptable here since the whole pass fits inside one day's
run in practice.

IMPORTANT — what this data may be used for: see
docs/decisions/0001-current-state-fetch-scope.md. Permitted for
universe-membership statics (disease category / MeSH, sponsor class,
start date) ONLY — never for a silence/model feature, which must keep
coming from the time-cut versioned-history path
(provisional_programs._best_trial_snapshot). audit/universe.py's
"current-state read boundary" check enforces this on provisional_programs.py.
"""
from __future__ import annotations

import argparse
import sys
import time

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import WAREHOUSE_DB


def trial_universe(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute("SELECT nct_ids FROM asset_candidates").fetchall()
    ids: set[str] = set()
    for (nct_ids,) in rows:
        ids.update(nct_ids)
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    universe = trial_universe(con)
    con.close()
    print(f"In-universe trials: {len(universe)}")

    client = CtgovClient()
    start = time.monotonic()
    fetched = errors = 0

    for i, nct_id in enumerate(universe):
        if args.max_seconds is not None and time.monotonic() - start > args.max_seconds:
            print(f"Time budget hit at {i}/{len(universe)}.")
            break
        if args.max_trials is not None and fetched >= args.max_trials:
            print(f"Trial budget hit at {i}/{len(universe)}.")
            break

        try:
            client.get_study(nct_id)  # cache-first; network only if not already fetched today
            fetched += 1
        except Exception as e:  # noqa: BLE001 — log and keep going, same policy as run_backfill
            errors += 1
            print(f"  ERROR {nct_id}: {e}")

        if args.progress_every and (i + 1) % args.progress_every == 0:
            elapsed = time.monotonic() - start
            print(f"  [{i + 1}/{len(universe)}] {fetched} fetched, {errors} errors, "
                  f"{elapsed:.0f}s elapsed", flush=True)

    print(f"\nDone: {fetched} fetched, {errors} errors, of {len(universe)} in-universe trials "
          f"({time.monotonic() - start:.0f}s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
