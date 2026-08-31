"""Append-only backfill: latest is_adc=no rows get in_scope=no / not_an_adc.

    python scripts/migrate_is_adc_in_scope.py --dry-run
    python scripts/migrate_is_adc_in_scope.py

Never edits an existing gold line. Does not flip is_adc — suffix hits stay
is_adc=no until a human re-decides them (see report_scope_migration.py).
"""
from __future__ import annotations

import argparse

from pharma_stats.labelling import migration
from pharma_stats.labelling import store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = store.load_records()
    to_backfill = migration.rows_needing_scope_backfill(records)
    print(f"Latest is_adc=no rows needing in_scope backfill: {len(to_backfill)}")
    inconsistent = [r for r in to_backfill if r.get("in_scope") == "yes"]
    if inconsistent:
        print(f"  of which is_adc=no + in_scope=yes (will be corrected): {len(inconsistent)}")
        for r in inconsistent:
            print(f"    - {r.get('proposed_name')} ({r['program_id']})")

    if args.dry_run:
        for r in to_backfill:
            print(f"  [dry-run] {r.get('proposed_name')} ({r['program_id']}) "
                  f"in_scope={r.get('in_scope')!r} reason={r.get('scope_reason')!r}")
        return

    for r in to_backfill:
        store.append_record(migration.build_scope_backfill_record(r))
    print(f"Appended {len(to_backfill)} gold line(s), session_id={migration.BACKFILL_SESSION_ID}.")


if __name__ == "__main__":
    main()
