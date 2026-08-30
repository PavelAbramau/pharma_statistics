"""Report supporting the is_adc / in_scope schema split:

    python scripts/report_scope_migration.py

1. Assets whose trials MeSH-classify as both heme and solid (see
   labelling/trial_scope.py) — scope is evaluated at trial level, not
   asset level, so these are the ones a whole-program in_scope=no would
   wrongly kill.
2. Rows saved under the old flag_invalid action that carry a known ADC
   naming suffix — re-decide these first under the corrected schema.
"""
from __future__ import annotations

from pharma_stats.labelling import migration
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store


def main() -> None:
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first, or "
              "pharma_stats.labelling.provisional_programs.materialize().")
        return

    print(f"Loaded {len(programs)} provisional programs.\n")

    spanning = [p for p in programs if p.get("spans_heme_and_solid")]
    print(f"=== Assets spanning heme + solid trials (MeSH-classified) ({len(spanning)}) ===")
    if not spanning:
        print("(none)")
    for p in spanning:
        print(f"- {p['proposed_name']} ({p['program_id']})")
        for nct_id, category in (p.get("trial_scope") or {}).items():
            print(f"    {category:12s} {nct_id}")
    print()

    records = store.load_records()
    to_migrate = migration.flag_invalid_migration_candidates(records)
    print(f"=== flag_invalid rows with an ADC naming suffix, needing re-decision ({len(to_migrate)}) ===")
    if not records:
        print("(no gold/labels.jsonl on disk yet — nothing has been reviewed)")
    elif not to_migrate:
        print("(none)")
    for r in to_migrate:
        print(f"- {r['proposed_name']} (program {r['program_id']}, event {r['event_id']}, "
              f"matched suffix {r['matched_suffix']!r})")


if __name__ == "__main__":
    main()
