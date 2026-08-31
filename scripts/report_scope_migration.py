"""Report supporting the is_adc / in_scope schema split:

    python scripts/report_scope_migration.py

1. Assets whose trials MeSH-classify as both heme and solid — scope is
   evaluated at trial level, not asset level, so a whole-asset
   in_scope=no would wrongly kill these.
2. Latest is_adc=no rows with an ADC naming suffix (re-decide by hand).
3. Latest is_adc=no rows with only a literal ADC/conjugate hit (near-miss).
"""
from __future__ import annotations

from pharma_stats.labelling import migration
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store


def _trial_line(program: dict, nct_id: str, category: str) -> str:
    trial = next((t for t in (program.get("trials") or []) if t.get("nct_id") == nct_id), {})
    conds = trial.get("conditions") or []
    cond = "; ".join(conds[:3]) if conds else ""
    extra = f" — {cond}" if cond else ""
    return f"    {category:12s} {nct_id}{extra}"


def main() -> None:
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first, or "
              "pharma_stats.labelling.provisional_programs.materialize().")
        return

    print(f"Loaded {len(programs)} provisional programs.\n")
    programs_by_id = {p["program_id"]: p for p in programs}
    records = store.load_records()
    latest = store.latest_by_program(records)

    spanning = [p for p in programs if p.get("spans_heme_and_solid")]
    print(f"=== Assets spanning heme + solid trials (MeSH-classified) ({len(spanning)}) ===")
    if not spanning:
        print("(none)")
    for p in spanning:
        gold = latest.get(p["program_id"])
        gold_s = "unreviewed"
        if gold:
            gold_s = (f"is_adc={gold.get('is_adc')} in_scope={gold.get('in_scope')} "
                      f"reason={gold.get('scope_reason')} gate={gold.get('gate_reached')}")
        print(f"- {p['proposed_name']} ({p['program_id']})  [{gold_s}]")
        trial_scope = p.get("trial_scope") or {}
        for nct_id, category in sorted(trial_scope.items(), key=lambda kv: (kv[1], kv[0])):
            if category in ("heme", "solid"):
                print(_trial_line(p, nct_id, category))
    print()

    suffix = migration.not_an_adc_suffix_candidates(records, programs_by_id)
    print(f"=== is_adc=no rows with an ADC naming suffix, needing re-decision ({len(suffix)}) ===")
    if not suffix:
        print("(none)")
    for r in suffix:
        print(f"- {r['proposed_name']} (program {r['program_id']}, event {r['event_id']}, "
              f"suffix {r['matched_suffix']!r} on {r['matched_on']!r})")
    print()

    literals = migration.not_an_adc_literal_candidates(records, programs_by_id)
    print(f"=== is_adc=no rows with a literal ADC/conjugate hit, no INN suffix ({len(literals)}) ===")
    if not literals:
        print("(none)")
    for r in literals:
        print(f"- {r['proposed_name']} (program {r['program_id']}, "
              f"{r['match_strength']} {r['matched_term']!r} on {r['matched_on']!r})")


if __name__ == "__main__":
    main()
