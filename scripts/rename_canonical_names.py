"""Re-picks each candidate's canonical proposed_name using the fixed
discovery.candidates._pick_proposed_name (strips dosing/formulation
qualifiers — up-titration, fixed dose, lyo-DP, for injection/infusion —
before scoring; see that module for the full rationale). Surgical UPDATE
of proposed_name/synonyms only — candidate_id is derived from an
internal union-find hash, never from proposed_name, so this never
touches program identity, gold labels, or staged decisions.

    python scripts/rename_canonical_names.py [--dry-run]
"""
from __future__ import annotations

import argparse

import duckdb

from pharma_stats.config import WAREHOUSE_DB
from pharma_stats.discovery.candidates import _pick_proposed_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = duckdb.connect(str(WAREHOUSE_DB))
    try:
        rows = con.execute("SELECT candidate_id, proposed_name, synonyms FROM asset_candidates").fetchall()
        renames = []
        for candidate_id, old_name, synonyms in rows:
            raw_names = [old_name] + list(synonyms or [])
            new_name = _pick_proposed_name(raw_names)
            if new_name != old_name:
                new_synonyms = [n for n in raw_names if n != new_name]
                renames.append((candidate_id, old_name, new_name, new_synonyms))

        print(f"{len(rows)} candidate(s) total; {len(renames)} would be renamed.")
        for candidate_id, old_name, new_name, _ in renames[:20]:
            print(f"  {candidate_id}: {old_name!r} -> {new_name!r}")
        if len(renames) > 20:
            print(f"  ... and {len(renames) - 20} more")

        if args.dry_run:
            print("\n[dry run] no changes written.")
            return

        for candidate_id, _old_name, new_name, new_synonyms in renames:
            con.execute(
                "UPDATE asset_candidates SET proposed_name = ?, synonyms = ? WHERE candidate_id = ?",
                [new_name, new_synonyms, candidate_id],
            )
        print(f"\nRenamed {len(renames)} candidate(s) in {WAREHOUSE_DB}::asset_candidates.")
        print("Run `python scripts/run_labelling_app.py --rebuild` (or restart the app) to pick this up "
              "in provisional_programs — program_id is unaffected, so no gold/staged data needs migrating.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
