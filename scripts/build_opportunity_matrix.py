"""Product B, first component: the opportunity matrix (B0 asset
attributes + B5 crowding/failure-density), on a payload chemotype x
tumour system axis pair (~10 payload classes x ~8 tumour systems, ~80
cells) — see attributes/matrix.py's module docstring and
docs/decisions/0005 for why the earlier target x specific-MeSH-indication
axes were rebuilt this way, and docs/decisions/0003 for why the
population stays thin (38 confirmed in-scope programs) rather than
waiting for more Gate-2 labels.

    python scripts/build_opportunity_matrix.py [--min-n 3]

Writes reports/opportunity_matrix_graveyard.md — the ranked graveyard
list, the only report this produces (a heatmap over ~80 mostly-thin cells
invites over-reading colour where there's no data; see matrix_report.py).
"""
from __future__ import annotations

import argparse
from collections import Counter

from pharma_stats.attributes import matrix as mx
from pharma_stats.attributes import matrix_report as mr
from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import provisional_programs as pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=3,
                     help="cells below this total (live+dead) are insufficient evidence, never graveyard")
    args = ap.parse_args()

    programs = pp.load_materialized()
    print(f"{len(programs)} materialized programs.")

    cells, program_attributes, stats = mx.build_matrix(programs, min_n=args.min_n)
    print(f"{stats['n_scope_confirmed']} program(s) confirmed is_adc=yes AND in_scope=yes.")
    print(f"  {stats['n_excluded_payload_undisclosed']} excluded: undisclosed payload chemotype "
          "(no INN suffix on file yet)")
    print(f"  {stats['n_excluded_system_unresolved']} excluded: no resolvable tumour system "
          "(no MeSH data, or only generic/site-agnostic MeSH terms)")
    print(f"{stats['n_in_population']} program(s) in the matrix's population.")
    print(f"{len(cells)} distinct (payload, tumour_system) cell(s) with >=1 program.")

    quadrant_by_key = {
        key: mx.classify_quadrant(cell.n_live, cell.n_dead, args.min_n)
        for key, cell in cells.items()
    }
    counts = Counter(quadrant_by_key.values())
    for q, n in counts.most_common():
        print(f"  {q}: {n} cell(s)")

    n_dead = sum(1 for a in program_attributes.values() if a["is_dead"])
    n_dead_proxy = sum(1 for a in program_attributes.values() if a["basis"] == "silence_proxy")
    print(f"\n{n_dead}/{stats['n_in_population']} program(s) classified dead "
          f"({n_dead_proxy} by silence-score proxy, {n_dead - n_dead_proxy} gold-confirmed).")

    population_note = (
        f"Population: {stats['n_in_population']} programs confirmed is_adc=yes AND in_scope=yes "
        f"(strict, gold-gated — see docs/decisions/0003), with a resolved payload chemotype "
        f"(not \"undisclosed\") AND a resolved tumour system (excludes generic/root MeSH terms "
        f"like \"Neoplasms\" and site-agnostic histology like \"Carcinoma\" — see "
        f"attributes/tumour_system.py and docs/decisions/0005). "
        f"{stats['n_excluded_payload_undisclosed']} program(s) were excluded for undisclosed "
        f"payload and {stats['n_excluded_system_unresolved']} for unresolved tumour system — "
        f"excluded entirely, never bucketed into a catch-all column/row. "
        f"{n_dead_proxy} of {n_dead} dead classifications are a silence-score proxy, not a "
        f"gold-confirmed kill — read the graveyard list's proxy flags before acting on them."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_out = REPORTS_DIR / "opportunity_matrix_graveyard.md"
    md_out.write_text(
        mr.render_graveyard_markdown(cells, quadrant_by_key, min_n=args.min_n, population_note=population_note),
        encoding="utf-8",
    )
    print(f"\nWrote {md_out}")


if __name__ == "__main__":
    main()
