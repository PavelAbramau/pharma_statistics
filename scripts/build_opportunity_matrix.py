"""Product B, first component: the opportunity matrix (B0 asset
attributes + B5 crowding/failure-density). See attributes/matrix.py's
module docstring for the population rule and the live/dead proxy, and
docs/decisions/0003 for why this ships thin (38 confirmed in-scope
programs) rather than waiting for more Gate-2 labels.

    python scripts/build_opportunity_matrix.py [--min-n 3]

Writes reports/opportunity_matrix.html (interactive) and
reports/opportunity_matrix_graveyard.md (the ranked graveyard list —
the actual deliverable).
"""
from __future__ import annotations

import argparse

from pharma_stats.attributes import matrix as mx
from pharma_stats.attributes import matrix_report as mr
from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import provisional_programs as pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=3,
                     help="cells below this total (live+dead) are greyed as insufficient evidence")
    args = ap.parse_args()

    programs = pp.load_materialized()
    print(f"{len(programs)} materialized programs.")

    cells, program_attributes = mx.build_matrix(programs, min_n=args.min_n)
    n_in_population = len(program_attributes)
    print(f"{n_in_population} program(s) confirmed is_adc=yes AND in_scope=yes (the matrix's population).")
    print(f"{len(cells)} distinct (target, indication) cell(s) with >=1 program.")

    quadrant_by_key = {
        key: mx.classify_quadrant(cell.n_live, cell.n_dead, args.min_n)
        for key, cell in cells.items()
    }
    from collections import Counter
    counts = Counter(quadrant_by_key.values())
    for q, n in counts.most_common():
        print(f"  {q}: {n} cell(s)")

    n_dead = sum(1 for a in program_attributes.values() if a["is_dead"])
    n_dead_proxy = sum(1 for a in program_attributes.values() if a["basis"] == "silence_proxy")
    print(f"\n{n_dead}/{n_in_population} program(s) classified dead "
          f"({n_dead_proxy} by silence-score proxy, {n_dead - n_dead_proxy} gold-confirmed).")
    n_no_target = sum(1 for a in program_attributes.values() if a["target_source"] == "unresolved")
    n_no_indication = sum(1 for a in program_attributes.values() if a["indication"] == "unknown")
    print(f"{n_no_target}/{n_in_population} program(s) have no resolved target (land in the "
          f"'undisclosed' column); {n_no_indication}/{n_in_population} have no MeSH indication "
          "(land in the 'unknown' row) — both inflate whichever cell they fall into, read those "
          "cells accordingly.")

    population_note = (
        f"Population: {n_in_population} programs confirmed is_adc=yes AND in_scope=yes "
        f"(strict, gold-gated — see docs/decisions/0003). This is thin by design: Gate-2 is "
        f"human-only, so this number only grows with more labelling, not more triage. "
        f"{n_dead_proxy} of {n_dead} dead classifications are a silence-score proxy, not a "
        f"gold-confirmed kill — read the graveyard list's proxy flags before acting on them."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_out = REPORTS_DIR / "opportunity_matrix.html"
    html_out.write_text(
        mr.render_matrix_html(cells, quadrant_by_key, min_n=args.min_n, population_note=population_note),
        encoding="utf-8",
    )
    print(f"\nWrote {html_out}")

    md_out = REPORTS_DIR / "opportunity_matrix_graveyard.md"
    md_out.write_text(mr.render_graveyard_markdown(cells, quadrant_by_key, min_n=args.min_n), encoding="utf-8")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
