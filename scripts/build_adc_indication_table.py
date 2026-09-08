"""The write-up's core artefact: two program-density tables — payload
chemotype x tumour system (coarse) and target antigen x specific MeSH
indication (fine) — reported as ranked TABLES, never a heatmap (an ~80-
to-several-hundred-cell grid against a ~230-program population invites
over-reading colour where there's no data; see attributes/matrix.py and
attributes/target_indication_matrix.py's module docstrings).

Every cell with n_live + n_dead >= 2 is shown (a lower bar than either
matrix module's own min_n, which only gates the QUADRANT label — see
--min-n), ranked by dead-program count, each row carrying: live count,
dead count, the kill-reason mix within its dead programs, and the
quadrant label. Population funnel (in-population count, and exclusion
counts by reason) is reported for each axis pair so a reader can judge
how much of the corpus each table actually speaks for.

Target resolution uses attributes.target's offline tiers PLUS the
data/target_residual.json map from scripts/resolve_target_residual.py's
ChEMBL + Open Targets pass, if present (falls back to offline-only with a
loud note if that file is missing or stale).

If data/matrix_extension_programs.json exists (built by
scripts/build_matrix_extension_universe.py — the 2000-2012/haematological
matrix-only universe extension, docs/decisions/0008), the COARSE view
below also gets a `dead (historical, inferred)` column: a deterministic-
rule inference, never a labelled outcome, kept in its own column and
never counted toward `dead`, `total`, or the quadrant label. The fine
view is not extended (see 0008 for why).

    python scripts/build_adc_indication_table.py [--min-n 3] [--show-n 2]

Writes reports/adc_indication_table.md.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from types import SimpleNamespace

import duckdb

from pharma_stats.attributes import matrix as mx
from pharma_stats.attributes import target_indication_matrix as tim
from pharma_stats.config import DATA_DIR, REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.discovery import matrix_extension as mxext
from pharma_stats.labelling import provisional_programs as pp

RESIDUAL_PATH = DATA_DIR / "target_residual.json"
EXTENSION_PATH = DATA_DIR / "matrix_extension_programs.json"


def _kill_reason_mix(dead_programs: list[dict]) -> str:
    counts = Counter(p.get("kill_reason") or p.get("status") or "unknown" for p in dead_programs)
    if not counts:
        return "(none)"
    return ", ".join(f"{reason}: {n}" for reason, n in counts.most_common())


def _render_table(
    title: str, axis1_label: str, axis2_label: str,
    cells: dict, min_n: int, show_n: int, population_note: str,
) -> list[str]:
    # A cell can clear the display bar entirely on dead_historical programs
    # (e.g. a payload x "heme" cell the gold-quality population never
    # populates at all) -- shown here for visibility, but `total` itself
    # (used everywhere else, including classify_quadrant below) stays
    # live+dead only, per docs/decisions/0008.
    rows = [(key, cell) for key, cell in cells.items() if cell.total + cell.n_dead_historical >= show_n]
    rows.sort(key=lambda kc: (-kc[1].n_dead, -kc[1].n_dead_historical))

    lines = [f"## {title}", "", population_note, "",
             f"{len(cells)} distinct cell(s) with >=1 program; {len(rows)} shown below with "
             f"n_live+n_dead >= {show_n}. Quadrant label uses min_n={min_n} (below that, a cell reads "
             "as insufficient_evidence even if shown here — it clears the display bar, not the "
             "confidence bar). `dead (historical, inferred)` is the matrix-only universe extension "
             "(docs/decisions/0008) — a deterministic rule's output, never a gold or proxy label, and "
             "never counted into `dead`, `total`, or the quadrant label.", "",
             f"| {axis1_label} | {axis2_label} | live | dead | dead (historical, inferred) | "
             "kill-reason mix (within dead) | quadrant |",
             "|---|---|---|---|---|---|---|"]
    for (a1, a2), cell in rows:
        quadrant = mx.classify_quadrant(cell.n_live, cell.n_dead, min_n)
        proxy_n = sum(1 for p in cell.dead_programs if p.get("basis") != "gold")
        proxy_note = f" ({proxy_n} proxy)" if proxy_n else ""
        lines.append(
            f"| {a1} | {a2} | {cell.n_live} | {cell.n_dead}{proxy_note} | {cell.n_dead_historical} | "
            f"{_kill_reason_mix(cell.dead_programs)} | {quadrant} |"
        )
    if not rows:
        lines.append("| *(none)* | | | | | | |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=3, help="quadrant-classification threshold")
    ap.add_argument("--show-n", type=int, default=2, help="minimum cell total to appear in the table")
    args = ap.parse_args()

    programs = pp.load_materialized()
    print(f"{len(programs)} materialized programs.")

    residual: dict = {}
    if RESIDUAL_PATH.exists():
        residual = json.loads(RESIDUAL_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(residual)} externally-resolved target(s) from {RESIDUAL_PATH}.")
    else:
        print(f"WARNING: {RESIDUAL_PATH} not found -- fine-view target coverage will be offline-tiers "
              "only. Run scripts/resolve_target_residual.py first for full coverage.")

    # --- Coarse: payload chemotype x tumour system -----------------------
    coarse_cells, _coarse_attrs, coarse_stats = mx.build_matrix(programs, min_n=args.min_n)
    coarse_note = (
        f"Population: {coarse_stats['n_in_population']} programs confirmed is_adc=yes AND "
        f"in_scope=yes, with a resolved payload chemotype AND a resolved tumour system, out of "
        f"{coarse_stats['n_scope_confirmed']} scope-confirmed programs. Excluded: "
        f"{coarse_stats['n_excluded_payload_undisclosed']} for undisclosed payload (no INN suffix "
        f"on file), {coarse_stats['n_excluded_system_unresolved']} for unresolved tumour system "
        "(no MeSH data, or only a generic/site-agnostic term). Live/dead is a gold label where one "
        "exists, otherwise a silence-score proxy (flagged per-cell above) -- never presented as more "
        "certain than that."
    )

    # --- Matrix-only universe extension (docs/decisions/0008) -------------
    # ADC trials from 2000-2012 and haematological indications, folded ONLY
    # into the coarse view's dead_historical column -- see that decision
    # for why the fine view isn't extended. INFERRED outcomes, never
    # labelled: this cohort never touches gold/labels.jsonl.
    ext_stats = None
    ext_programs = []
    if EXTENSION_PATH.exists():
        raw = json.loads(EXTENSION_PATH.read_text(encoding="utf-8"))
        ext_programs = [SimpleNamespace(**r) for r in raw]
        ext_stats = mxext.fold_into_coarse_matrix(coarse_cells, ext_programs)
        n_dead_historical_total = sum(1 for p in ext_programs if p.dead_historical == "dead_historical")
        print(f"\nMatrix-only universe extension: {len(ext_programs)} program(s) "
              f"({n_dead_historical_total} dead_historical, "
              f"{len(ext_programs) - n_dead_historical_total} not classified), "
              f"loaded from {EXTENSION_PATH}.")
        print(f"  folded into coarse matrix: {ext_stats['n_dead_historical_folded_in']} "
              f"(excluded: {ext_stats['n_excluded_payload_undisclosed']} undisclosed payload, "
              f"{ext_stats['n_excluded_tumour_unresolved']} unresolved tumour bucket)")
    else:
        print(f"\nNo matrix extension file at {EXTENSION_PATH} -- run "
              "scripts/build_matrix_extension_universe.py first for the 2000-2012/heme cohort. "
              "Coarse view below is gold-quality population only.")

    # --- Fine: target antigen x specific MeSH indication ------------------
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        fine_cells, _fine_attrs, fine_stats = tim.build_target_indication_matrix(programs, con, residual=residual)
    finally:
        con.close()
    src_counts = fine_stats["target_source_counts"]
    src_line = ", ".join(f"{k}: {v}" for k, v in sorted(src_counts.items(), key=lambda kv: -kv[1]))
    fine_note = (
        f"Population: {fine_stats['n_in_population']} programs confirmed is_adc=yes AND in_scope=yes, "
        f"with a resolved target antigen AND a resolved specific MeSH indication term, out of "
        f"{fine_stats['n_scope_confirmed']} scope-confirmed programs. Excluded: "
        f"{fine_stats['n_excluded_target_unresolved']} for unresolved target, "
        f"{fine_stats['n_excluded_indication_unresolved']} for unresolved indication. "
        f"Target resolution source breakdown across all scope-confirmed programs: {src_line}. "
        "`chembl_opentargets_verified` and `trial_text_majority` are lower-confidence tiers than a "
        "direct antibody-stem/name match or an unambiguous single trial-text hit -- see "
        "attributes/target.py and attributes/target_external.py."
    )

    lines = [
        "# ADC program density tables — payload x tumour system, target x indication", "",
        "Ranked tables, not heatmaps: both populations are small enough (low hundreds of programs "
        "against tens to low hundreds of cells) that a colour grid would invite reading signal into "
        "cells that only have 2-3 programs in them. Every number here is a raw count against a "
        "stated, exclusion-audited population -- read the population note before citing any cell.",
        "",
    ]
    lines += _render_table(
        "Coarse view: payload chemotype x tumour system", "payload", "tumour system",
        coarse_cells, args.min_n, args.show_n, coarse_note,
    )
    lines += _render_table(
        "Fine view: target antigen x specific MeSH indication", "target (HGNC)", "indication (MeSH)",
        fine_cells, args.min_n, args.show_n, fine_note,
    )

    if ext_stats is not None:
        n_dead_historical_total = sum(1 for p in ext_programs if p.dead_historical == "dead_historical")
        n_pre_2012 = sum(1 for p in ext_programs if "pre_2012" in p.extension_reasons)
        n_heme = sum(1 for p in ext_programs if "heme" in p.extension_reasons)
        lines += [
            "## Matrix-only universe extension: 2000–2012 and haematological indications "
            "(docs/decisions/0008)", "",
            "**Every outcome in this section is an INFERENCE from a deterministic rule "
            "(labelling/dead_historical.py), never a labelled or gold outcome** — this cohort is "
            "never manually labelled and never enters gold/labels.jsonl. It exists solely to "
            "populate the coarse matrix's density tables above (folded into that table's "
            "`dead (historical, inferred)` column, kept visually and numerically separate from "
            "`dead` everywhere).",
            "",
            f"{len(ext_programs)} candidate program(s) discovered outside the main 2012+/solid-tumour "
            f"scope ({n_pre_2012} pre-2012, {n_heme} haematological; a program can be both), all "
            "industry-sponsored, confirmed ADC by the same Layer-1 rule the main pipeline uses. "
            f"{n_dead_historical_total} classify dead_historical (no trial in >=10 years, not a "
            "known-approved compound, no successor asset from the same sponsor on the same "
            f"resolved target); {len(ext_programs) - n_dead_historical_total} are NOT classified "
            "(recent trial, a possible successor, or a known approval) and contribute nothing to "
            "the density tables.",
            "",
            f"Folded into the coarse matrix: {ext_stats['n_dead_historical_folded_in']} "
            f"(excluded: {ext_stats['n_excluded_payload_undisclosed']} for undisclosed payload, "
            f"{ext_stats['n_excluded_tumour_unresolved']} for an unresolved tumour-system/heme "
            "bucket — same never-guess exclusion discipline as the main matrix). Tumour-system/heme "
            "classification here is a TEXT heuristic on raw condition strings "
            "(discovery/heme_scope_text.py), not the MeSH-ID pipeline the main matrix uses — lower "
            "confidence, by design; see 0008.",
            "",
            "The fine (target x specific MeSH indication) view is NOT extended by this cohort — that "
            "axis has no non-MeSH proxy available without a current-state fetch per trial, which "
            "this bounded pass does not do (0008).",
            "",
        ]

    text = "\n".join(lines)
    print("\n" + text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "adc_indication_table.md"
    out.write_text(text, encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
