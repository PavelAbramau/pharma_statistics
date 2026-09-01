"""Report MeSH coverage vs heme/solid/both/ambiguous mix.

    python scripts/report_mesh_coverage.py

Coverage (has any conditionBrowseModule data) and classification
(heme/solid/ambiguous) are different numbers — ambiguous also fires when
MeSH is present but the dictionary can't classify it. heme_only
auto-exclusion requires every trial to classify heme, so a high coverage
rate with a dominant ambiguous share means the rule is weaker than it
looks (under-excludes), not more aggressive.
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import trial_scope as ts


def _brentuximab(programs: list[dict]) -> list[dict]:
    hits = []
    for p in programs:
        names = " ".join([p.get("proposed_name") or ""] + list(p.get("synonyms") or []))
        if "brentuximab" in names.lower():
            hits.append(p)
    return hits


def format_report(programs: list[dict]) -> str:
    coverage = ts.mesh_coverage(programs)
    dist = ts.scope_distribution(programs)
    trials = dist["trials"]
    assets = dist["assets"]
    lines = [
        "# MeSH coverage and scope-class mix",
        "",
        "## Coverage (any conditionBrowseModule data)",
        "",
        f"- {coverage['covered']} / {coverage['total']} trials "
        f"({coverage['coverage_rate']:.1%})",
        f"- Gate: {ts.MESH_COVERAGE_THRESHOLD:.0%} "
        f"({'PASS' if coverage['coverage_rate'] >= ts.MESH_COVERAGE_THRESHOLD else 'FAIL'})",
        "",
        "## Trial-level classification",
        "",
        f"- heme: {trials['heme']} ({trials['rates']['heme']:.1%})",
        f"- solid: {trials['solid']} ({trials['rates']['solid']:.1%})",
        f"- non_oncology: {trials['non_oncology']} ({trials['rates']['non_oncology']:.1%})",
        f"- ambiguous: {trials['ambiguous']} ({trials['rates']['ambiguous']:.1%})",
        f"  - of which MeSH present (dictionary gap / mix / basket): {trials['ambiguous_with_mesh']}",
        f"  - of which no MeSH at all: {trials['ambiguous_without_mesh']}",
        "",
        f"Ambiguous dominates: **{'yes' if dist['ambiguous_dominates_trials'] else 'no'}**.",
        "",
        "## Asset-level rollup",
        "",
        f"- heme_only (every trial heme): {assets['heme_only']}",
        f"- solid_only: {assets['solid_only']}",
        f"- both (spans heme and solid): {assets['both']}",
        f"- all_ambiguous: {assets['all_ambiguous']}",
        f"- mixed_other (e.g. solid+ambiguous, heme+ambiguous): {assets['mixed_other']}",
        f"- non_oncology_only: {assets['non_oncology_only']}",
        f"- no_trials: {assets['no_trials']}",
        "",
        "heme_only auto-exclusion only fires on the first bucket. If ambiguous",
        "dominates at trial level, most assets land in all_ambiguous or mixed_other",
        "and stay in the manual queue — the rule under-excludes.",
        "",
    ]
    for p in _brentuximab(programs):
        scope = p.get("trial_scope") or {}
        counts: dict[str, int] = {}
        for cat in scope.values():
            counts[cat] = counts.get(cat, 0) + 1
        n = len(scope)
        lines += [
            f"## Brentuximab — `{p['proposed_name']}`",
            "",
            f"- program_id: `{p['program_id']}`",
            f"- trials classified: {n}",
            *[f"- {cat}: {n_cat}" for cat, n_cat in sorted(counts.items())],
            f"- asset bucket: {ts.asset_scope_bucket(list(scope.values()))}",
            f"- classify_asset (auto-exclusion input): {p.get('scope_category')}",
            f"- spans_heme_and_solid: {p.get('spans_heme_and_solid')}",
            "",
        ]
    if coverage["coverage_rate"] < ts.MESH_COVERAGE_THRESHOLD:
        lines += [
            "## Implication for auto-exclusion",
            "",
            f"MeSH coverage is {coverage['coverage_rate']:.1%}, below the "
            f"{ts.MESH_COVERAGE_THRESHOLD:.0%} gate. Almost all "
            f"\"ambiguous\" trials ({trials['ambiguous_without_mesh']}/"
            f"{trials['ambiguous']}) have **no MeSH at all** — this is a "
            "coverage gap, not a dictionary gap "
            f"({trials['ambiguous_with_mesh']} trials have MeSH but still "
            "classify ambiguous). heme_only auto-exclusion is OFF. The "
            f"{assets['heme_only']} assets that classify heme_only anyway "
            "are the small leftover with complete MeSH, not the real "
            "haematology cohort, and must not be used to filter the queue.",
            "",
        ]
    elif dist["ambiguous_dominates_trials"]:
        lines += [
            "## Implication for auto-exclusion",
            "",
            "Do **not** run heme_only auto-exclusion on this mix. Coverage can still",
            "clear the 90% gate while most trials classify ambiguous (dictionary gap,",
            "heme+solid mix on one record, or generic basket terms). The qualifying",
            "heme_only set would be the small leftover that happened to classify",
            "cleanly, not the real haematology cohort.",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized.")
        return
    text = format_report(programs)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "mesh_coverage.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
