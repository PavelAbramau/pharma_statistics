"""Coverage report for B0 asset attributes (payload chemotype, target
antigen) over the in-scope population — the gate the user asked for
before building the opportunity matrix: report coverage, don't draw a
mostly-empty grid.

"In-scope" here = effective is_adc=yes AND in_scope=yes, preferring a
gold label when one exists, falling back to the latest triage-staged
verdict otherwise — the same population B5's crowding/failure-density
cells will eventually be built over.

    python scripts/report_attribute_coverage.py
"""
from __future__ import annotations

import duckdb

from pharma_stats.attributes import payload as payload_attr
from pharma_stats.attributes import target as target_attr
from pharma_stats.config import REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.triage import evidence as tev
from pharma_stats.triage import staging


def in_scope_programs(programs: list[dict], gold_records: list[dict], staged_records: list[dict]) -> list[dict]:
    gold_latest = store.latest_by_program(gold_records)
    staged_latest = staging.latest_by_program(staged_records)
    out = []
    for p in programs:
        pid = p["program_id"]
        g = gold_latest.get(pid)
        if g is not None:
            if g.get("is_adc") == "yes" and g.get("in_scope") == "yes":
                out.append(p)
            continue
        s = staged_latest.get(pid)
        if s is not None and not s.get("manual_overflow") and s.get("is_adc") == "yes" and s.get("in_scope") == "yes":
            out.append(p)
    return out


def main() -> None:
    programs = pp.load_materialized()
    gold_records = store.load_records()
    staged_records = staging.load_records()
    scoped = in_scope_programs(programs, gold_records, staged_records)
    print(f"In-scope population (is_adc=yes, in_scope=yes, gold-first): {len(scoped)} / {len(programs)} programs")

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        payload_undisclosed = []
        target_unresolved = []
        target_source_counts = {"antibody_stem": 0, "trial_text": 0, "name": 0, "unresolved": 0}
        payload_resolved = 0

        for p in scoped:
            name = p.get("proposed_name")
            synonyms = p.get("synonyms") or []
            chemotype = payload_attr.derive_payload_chemotype(name, synonyms)
            if chemotype == "undisclosed":
                payload_undisclosed.append(p)
            else:
                payload_resolved += 1

            evidence = tev.build_layer2_evidence(p, con)
            target, source = target_attr.derive_target(name, synonyms, evidence.get("text_snippets"))
            target_source_counts[source] += 1
            if target is None:
                target_unresolved.append(p)
    finally:
        con.close()

    n = len(scoped) or 1
    payload_rate = payload_resolved / n
    target_rate = (n - len(target_unresolved)) / n

    lines = [
        "# B0 attribute coverage (payload chemotype, target antigen)",
        "",
        f"In-scope population: {len(scoped)} programs (is_adc=yes, in_scope=yes; gold-first, "
        "triage-staged fallback).",
        "",
        "## Payload chemotype",
        "",
        f"- resolved (not undisclosed): {payload_resolved} / {len(scoped)} ({payload_rate:.1%})",
        f"- undisclosed (no INN suffix — bare dev code): {len(payload_undisclosed)}",
        "",
        "## Target antigen",
        "",
        f"- resolved: {n - len(target_unresolved)} / {len(scoped)} ({target_rate:.1%})",
        f"  - via antibody-stem dictionary: {target_source_counts['antibody_stem']}",
        f"  - via trial text: {target_source_counts['trial_text']}",
        f"  - via candidate name: {target_source_counts['name']}",
        f"- unresolved (flagged for review, not guessed): {target_source_counts['unresolved']}",
        "",
        f"Gate: {'PASS' if payload_rate >= 0.6 and target_rate >= 0.6 else 'FAIL'} (both need >= 60% "
        "per the user's threshold for the opportunity matrix to be worth drawing).",
        "",
    ]
    if target_unresolved:
        lines += ["## Unresolved-target review queue (sample, first 30)", ""]
        for p in target_unresolved[:30]:
            lines.append(f"- `{p['program_id']}` — {p.get('proposed_name')}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "attribute_coverage.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
