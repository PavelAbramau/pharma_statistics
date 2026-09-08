"""Residual target-antigen resolution for the in-scope population's
still-unresolved candidates (after attributes/target.py's offline
dictionaries + trial-text extraction) via ChEMBL + Open Targets — see
attributes/target_external.py's module docstring for the two-step
verified-identity-then-mechanism design.

Writes reports/target_residual_resolution.md (full audit trail: which
candidate matched which ChEMBL id via which field, and the mechanism text
behind each resolved target) and data/target_residual.json (program_id ->
{target, chembl_id, matched_name, matched_via, mechanism_of_action}) for
downstream table-building to consume without re-hitting the network.

    python scripts/resolve_target_residual.py
"""
from __future__ import annotations

import json
import time

import duckdb

from pharma_stats.attributes import payload as payload_attr
from pharma_stats.attributes import target as target_attr
from pharma_stats.attributes import target_external as text
from pharma_stats.clients.chembl import ChemblClient
from pharma_stats.clients.opentargets import OpenTargetsClient
from pharma_stats.config import DATA_DIR, REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.triage import evidence as tev
from pharma_stats.triage import staging

OUT_JSON = DATA_DIR / "target_residual.json"
OUT_MD = REPORTS_DIR / "target_residual_resolution.md"


def in_scope_programs(programs: list[dict], gold_records: list[dict], staged_records: list[dict]) -> list[dict]:
    """Same population as report_attribute_coverage.py's own definition:
    effective is_adc=yes AND in_scope=yes, gold-first, triage-staged
    fallback -- kept in sync by hand rather than cross-imported, since
    scripts/ entries are meant to be standalone (python scripts/x.py adds
    only that file's own directory to sys.path, not scripts/ itself as a
    package)."""
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
    print(f"In-scope population: {len(scoped)} programs.")

    chembl = ChemblClient()
    opentargets = OpenTargetsClient()

    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    unresolved: list[dict] = []
    try:
        for p in scoped:
            name = p.get("proposed_name")
            synonyms = p.get("synonyms") or []
            ev = tev.build_layer2_evidence(p, con)
            t, source = target_attr.derive_target(name, synonyms, ev.get("text_snippets"))
            if t is None:
                unresolved.append(p)
    finally:
        con.close()

    print(f"{len(unresolved)} candidates still unresolved after offline dictionaries + trial text.")
    print("Querying ChEMBL + Open Targets for each (rate-limited, ~0.34s/request)...")

    resolved: dict[str, dict] = {}
    not_found_in_chembl = 0
    chembl_found_but_no_binding_mechanism = 0
    t0 = time.monotonic()
    for i, p in enumerate(unresolved):
        name = p.get("proposed_name")
        synonyms = p.get("synonyms") or []
        hit = text.resolve_target_external(name, synonyms, chembl_client=chembl, opentargets_client=opentargets)
        if hit is not None:
            resolved[p["program_id"]] = {
                "target": hit.target, "chembl_id": hit.chembl_id, "matched_name": hit.matched_name,
                "matched_via": hit.matched_via, "mechanism_of_action": hit.mechanism_of_action,
            }
        else:
            # Distinguish the two failure modes for an honest report: did
            # ChEMBL even have this compound on file at all?
            verified = text._verified_chembl_id(name, synonyms, client=chembl)
            if verified is None:
                not_found_in_chembl += 1
            else:
                chembl_found_but_no_binding_mechanism += 1
        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {i + 1}/{len(unresolved)} checked ({elapsed:.0f}s elapsed, {len(resolved)} resolved so far)")

    print(f"\nResolved via ChEMBL+Open Targets: {len(resolved)} / {len(unresolved)}")
    print(f"  not found in ChEMBL at all: {not_found_in_chembl}")
    print(f"  found in ChEMBL, no single-target BINDING AGENT mechanism on Open Targets: "
          f"{chembl_found_but_no_binding_mechanism}")

    still_unresolved = len(unresolved) - len(resolved)
    n = len(scoped) or 1
    new_target_resolved = (len(scoped) - len(unresolved)) + len(resolved)
    print(f"\nTarget coverage: {new_target_resolved} / {len(scoped)} ({new_target_resolved / n:.1%}) "
          f"after external resolution (was {len(scoped) - len(unresolved)} / {len(scoped)} "
          f"({(len(scoped) - len(unresolved)) / n:.1%}) from offline tiers alone).")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    lines = [
        "# Residual target-antigen resolution (ChEMBL + Open Targets)", "",
        f"In-scope population: {len(scoped)}. Unresolved after offline dictionaries + trial text: "
        f"{len(unresolved)}.", "",
        f"- Resolved via verified ChEMBL identity match + Open Targets single-target BINDING AGENT "
        f"mechanism: **{len(resolved)}**",
        f"- Not found in ChEMBL at all (real data-availability ceiling, not a matching failure): "
        f"{not_found_in_chembl}",
        f"- Found in ChEMBL, but Open Targets has no single-target antibody-binding mechanism on file "
        f"(zero, multiple, or multi-target BINDING AGENT rows -- ambiguous, not guessed): "
        f"{chembl_found_but_no_binding_mechanism}",
        "",
        f"**New target coverage: {new_target_resolved} / {len(scoped)} ({new_target_resolved / n:.1%})**",
        "", "## Resolved (full audit trail)", "",
        "| program_id | target | ChEMBL id | matched via | mechanism of action |",
        "|---|---|---|---|---|",
    ]
    by_id = {p["program_id"]: p for p in scoped}
    for pid, r in sorted(resolved.items(), key=lambda kv: kv[1]["target"]):
        pname = by_id[pid].get("proposed_name", pid)
        lines.append(
            f"| `{pid}` ({pname}) | {r['target']} | {r['chembl_id']} | "
            f"{r['matched_via']} (\"{r['matched_name']}\") | {r['mechanism_of_action']} |"
        )
    lines.append("")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
