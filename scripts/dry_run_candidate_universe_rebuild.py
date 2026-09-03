"""Dry-run risk check for rebuilding asset_candidates (the fix for the
null discovery_strategy/match_strength/matched_term bug — see
build_candidate_universe.py). Runs the SAME live CT.gov discovery crawl
build_candidate_universe.py would (same cost, since the crawl itself is
the expensive part, not the table write) but never touches
asset_candidates — reports a diff against the CURRENT live table's
candidate_ids instead, so the real rebuild's impact is known before
anything is committed.

    python scripts/dry_run_candidate_universe_rebuild.py

Writes reports/candidate_universe_rebuild_dry_run.md. Never calls
duckdb.connect in read-write mode on WAREHOUSE_DB.
"""
from __future__ import annotations

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.discovery.candidates import (
    build_candidate_table,
    iter_pattern_matches,
    iter_seed_matches,
    iter_sponsor_matches,
    load_seed_assets,
)


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        existing = set(
            r[0] for r in con.execute("SELECT candidate_id FROM asset_candidates").fetchall()
        )
        existing_by_id = {
            r[0]: r[1] for r in con.execute(
                "SELECT candidate_id, proposed_name FROM asset_candidates"
            ).fetchall()
        }
    finally:
        con.close()
    print(f"Current live asset_candidates: {len(existing)} rows.")

    client = CtgovClient()
    seed_assets = load_seed_assets()
    print("Running live discovery crawl (pattern + seed + sponsor matching) — same cost as the real rebuild...")

    mentions = list(iter_pattern_matches(client))
    print(f"  pattern matches: {len(mentions)}")
    seed_mentions = list(iter_seed_matches(client, seed_assets))
    print(f"  seed matches: {len(seed_mentions)}")
    mentions += seed_mentions

    known_normalized = {" ".join((m.intervention_name or "").strip().lower().split()) for m in mentions}
    sponsor_names = {m.lead_sponsor for m in mentions if m.lead_sponsor and m.lead_sponsor_class == "INDUSTRY"}
    sponsor_mentions = list(iter_sponsor_matches(client, sponsor_names, known_normalized))
    print(f"  sponsor-expansion matches: {len(sponsor_mentions)}")
    mentions += sponsor_mentions

    new_candidates = build_candidate_table(mentions)
    new_ids = {c.candidate_id for c in new_candidates}
    new_by_id = {c.candidate_id: c.proposed_name for c in new_candidates}

    added = new_ids - existing
    removed = existing - new_ids
    kept = new_ids & existing
    renamed = [
        (cid, existing_by_id[cid], new_by_id[cid])
        for cid in kept if existing_by_id[cid] != new_by_id[cid]
    ]

    lines = [
        "# Candidate universe rebuild — dry run",
        "",
        f"Current live table: {len(existing)} candidates.",
        f"Fresh crawl result: {len(new_ids)} candidates.",
        "",
        f"- Kept (same candidate_id): {len(kept)}",
        f"- Would be ADDED (new candidate_id, not in live table): {len(added)}",
        f"- Would be REMOVED (in live table, not in fresh crawl): {len(removed)}",
        f"- Renamed (same candidate_id, different proposed_name — expected, includes the "
        f"qualifier-stripping fix): {len(renamed)}",
        "",
    ]
    if removed:
        lines += ["## Candidates that would disappear", ""]
        for cid in sorted(removed)[:50]:
            lines.append(f"- `{cid}` — {existing_by_id[cid]!r}")
        if len(removed) > 50:
            lines.append(f"- ... and {len(removed) - 50} more")
        lines.append("")
    if added:
        lines += ["## New candidates that would appear", ""]
        for cid in sorted(added)[:50]:
            lines.append(f"- `{cid}` — {new_by_id[cid]!r}")
        if len(added) > 50:
            lines.append(f"- ... and {len(added) - 50} more")
        lines.append("")

    impact = "LOW" if not removed else ("HIGH — real programs would disappear from the universe" if len(removed) > 5 else "MODERATE")
    lines.append(f"**Risk assessment: {impact}**. `removed` candidate_ids would orphan any gold "
                  "label/staged decision/session reference pointing at them — check those ids "
                  "against gold/labels.jsonl before committing to a real rebuild if this list is non-empty.")

    text = "\n".join(lines)
    print("\n" + text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "candidate_universe_rebuild_dry_run.md"
    out.write_text(text, encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
