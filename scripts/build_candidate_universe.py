"""Build the ADC candidate-asset universe from CT.gov: union of pattern
matching, seed-list expansion, and sponsor-based expansion.

Writes:
- data/warehouse.duckdb: table asset_candidates
- reports/candidate_universe.csv: one row per candidate, for spreadsheet review
- reports/discovery_audit.md: coverage + audit report

Usage: python scripts/build_candidate_universe.py
"""
import csv
import dataclasses
import json
import sys
import time

import duckdb

from pharma_stats.clients.ctgov import CtgovClient
from pharma_stats.config import REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.discovery.audit import render_report
from pharma_stats.discovery.candidates import (
    Mention,
    build_candidate_table,
    iter_pattern_matches,
    iter_seed_matches,
    iter_sponsor_matches,
    load_seed_assets,
)


def main() -> None:
    client = CtgovClient()
    seed_assets = load_seed_assets()

    all_mentions: list[Mention] = []

    t0 = time.time()
    print("Strategy 1/3: pattern matching (naming suffixes + literal ADC terms)...")
    pattern_mentions = list(iter_pattern_matches(client))
    all_mentions.extend(pattern_mentions)
    print(f"  {len(pattern_mentions)} mentions, "
          f"{len({m.nct_id for m in pattern_mentions})} distinct trials "
          f"[{time.time() - t0:.0f}s]")

    t0 = time.time()
    print("Strategy 2/3: seed-list expansion...")
    seed_mentions = list(iter_seed_matches(client, seed_assets))
    all_mentions.extend(seed_mentions)
    print(f"  {len(seed_mentions)} mentions, "
          f"{len({m.nct_id for m in seed_mentions})} distinct trials "
          f"[{time.time() - t0:.0f}s]")

    t0 = time.time()
    print("Strategy 3/3: sponsor-based expansion...")
    known_normalized = {
        " ".join((m.intervention_name or "").strip().lower().split())
        for m in all_mentions
    }
    sponsor_names = {
        m.lead_sponsor for m in all_mentions
        if m.lead_sponsor and m.lead_sponsor_class == "INDUSTRY"
    }
    print(f"  scanning {len(sponsor_names)} industry sponsors with a confirmed ADC hit...")
    sponsor_mentions = list(iter_sponsor_matches(client, sponsor_names, known_normalized))
    all_mentions.extend(sponsor_mentions)
    print(f"  {len(sponsor_mentions)} mentions, "
          f"{len({m.nct_id for m in sponsor_mentions})} distinct trials "
          f"[{time.time() - t0:.0f}s]")

    print(f"\nTotal mentions across all strategies: {len(all_mentions)}")
    candidates = build_candidate_table(all_mentions)
    print(f"Clustered into {len(candidates)} candidate assets")

    # -- write warehouse table -------------------------------------------
    con = duckdb.connect(str(WAREHOUSE_DB))
    con.execute("DROP TABLE IF EXISTS asset_candidates")
    con.execute(
        """
        CREATE TABLE asset_candidates (
            candidate_id VARCHAR PRIMARY KEY,
            proposed_name VARCHAR,
            synonyms VARCHAR[],
            sponsors_over_time JSON,
            trial_count INTEGER,
            nct_ids VARCHAR[],
            first_trial_start_date VARCHAR,
            last_trial_start_date VARCHAR,
            strategies VARCHAR[],
            ambiguous BOOLEAN,
            dev_code_only BOOLEAN,
            discovery_strategy VARCHAR,
            match_strength VARCHAR,
            matched_term VARCHAR,
            review_status VARCHAR
        )
        """
    )
    for c in candidates:
        con.execute(
            """
            INSERT INTO asset_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                c.candidate_id, c.proposed_name, c.synonyms,
                json.dumps(c.sponsors_over_time), c.trial_count, c.nct_ids,
                c.first_trial_start_date, c.last_trial_start_date,
                c.strategies, c.ambiguous, c.dev_code_only,
                c.discovery_strategy, c.match_strength, c.matched_term,
                c.review_status,
            ],
        )
    con.close()
    print(f"\nWrote {len(candidates)} rows to {WAREHOUSE_DB}::asset_candidates")

    # -- write CSV for spreadsheet review ---------------------------------
    csv_path = REPORTS_DIR / "candidate_universe.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "proposed_name", "synonyms", "sponsors",
            "trial_count", "first_trial_start_date", "last_trial_start_date",
            "strategies", "ambiguous", "dev_code_only",
            "discovery_strategy", "match_strength", "matched_term",
            "review_status", "nct_ids",
        ])
        for c in candidates:
            sponsor_str = "; ".join(
                f"{s['sponsor']} ({s['first_seen']}–{s['last_seen']})"
                for s in c.sponsors_over_time
            )
            writer.writerow([
                c.candidate_id, c.proposed_name, "; ".join(c.synonyms), sponsor_str,
                c.trial_count, c.first_trial_start_date, c.last_trial_start_date,
                "; ".join(c.strategies), c.ambiguous, c.dev_code_only,
                c.discovery_strategy, c.match_strength, c.matched_term,
                c.review_status, "; ".join(c.nct_ids),
            ])
    print(f"Wrote {csv_path}")

    # -- write audit report -------------------------------------------------
    report = render_report(
        candidates, seed_assets,
        fields_used={
            "suffix_terms": "vedotin, deruxtecan, govitecan, emtansine, tesirine, mafodotin, "
                             "ozogamicin, soravtansine, ravtansine, duocarmazine, tirumotecan",
            "literal_terms": "antibody-drug conjugate, antibody drug conjugate, "
                              "immunoconjugate, ADC, conjugate",
            "oncology_filter": "query.cond=cancer applied to every discovery search",
            "date_filter": "studies with a parseable start date before 2012-01-01 excluded; "
                            "studies with no/unparseable start date kept",
            "sponsor_class_filter": "not applied at trial level in discovery (kept for later "
                                     "normalisation); strategy 3 only scans sponsors whose lead "
                                     "class is INDUSTRY",
            "study_type_filter": "none applied — observational studies are included",
        },
    )
    report_path = REPORTS_DIR / "discovery_audit.md"
    report_path.write_text(report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    sys.exit(main())
