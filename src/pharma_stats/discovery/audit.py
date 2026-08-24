"""Coverage/audit report over the candidate asset table — the thing a human
actually reads to sanity-check discovery before trusting it."""
from __future__ import annotations

from collections import Counter
from typing import Optional

from pharma_stats.discovery.candidates import CandidateAsset


def single_strategy_candidates(candidates: list[CandidateAsset]) -> list[CandidateAsset]:
    return [c for c in candidates if len(c.strategies) == 1]


def ambiguous_candidates(candidates: list[CandidateAsset]) -> list[CandidateAsset]:
    return [c for c in candidates if c.ambiguous]


def dev_code_only_candidates(candidates: list[CandidateAsset]) -> list[CandidateAsset]:
    return [c for c in candidates if c.dev_code_only]


def seed_recovery(candidates: list[CandidateAsset], seed_assets: list[dict]) -> list[dict]:
    """For each seed asset, did discovery actually find it? Sanity check
    against ADCs the user already knows exist."""
    all_synonym_norms: list[set[str]] = []
    for c in candidates:
        names = {c.proposed_name.lower()} | {s.lower() for s in c.synonyms}
        all_synonym_norms.append(names)

    rows = []
    for seed in seed_assets:
        seed_terms = {seed["name"].lower()} | {s.lower() for s in seed.get("synonyms", [])}
        found = any(names & seed_terms for names in all_synonym_norms)
        rows.append({"seed_name": seed["name"], "found": found})
    return rows


def render_report(
    candidates: list[CandidateAsset],
    seed_assets: list[dict],
    *,
    fields_used: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    lines.append("# ADC candidate-universe discovery — audit report\n")

    lines.append(f"**Total candidate assets:** {len(candidates)}\n")
    strategy_counts = Counter()
    for c in candidates:
        for s in c.strategies:
            strategy_counts[s] += 1
    lines.append("**By strategy (a candidate can be counted in more than one):**\n")
    for strat, n in sorted(strategy_counts.items()):
        lines.append(f"- {strat}: {n}")
    lines.append("")

    trial_counts = [c.trial_count for c in candidates]
    if trial_counts:
        lines.append(
            f"**Trials per candidate:** min={min(trial_counts)}, "
            f"median={sorted(trial_counts)[len(trial_counts) // 2]}, max={max(trial_counts)}\n"
        )
    total_trials = len({nct for c in candidates for nct in c.nct_ids})
    lines.append(f"**Distinct trials touched:** {total_trials}\n")

    lines.append("## Seed-list recovery (sanity check)\n")
    lines.append(
        "Did discovery actually find each asset the user already knows exists? "
        "A seed missing here means the union of the three strategies has a real gap.\n"
    )
    recovery = seed_recovery(candidates, seed_assets)
    missed = [r for r in recovery if not r["found"]]
    lines.append(f"Recovered {len(recovery) - len(missed)}/{len(recovery)} seed assets.\n")
    if missed:
        lines.append("**Missed:**\n")
        for r in missed:
            lines.append(f"- {r['seed_name']}")
        lines.append("")

    lines.append("## Candidates caught by only one strategy\n")
    lines.append(
        "These are the least corroborated — a single search path found them and nothing "
        "else confirmed them. Review by hand. Split into two groups below: candidates with "
        "at least some naming-pattern/seed evidence they're an ADC, and pure sponsor-expansion "
        "dev-code guesses that carry no ADC signal at all beyond \"looks like a compound code "
        "from a company that also makes an ADC\" (see the dedicated section further down — most "
        "of the volume, and most of the noise, is there).\n"
    )
    single = single_strategy_candidates(candidates)
    single_evidenced = [c for c in single if not c.dev_code_only]
    lines.append(
        f"{len(single)} total single-strategy candidates "
        f"({len(single_evidenced)} with naming evidence, "
        f"{len(single) - len(single_evidenced)} dev-code-only — see below).\n"
    )
    lines.append("| candidate | strategy | trials | synonyms |")
    lines.append("|---|---|---|---|")
    for c in single_evidenced:
        syn = ", ".join(c.synonyms[:4]) + (" ..." if len(c.synonyms) > 4 else "")
        lines.append(f"| {c.proposed_name} | {c.strategies[0]} | {c.trial_count} | {syn} |")
    lines.append("")

    dev_code_only = dev_code_only_candidates(candidates)
    lines.append("## Sponsor-expansion dev-code-only candidates (likely mostly noise)\n")
    lines.append(
        "These carry **no ADC signal at all** — they were pulled in only because they're an "
        "unnamed development-stage compound code (e.g. \"ABBV-011\", \"AZD1775\") from a sponsor "
        "that has a confirmed ADC elsewhere in their pipeline. The dev-code heuristic can't tell "
        "an ADC apart from any other modality (kinase inhibitor, bispecific, cytokine, ...) that "
        "same sponsor is also developing, so most of this bucket is expected to be false "
        "positives — it exists because strategy 3 (\"scan their full oncology trial list for "
        "candidates\") is explicitly recall-first, per your instruction, and true early-stage "
        "ADCs without public names are otherwise undetectable from CT.gov alone. This is the "
        "least-reliable part of this candidate universe; consider it a shortlist to skim rather "
        "than review row-by-row. Full list with sponsor/trial detail is in "
        "candidate_universe.csv.\n"
    )
    lines.append(f"{len(dev_code_only)} candidates ({sum(c.trial_count for c in dev_code_only)} "
                 f"mention-trials).\n")
    lines.append("Top 30 by trial count (more trials = slightly more likely to be a real, "
                 "actively-developed asset worth a closer look):\n")
    lines.append("| candidate | sponsor(s) | trials |")
    lines.append("|---|---|---|")
    for c in sorted(dev_code_only, key=lambda c: -c.trial_count)[:30]:
        sponsors = ", ".join(s["sponsor"] for s in c.sponsors_over_time[:2])
        lines.append(f"| {c.proposed_name} | {sponsors} | {c.trial_count} |")
    lines.append("")

    lines.append("## Ambiguous candidates (literal-term-only match)\n")
    lines.append(
        "Caught only via a weak literal term (\"ADC\", \"conjugate\", ...) with no "
        "suffix-pattern or seed corroboration. Some of these are real; some are noise "
        "(vaccine conjugates, unrelated chemistry). Review by hand.\n"
    )
    ambiguous = ambiguous_candidates(candidates)
    lines.append(f"{len(ambiguous)} candidates.\n")
    lines.append("| candidate | trials | synonyms |")
    lines.append("|---|---|---|")
    for c in ambiguous:
        syn = ", ".join(c.synonyms[:4]) + (" ..." if len(c.synonyms) > 4 else "")
        lines.append(f"| {c.proposed_name} | {c.trial_count} | {syn} |")
    lines.append("")

    if fields_used:
        lines.append("## Discovery parameters\n")
        for k, v in fields_used.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    return "\n".join(lines)
