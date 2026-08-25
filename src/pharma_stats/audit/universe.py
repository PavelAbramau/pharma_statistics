"""Universe stage: discovery coverage — recall probe against known ADCs,
duplicate/orphan clustering issues, review backlog, and the strategy
saturation curve (is discovery actually done, or would a 4th strategy
still find a lot?)."""
from __future__ import annotations

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok, warn
from pharma_stats.config import REPO_ROOT, WAREHOUSE_DB
from pharma_stats.discovery.candidates import load_seed_assets

STAGE = "universe"
KNOWN_ADCS_PATH = REPO_ROOT / "tests" / "fixtures" / "known_adcs.txt"
STRATEGY_ORDER = ["pattern_match", "seed_expansion", "sponsor_expansion"]


def run() -> list[Check]:
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)
    try:
        candidates = _load_candidates(con)
    finally:
        con.close()

    checks: list[Check] = []
    checks += _recall_probe(candidates)
    checks += _clustering_and_backlog(candidates)
    checks += _saturation_curve(candidates)
    return checks


def _load_candidates(con) -> list[dict]:
    cols = ["candidate_id", "proposed_name", "synonyms", "nct_ids", "strategies", "review_status", "ambiguous"]
    rows = con.execute(f"SELECT {', '.join(cols)} FROM asset_candidates").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _load_known_adcs() -> list[list[str]]:
    if not KNOWN_ADCS_PATH.exists():
        return []
    out = []
    for line in KNOWN_ADCS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names = [n.strip() for n in line.split("|") if n.strip()]
        if names:
            out.append(names)
    return out


def _recall_probe(candidates: list[dict]) -> list[Check]:
    known = _load_known_adcs()
    if not known:
        return [info(
            STAGE, "recall probe vs known ADC list",
            expected=">=1 hand-maintained known asset in tests/fixtures/known_adcs.txt",
            actual="fixture missing or empty", detail="",
        )]

    candidate_name_sets = [
        {c["proposed_name"].lower()} | {s.lower() for s in (c["synonyms"] or [])}
        for c in candidates
    ]

    missing = []
    for names in known:
        norm = {n.lower() for n in names}
        if not any(norm & cand_names for cand_names in candidate_name_sets):
            missing.append(names[0])

    checks = [(warn if missing else ok)(
        STAGE, "recall probe: known ADCs found in asset_candidates",
        expected=f"{len(known)} / {len(known)} found",
        actual=f"{len(known) - len(missing)} / {len(known)} found",
        detail="missing: " + ", ".join(missing) if missing else "",
    )]
    checks.append(_independence_check(known))
    return checks


def _independence_check(known: list[list[str]]) -> Check:
    """Does known_adcs.txt actually differ from the seed list fed into
    discovery's seed_expansion? If it's identical, a "found" result is
    circular — of course seed_expansion finds exactly what it was told
    to search for. Compares actual file contents each run rather than a
    hardcoded claim, so this stays accurate as either file changes."""
    seed_names = {
        n.lower() for seed in load_seed_assets()
        for n in [seed["name"], *seed.get("synonyms", [])]
    }
    known_flat = {n.lower() for names in known for n in names}
    overlap = known_flat & seed_names
    identical_or_subset = known_flat and known_flat <= seed_names

    if identical_or_subset:
        return warn(
            STAGE, "known_adcs.txt independence from discovery/seed_assets.json",
            expected="contains names not present in seed_assets.json",
            actual="every name in known_adcs.txt also appears in seed_assets.json — "
                   "the recall probe above is circular",
            detail="",
        )
    return ok(
        STAGE, "known_adcs.txt independence from discovery/seed_assets.json",
        expected="contains names not present in seed_assets.json",
        actual=f"{len(known_flat - seed_names)} / {len(known_flat)} names are not in "
               f"seed_assets.json ({len(overlap)} overlap, expected for well-known ADCs)",
        detail="",
    )


def _clustering_and_backlog(candidates: list[dict]) -> list[Check]:
    by_id = {c["candidate_id"]: c for c in candidates}
    zero_trial = [c["candidate_id"] for c in candidates if not c["nct_ids"]]

    nct_to_candidates: dict[str, set[str]] = {}
    for c in candidates:
        for nct_id in c["nct_ids"] or []:
            nct_to_candidates.setdefault(nct_id, set()).add(c["candidate_id"])
    overlaps = {nct: cs for nct, cs in nct_to_candidates.items() if len(cs) > 1}

    def is_verified(cid: str) -> bool:
        """A pattern/seed hit only counts as verified if it wasn't merely
        a weak literal-term match (e.g. a generic "TROP2 ADC" arm-slot
        label matches the literal "adc" term and gets tagged
        pattern_match, but candidates.py itself flags that as ambiguous —
        respect that flag here rather than trusting the strategy tag
        alone, or a generic label reads as "verified"."""
        c = by_id[cid]
        return bool(set(c["strategies"] or []) & {"pattern_match", "seed_expansion"}) and not c["ambiguous"]

    genuine_combo = {  # every claimant independently pattern/seed-verified — likely a real shared trial
        nct: cs for nct, cs in overlaps.items() if all(is_verified(c) for c in cs)
    }
    likely_noise = {nct: cs for nct, cs in overlaps.items() if nct not in genuine_combo}

    unreviewed = [c["candidate_id"] for c in candidates if c["review_status"] == "unreviewed"]

    return [
        (warn if zero_trial else ok)(
            STAGE, "candidate assets with zero linked trials",
            expected="0", actual=str(len(zero_trial)), detail=", ".join(zero_trial[:10]),
        ),
        info(
            STAGE, "trials shared by 2+ independently-verified candidates (likely genuine combination trials)",
            expected="modelled as many-to-many trial<->asset links once the five-entity warehouse "
                     "exists; not a clustering bug on its own",
            actual=f"{len(genuine_combo)} trials",
            detail="; ".join(f"{n}: {sorted(cs)}" for n, cs in list(genuine_combo.items())[:10]),
        ),
        (warn if likely_noise else ok)(
            STAGE, "trials shared where at least one claimant is unverified (sponsor_expansion-only "
                   "dev-code guess) — likely clustering noise, not a real shared trial",
            expected="0", actual=f"{len(likely_noise)} trials",
            detail="; ".join(f"{n}: {sorted(cs)}" for n, cs in list(likely_noise.items())[:10]),
        ),
        (fail if unreviewed else ok)(
            STAGE, "unreviewed candidates in the human review queue",
            expected="0 — this is Phase 0's exit condition; downstream stages should not run on "
                     "an unreviewed candidate universe",
            actual=f"{len(unreviewed)} / {len(candidates)}", detail="",
        ),
    ]


def _saturation_curve(candidates: list[dict]) -> list[Check]:
    counts = {s: 0 for s in STRATEGY_ORDER}
    unrecognised = 0
    for c in candidates:
        strategies = c["strategies"] or []
        found_by = next((s for s in STRATEGY_ORDER if s in strategies), None)
        if found_by is None:
            unrecognised += 1
        else:
            counts[found_by] += 1

    total = sum(counts.values())
    lines, cumulative = [], 0
    last_share = 0.0
    for s in STRATEGY_ORDER:
        cumulative += counts[s]
        share = counts[s] / total if total else 0.0
        lines.append(f"{s}: +{counts[s]} new candidates ({share:.0%}), cumulative {cumulative}")
        last_share = share

    level = warn if last_share > 0.15 else info
    detail = " | ".join(lines)
    if unrecognised:
        detail += f" | {unrecognised} candidates with no recognised strategy tag"

    return [level(
        STAGE, "discovery saturation curve (new candidates per additional search strategy, in run order)",
        expected="the last strategy (sponsor_expansion) contributing a small, shrinking share",
        actual=f"sponsor_expansion added {last_share:.0%} of all {total} candidates",
        detail=detail,
    )]
