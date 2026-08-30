"""Universe stage: discovery coverage — recall probe against known ADCs,
duplicate/orphan clustering issues, review backlog, and the strategy
saturation curve (is discovery actually done, or would a 4th strategy
still find a lot?)."""
from __future__ import annotations

import re

import duckdb

from pharma_stats.audit.types import Check, fail, info, ok, warn
from pharma_stats.config import REPO_ROOT, WAREHOUSE_DB
from pharma_stats.discovery.candidates import genuine_combo_trial_ids, load_seed_assets
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.labelling import trial_scope as ts

STAGE = "universe"
KNOWN_ADCS_PATH = REPO_ROOT / "tests" / "fixtures" / "known_adcs.txt"
STRATEGY_ORDER = ["pattern_match", "seed_expansion", "sponsor_expansion"]


def _word_boundary_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


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
    checks += _heme_solid_span_report()
    checks += _heme_only_auto_exclusion_agreement()
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

    candidate_names = [
        n.lower() for c in candidates
        for n in [c["proposed_name"], *(c["synonyms"] or [])]
    ]

    missing = []
    for names in known:
        norm = [n.lower() for n in names]
        # exact match OR word-boundary substring either direction — a
        # candidate's proposed_name is often the raw CT.gov intervention
        # string plus a parenthetical ("Enapotamab vedotin (HuMax-AXL-
        # ADC)"), so an exact-set match alone produces false "missing"
        # verdicts for assets discovery actually found. Diagnosed
        # 2026-08-26: 2 of 26 originally-"missing" assets were exactly
        # this — already correctly discovered, just under a longer name.
        found = any(
            n == cn or _word_boundary_contains(cn, n) or _word_boundary_contains(n, cn)
            for n in norm for cn in candidate_names
        )
        if not found:
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
    zero_trial = [c["candidate_id"] for c in candidates if not c["nct_ids"]]

    nct_to_candidates: dict[str, set[str]] = {}
    for c in candidates:
        for nct_id in c["nct_ids"] or []:
            nct_to_candidates.setdefault(nct_id, set()).add(c["candidate_id"])
    overlaps = {nct: cs for nct, cs in nct_to_candidates.items() if len(cs) > 1}

    # shared source of truth with the labelling app's provisional program
    # builder (pharma_stats.labelling.provisional_programs) — both need
    # the same answer to "is this shared trial real or noise"
    genuine_combo_ids = genuine_combo_trial_ids(candidates)
    genuine_combo = {nct: cs for nct, cs in overlaps.items() if nct in genuine_combo_ids}
    likely_noise = {nct: cs for nct, cs in overlaps.items() if nct not in genuine_combo_ids}

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


def _heme_solid_span_report() -> list[Check]:
    """Assets whose trials MeSH-classify as both heme and solid — the ones
    a naive whole-asset keyword filter (or a careless whole-asset in_scope
    call) would wrongly kill on their heme trials alone. See
    labelling/trial_scope.py / discovery/mesh_categories.py."""
    programs = pp.load_materialized()
    if not programs:
        return [info(
            STAGE, "assets spanning both heme and solid-tumour trials (MeSH-classified)",
            expected="provisional_programs materialized", actual="not materialized yet",
            detail="run the labelling app once, or pharma_stats.labelling.provisional_programs.materialize()",
        )]
    spanning = [p for p in programs if p.get("spans_heme_and_solid")]
    return [info(
        STAGE, "assets spanning both heme and solid-tumour trials (MeSH-classified)",
        expected="n/a — informational: a naive whole-asset filter would have wrongly excluded "
                 "every one of these on their heme trials alone",
        actual=f"{len(spanning)} / {len(programs)} assets",
        detail=", ".join(p["proposed_name"] for p in spanning[:15]),
    )]


def _heme_only_auto_exclusion_agreement() -> list[Check]:
    """The kill-switch: agreement between the classifier's heme_only
    prediction and the reviewer's own blind decision on the held-out
    validation sample (scripts/apply_auto_scope_exclusions.py). Below
    trial_scope.AGREEMENT_THRESHOLD, that script refuses to write further
    auto-exclusions — this check is what would tell you that's happening."""
    sample = ts.load_validation_sample()
    name = "heme_only auto-exclusion: blind validation agreement"
    if not sample:
        return [info(
            STAGE, name,
            expected=f">={ts.AGREEMENT_THRESHOLD:.0%} agreement before auto-exclusion is trusted",
            actual="no validation sample yet",
            detail="run scripts/apply_auto_scope_exclusions.py to draw one",
        )]

    records = store.load_records()
    result = ts.validation_agreement(sample, records)
    if result["agreement_rate"] is None:
        return [info(
            STAGE, name,
            expected=f">={ts.AGREEMENT_THRESHOLD:.0%} agreement before auto-exclusion is trusted",
            actual=f"0 / {len(sample)} of the held-out sample reviewed yet", detail="",
        )]

    level = ok if result["agreement_rate"] >= ts.AGREEMENT_THRESHOLD else fail
    return [level(
        STAGE, name,
        expected=f">={ts.AGREEMENT_THRESHOLD:.0%} agreement between the classifier's heme_only "
                 "prediction and the reviewer's own (blind) scope call",
        actual=f"{result['agreements']} / {result['compared']} ({result['agreement_rate']:.0%}) "
               f"of {len(sample)} held out",
        detail="below threshold: scripts/apply_auto_scope_exclusions.py refuses to write further "
               "auto-exclusions until this recovers",
    )]


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
