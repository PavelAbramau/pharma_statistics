"""Audit CT.gov's sponsor-selected leadSponsor.class field in both
directions before trusting it for scope decisions.

    python scripts/report_sponsor_class_candidates.py

CT.gov's class field is self-reported by whoever filled in the registry
record, not derived — it gets miscoded both ways. Prompted by "Shanghai
Institute Of Biological Products" — a state-owned commercial biologics
manufacturer under Sinopharm/CNBG despite the academic-sounding name; on
this universe's data it's actually already classed INDUSTRY, which is
exactly the kind of name/class mismatch this report exists to surface for
a human to confirm rather than trust silently either way. This matters
beyond individual cases: the EDA
found Chinese-sponsored programs differ on 6 of 8 features, so
systematically excluding state-owned Chinese sponsors on name appearance
alone would bias that whole stratum invisibly.

Two lists, both restricted to sponsors NOT already adjudicated in
sponsor_class_overrides.json (see trial_scope.py):

1. Possible false academics — classed something other than INDUSTRY, but
   either the name reads as commercial (contains a company-form or
   biologics/pharma term, or is affiliated with a known state-owned
   pharma group), or the sponsor is running a Phase 3 trial with over 100
   participants (academic sponsors rarely do that with an ADC).
2. Possible false industry — classed INDUSTRY, but the name reads as
   academic/nonprofit (university, hospital, cancer center, institute of,
   medical center, foundation).

Trial/phase counts are attributed per-trial (each trial's own
TrialSummary.sponsor — the lead sponsor CT.gov actually recorded on that
specific record), not by treating every trial on a multi-sponsor asset as
belonging to every sponsor that asset has ever had — the latter made a
first version of this report claim every academic collaborator on a
multi-sponsor ADC program was "running" that program's Phase 3 trial.
sponsors_over_time (asset-level) is used only to look up each sponsor
name's CT.gov class, which is a genuine 1:1 fact per name.

This script only ever PROPOSES candidates — it never writes
sponsor_class_overrides.json itself. Adjudicate by hand and add entries
there, keyed by the exact sponsor name, e.g.:

    {
      "Shanghai Institute Of Biological Products": {
        "override_class": "INDUSTRY",
        "reason": "State-owned commercial biologics manufacturer under "
                   "Sinopharm/CNBG, despite the academic-sounding name.",
        "decided_by": "<you>",
        "decided_at": "2026-08-30"
      }
    }
"""
from __future__ import annotations

import html
import re
from collections import Counter, defaultdict

from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import trial_scope as ts

FALSE_ACADEMIC_NAME_PATTERNS = [
    "pharmaceutical", "biologics", "biological products", "biotech",
    "co., ltd", "co.,ltd", "co ltd", "inc", "group",
    "sinopharm", "cnbg", "china resources", "shanghai pharma",
]
FALSE_INDUSTRY_NAME_PATTERNS = [
    "university", "hospital", "cancer center", "institute of",
    "medical center", "foundation",
]
LARGE_PHASE3_ENROLLMENT_THRESHOLD = 100


def _name_matches(name: str, patterns: list[str]) -> list[str]:
    lowered = (name or "").lower()
    hits = []
    for p in patterns:
        pl = p.lower()
        if pl[0].isalnum() and pl[-1].isalnum():
            # word-boundary regex for bare-word/phrase patterns — "inc" must
            # not match "Incyte", "group" must not match "Groupama"
            regex = r"\b" + re.escape(pl).replace(r"\ ", r"\s+") + r"\b"
            if re.search(regex, lowered):
                hits.append(p)
        elif pl in lowered:
            # punctuated patterns like "co., ltd" — \b doesn't behave
            # usefully around punctuation, so fall back to plain substring
            hits.append(p)
    return hits


def _build_sponsor_index(programs: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = defaultdict(lambda: {
        "raw_class": None, "programs": set(), "nct_ids": set(),
        "phases": Counter(), "has_large_phase3": False,
    })
    for p in programs:
        # sponsors_over_time is asset-level: a sponsor's CT.gov class is a
        # genuine 1:1 fact per name, so this source is fine for class and
        # for "which programs mention this sponsor at all".
        for s in p.get("sponsors_over_time") or []:
            name = s.get("sponsor")
            if not name:
                continue
            # normalize on the HTML-unescaped name — the same sponsor shows
            # up both escaped and not across different raw CT.gov snapshots
            # (see trial_scope.apply_sponsor_class_overrides); without this,
            # discovery's name and this trial's own recorded name silently
            # fail to join and the sponsor looks like it "has no class".
            entry = index[html.unescape(name)]
            entry["programs"].add(p["candidate_id"])
            if entry["raw_class"] is None:
                entry["raw_class"] = s.get("class")

        # Trial/phase/enrollment counts are attributed per-trial, via that
        # trial's OWN recorded lead sponsor (TrialSummary.sponsor) — never
        # by crediting every trial on a multi-sponsor asset to every
        # sponsor that asset has ever had (that inflated academic
        # collaborators on big multi-sponsor programs into false "running
        # a Phase 3 trial" hits in an earlier version of this script).
        for t in p.get("trials") or []:
            name = t.get("sponsor")
            if not name:
                continue
            entry = index[html.unescape(name)]
            entry["programs"].add(p["candidate_id"])
            entry["nct_ids"].add(t["nct_id"])
            for ph in t.get("phases") or []:
                entry["phases"][ph] += 1
            if "PHASE3" in (t.get("phases") or []) and \
                    (t.get("enrollment_count") or 0) > LARGE_PHASE3_ENROLLMENT_THRESHOLD:
                entry["has_large_phase3"] = True
    return dict(index)


def _print_list(title: str, rows: list[tuple]) -> None:
    print(f"=== {title} ({len(rows)}) ===")
    if not rows:
        print("(none)")
        print()
        return
    for name, raw_class, reasons, n_programs, n_trials, phases in sorted(rows, key=lambda r: -r[4]):
        phase_str = ", ".join(f"{ph}:{c}" for ph, c in sorted(phases.items())) or "(no phase on file)"
        print(f"- {name}")
        print(f"    class={raw_class!r}  reasons={reasons}")
        print(f"    programs={n_programs}  trials={n_trials}  phases={{{phase_str}}}")
    print()


def main() -> None:
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    overrides = ts.load_sponsor_class_overrides()
    index = _build_sponsor_index(programs)

    false_academics, false_industry, no_class_on_file = [], [], []
    for name, entry in index.items():
        if name in overrides:
            continue  # already adjudicated
        if entry["raw_class"] is None:
            # this exact sponsor-name string never appears in any
            # sponsors_over_time row — usually a spelling/legal-entity
            # variant between what discovery captured and what this
            # trial's own (possibly newer) snapshot recorded as the lead
            # sponsor. We genuinely don't know its class, so it's neither
            # a false academic nor a false industry claim — don't guess.
            no_class_on_file.append(name)
            continue
        raw_class = entry["raw_class"].upper()

        if raw_class != "INDUSTRY":
            reasons = _name_matches(name, FALSE_ACADEMIC_NAME_PATTERNS)
            if entry["has_large_phase3"]:
                reasons = reasons + [f"Phase 3 trial with >{LARGE_PHASE3_ENROLLMENT_THRESHOLD} participants"]
            if reasons:
                false_academics.append((
                    name, raw_class, reasons, len(entry["programs"]),
                    len(entry["nct_ids"]), entry["phases"],
                ))

        if raw_class == "INDUSTRY":
            reasons = _name_matches(name, FALSE_INDUSTRY_NAME_PATTERNS)
            if reasons:
                false_industry.append((
                    name, raw_class, reasons, len(entry["programs"]),
                    len(entry["nct_ids"]), entry["phases"],
                ))

    print(f"Loaded {len(programs)} provisional programs, {len(index)} distinct sponsors, "
          f"{len(overrides)} already adjudicated in {ts.SPONSOR_CLASS_OVERRIDES_PATH}.")
    if no_class_on_file:
        print(f"{len(no_class_on_file)} sponsor name(s) appear on a trial but have no class on file at all "
              f"(likely a name/legal-entity spelling variant between discovery and that trial's own "
              f"snapshot) — excluded from both lists below, not guessed into either: "
              f"{sorted(no_class_on_file)[:10]}{'...' if len(no_class_on_file) > 10 else ''}")
    print()
    _print_list("Possible false academics (classed non-INDUSTRY, reads commercial)", false_academics)
    _print_list("Possible false industry (classed INDUSTRY, reads academic/nonprofit)", false_industry)


if __name__ == "__main__":
    main()
