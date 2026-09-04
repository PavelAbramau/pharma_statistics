"""B5: crowding and failure density — the opportunity matrix.

Axes: payload chemotype (~10 CLAUDE.md classes) x tumour system
(~8 coarse organ/tissue groups, attributes/tumour_system.py) — around 80
cells. This replaces an earlier target x specific-MeSH-indication version:
~300 in-scope programs spread across ~350 HGNC targets x ~150 specific
MeSH terms can never produce a dense matrix — that's arithmetic, not a
data gap, and no amount of further labelling fixes it. Coarsening both
axes is the fix. See docs/decisions/0005 for the full rationale.

Both axes exclude unresolved values entirely rather than bucketing them
into a catch-all column/row ("undisclosed" payload, "unknown" system) —
a catch-all cell would itself become the single largest cell in the
matrix and would misrepresent "we don't know" as "this combination is
common":
  - payload chemotype "undisclosed" (attributes/payload.py — no INN
    suffix on file yet, the common case for early dev-code-only
    candidates) -> excluded.
  - tumour system unresolved (attributes/tumour_system.py — no MeSH data,
    or MeSH present but only a generic/root term like "Neoplasms" or a
    site-agnostic histology term like "Carcinoma") -> excluded. This is
    the "minimum MeSH tree depth" rule: a term that says nothing about
    body site is too shallow to be a matrix cell.

Population: programs confirmed is_adc=yes AND in_scope=yes (gold-first,
triage-staged fallback) — the same strict population attribute_coverage
reporting used, AND resolvable on both axes per the exclusion rules
above. Deliberately not widened to the broader is_adc=yes-only pool: that
pool still includes candidates that will turn out heme-only or
non-industry on Gate-2 review (~22% historically), which would
contaminate the very cells this matrix is trying to read honestly.

Live/dead is a PROXY, not a certain fact, for any program without a
gold gate-3 label (only 38 of ~1093 candidates have one):
  - gold status active/approved -> live
  - gold status dead_confirmed/dormant_suspected/superseded -> dead
    (superseded counts as dead FOR THIS PROGRAM even though the asset
    may continue elsewhere — Program is asset x indication x line, and
    this specific indication was dropped)
  - gold status unknown, or no gold label at all -> live UNLESS the
    silence-score heuristic's band is high (>= DEAD_PROXY_BAND_THRESHOLD)
    AND the program's history_coverage is "full" (never call something
    dead off an incomplete timeline) -> dead-by-proxy
This is exactly what makes the graveyard cells even computable before
enough Gate-3 labels exist — and exactly why it must never be reported
as more certain than "proxy."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pharma_stats.attributes import payload as pl
from pharma_stats.attributes import tumour_system as tsys
from pharma_stats.labelling import store

DEAD_PROXY_BAND_THRESHOLD = 3  # SCORE_BANDS index (0-4); band 3 = 60-80, band 4 = 80-100
DEAD_GOLD_STATUSES = {"dead_confirmed", "dormant_suspected", "superseded"}
LIVE_GOLD_STATUSES = {"active", "approved"}


@dataclass
class ProgramStatus:
    program_id: str
    is_dead: bool
    basis: str  # "gold" | "silence_proxy" | "assumed_live"
    status: Optional[str] = None
    kill_reason: Optional[str] = None


def program_live_dead_status(program: dict, gold_latest: dict) -> Optional[ProgramStatus]:
    """None if this program isn't confirmed in-scope at all (excluded
    from the matrix's population). in_scope (Gate 2) is gold-only by
    construction — Layer 2/3 triage only ever resolves is_adc (Gate 1),
    never in_scope (see triage_serve.py) — so a program with no gold
    record can never be in this matrix's population, regardless of what
    triage has staged for it."""
    pid = program["program_id"]
    g = gold_latest.get(pid)
    if g is None or not (g.get("is_adc") == "yes" and g.get("in_scope") == "yes"):
        return None

    status = g.get("status")
    if status in DEAD_GOLD_STATUSES:
        return ProgramStatus(pid, True, "gold", status, g.get("kill_reason"))
    if status in LIVE_GOLD_STATUSES:
        return ProgramStatus(pid, False, "gold", status)
    # gate-3 gold record exists but status is "unknown" or missing —
    # fall through to the silence proxy below rather than guessing

    if program.get("history_coverage") != "full" or program.get("band") is None:
        return ProgramStatus(pid, False, "assumed_live")  # incomplete timeline -> never call it dead
    is_dead = program["band"] >= DEAD_PROXY_BAND_THRESHOLD
    return ProgramStatus(pid, is_dead, "silence_proxy" if is_dead else "assumed_live")


@dataclass
class Cell:
    payload: str
    tumour_system: str
    live_programs: list[dict] = field(default_factory=list)
    dead_programs: list[dict] = field(default_factory=list)

    @property
    def n_live(self) -> int:
        return len(self.live_programs)

    @property
    def n_dead(self) -> int:
        return len(self.dead_programs)

    @property
    def total(self) -> int:
        return self.n_live + self.n_dead


def classify_quadrant(n_live: int, n_dead: int, min_n: int) -> str:
    """untested_white_space (0 evidence at all) is distinct from
    insufficient_evidence (some evidence, just not enough to trust —
    greyed out of the graveyard ranking, not read as white space, per the
    user's own framing). Above min_n: red_ocean (many live, no deaths
    seen), graveyard (few/no live, many dead), contested_and_hard
    (meaningful amounts of both) — a mixed cell that clears min_n but has
    neither side individually >= min_n still counts as contested_and_hard,
    the closest of the four to "some of both, no dominant read.\""""
    total = n_live + n_dead
    if total == 0:
        return "untested_white_space"
    if total < min_n:
        return "insufficient_evidence"
    many_live = n_live >= min_n
    many_dead = n_dead >= min_n
    if many_live and not many_dead:
        return "red_ocean"
    if not many_live and many_dead:
        return "graveyard"
    return "contested_and_hard"  # many_live and many_dead, or neither individually >= min_n


def build_matrix(
    programs: list[dict], *, min_n: int = 5,
) -> tuple[dict[tuple[str, str], Cell], dict[str, dict], dict[str, int]]:
    """(cells, program_attributes, population_stats).

    program_attributes keys every in-population program_id to
    {"payload", "tumour_system", ...} for caller-side reporting (e.g. the
    graveyard ranked list) without recomputing attribute derivation
    twice. population_stats counts why programs left the population at
    each stage, so callers can report the exclusion rules honestly
    instead of just a final headline number."""
    gold_records = store.load_records()
    gold_latest = store.latest_by_program(gold_records)

    cells: dict[tuple[str, str], Cell] = {}
    program_attributes: dict[str, dict] = {}
    n_scope_confirmed = 0
    n_excluded_payload_undisclosed = 0
    n_excluded_system_unresolved = 0

    for p in programs:
        pstatus = program_live_dead_status(p, gold_latest)
        if pstatus is None:
            continue
        n_scope_confirmed += 1

        name = p.get("proposed_name")
        synonyms = p.get("synonyms") or []

        chemotype = pl.derive_payload_chemotype(name, synonyms)
        if chemotype == "undisclosed":
            n_excluded_payload_undisclosed += 1
            continue

        system = tsys.program_tumour_system(p)
        if system is None:
            n_excluded_system_unresolved += 1
            continue

        key = (chemotype, system)
        cell = cells.setdefault(key, Cell(payload=chemotype, tumour_system=system))
        entry = {
            "program_id": p["program_id"], "proposed_name": name,
            "status": pstatus.status, "kill_reason": pstatus.kill_reason, "basis": pstatus.basis,
        }
        if pstatus.is_dead:
            cell.dead_programs.append(entry)
        else:
            cell.live_programs.append(entry)

        program_attributes[p["program_id"]] = {
            "payload": chemotype, "tumour_system": system,
            "is_dead": pstatus.is_dead, "basis": pstatus.basis,
            "status": pstatus.status, "kill_reason": pstatus.kill_reason,
        }

    population_stats = {
        "n_scope_confirmed": n_scope_confirmed,
        "n_excluded_payload_undisclosed": n_excluded_payload_undisclosed,
        "n_excluded_system_unresolved": n_excluded_system_unresolved,
        "n_in_population": len(program_attributes),
    }
    return cells, program_attributes, population_stats
