"""Render the B5 opportunity matrix's ranked graveyard-cell list — the
actual deliverable, per the user's explicit call that a ~300-in-scope-
program universe crossed against two coarsened axes (~10 payload classes
x ~8 tumour systems, ~80 cells) is still too sparse to read honestly as a
colour heatmap; report it as a ranked list instead. See
attributes/matrix.py's module docstring for the axis-coarsening
rationale, the exclusion rules, and the live/dead proxy.
"""
from __future__ import annotations


def render_graveyard_markdown(
    cells: dict, quadrant_by_key: dict, *, min_n: int, population_note: str = "",
) -> str:
    graveyard = [
        (key, cell) for key, cell in cells.items()
        if quadrant_by_key[key] == "graveyard"
    ]
    graveyard.sort(key=lambda kc: -kc[1].n_dead)

    lines = ["# Opportunity matrix — graveyard cells", ""]
    if population_note:
        lines += [population_note, ""]
    lines += [
        f"{len(graveyard)} cell(s) classified graveyard (few/no live programs, "
        f">= {min_n} dead), ranked by dead-program count, out of the payload "
        "chemotype x tumour system cells with >=1 program. Cells below min_n "
        "are insufficient_evidence, not graveyard — greyed out of this "
        "ranking, not read as white space. This list is the deliverable, not "
        "a heatmap: public data alone can't distinguish \"never tried\" from "
        "\"tried and quietly failed\"; this project's failure denominator can.",
        "",
    ]
    for (payload, system), cell in graveyard:
        lines.append(f"## {payload} × {system} — {cell.n_dead} dead, {cell.n_live} live")
        lines.append("")
        for p in sorted(cell.dead_programs, key=lambda x: x.get("proposed_name") or ""):
            reason = p.get("kill_reason") or p.get("status") or "unknown"
            proxy_note = "" if p.get("basis") == "gold" else " *(silence-score proxy, not gold-confirmed)*"
            lines.append(f"- **{p.get('proposed_name') or p['program_id']}** — {reason}{proxy_note}")
        lines.append("")
    if not graveyard:
        lines.append("(none at this min_n — try lowering it, or this is genuinely good news)")
    return "\n".join(lines)
