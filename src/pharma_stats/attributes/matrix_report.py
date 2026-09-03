"""Render the B5 opportunity matrix as a self-contained interactive HTML
page, plus the ranked graveyard-cell list (the actual deliverable — see
matrix.py's module docstring on why the graveyard quadrant is the point).
No external libraries — vanilla JS/CSS, same convention as
triage/report.py's blind validation page.
"""
from __future__ import annotations

import html
import json

from pharma_stats.attributes.matrix import Cell

QUADRANT_COLORS = {
    "red_ocean": "#c0392b",
    "contested_and_hard": "#e67e22",
    "graveyard": "#7f8c8d",
    "untested_white_space": "#2c3e50",
    "insufficient_evidence": "#1a1d22",
}
QUADRANT_LABELS = {
    "red_ocean": "Red ocean",
    "contested_and_hard": "Contested & hard",
    "graveyard": "Graveyard",
    "untested_white_space": "Untested white space",
    "insufficient_evidence": "Insufficient evidence",
}


def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def render_matrix_html(
    cells: dict, quadrant_by_key: dict, *, min_n: int, population_note: str,
) -> str:
    targets = sorted({k[0] for k in cells})
    indications = sorted({k[1] for k in cells})

    cell_data = {}
    for key, cell in cells.items():
        t, i = key
        cell_data[f"{t}|||{i}"] = {
            "quadrant": quadrant_by_key[key],
            "n_live": cell.n_live, "n_dead": cell.n_dead,
            "live": cell.live_programs, "dead": cell.dead_programs,
        }

    rows_html = []
    for indication in indications:
        cells_html = []
        for target in targets:
            key = f"{target}|||{indication}"
            data = cell_data.get(key)
            if data is None:
                cells_html.append('<td class="cell empty"></td>')
                continue
            q = data["quadrant"]
            color = QUADRANT_COLORS[q]
            label = f'{data["n_live"]}L / {data["n_dead"]}D' if q != "untested_white_space" else "—"
            cells_html.append(
                f'<td class="cell" style="background:{color}" '
                f'data-key="{_esc(key)}" data-quadrant="{_esc(q)}">{_esc(label)}</td>'
            )
        rows_html.append(f'<tr><th class="row-label">{_esc(indication)}</th>{"".join(cells_html)}</tr>')

    header_cells = "".join(f'<th class="col-label">{_esc(t)}</th>' for t in targets)
    legend = "".join(
        f'<span class="legend-item"><span class="swatch" style="background:{c}"></span>{_esc(QUADRANT_LABELS[q])}</span>'
        for q, c in QUADRANT_COLORS.items()
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ADC opportunity matrix</title>
<style>
  body {{ background:#0f1216; color:#e6e9ef; font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif; margin:0; padding:20px; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  .note {{ color:#8b94a3; font-size:12.5px; margin-bottom:16px; max-width:900px; }}
  .note b {{ color:#e8b04b; }}
  .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:14px; font-size:12px; }}
  .legend-item {{ display:flex; align-items:center; gap:5px; }}
  .swatch {{ width:12px; height:12px; border-radius:2px; display:inline-block; }}
  table {{ border-collapse:collapse; font-size:11.5px; }}
  th, td {{ border:1px solid #2a313c; padding:0; }}
  th.col-label {{ writing-mode:vertical-rl; transform:rotate(180deg); padding:6px 3px; max-width:22px; font-weight:500; color:#8b94a3; }}
  th.row-label {{ text-align:right; padding:4px 8px; white-space:nowrap; color:#8b94a3; font-weight:500; position:sticky; left:0; background:#0f1216; }}
  td.cell {{ width:26px; height:22px; text-align:center; font-size:9.5px; cursor:pointer; color:#fff; }}
  td.cell.empty {{ background:transparent; border-color:#1a1d22; }}
  #tooltip {{ position:fixed; display:none; background:#171b21; border:1px solid #2a313c; border-radius:6px;
    padding:10px 12px; max-width:420px; font-size:12px; z-index:50; box-shadow:0 4px 16px rgba(0,0,0,.4); }}
  #tooltip h3 {{ margin:0 0 6px; font-size:12.5px; }}
  #tooltip .prog {{ padding:2px 0; border-top:1px dashed #2a313c; }}
  #tooltip .prog:first-of-type {{ border-top:none; }}
  #tooltip .dead {{ color:#ff5d6c; }}
  #tooltip .live {{ color:#35c98f; }}
</style></head>
<body>
<h1>ADC opportunity matrix — target × indication</h1>
<div class="note">
  {population_note}<br>
  min_n = <b>{min_n}</b> (cells below this total are greyed as insufficient evidence, not shown as white space).
  Live/dead status beyond gold gate-3 labels is a <b>silence-score proxy</b>, not a certain fact — see
  docs/decisions/0003 and attributes/matrix.py's module docstring.
</div>
<div class="legend">{legend}</div>
<div style="overflow:auto; max-height:82vh">
<table>
<tr><th></th>{header_cells}</tr>
{"".join(rows_html)}
</table>
</div>
<div id="tooltip"></div>
<script>
const CELLS = {json.dumps(cell_data)};
const tip = document.getElementById("tooltip");
document.querySelectorAll("td.cell[data-key]").forEach(td => {{
  td.addEventListener("mouseenter", () => {{
    const d = CELLS[td.dataset.key];
    const [target, indication] = td.dataset.key.split("|||");
    let h = `<h3>${{target}} × ${{indication}}</h3>`;
    h += `<div>${{d.n_live}} live, ${{d.n_dead}} dead — ${{td.dataset.quadrant.replace(/_/g," ")}}</div>`;
    if (d.dead.length) {{
      h += d.dead.map(p => `<div class="prog dead">✖ ${{p.proposed_name || p.program_id}}` +
        (p.kill_reason ? ` — ${{p.kill_reason}}` : (p.status ? ` — ${{p.status}}` : "")) +
        (p.basis !== "gold" ? " (proxy)" : "") + `</div>`).join("");
    }}
    if (d.live.length) {{
      h += d.live.map(p => `<div class="prog live">● ${{p.proposed_name || p.program_id}}</div>`).join("");
    }}
    tip.innerHTML = h;
    tip.style.display = "block";
  }});
  td.addEventListener("mousemove", e => {{
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 440) + "px";
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - 200) + "px";
  }});
  td.addEventListener("mouseleave", () => {{ tip.style.display = "none"; }});
}});
</script>
</body></html>
"""


def render_graveyard_markdown(cells: dict, quadrant_by_key: dict, *, min_n: int) -> str:
    graveyard = [
        (key, cell) for key, cell in cells.items()
        if quadrant_by_key[key] == "graveyard"
    ]
    graveyard.sort(key=lambda kc: -kc[1].n_dead)

    lines = [
        "# Opportunity matrix — graveyard cells",
        "",
        f"{len(graveyard)} cell(s) classified graveyard (few/no live programs, "
        f">= {min_n} dead), ranked by dead-program count. This list is the deliverable — "
        "public data alone can't distinguish \"never tried\" from \"tried and quietly failed\"; "
        "this project's failure denominator can.",
        "",
    ]
    for (target, indication), cell in graveyard:
        lines.append(f"## {target} × {indication} — {cell.n_dead} dead, {cell.n_live} live")
        lines.append("")
        for p in sorted(cell.dead_programs, key=lambda x: x.get("proposed_name") or ""):
            reason = p.get("kill_reason") or p.get("status") or "unknown"
            proxy_note = "" if p.get("basis") == "gold" else " *(silence-score proxy, not gold-confirmed)*"
            lines.append(f"- **{p.get('proposed_name') or p['program_id']}** — {reason}{proxy_note}")
        lines.append("")
    if not graveyard:
        lines.append("(none at this min_n — try lowering it, or this is genuinely good news)")
    return "\n".join(lines)
