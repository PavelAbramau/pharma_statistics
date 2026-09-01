"""HTML reports for the Layer 2 pilot and the blind validation sample."""
from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pharma_stats.triage import grounding

_CSS = """
:root { color-scheme: dark; --bg:#0f1216; --panel:#171b21; --border:#2a313c;
  --text:#e6e9ef; --muted:#8b94a3; --accent:#4da3ff; --good:#35c98f;
  --bad:#ff5d6c; --warn:#e8b04b; --mono:"SFMono-Regular",Consolas,Menlo,monospace; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:14px/1.45 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 22px; margin: 0 0 6px; }
.meta { color: var(--muted); margin-bottom: 18px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin: 16px 0 22px; }
.stat { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }
.stat b { display:block; font-size:20px; }
.stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
.filters { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
.filters button { background:var(--panel); color:var(--text); border:1px solid var(--border);
  border-radius:6px; padding:5px 10px; cursor:pointer; }
.filters button.on { border-color:var(--accent); color:var(--accent); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; vertical-align:top; padding:7px 8px; border-bottom:1px solid var(--border); }
th { color:var(--muted); font-weight:500; position:sticky; top:0; background:var(--bg); }
.yes { color:var(--good); } .no { color:var(--bad); } .unsure { color:var(--warn); }
.flag { color:var(--warn); font-size:11px; }
.quote { font-family:var(--mono); font-size:12px; color:var(--muted); }
.recall { color:var(--warn); }
.l3 { color:var(--accent); }
"""


def parse_pilot_markdown(path: Path) -> tuple[str, list[dict]]:
    """(header_line, rows) from reports/triage_pilot.md."""
    header = ""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Run "):
            header = line
            continue
        if not line.startswith("| ") or line.startswith("| name") or line.startswith("|---"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        # Old: name, verdict, confidence, quote, from_recall, l3
        # New: name, verdict, confidence, evidence, quote, from_recall, l3
        if len(parts) >= 7:
            quote, fr_s, l3_s = parts[4], parts[5], parts[6]
            evidence = parts[3]
        elif len(parts) >= 6:
            quote, fr_s, l3_s = parts[3], parts[4], parts[5]
            evidence = None
        else:
            continue
        rows.append({
            "name": parts[0],
            "verdict": parts[1],
            "confidence": parts[2],
            "evidence_source": evidence,
            "quote": quote if quote else None,
            "from_recall": fr_s == "True",
            "routed_to_layer3": l3_s == "True",
        })
    return header, rows


def _annotate(row: dict) -> dict:
    """Derive evidence_source + grounding flags for a stored pilot row
    (which may still have the old text/recall 'confidence' column)."""
    out = dict(row)
    quote = out.get("quote")
    verdict = out.get("verdict")
    from_recall = bool(out.get("from_recall"))
    forced = False
    if verdict in ("yes", "no") and not from_recall:
        from_recall, forced = grounding.apply_grounding(verdict, from_recall, quote)
    src = grounding.evidence_source(verdict, from_recall, quote)
    conf = out.get("confidence") or ""
    if conf in ("text", "recall", ""):
        # old column duplicated from_recall — don't show it as confidence
        conf = out.get("k_confidence") or "—"
    out["from_recall"] = from_recall
    out["evidence_source"] = src
    out["confidence"] = conf
    out["grounding_forced_recall"] = forced
    out["would_route_l3"] = bool(
        out.get("routed_to_layer3") or from_recall or verdict == "unsure"
    )
    return out


def _stat_block(rows: list[dict]) -> dict:
    n = len(rows)
    verdicts = Counter(r["verdict"] for r in rows)
    sources = Counter(r["evidence_source"] for r in rows)
    n_l3 = sum(1 for r in rows if r["would_route_l3"])
    n_forced = sum(1 for r in rows if r.get("grounding_forced_recall"))
    n_empty_unsure = sum(
        1 for r in rows
        if r["verdict"] == "unsure" and r["evidence_source"] == "no_usable_evidence"
    )
    return {
        "n": n,
        "yes": verdicts.get("yes", 0),
        "no": verdicts.get("no", 0),
        "unsure": verdicts.get("unsure", 0),
        "text": sources.get("text", 0),
        "recall": sources.get("recall", 0),
        "no_usable_evidence": sources.get("no_usable_evidence", 0),
        "n_l3": n_l3,
        "l3_rate": (n_l3 / n) if n else 0.0,
        "n_forced": n_forced,
        "n_empty_unsure": n_empty_unsure,
    }


def render_pilot_html(
    rows: list[dict], *, header: str = "", spend: Optional[str] = None,
) -> str:
    annotated = [_annotate(r) for r in rows]
    s = _stat_block(annotated)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    for k, label in (
        ("n", "candidates"),
        ("yes", "yes"),
        ("no", "no"),
        ("unsure", "unsure"),
        ("text", "text-grounded"),
        ("recall", "recall"),
        ("no_usable_evidence", "no usable evidence"),
        ("n_l3", "→ Layer 3"),
        ("n_forced", "grounding forced recall"),
        ("n_empty_unsure", "unsure, empty quote"),
    ):
        val = s[k]
        if k == "n_l3":
            val = f"{s['n_l3']} ({s['l3_rate']:.0%})"
        cards.append(f'<div class="stat"><span>{html.escape(label)}</span><b>{val}</b></div>')

    body_rows = []
    for r in annotated:
        v = r["verdict"]
        flags = []
        if r.get("grounding_forced_recall"):
            flags.append("quote does not ground verdict — treat as recall, route to Layer 3")
        if v == "unsure" and r["evidence_source"] == "no_usable_evidence":
            flags.append("unsure with no quote: not text-grounded")
        flag_html = "".join(f'<div class="flag">{html.escape(f)}</div>' for f in flags)
        l3 = "yes" if r["would_route_l3"] else "no"
        body_rows.append(
            f'<tr data-verdict="{html.escape(v)}" data-src="{html.escape(r["evidence_source"])}" data-l3="{l3}">'
            f'<td>{html.escape(r["name"])}</td>'
            f'<td class="{html.escape(v)}">{html.escape(v)}</td>'
            f'<td>{html.escape(str(r["confidence"]))}</td>'
            f'<td class="{"recall" if r["evidence_source"]!="text" else ""}">{html.escape(r["evidence_source"])}</td>'
            f'<td class="quote">{html.escape(r.get("quote") or "")}</td>'
            f'<td>{r["from_recall"]}</td>'
            f'<td class="{"l3" if r["would_route_l3"] else ""}">{r["would_route_l3"]}</td>'
            f'<td>{flag_html}</td></tr>'
        )

    spend_note = f" Spend {html.escape(spend)}." if spend else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Triage pilot</title>
<style>{_CSS}</style></head><body><main>
<h1>Triage Layer 2 pilot</h1>
<p class="meta">{html.escape(header)}{spend_note} Generated {generated}.
Review quotes and grounding, not whether the verdict looks plausible.
Confidence is unanimous / escalated-and-resolved / escalated-and-split
when the live run recorded it; older rows show — because the markdown
column was duplicating from_recall as text/recall.</p>
<div class="stats">{''.join(cards)}</div>
<div class="filters">
  <button class="on" data-filter="all">all</button>
  <button data-filter="yes">yes</button>
  <button data-filter="no">no</button>
  <button data-filter="unsure">unsure</button>
  <button data-filter="l3">→ Layer 3</button>
  <button data-filter="forced">grounding flags</button>
</div>
<table>
<thead><tr><th>name</th><th>verdict</th><th>confidence</th><th>evidence</th>
<th>quote</th><th>from_recall</th><th>→ L3</th><th>flags</th></tr></thead>
<tbody>
{''.join(body_rows)}
</tbody></table>
<script>
document.querySelectorAll('.filters button').forEach(btn => btn.addEventListener('click', () => {{
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  const f = btn.dataset.filter;
  document.querySelectorAll('tbody tr').forEach(tr => {{
    let show = true;
    if (f === 'yes' || f === 'no' || f === 'unsure') show = tr.dataset.verdict === f;
    if (f === 'l3') show = tr.dataset.l3 === 'yes';
    if (f === 'forced') show = tr.querySelector('.flag') !== null;
    tr.style.display = show ? '' : 'none';
  }});
}});
</script>
</main></body></html>
"""


def write_pilot_html(rows: list[dict], path: Path, *, header: str = "", spend: Optional[str] = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_pilot_html(rows, header=header, spend=spend), encoding="utf-8")
    return _stat_block([_annotate(r) for r in rows])


def render_validation_blind_html(sample: list[dict]) -> str:
    """No verdicts — the reviewer must not see the triage answer."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    for i, d in enumerate(sample, 1):
        name = html.escape(d.get("proposed_name") or d.get("name") or d["program_id"])
        pid = html.escape(d["program_id"])
        cards.append(
            f'<div class="stat" style="grid-column:span 2"><span>#{i}</span>'
            f'<b>{name}</b><div class="meta">{pid}</div></div>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Triage validation sample (blind)</title>
<style>{_CSS}</style></head><body><main>
<h1>Blind validation sample — {len(sample)} candidates</h1>
<p class="meta">Generated {generated}. Verdicts are withheld. Label these in the
labelling app (they stay in the normal queue, no auto-derived banner).
Agreement is computed later against gold, per stratum, never blended.</p>
<div class="stats">{''.join(cards)}</div>
</main></body></html>
"""
