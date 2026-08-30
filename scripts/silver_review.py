"""Self-contained HTML inspection page for silver/labels.jsonl.

    python scripts/silver_review.py

One card per silver record: the verdict, each decomposed question with its
answer, cited snippet, and citation-gate verdict, where the k=5 samples
disagreed, the Red Team objection (if one fired), and the deterministic
rule path. Where a program already has a gold label, it's shown side by
side — clearly marked reference-only, never as a computed accuracy figure
(13 programs is not a measurement). Reading the reasoning is the point of
this page, not a score.

No external assets, no charts — pure HTML/CSS, opens with a double-click.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import store as gold_store
from pharma_stats.silver import store as silver_store

OUT_PATH = REPORTS_DIR / "silver_review.html"

PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Silver review</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
          max-width: 1000px; margin: 0 auto; padding: 24px; background: Canvas; color: CanvasText; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: GrayText; font-size: 13px; margin-bottom: 24px; }}
  .card {{ border: 1px solid color-mix(in srgb, CanvasText 18%, transparent); border-radius: 10px;
           padding: 16px 20px; margin-bottom: 20px; }}
  .card h2 {{ margin: 0 0 6px; font-size: 17px; }}
  .card .pid {{ color: GrayText; font-size: 12px; font-family: monospace; }}
  .verdict {{ font-size: 15px; font-weight: 600; padding: 8px 10px; border-radius: 6px; margin: 10px 0; }}
  .verdict.abstain {{ background: color-mix(in srgb, orange 18%, transparent); }}
  .verdict.labelled {{ background: color-mix(in srgb, seagreen 18%, transparent); }}
  .rule-path {{ font-family: monospace; font-size: 12px; color: GrayText; }}
  .reference {{ border: 1px dashed color-mix(in srgb, CanvasText 30%, transparent); border-radius: 6px;
                padding: 8px 10px; margin: 10px 0; font-size: 13px; }}
  .reference .tag {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: GrayText; }}
  .question {{ border-top: 1px solid color-mix(in srgb, CanvasText 10%, transparent); padding: 10px 0; }}
  .question summary {{ cursor: pointer; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
            border: 1px solid color-mix(in srgb, CanvasText 25%, transparent); margin-left: 6px; }}
  .badge.pass {{ color: seagreen; border-color: seagreen; }}
  .badge.fail {{ color: crimson; border-color: crimson; }}
  .badge.disagree {{ color: darkorange; border-color: darkorange; }}
  .badge.agree {{ color: seagreen; border-color: seagreen; }}
  .quote {{ font-style: italic; background: color-mix(in srgb, CanvasText 6%, transparent);
            padding: 4px 8px; border-radius: 4px; display: inline-block; margin: 4px 0; }}
  .votes {{ font-family: monospace; font-size: 12px; color: GrayText; }}
  pre {{ white-space: pre-wrap; font-size: 11px; background: color-mix(in srgb, CanvasText 5%, transparent);
         padding: 8px; border-radius: 6px; max-height: 300px; overflow-y: auto; }}
  .red-team {{ border-top: 1px solid color-mix(in srgb, CanvasText 10%, transparent); padding-top: 10px; margin-top: 10px; }}
  .red-team.forced {{ background: color-mix(in srgb, crimson 10%, transparent); border-radius: 6px; padding: 8px 10px; }}
</style>
</head><body>
<h1>Silver review</h1>
<p class="meta">Generated {generated_at}. {n} silver record(s). Reasoning, not a score — gold labels \
shown for reference only where they exist, never as a computed accuracy figure.</p>
{cards}
</body></html>
"""


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def _render_citation_verdict(verdict) -> str:
    if not verdict:
        return ""
    if verdict.get("passed"):
        return _badge("citation verified", "pass")
    reason = verdict.get("reason") or "no usable citation"
    return _badge(f"citation failed: {reason}", "fail")


def _render_question_log(title: str, log: dict) -> str:
    if not log.get("prompt"):
        note = log.get("note", "")
        return f"""<div class="question"><b>{html.escape(title)}</b> — {html.escape(note)}</div>"""

    disagreement = log.get("disagreement", False)
    agree_badge = _badge("k=5 disagreed → abstain", "disagree") if disagreement else _badge("k=5 unanimous", "agree")
    citation_badge = _render_citation_verdict(log.get("citation_verdict"))
    votes = ", ".join(str(v) for v in log.get("votes", []))
    samples_html = "\n".join(
        f"--- sample {i + 1} ---\n{r}" for i, r in enumerate(log.get("raw_responses", []))
    )
    quote = ""
    verdict = log.get("citation_verdict")
    if verdict and verdict.get("citation"):
        quote = f'<div class="quote">&ldquo;{html.escape(verdict["citation"]["quote"])}&rdquo; ' \
                f'({html.escape(verdict["citation"]["locator"])})</div>'

    return f"""
<details class="question">
  <summary>{html.escape(title)} {agree_badge} {citation_badge}</summary>
  <div class="votes">k={log.get("k")} votes: [{html.escape(votes)}]</div>
  {quote}
  <details><summary>prompt + all {log.get("k", "?")} raw responses</summary>
    <pre>PROMPT:\n{html.escape(log.get("prompt") or "")}\n\n{html.escape(samples_html)}</pre>
  </details>
</details>
"""


def _render_red_team(log) -> str:
    if not log:
        return ""  # legacy record predating the status gate — nothing logged either way
    if log.get("skipped"):
        return f"""
<div class="red-team">
  <b>Red Team</b> {_badge("gated — not run", "agree")}
  <div style="color:var(--muted, #888)">{html.escape(log.get("reason", ""))}</div>
</div>
"""
    objection = log.get("objection") or {}
    forced = objection.get("strength") == "strong" and objection.get("citations")
    cls = "red-team forced" if forced else "red-team"
    citation_html = ""
    if objection.get("citations"):
        c = objection["citations"][0]
        citation_html = f'<div class="quote">&ldquo;{html.escape(c["quote"])}&rdquo; ({html.escape(c["locator"])})</div>'
    return f"""
<div class="{cls}">
  <b>Red Team objection</b> {_badge(objection.get("strength", "?"), "fail" if forced else "agree")}
  {"— FORCED ABSTENTION" if forced else ""}
  <div>{html.escape(objection.get("argument", ""))}</div>
  {citation_html}
  <details><summary>prompt + raw response</summary>
    <pre>PROMPT:\n{html.escape(log.get("prompt") or "")}\n\nRESPONSE:\n{html.escape(log.get("raw_response") or "")}</pre>
  </details>
</div>
"""


def _render_gold_reference(gold_record) -> str:
    if not gold_record:
        return ""
    return f"""
<div class="reference">
  <div class="tag">Your gold label (reference only — not compared as accuracy)</div>
  status={html.escape(str(gold_record.get("status")))},
  kill_reason={html.escape(str(gold_record.get("kill_reason")))},
  public_confirmation_date={html.escape(str(gold_record.get("public_confirmation_date")))}
</div>
"""


def render_card(record: dict, gold_by_pid: dict) -> str:
    answers = record.get("answers") or {}
    verdict_cls = "abstain" if record.get("abstained") else "labelled"
    verdict_text = (
        f"ABSTAIN — {html.escape(record.get('abstain_reason') or '')}" if record.get("abstained") else
        f"{html.escape(str(record.get('status')))}"
        + (f" / {html.escape(str(record.get('kill_reason')))}" if record.get("kill_reason") else "")
        + (f" (confirmed {html.escape(str(record.get('public_confirmation_date')))})"
           if record.get("public_confirmation_date") else "")
    )

    questions_html = "".join([
        _render_question_log("Q1 — trial initiated since cutoff?", answers.get("trial_initiated_since", {})),
        _render_question_log("Q2 — public discontinuation statement?", answers.get("discontinuation_statement", {})),
        _render_question_log("Q3 — stop reason category?", answers.get("stop_reason", {})),
        _render_question_log("Q4 — successor asset?", answers.get("successor_asset", {})),
    ])

    cost_line = (
        f'cost: ${record.get("cost_usd", 0.0):.4f} ({record.get("calls", 0)} call(s), '
        f'{record.get("input_tokens", 0)}+{record.get("output_tokens", 0)} tokens)'
    )
    return f"""
<div class="card">
  <h2>{html.escape(record.get("proposed_name") or "(unnamed)")}</h2>
  <div class="pid">{html.escape(record["program_id"])} — silver event {html.escape(record["event_id"])}</div>
  <div class="verdict {verdict_cls}">{verdict_text}</div>
  <div class="rule-path">rule_path: {html.escape(record.get("rule_path") or "")}</div>
  <div class="rule-path">{html.escape(cost_line)}</div>
  {_render_gold_reference(gold_by_pid.get(record["program_id"]))}
  {questions_html}
  {_render_red_team(record.get("red_team_objection"))}
</div>
"""


def main() -> None:
    silver_records = silver_store.load_records()
    if not silver_records:
        print(f"No silver records at {silver_store.SILVER_LABELS_PATH} — run "
              "scripts/run_silver_labelling.py first.")
        return

    gold_by_pid = gold_store.latest_by_program(gold_store.load_records())

    # latest silver record per program, in case of reruns
    latest_by_pid: dict[str, dict] = {}
    for r in silver_records:
        pid = r["program_id"]
        if pid not in latest_by_pid or r["timestamp"] > latest_by_pid[pid]["timestamp"]:
            latest_by_pid[pid] = r
    records = sorted(latest_by_pid.values(), key=lambda r: r.get("proposed_name") or "")

    cards = "".join(render_card(r, gold_by_pid) for r in records)
    page = PAGE_TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).isoformat(), n=len(records), cards=cards,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(records)} programs)")


if __name__ == "__main__":
    main()
