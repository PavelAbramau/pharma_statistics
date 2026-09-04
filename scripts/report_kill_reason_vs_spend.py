"""B1: kill reason vs. spend — report.

    python scripts/report_kill_reason_vs_spend.py

Answers the question docs/decisions/0004 exists to keep honest: "did this
program die cheap or expensive," never "did it die for a good reason."
Reads gold/labels.jsonl (dead_confirmed, kill_reason) and the money-layer
feature panel (pharma_stats.finance.panel, built from the already-
populated financial_events table) — estimated_cumulative_spend and
conviction_ratio resolved as of each program's own label_evidence_date,
never a later date. See audit/leakage.md for the knowability-date
contract both features carry.
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.finance import panel as money_panel
from pharma_stats.labelling import store
from pharma_stats.productb import kill_reason_spend as krs


def _fmt(v) -> str:
    return f"{v:,.0f}" if v is not None else "—"


def render(rows: list[krs.KillReasonSpendRow], summary: dict[str, dict]) -> str:
    with_spend = [r for r in rows if r.estimated_cumulative_spend is not None]

    lines = [
        "# Kill reason vs. spend",
        "",
        f"{len(rows)} dead_confirmed program(s) with a kill_reason; {len(with_spend)} have a "
        "resolvable estimated_cumulative_spend as of their label_evidence_date (the rest have no "
        "financial_events history for any trial on the program as of that date).",
        "",
        "Per docs/decisions/0004: spend and survival are jointly determined. This table is "
        "descriptive — how much a sponsor had already committed when a program died, broken out "
        "by the stated reason — never a claim that spend caused, or should have prevented, any "
        "one outcome.",
        "",
        "| kill_reason | n | n with spend data | median spend | mean spend | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for reason in sorted(summary, key=lambda r: -summary[r]["n"]):
        s = summary[reason]
        lines.append(
            f"| {reason} | {s['n']} | {s['n_with_spend_data']} | {_fmt(s['median_spend'])} | "
            f"{_fmt(s['mean_spend'])} | {_fmt(s['min_spend'])} | {_fmt(s['max_spend'])} |"
        )
    lines.append("")

    if with_spend:
        lines += ["## Programs, ranked by spend at death", ""]
        for r in sorted(with_spend, key=lambda r: -r.estimated_cumulative_spend):
            conv = f", conviction_ratio={r.conviction_ratio:.2f}" if r.conviction_ratio is not None else ""
            lines.append(
                f"- **{r.proposed_name or r.program_id}** — {r.kill_reason}, "
                f"spend={r.estimated_cumulative_spend:,.0f} as of {r.label_evidence_date}{conv}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    gold_records = store.load_records()
    dead = krs.dead_confirmed_records(gold_records)
    print(f"{len(dead)} dead_confirmed program(s) with a kill_reason.")

    panel = money_panel.build_money_layer_panel()
    print(f"{len(panel)} money-layer feature panel row(s).")

    rows = krs.spend_at_death_rows(dead, panel)
    summary = krs.summarize_by_kill_reason(rows)

    text = render(rows, summary)
    print(text)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "kill_reason_vs_spend.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
