"""Render reports/triage_pilot.html from the markdown the pipeline wrote.

    python scripts/report_triage_pilot.py
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.triage import report as trep


def main() -> None:
    md = REPORTS_DIR / "triage_pilot.md"
    if not md.exists():
        print(f"No {md} — run the Layer 2 pilot first.")
        return
    header, rows = trep.parse_pilot_markdown(md)
    out = REPORTS_DIR / "triage_pilot.html"
    stats = trep.write_pilot_html(rows, out, header=header)
    print(f"Wrote {out} — {stats['n']} rows, {stats['l3_rate']:.0%} → Layer 3, "
          f"{stats['n_forced']} grounding-forced-recall, "
          f"{stats['n_empty_unsure']} unsure with no usable evidence.")


if __name__ == "__main__":
    main()
