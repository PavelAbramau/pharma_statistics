"""Descriptive statistics over the whole provisional-program corpus —
the full population, no sampling involved (contrast with
report_label_statistics.py, which reports on the gold label SAMPLE and
must weight accordingly).

    python scripts/report_corpus_statistics.py
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.stats import corpus_statistics as cs


def _render(summary: dict) -> str:
    lines = [
        "# Corpus statistics",
        "",
        f"Total programs (provisional, candidate-asset level): {summary['total_programs']}",
        "",
        "## Silence-score band distribution",
        "",
    ]
    for row in summary["band_distribution"]:
        lines.append(f"- band {row['band_label']}: {row['count']} ({row['share']:.1%})")

    lines += ["", "## Archetype distribution", ""]
    for row in summary["archetype_distribution"]:
        lines.append(f"- {row['archetype']}: {row['count']} ({row['share']:.1%})")

    lines += ["", "## History coverage", ""]
    for row in summary["history_coverage_distribution"]:
        lines.append(f"- {row['history_coverage']}: {row['count']} ({row['share']:.1%})")

    lines += ["", "## Latest registry status (raw CT.gov overallStatus)", ""]
    for row in summary["latest_status_distribution"]:
        lines.append(f"- {row['latest_status']}: {row['count']} ({row['share']:.1%})")

    lines += ["", "## Sponsor distribution (top 20, lead sponsor by most-recently-seen)", ""]
    for row in summary["sponsor_distribution"]:
        lines.append(f"- {row['sponsor']}: {row['count']} ({row['share']:.1%})")

    lines += ["", "## Discovery strategy", ""]
    for row in summary["discovery_strategy_distribution"]:
        lines.append(f"- {row['discovery_strategy']}: {row['count']} ({row['share']:.1%})")

    t = summary["trial_count_stats"]
    lines += [
        "", "## Trial counts", "",
        f"- distinct trials referenced: {t['distinct_trials']}",
        f"- total trial references (a shared trial counts once per program): {t['total_trial_references']}",
        f"- mean trials/program: {t['mean_trials_per_program']:.2f}",
        f"- median trials/program: {t['median_trials_per_program']:.1f}",
        f"- max trials on one program: {t['max_trials_per_program']}",
        f"- programs with zero resolvable trials: {t['zero_trial_programs']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    programs = cs.load_corpus()
    if not programs:
        print("No provisional_programs materialized — run the labelling app once, or "
              "pharma_stats.labelling.provisional_programs.materialize().")
        return
    summary = cs.summary(programs)
    text = _render(summary)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "corpus_statistics.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
