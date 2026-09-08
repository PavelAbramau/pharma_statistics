"""Population-level statistics from gold/labels.jsonl — inverse-
probability-weighted by (silence-score band x archetype) stratum, never
a raw percentage of the labelled sample (see pharma_stats.stats.label_statistics
for why that distinction matters), plus the stated-vs-true kill-reason
divergence: how often CT.gov's own why_stopped text would have given a
different kill reason than the reviewer's full-evidence judgement.

    python scripts/report_label_statistics.py
"""
from __future__ import annotations

from pharma_stats.config import REPORTS_DIR
from pharma_stats.labelling import provisional_programs as pp
from pharma_stats.labelling import store
from pharma_stats.stats import label_statistics as ls


def _render(summary: dict) -> str:
    lines = [
        "# Label statistics (gold/labels.jsonl)",
        "",
        f"Gate-3 (real) labels: {summary['n_gate3_labels']}",
        "",
    ]

    if "population_estimates_refused" in summary:
        lines += [
            "## Population estimates: REFUSED",
            "",
            "Every population-level estimate in this module is inverse-probability-"
            "weighted by (band, archetype) stratum; at least one stratum with "
            "population > 0 has zero labels, so there is no weight to assign its "
            "share of the corpus. Label the strata below before a population "
            "estimate can be trusted.",
            "",
            summary["population_estimates_refused"],
            "",
        ]
    else:
        lines += ["## Stratum coverage (population / labelled / IPW weight)", ""]
        for row in summary["stratum_coverage"]:
            lines.append(
                f"- band {row['band']} / {row['archetype']}: "
                f"{row['labelled']} labelled of {row['population']} in corpus "
                f"(weight {row['weight']:.2f})"
            )

        lines += ["", "## Weighted program-status distribution (population estimate)", ""]
        for row in summary["weighted_status_distribution"]:
            lines.append(
                f"- {row['status']}: {row['weighted_share']:.1%} "
                f"(raw labelled n={row['raw_labelled_n']})"
            )

        lines += ["", "## Weighted kill-reason distribution, within dead_confirmed (population estimate)", ""]
        for row in summary["weighted_kill_reason_distribution"]:
            lines.append(
                f"- {row['kill_reason']}: {row['weighted_share']:.1%} "
                f"(raw labelled n={row['raw_labelled_n']})"
            )

        d = summary["weighted_dead_confirmed_share_ci"]
        lines += [
            "", "## dead_confirmed share of the corpus (sponsor-cluster-bootstrap 95% CI)", "",
            f"- point estimate: {d['point_estimate']:.1%}"
            if d["point_estimate"] is not None else "- point estimate: n/a",
            f"- 95% CI: [{d['ci_lo']:.1%}, {d['ci_hi']:.1%}]"
            if d["ci_lo"] is not None else "- 95% CI: n/a",
            f"- n_labelled={d['n_labelled']}, n_sponsors={d['n_sponsors']}, resamples={d['n_resamples']}",
        ]

        m = summary["weighted_kill_reason_divergence_ci"]
        lines += [
            "", "## Stated-vs-true kill-reason mismatch rate, population estimate", "",
            "How often would trusting CT.gov's own why_stopped text alone (mapped "
            "via keyword to the kill-reason taxonomy) give a DIFFERENT kill reason "
            "than the reviewer's full-evidence judgement, across dead_confirmed "
            "programs in the whole corpus:",
            "",
            f"- point estimate: {m['point_estimate_mismatch_rate']:.1%}"
            if m["point_estimate_mismatch_rate"] is not None else "- point estimate: n/a",
            f"- 95% CI: [{m['ci_lo']:.1%}, {m['ci_hi']:.1%}]"
            if m["ci_lo"] is not None else "- 95% CI: n/a",
            f"- n_dead_confirmed_labelled={m['n_dead_confirmed_labelled']}, "
            f"n_sponsors={m['n_sponsors']}, resamples={m['n_resamples']}",
        ]

    s = summary["kill_reason_divergence_sample"]
    lines += [
        "",
        "## Stated-vs-true kill-reason divergence (labelled SAMPLE, not weighted — "
        "describes the gold set as sampled, not the corpus)",
        "",
        f"n={s['n']}, raw agreement rate={s['agreement_rate']:.1%}" if s["agreement_rate"] is not None
        else f"n={s['n']} (no dead_confirmed labels with a kill_reason yet)",
        "",
        "True kill_reason (reviewer judgement), raw counts:",
    ]
    for reason, n in s["true_kill_reason_counts"].items():
        lines.append(f"- {reason}: {n}")
    lines += ["", "Stated kill_reason (why_stopped text, keyword-classified), raw counts:"]
    for reason, n in s["stated_kill_reason_counts"].items():
        lines.append(f"- {reason}: {n}")
    lines += ["", "Confusion (stated x true), raw counts:"]
    for key, n in sorted(s["confusion_matrix"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {key}: {n}")

    a = summary["agreement_rate_sample_ci"]
    lines += [
        "",
        "### Raw agreement rate, sponsor-cluster-bootstrap 95% CI (sample, unweighted)",
        "",
        f"- point estimate: {a['point_estimate']:.1%}" if a["point_estimate"] is not None
        else "- point estimate: n/a",
        f"- 95% CI: [{a['ci_lo']:.1%}, {a['ci_hi']:.1%}]" if a["ci_lo"] is not None else "- 95% CI: n/a",
        f"- n={a['n']}, n_sponsors={a['n_sponsors']}, resamples={a['n_resamples']}",
    ]

    rows = summary["kill_reason_divergence_sample_rows"]
    not_stated = [r for r in rows if r["stated_kill_reason"] is None]
    classified = [r for r in rows if r["stated_kill_reason"] is not None]
    lines += [
        "",
        "### Every case, raw why_stopped text next to both kill-reason labels "
        f"(n={len(rows)})",
        "",
        f"{len(not_stated)} of {len(rows)} have no usable why_stopped text at all "
        "(blank, or under the 15-char floor) — these need no classifier judgement "
        "call and can't be a source of classifier-manufactured disagreement; listed "
        "separately below, not mixed in with the classified cases.",
        "",
        f"#### Classified cases (n={len(classified)}) — stated_kill_reason came from "
        "the keyword classifier",
        "",
    ]
    for r in sorted(classified, key=lambda r: r["program_id"]):
        agree = "AGREE" if r["stated_kill_reason"] == r["true_kill_reason"] else "disagree"
        lines.append(
            f"- **{r['program_id']}** [{agree}] true={r['true_kill_reason']} "
            f"stated={r['stated_kill_reason']} — why_stopped: "
            f"\"{r['why_stopped_text']}\""
        )
    lines += ["", f"#### not_stated cases (n={len(not_stated)}) — no why_stopped text to classify", ""]
    for r in sorted(not_stated, key=lambda r: r["program_id"]):
        text = r["why_stopped_text"]
        shown = f"\"{text}\"" if text else "(empty)"
        lines.append(f"- **{r['program_id']}** true={r['true_kill_reason']} — why_stopped: {shown}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    programs = pp.load_materialized()
    records = store.load_records()
    if not programs:
        print("No provisional_programs materialized — run the labelling app once, or "
              "pharma_stats.labelling.provisional_programs.materialize().")
        return
    if not records:
        print("gold/labels.jsonl is empty — nothing to report yet.")
        return

    summary = ls.summary(programs, records)
    text = _render(summary)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "label_statistics.md"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
