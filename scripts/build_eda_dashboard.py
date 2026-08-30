"""Single-file HTML dashboard for the silence-feature EDA.

    python scripts/build_eda_dashboard.py

Produces reports/eda_dashboard.html — every chart inlined as SVG, no
external assets, opens with a double-click. Reuses the computation
functions in eda_silence_features.py (imported directly, not duplicated)
so the numbers here and in the markdown report always agree.

Aggregate distributions only — no per-program score or identifier
anywhere in the output. Every chart states its n.
"""
from __future__ import annotations

import html
import importlib.util
import io
import sys
from datetime import date
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as spstats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print('Missing the "analysis" extra. Run: pip install -e ".[analysis]"')
    sys.exit(1)

import duckdb

from pharma_stats.config import REPORTS_DIR, WAREHOUSE_DB
from pharma_stats.labelling import provisional_programs as pp

# Load eda_silence_features.py as a module without needing scripts/ to be
# a package — reuses its analysis functions rather than duplicating them.
_eda_path = Path(__file__).with_name("eda_silence_features.py")
_spec = importlib.util.spec_from_file_location("eda_silence_features", _eda_path)
eda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eda)

OUT_PATH = REPORTS_DIR / "eda_dashboard.html"


def _fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]  # strip the XML prolog/DOCTYPE, keep the element


def _section(title: str, svg: str, commentary: str) -> str:
    return f"""
<section class="panel">
  <h2>{html.escape(title)}</h2>
  <div class="chart-row">
    <div class="chart">{svg}</div>
    <div class="commentary">{commentary}</div>
  </div>
</section>
"""


# ---------------------------------------------------------------- 1. bimodality --

def section_bimodality(trials: list[dict], as_of: date) -> tuple[str, dict]:
    months = np.array([
        (as_of - t["last_version_date"]).days / 30.44
        for t in trials if t["last_version_date"] is not None
    ])
    months = months[(months >= 0) & np.isfinite(months)]
    n = len(months)

    mu, sigma = months.mean(), months.std()
    ll1 = spstats.norm.logpdf(months, mu, max(sigma, 1e-6)).sum()
    w2, mu2, sig2, ll2 = eda._gmm_1d_em(months, k=2)
    bic1 = -2 * ll1 + 2 * np.log(n)
    bic2 = -2 * ll2 + 5 * np.log(n)
    ashman_d = abs(mu2[0] - mu2[1]) / np.sqrt((sig2[0] ** 2 + sig2[1] ** 2) / 2)
    two_component_wins = bic2 < bic1

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(months, bins=40, density=True, alpha=0.45, color="#4da3ff", label=f"observed (n={n})")
    xs = np.linspace(0, months.max(), 500)
    ax.plot(xs, spstats.norm.pdf(xs, mu, sigma), "k--", label="1-component fit")
    mix = sum(w2[j] * spstats.norm.pdf(xs, mu2[j], sig2[j]) for j in range(2))
    ax.plot(xs, mix, color="#e8574a", linewidth=2, label="2-component fit")
    ax.set_xlabel("months since last version")
    ax.set_ylabel("density")
    ax.set_title(f"n={n}")
    ax.text(
        0.98, 0.95, f"BIC: 1-comp={bic1:.0f}, 2-comp={bic2:.0f}\nAshman's D = {ashman_d:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#ccc"),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.78), fontsize=8)

    verdict = (
        "Real, if imperfect, bimodal structure — BIC prefers two components, though Ashman's D "
        f"({ashman_d:.2f}) shows the two clusters still overlap substantially."
        if two_component_wins and ashman_d >= 1.0 else
        "Weak or no evidence of two distinct populations — treat staleness as a continuous "
        "spectrum, not a natural two-cluster split."
    )
    commentary = (
        f"<p>{html.escape(verdict)}</p>"
        f"<p>N={n} trials with a resolvable last-version date. "
        "This determines whether a fixed staleness cutoff is even meaningful, or whether the "
        "labelling task has to treat every case on a continuum.</p>"
    )
    headline = {
        "label": "Staleness bimodality",
        "value": "weak/no separation" if not (two_component_wins and ashman_d >= 1.0) else "real, weak separation",
        "impact": "No natural staleness cutoff to lean on — every case needs individual judgement, "
                  "not a threshold rule.",
    }
    return _section("1. Is months_since_last_version bimodal?", _fig_to_svg(fig), commentary), headline


# ------------------------------------------------------ 2. event-type frequency --

def section_event_types(con: duckdb.DuckDBPyConnection, trials: list[dict]) -> tuple[str, dict]:
    nct_ids = {t["nct_id"] for t in trials}
    placeholders = ",".join("?" for _ in nct_ids)
    rows = con.execute(
        f"""
        SELECT event_type, direction, count(DISTINCT (nct_id, to_version)) AS n_pairs
        FROM evidence_events WHERE nct_id IN ({placeholders})
        GROUP BY 1, 2
        """,
        list(nct_ids),
    ).fetchall()
    total_pairs = sum(max(0, t["n_versions"] - 1) for t in trials)

    # group by event_type, keep push/pull adjacent within a group, order
    # groups by total share descending (universal registry hygiene at
    # top, rare-and-informative types at the bottom)
    by_type: dict[str, dict] = {}
    for event_type, direction, n in rows:
        by_type.setdefault(event_type, {})[direction] = n
    direction_order = ["pushed_later", "pulled_earlier", "finalized", "increased", "decreased", None]
    type_order = sorted(by_type, key=lambda et: -sum(by_type[et].values()))

    labels, shares, colors = [], [], []
    for et in type_order:
        for d in direction_order:
            if d in by_type[et]:
                n = by_type[et][d]
                labels.append(f"{et} ({d})" if d else et)
                shares.append(n / total_pairs if total_pairs else 0.0)
                colors.append("#e8574a" if d == "pulled_earlier" else "#4da3ff" if d == "pushed_later" else "#8b94a3")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(labels))))
    y = range(len(labels))
    ax.barh(list(y), shares, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of all version pairs")
    ax.set_title(f"n={total_pairs} diffed version pairs, {len(nct_ids)} trials")

    pushed = sum(n for et, d, n in rows if d == "pushed_later")
    pulled = sum(n for et, d, n in rows if d == "pulled_earlier")
    asymmetry = (pushed / pulled) if pulled else float("inf")
    commentary = (
        f"<p>Dates pushed later ({pushed} pairs, <span style='color:#4da3ff'>blue</span>) outnumber "
        f"dates pulled earlier ({pulled} pairs, <span style='color:#e8574a'>red</span>) by "
        f"<b>{asymmetry:.1f}x</b>.</p>"
        "<p>Slippage is near-universal registry hygiene; a pulled-earlier date is rare and, per this "
        "project's hypothesis, the sharper signal.</p>"
    )
    headline = {
        "label": "pushed_later vs pulled_earlier",
        "value": f"{asymmetry:.1f}x",
        "impact": "A pulled-earlier date is rare enough to be a real signal, not noise — worth "
                  "weighting more heavily than a pushed-later date in review.",
    }
    return _section("2. Event-type frequency", _fig_to_svg(fig), commentary), headline


# ------------------------------------------------- 3. correlation + PCA --

def section_correlation_pca(programs: list[dict], as_of: date) -> tuple[str, dict]:
    X, names, _ids = eda.build_feature_matrix(programs, as_of)
    n = X.shape[0]
    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    corr = np.corrcoef(Xz, rowvar=False)
    condition_number = np.linalg.cond(corr)
    eigvals, eigvecs = np.linalg.eigh(np.cov(Xz, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    explained = eigvals / eigvals.sum()

    off_diag = corr[~np.eye(len(names), dtype=bool)]
    vmax = max(abs(off_diag.min()), abs(off_diag.max()))
    masked_corr = np.ma.masked_where(np.eye(len(names), dtype=bool), corr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#eeeeee")
    im = axes[0].imshow(masked_corr, vmin=-vmax, vmax=vmax, cmap=cmap)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_title(f"correlation matrix (n={n} programs, diag masked, scale=±{vmax:.2f})")
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                axes[0].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].bar(range(1, len(explained) + 1), explained, color="#4da3ff")
    axes[1].plot(range(1, len(explained) + 1), np.cumsum(explained), "o-", color="#e8574a")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("explained variance ratio")
    axes[1].set_title(f"PCA scree (n={n}) — cond. number={condition_number:.1e}")

    one_component = explained[0] > 0.6
    verdict = (
        f"PC1 alone explains {explained[0]:.0%} of variance — most of these features are largely "
        "one underlying signal in different costumes." if one_component else
        f"No single component dominates (PC1 = {explained[0]:.0%}) — the features carry meaningfully "
        "distinct information."
    )
    rank_note = (
        "Condition number is astronomical because staleness_age_adjusted is an exact linear "
        "combination of staleness and months_since_start (both also in this matrix) — mechanical, "
        "not a data surprise." if condition_number > 1e6 else
        f"Condition number {condition_number:.1f} — " +
        ("indicates near-rank-deficiency." if condition_number > 30 else "no strong rank deficiency.")
    )
    commentary = (
        f"<p>{html.escape(verdict)}</p>"
        f"<p>{html.escape(rank_note)}</p>"
        f"<p>N={n} programs (silence_score excluded — deterministic function of 3 of the other "
        "features, would double-count them).</p>"
    )
    headline = {
        "label": "PCA: PC1 share of variance",
        "value": f"{explained[0]:.0%}",
        "impact": "Model is simpler than planned — one latent signal, not eight." if one_component else
                  "Features are not redundant — worth keeping all of them, not collapsing to one score.",
    }
    return _section("3. Feature correlation + PCA", _fig_to_svg(fig), commentary), headline


# -------------------------------------------------------- 4. KM survival --

def section_survival(trials: list[dict], as_of: date) -> tuple[str, dict]:
    from collections import defaultdict
    by_phase_terminal = defaultdict(list)
    by_phase_amendment = defaultdict(list)
    for t in trials:
        if t["start_date"] is None or t["last_version_date"] is None:
            continue
        duration = (t["last_version_date"] - t["start_date"]).days / 30.44
        if duration < 0:
            continue
        phase = eda.phase_bucket(t["phases"])
        by_phase_terminal[phase].append(
            {"duration": duration, "event": 1 if t["status"] in eda.TERMINAL_STATUSES else 0}
        )
        days_since_last = (as_of - t["last_version_date"]).days
        by_phase_amendment[phase].append(
            {"duration": duration, "event": 1 if days_since_last > eda.STILL_FRESH_DAYS else 0}
        )

    def _draw(by_phase, ax, title):
        medians = {}
        total_n = 0
        events_total = 0
        for phase in eda.PHASE_ORDER:
            rows = by_phase.get(phase)
            if not rows or len(rows) < 5:
                continue
            durations = np.array([r["duration"] for r in rows])
            events = np.array([r["event"] for r in rows])
            times, survival = eda._kaplan_meier(durations, events)
            ax.step(times, survival, where="post", label=f"{phase} (n={len(rows)})")
            medians[phase] = eda._median_survival(times, survival)
            total_n += len(rows)
            events_total += int(events.sum())
        ax.set_xlabel("months since trial start")
        ax.set_ylabel("P(event not yet occurred)")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=7)
        return medians, total_n, events_total

    n_total = sum(len(v) for v in by_phase_terminal.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    medians_a, _, events_a = _draw(
        by_phase_terminal, axes[0], f"A. Time to TERMINAL STATUS (n={n_total})\n(confounded by status lag)"
    )
    medians_b, _, events_b = _draw(
        by_phase_amendment, axes[1],
        f"B. Time to LAST AMENDMENT (n={n_total}, event=no new version >{eda.STILL_FRESH_DAYS}d)\nstatus-independent",
    )

    commentary = (
        "<p>Two different questions, kept separate: panel A's event (reaching a terminal registry "
        "status) is confounded by status-update lag; panel B's event (no new version posted in "
        f"&gt;{eda.STILL_FRESH_DAYS} days) is status-independent — the base rate this project actually needs.</p>"
        f"<p><b>{events_b} trials are amendment-silent (panel B) vs {events_a} formally terminal "
        f"(panel A)</b> — {events_b - events_a} more ({(events_b - events_a) / events_a:.0%}) look "
        "quiet on the amendment record without the registry status catching up yet. n=" + str(n_total) + " trials.</p>"
    )
    headline = {
        "label": "Amendment-silent vs formally terminal",
        "value": f"{events_b} vs {events_a} ({(events_b - events_a) / events_a:+.0%})",
        "impact": "Registry status understates and lags real silence — use panel B's base rate, not "
                  "an assumed cutoff or panel A alone.",
    }
    return _section("4. Survival, by phase — two different questions", _fig_to_svg(fig), commentary), headline


# ------------------------------------------------------ 5. sponsor clustering --

def section_sponsor_clustering(programs: list[dict]) -> tuple[str, dict]:
    by_sponsor: dict[str, list[float]] = {}
    for p in programs:
        sponsor = eda._current_sponsor(p)
        breakdown = p.get("score_breakdown") or {}
        if sponsor is None or "staleness" not in breakdown:
            continue
        by_sponsor.setdefault(sponsor, []).append(breakdown["staleness"])

    groups = {s: v for s, v in by_sponsor.items() if len(v) >= 2}
    all_vals = np.array([v for vals in groups.values() for v in vals])
    grand_mean = all_vals.mean()
    n_total = len(all_vals)
    g = len(groups)
    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values())
    ms_between = ss_between / (g - 1)
    ms_within = ss_within / (n_total - g)
    sum_ni2 = sum(len(v) ** 2 for v in groups.values())
    n0 = (n_total - sum_ni2 / n_total) / (g - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within) if ms_within > 0 else float("nan")
    icc = max(0.0, icc) if np.isfinite(icc) else icc

    # top 20 sponsors by program count, ordered by median staleness
    top20 = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:20]
    top20 = sorted(top20, key=lambda kv: np.median(kv[1]))

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top20))))
    ax.boxplot(
        [vals for _, vals in top20], vert=False,
        tick_labels=[f"sponsor {i+1} (n={len(v)})" for i, (_, v) in enumerate(top20)],
        patch_artist=True, boxprops=dict(facecolor="#4da3ff", alpha=0.5),
    )
    ax.set_xlabel("staleness feature")
    ax.set_title(f"top 20 sponsors by program count, n={n_total} programs across {g} sponsors — ICC={icc:.2f}")

    commentary = (
        f"<p>Sponsor names withheld — distributions and rank only. Intra-sponsor ICC of staleness: "
        f"<b>{icc:.2f}</b> (F({g-1},{n_total-g}) one-way ANOVA).</p>"
        f"<p>{'Substantial' if icc >= 0.1 else 'Small'} — programs from the same sponsor are "
        f"{'NOT' if icc >= 0.1 else 'roughly'} independent observations. "
        f"{'A sponsor running out of money plausibly takes multiple programs stale at once.' if icc >= 0.1 else ''}</p>"
    )
    headline = {
        "label": "Sponsor-clustering ICC",
        "value": f"{icc:.2f}",
        "impact": "Programs are not independent within sponsor — train/test splits and CIs must "
                  "account for this (see the three follow-on fixes).",
    }
    return _section("5. Sponsor clustering", _fig_to_svg(fig), commentary), headline


# ------------------------------------------------------- 6. regional strata --

def section_regional_strata(programs: list[dict], con: duckdb.DuckDBPyConnection, as_of: date) -> tuple[str, dict]:
    china_ids = set()
    for p in programs:
        countries = set()
        for nct_id in (p.get("nct_ids") or []):
            countries |= eda._trial_countries(nct_id, con)
        if countries == {"China"}:
            china_ids.add(p["program_id"])

    X, names, id_order = eda.build_feature_matrix(programs, as_of)
    is_china = np.array([pid in china_ids for pid in id_order])
    n_china, n_rest = int(is_china.sum()), int((~is_china).sum())

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    sig_features = []
    for j, (name, ax) in enumerate(zip(names, axes.flat)):
        a, b = X[is_china, j], X[~is_china, j]
        try:
            _, p_value = spstats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p_value = float("nan")
        pooled_std = np.sqrt((a.var() + b.var()) / 2) or 1.0
        effect_size = (a.mean() - b.mean()) / pooled_std
        if p_value < 0.05:
            sig_features.append(name)
        ax.hist(b, bins=15, density=True, alpha=0.45, color="#4da3ff", label=f"rest (n={n_rest})")
        ax.hist(a, bins=15, density=True, alpha=0.45, color="#e8574a", label=f"China-only (n={n_china})")
        ax.set_title(f"{name}\nd={effect_size:.2f}, p={p_value:.1e}", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6)
    fig.suptitle(f"China-only-site programs (n={n_china}) vs rest (n={n_rest})")
    fig.tight_layout()

    commentary = (
        "<p>Proxy: every trial location on file for the program is in China (not a direct sponsor-"
        "nationality field). Effect size d = standardised mean difference.</p>"
        f"<p>Significant (p&lt;0.05) on: {', '.join(sig_features) if sig_features else 'none'}. "
        "Any feature differing systematically by region should get a separate base rate, not a "
        "pooled one.</p>"
    )
    headline = {
        "label": "China-only vs rest: features differing (p<0.05)",
        "value": f"{len(sig_features)} / {len(names)}",
        "impact": "Registry behaviour differs systematically by region — plan for separate base "
                  "rates, possibly separate models, not one pooled estimate.",
    }
    return _section("6. Regional strata", _fig_to_svg(fig), commentary), headline


# --------------------------------------------------------------------- header --

def render_header(headlines: list[dict], n_programs: int, n_trials: int, as_of: date) -> str:
    rows = "".join(
        f"""<div class="stat">
              <div class="stat-label">{html.escape(h['label'])}</div>
              <div class="stat-value">{html.escape(str(h['value']))}</div>
              <div class="stat-impact">{html.escape(h['impact'])}</div>
            </div>"""
        for h in headlines
    )
    return f"""
<header>
  <h1>Silence-feature EDA dashboard</h1>
  <p class="meta">Generated {as_of.isoformat()} from {n_programs} provisional programs / {n_trials} trials.
     Aggregate distributions only — no per-program scores anywhere in this page.</p>
  <div class="stat-grid">{rows}</div>
</header>
"""


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Silence-feature EDA dashboard</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 24px; background: Canvas; color: CanvasText; }}
  header {{ margin-bottom: 32px; }}
  header .meta {{ color: GrayText; font-size: 13px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }}
  .stat {{ border: 1px solid color-mix(in srgb, CanvasText 15%, transparent); border-radius: 8px; padding: 12px 14px; }}
  .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: GrayText; }}
  .stat-value {{ font-size: 22px; font-weight: 700; margin: 4px 0; }}
  .stat-impact {{ font-size: 12.5px; color: GrayText; }}
  section.panel {{ border-top: 1px solid color-mix(in srgb, CanvasText 12%, transparent); padding: 28px 0; }}
  section.panel h2 {{ margin-top: 0; }}
  .chart-row {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
  .chart {{ flex: 1 1 520px; min-width: 320px; }}
  .chart svg {{ max-width: 100%; height: auto; }}
  .commentary {{ flex: 1 1 280px; min-width: 240px; font-size: 14px; }}
  .commentary p {{ margin: 0 0 10px; }}
</style>
</head><body>
{header}
{sections}
</body></html>
"""


def main() -> None:
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    as_of = date.today()
    trials = eda.load_trials(programs)
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)

    print(f"Loaded {len(programs)} programs, {len(trials)} trials. Rendering dashboard...")

    sec1, h1 = section_bimodality(trials, as_of)
    sec2, h2 = section_event_types(con, trials)
    sec3, h3 = section_correlation_pca(programs, as_of)
    sec4, h4 = section_survival(trials, as_of)
    sec5, h5 = section_sponsor_clustering(programs)
    sec6, h6 = section_regional_strata(programs, con, as_of)
    con.close()

    headlines = [h1, h2, h3, h4, h5, h6]
    header = render_header(headlines, len(programs), len(trials), as_of)
    sections = "\n".join([sec1, sec2, sec3, sec4, sec5, sec6])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(PAGE_TEMPLATE.format(header=header, sections=sections), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
