"""Exploratory analysis of silence features, ahead of blind labelling.

    python scripts/eda_silence_features.py

Deliberately aggregate-only: NO per-program score or identifier appears in
the output the user reads (report + terminal). Charts and group summaries
only — the user is about to label these programs blind and must not see
individual scores first. Requires the optional "analysis" extra
(numpy/scipy/matplotlib): pip install -e ".[analysis]"

Answers seven questions, in order:
1. Is months_since_last_version bimodal? (2-component vs 1-component
   Gaussian mixture, likelihood ratio + BIC + Ashman's D)
2. Event-type frequency table (share of version pairs firing each type,
   with pushed_later/pulled_earlier direction asymmetry)
3. Feature correlation matrix + PCA over candidate silence features
4. Kaplan-Meier time-to-last-amendment from trial start, by phase
5. Sponsor clustering: intra-sponsor ICC of staleness
6. Regional strata: China-located trials vs the rest, per feature
7. Crude UNKNOWN-status base rate, with an explicit bias statement
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, timedelta

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

OUT_DIR = REPORTS_DIR / "eda_silence_features"
REPORT_PATH = REPORTS_DIR / "eda_silence_features.md"

TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "WITHDRAWN"}
ACTIVE_STATUSES = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING",
}
PHASE_ORDER = ["PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]


# ---------------------------------------------------------------- loading --

def _parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def load_trials(programs: list[dict]) -> list[dict]:
    """One row per trial, flattened from every program's embedded trial
    list (combo-excluded shared trials are skipped — a small, acceptable
    gap for exploratory work)."""
    trials = []
    for p in programs:
        for t in p.get("trials", []):
            history = t.get("history") or []
            last_version_date = _parse_date(history[-1]["posted_date"]) if history else None
            trials.append({
                "nct_id": t["nct_id"],
                "program_id": p["program_id"],
                "status": t.get("status"),
                "phases": t.get("phases") or [],
                "start_date": _parse_date(t.get("start_date")),
                "last_update_post_date": _parse_date(t.get("last_update_post_date")),
                "last_version_date": last_version_date or _parse_date(t.get("last_update_post_date")),
                "n_versions": len(history),
                "has_results": t.get("has_results"),
            })
    return trials


def phase_bucket(phases: list[str]) -> str:
    for p in PHASE_ORDER[:-1]:
        if p in (phases or []):
            return p
    return "NA"


# --------------------------------------------------------- 1. bimodality --

def _gmm_1d_em(x: np.ndarray, k: int, seed: int = 0, n_iter: int = 200, tol: float = 1e-8):
    """Minimal EM for a k-component 1D Gaussian mixture (no sklearn
    dependency). Returns (weights, means, stds, log_likelihood)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    means = rng.choice(x, size=k, replace=False).astype(float)
    stds = np.full(k, x.std() + 1e-6)
    weights = np.full(k, 1.0 / k)
    prev_ll = -np.inf

    for _ in range(n_iter):
        # E-step
        dens = np.array([
            weights[j] * spstats.norm.pdf(x, means[j], max(stds[j], 1e-6)) for j in range(k)
        ])
        total = dens.sum(axis=0)
        total[total <= 0] = 1e-300
        resp = dens / total
        ll = np.log(total).sum()
        if abs(ll - prev_ll) < tol:
            prev_ll = ll
            break
        prev_ll = ll

        # M-step
        nk = resp.sum(axis=1)
        weights = nk / n
        means = (resp * x).sum(axis=1) / np.maximum(nk, 1e-12)
        variances = (resp * (x[None, :] - means[:, None]) ** 2).sum(axis=1) / np.maximum(nk, 1e-12)
        stds = np.sqrt(np.maximum(variances, 1e-6))

    return weights, means, stds, prev_ll


def analyze_bimodality(trials: list[dict], as_of: date) -> str:
    months = np.array([
        (as_of - t["last_version_date"]).days / 30.44
        for t in trials if t["last_version_date"] is not None
    ])
    months = months[(months >= 0) & np.isfinite(months)]

    mu, sigma = months.mean(), months.std()
    ll1 = spstats.norm.logpdf(months, mu, max(sigma, 1e-6)).sum()

    w2, mu2, sig2, ll2 = _gmm_1d_em(months, k=2)

    lr_stat = 2 * (ll2 - ll1)
    df = 3  # 2-comp GMM has 5 free params vs 1-comp's 2
    p_value = spstats.chi2.sf(lr_stat, df)

    n = len(months)
    bic1 = -2 * ll1 + 2 * np.log(n)
    bic2 = -2 * ll2 + 5 * np.log(n)

    ashman_d = abs(mu2[0] - mu2[1]) / np.sqrt((sig2[0] ** 2 + sig2[1] ** 2) / 2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(months, bins=40, density=True, alpha=0.5, color="#4da3ff", label="observed")
    xs = np.linspace(0, months.max(), 500)
    ax.plot(xs, spstats.norm.pdf(xs, mu, sigma), "k--", label="1-component fit")
    mix = sum(w2[j] * spstats.norm.pdf(xs, mu2[j], sig2[j]) for j in range(2))
    ax.plot(xs, mix, color="#e8574a", linewidth=2, label="2-component fit")
    ax.set_xlabel("months since last version")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("months_since_last_version: 1- vs 2-component fit")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_bimodality.png", dpi=140)
    plt.close(fig)

    verdict = (
        "some separation, but weak (Ashman's D < 2 — components overlap substantially)"
        if ashman_d < 2 else
        "clear separation (Ashman's D >= 2)"
    )

    return f"""## 1. Is months_since_last_version bimodal?

N = {n} trials with a resolvable last-version date, as of {as_of.isoformat()}.

- 1-component fit: mu={mu:.1f}mo, sigma={sigma:.1f}mo, log-likelihood={ll1:.1f}, BIC={bic1:.1f}
- 2-component fit: weights={np.round(w2, 2).tolist()}, means={np.round(mu2, 1).tolist()}mo, \
stds={np.round(sig2, 1).tolist()}mo, log-likelihood={ll2:.1f}, BIC={bic2:.1f}
- Likelihood ratio stat={lr_stat:.1f}, naive chi2(df={df}) p={p_value:.2e} — **caveat**: standard \
LRT asymptotics don't strictly hold for mixture models (boundary/label-switching issues), so treat \
this p-value as directional, not exact.
- BIC prefers the **{"2-component" if bic2 < bic1 else "1-component"}** model \
(lower BIC by {abs(bic1 - bic2):.1f}).
- Ashman's D = {ashman_d:.2f} — {verdict}.

**Plain-language verdict**: {"The distribution shows real, if imperfect, bimodal structure — there "
"appear to be two population regimes (a 'recently touched' cluster and a 'gone quiet' cluster), not "
"one continuous spread." if bic2 < bic1 and ashman_d >= 1.0 else
"The evidence for two distinct populations is weak. BIC/LRT/Ashman's D do not agree on a clean "
"separation — treat staleness as a continuous spectrum, not a natural two-cluster split. This makes "
"the labelling task harder: there's no natural staleness cutoff to lean on."}

![bimodality]({OUT_DIR.name}/01_bimodality.png)
"""


# ------------------------------------------------------ 2. event-type freq --

def analyze_event_types(con: duckdb.DuckDBPyConnection, trials: list[dict]) -> str:
    nct_ids = {t["nct_id"] for t in trials}
    if not nct_ids:
        return "## 2. Event-type frequency table\n\n(no trials)\n"

    placeholders = ",".join("?" for _ in nct_ids)
    rows = con.execute(
        f"""
        SELECT event_type, direction, count(DISTINCT (nct_id, to_version)) AS n_pairs
        FROM evidence_events WHERE nct_id IN ({placeholders})
        GROUP BY 1, 2 ORDER BY n_pairs DESC
        """,
        list(nct_ids),
    ).fetchall()

    total_pairs = sum(max(0, t["n_versions"] - 1) for t in trials)

    lines = ["| event_type | direction | version pairs | share of all version pairs |",
             "|---|---|---:|---:|"]
    for event_type, direction, n_pairs in rows:
        share = n_pairs / total_pairs if total_pairs else 0.0
        lines.append(f"| {event_type} | {direction or '—'} | {n_pairs} | {share:.1%} |")
    table = "\n".join(lines)

    pushed = sum(n for et, d, n in rows if d == "pushed_later")
    pulled = sum(n for et, d, n in rows if d == "pulled_earlier")
    asymmetry = (pushed / pulled) if pulled else float("inf")

    return f"""## 2. Event-type frequency table

Total diffed version pairs across {len(nct_ids)} trials: {total_pairs}.

{table}

**Direction asymmetry**: dates pushed later ({pushed} pairs) outnumber dates pulled earlier \
({pulled} pairs) by {asymmetry:.1f}x. Slippage is common registry hygiene; a pulled-earlier date is \
rarer and — per the hypothesis this project is built on — the sharper signal.
"""


# ------------------------------------------------- 3. feature correlation --

FEATURE_NAMES = [
    "silence_score", "staleness", "status_ambiguity", "enrollment_signal",
    "verification_lapse", "amendment_count", "months_since_start", "has_results",
]


def build_feature_matrix(programs: list[dict], as_of: date) -> "tuple[np.ndarray, list[str]]":
    rows = []
    for p in programs:
        breakdown = p.get("score_breakdown") or {}
        if "staleness" not in breakdown:
            continue
        trials = p.get("trials") or []
        amendment_count = sum(len(t.get("history") or []) for t in trials)
        start_dates = [_parse_date(t.get("start_date")) for t in trials]
        start_dates = [d for d in start_dates if d is not None]
        months_since_start = (as_of - min(start_dates)).days / 30.44 if start_dates else None
        has_results = 1.0 if any(t.get("has_results") for t in trials) else 0.0
        if months_since_start is None:
            continue
        rows.append([
            p.get("silence_score", 0.0),
            breakdown.get("staleness", 0.0),
            breakdown.get("status_ambiguity", 0.0),
            breakdown.get("enrollment_signal", 0.0),
            breakdown.get("verification_lapse", 0.0),
            amendment_count,
            months_since_start,
            has_results,
        ])
    return np.array(rows, dtype=float), FEATURE_NAMES


def analyze_correlation_and_pca(programs: list[dict], as_of: date) -> str:
    X, names = build_feature_matrix(programs, as_of)
    n = X.shape[0]
    if n < 10:
        return "## 3. Feature correlation structure\n\n(too few programs with complete features)\n"

    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    corr = np.corrcoef(Xz, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(np.cov(Xz, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    explained = eigvals / eigvals.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=8)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_title("feature correlation matrix")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].bar(range(1, len(explained) + 1), explained, color="#4da3ff")
    axes[1].plot(range(1, len(explained) + 1), np.cumsum(explained), "o-", color="#e8574a")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("explained variance ratio")
    axes[1].set_title("PCA scree plot (bars) + cumulative (line)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_correlation_pca.png", dpi=140)
    plt.close(fig)

    corr_lines = ["| |" + "|".join(names) + "|", "|---|" + "---|" * len(names)]
    for i, row_name in enumerate(names):
        corr_lines.append(
            "| " + row_name + " |" + "|".join(f"{corr[i, j]:.2f}" for j in range(len(names))) + "|"
        )
    corr_table = "\n".join(corr_lines)

    top_loadings = "\n".join(
        f"- PC{k + 1} ({explained[k]:.0%} var): "
        + ", ".join(f"{names[i]}={eigvecs[i, k]:+.2f}" for i in range(len(names)))
        for k in range(min(3, len(explained)))
    )

    one_component = explained[0] > 0.6
    verdict = (
        f"PC1 alone explains {explained[0]:.0%} of variance — most of these 'eight' features are "
        "largely one underlying signal wearing different costumes. The model is simpler than the "
        "feature list suggests." if one_component else
        f"No single component dominates (PC1 = {explained[0]:.0%}); the features carry meaningfully "
        "distinct information, not one redundant signal."
    )

    return f"""## 3. Feature correlation structure

N = {n} programs. Features: {", ".join(names)}.

{corr_table}

### PCA
{top_loadings}

**Verdict**: {verdict}

![correlation_pca]({OUT_DIR.name}/03_correlation_pca.png)
"""


# -------------------------------------------------------- 4. KM survival --

def _kaplan_meier(durations: np.ndarray, events: np.ndarray):
    """Standard product-limit estimator. Returns (times, survival)."""
    order = np.argsort(durations)
    durations, events = durations[order], events[order]
    times, survival = [0.0], [1.0]
    s = 1.0
    unique_times = np.unique(durations[events == 1])
    for t in unique_times:
        n_at_risk = (durations >= t).sum()
        d = ((durations == t) & (events == 1)).sum()
        if n_at_risk > 0:
            s *= (1 - d / n_at_risk)
        times.append(t)
        survival.append(s)
    return np.array(times), np.array(survival)


def _median_survival(times: np.ndarray, survival: np.ndarray) -> str:
    below = np.where(survival <= 0.5)[0]
    if len(below) == 0:
        return "not reached"
    return f"{times[below[0]]:.0f}mo"


def analyze_survival(trials: list[dict], as_of: date) -> str:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if t["start_date"] is None or t["last_version_date"] is None:
            continue
        duration = (t["last_version_date"] - t["start_date"]).days / 30.44
        if duration < 0:
            continue
        event = 1 if t["status"] in TERMINAL_STATUSES else 0  # censored if still active/unknown
        by_phase[phase_bucket(t["phases"])].append({"duration": duration, "event": event})

    fig, ax = plt.subplots(figsize=(7.5, 5))
    lines = ["| phase | n | events (not censored) | median time-to-last-amendment |",
             "|---|---:|---:|---:|"]
    for phase in PHASE_ORDER:
        rows = by_phase.get(phase)
        if not rows or len(rows) < 5:
            continue
        durations = np.array([r["duration"] for r in rows])
        events = np.array([r["event"] for r in rows])
        times, survival = _kaplan_meier(durations, events)
        ax.step(times, survival, where="post", label=f"{phase} (n={len(rows)})")
        lines.append(f"| {phase} | {len(rows)} | {events.sum()} | {_median_survival(times, survival)} |")

    ax.set_xlabel("months since trial start")
    ax.set_ylabel("probability not yet at its last amendment")
    ax.set_title("Kaplan-Meier: time-to-last-amendment by phase\n(event = reached a terminal "
                  "registry status; censored = still active/unknown)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_survival_by_phase.png", dpi=140)
    plt.close(fig)

    table = "\n".join(lines)
    return f"""## 4. Time-to-last-amendment survival, by phase

Event = the trial reached a terminal registry status (COMPLETED/TERMINATED/WITHDRAWN) — treated as \
"we're confident this is the last amendment". Still-active/unknown trials are right-censored at their \
current staleness, since more amendments could still come.

{table}

This is the base rate for "how long do amendments normally keep happening" — read "unusually stale" \
against these medians, not an assumed fixed cutoff.

![survival_by_phase]({OUT_DIR.name}/04_survival_by_phase.png)
"""


# ------------------------------------------------------ 5. sponsor cluster --

def _current_sponsor(program: dict) -> "str | None":
    sponsors = program.get("sponsors_over_time") or []
    if not sponsors:
        return None
    with_last_seen = [s for s in sponsors if s.get("last_seen")]
    pool = with_last_seen or sponsors
    return max(pool, key=lambda s: s.get("last_seen") or "")["sponsor"]


def analyze_sponsor_clustering(programs: list[dict]) -> str:
    by_sponsor: dict[str, list[float]] = defaultdict(list)
    for p in programs:
        sponsor = _current_sponsor(p)
        breakdown = p.get("score_breakdown") or {}
        if sponsor is None or "staleness" not in breakdown:
            continue
        by_sponsor[sponsor].append(breakdown["staleness"])

    groups = {s: v for s, v in by_sponsor.items() if len(v) >= 2}
    if len(groups) < 3:
        return "## 5. Sponsor clustering\n\n(too few sponsors with >=2 programs to estimate ICC)\n"

    all_vals = np.array([v for vals in groups.values() for v in vals])
    grand_mean = all_vals.mean()
    n_total = len(all_vals)
    g = len(groups)

    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    ss_within = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in groups.values())
    ms_between = ss_between / (g - 1)
    ms_within = ss_within / (n_total - g)

    sum_ni2 = sum(len(v) ** 2 for v in groups.values())
    n0 = (n_total - sum_ni2 / n_total) / (g - 1)  # Fisher's average-group-size correction

    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within) if ms_within > 0 else float("nan")
    icc = max(0.0, icc) if np.isfinite(icc) else icc

    f_stat = ms_between / ms_within if ms_within > 0 else float("inf")
    p_value = spstats.f.sf(f_stat, g - 1, n_total - g)

    return f"""## 5. Sponsor clustering (independence check)

{g} sponsors with >=2 programs, {n_total} programs total (of {sum(len(v) for v in by_sponsor.values())} \
sponsor-attributable programs).

- Intra-sponsor ICC of the staleness feature: **{icc:.2f}**
- One-way ANOVA: F({g - 1}, {n_total - g}) = {f_stat:.2f}, p = {p_value:.2e}

**Interpretation**: an ICC of {icc:.2f} means roughly {icc:.0%} of the total variance in staleness is \
"explained" by which sponsor a program belongs to, rather than program-specific circumstances. \
{"This is substantial — programs from the same sponsor are NOT independent observations. A sponsor "
"that runs out of money plausibly does take multiple programs stale at once, which inflates the "
"effective variance of any aggregate estimate (and narrows real degrees of freedom below the raw "
"program count)." if icc >= 0.1 else
"This is small — sponsor identity doesn't explain much of the staleness variance, so treating "
"programs as roughly independent for confidence-interval purposes is a reasonable approximation."}
"""


# ------------------------------------------------------- 6. regional strata --

def _trial_countries(nct_id: str, con: duckdb.DuckDBPyConnection) -> set[str]:
    found = pp._best_trial_snapshot(nct_id, con)
    if found is None:
        return set()
    study, _source = found
    locations = (study.get("protocolSection", {}).get("contactsLocationsModule", {}) or {}).get("locations", [])
    return {loc.get("country") for loc in locations if loc.get("country")}


def analyze_regional_strata(programs: list[dict], con: duckdb.DuckDBPyConnection, as_of: date) -> str:
    china_program_ids = set()
    for p in programs:
        nct_ids = p.get("nct_ids") or []
        countries = set()
        for nct_id in nct_ids:
            countries |= _trial_countries(nct_id, con)
        if countries and countries == {"China"}:
            china_program_ids.add(p["program_id"])
        elif "China" in countries and len(countries) == 1:
            china_program_ids.add(p["program_id"])

    X, names = build_feature_matrix(programs, as_of)
    id_order = [p["program_id"] for p in programs
                if "staleness" in (p.get("score_breakdown") or {})
                and any(_parse_date(t.get("start_date")) for t in (p.get("trials") or []))]
    is_china = np.array([pid in china_program_ids for pid in id_order])

    if is_china.sum() < 5 or (~is_china).sum() < 5:
        return (f"## 6. Regional strata\n\n{int(is_china.sum())} China-only-site programs identified "
                f"— too few for a reliable comparison.\n")

    lines = ["| feature | China-only median | rest median | Mann-Whitney p |", "|---|---:|---:|---:|"]
    for j, name in enumerate(names):
        a, b = X[is_china, j], X[~is_china, j]
        try:
            _, p_value = spstats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            p_value = float("nan")
        lines.append(f"| {name} | {np.median(a):.1f} | {np.median(b):.1f} | {p_value:.2e} |")
    table = "\n".join(lines)

    return f"""## 6. Regional strata: China-only-site programs vs. the rest

Proxy used: a program counts as "China-only-site" when every trial location on file for it is in \
China (a proxy for sponsor region, not a direct sponsor-nationality field — CT.gov doesn't expose \
sponsor HQ country directly). {int(is_china.sum())} programs qualify, vs {int((~is_china).sum())} rest.

{table}

Any feature with p < 0.05 above differs systematically by region — if so, that base rate (and \
possibly the model) should be split by region rather than pooled.
"""


# ------------------------------------------------------- 7. UNKNOWN base rate --

def analyze_unknown_base_rate(programs: list[dict]) -> str:
    total = len(programs)
    unknown = sum(1 for p in programs if p.get("latest_status") == "UNKNOWN")
    rate = unknown / total if total else 0.0
    return f"""## 7. Crude base rate from the UNKNOWN status proxy

{unknown} / {total} programs ({rate:.1%}) currently show CT.gov status UNKNOWN (verification lapsed).

**This is a biased proxy, in both directions**:
- **Understates** true silent death: a sponsor can leave a record as ACTIVE_NOT_RECRUITING or even \
COMPLETED indefinitely without CT.gov ever flipping it to UNKNOWN — UNKNOWN requires the status- \
verification date to lapse past CT.gov's own threshold, which many abandoned trials never trigger.
- **Overstates** it in the other direction: some UNKNOWN trials are just administrative lag (a CRA \
missed a routine re-verification on an otherwise perfectly healthy trial), not genuine abandonment.

Treat {rate:.1%} as a floor-ish anchor, not an estimate of the true silent-discontinuation rate — the \
real rate is almost certainly higher, by an unknown amount that only human labelling can pin down.
"""


# --------------------------------------------------------------------- main --

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    programs = pp.load_materialized()
    if not programs:
        print("provisional_programs not materialized — run "
              "`python scripts/run_labelling_app.py --rebuild` first.")
        return

    as_of = date.today()
    trials = load_trials(programs)
    con = duckdb.connect(str(WAREHOUSE_DB), read_only=True)

    print(f"Loaded {len(programs)} programs, {len(trials)} trials. Running analyses...")
    sections = [
        analyze_bimodality(trials, as_of),
        analyze_event_types(con, trials),
        analyze_correlation_and_pca(programs, as_of),
        analyze_survival(trials, as_of),
        analyze_sponsor_clustering(programs),
        analyze_regional_strata(programs, con, as_of),
        analyze_unknown_base_rate(programs),
    ]
    con.close()

    report = ("# Silence-feature EDA\n\n"
               f"Generated {as_of.isoformat()} from {len(programs)} provisional programs / "
               f"{len(trials)} trials. Aggregate distributions only — no per-program scores.\n\n"
               + "\n---\n\n".join(sections))
    REPORT_PATH.write_text(report)
    print(f"\nWrote {REPORT_PATH} and charts under {OUT_DIR}/")


if __name__ == "__main__":
    main()
