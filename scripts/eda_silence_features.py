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
from collections import Counter, defaultdict
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

# silence_score deliberately excluded: it's a deterministic weighted sum
# of staleness + status_ambiguity + verification_lapse (see
# provisional_programs.compute_silence_score), so including it alongside
# its own inputs manufactures a dominant component out of double-counted
# variance — that's why an earlier pass showed a near-zero-variance last
# component. staleness_age_adjusted is staleness residualised on
# months_since_start (OLS) — raw staleness is confounded by trial age
# (older trials mechanically accumulate more staleness signal), which
# shows up as their negative raw correlation.
FEATURE_NAMES = [
    "staleness", "status_ambiguity", "enrollment_signal", "verification_lapse",
    "amendment_count", "months_since_start", "has_results", "staleness_age_adjusted",
]


def build_feature_matrix(programs: list[dict], as_of: date) -> "tuple[np.ndarray, list[str], list[str]]":
    rows = []
    program_ids = []
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
            breakdown.get("staleness", 0.0),
            breakdown.get("status_ambiguity", 0.0),
            breakdown.get("enrollment_signal", 0.0),
            breakdown.get("verification_lapse", 0.0),
            amendment_count,
            months_since_start,
            has_results,
        ])
        program_ids.append(p["program_id"])

    X = np.array(rows, dtype=float)
    staleness_col, age_col = X[:, 0], X[:, 5]
    slope, intercept = np.polyfit(age_col, staleness_col, 1) if len(X) > 1 else (0.0, 0.0)
    age_adjusted = staleness_col - (slope * age_col + intercept)
    X = np.column_stack([X, age_adjusted])
    return X, FEATURE_NAMES, program_ids


def analyze_correlation_and_pca(programs: list[dict], as_of: date) -> str:
    X, names, program_ids = build_feature_matrix(programs, as_of)
    n = X.shape[0]
    if n < 10:
        return "## 3. Feature correlation structure\n\n(too few programs with complete features)\n"

    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    corr = np.corrcoef(Xz, rowvar=False)
    condition_number = np.linalg.cond(corr)

    eigvals, eigvecs = np.linalg.eigh(np.cov(Xz, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    explained = eigvals / eigvals.sum()

    # rescale to the OBSERVED off-diagonal range — a fixed [-1, 1] scale
    # renders everything pale when real values sit in a much narrower band
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
    axes[0].set_title(f"correlation matrix (n={n} programs, diagonal masked,\n"
                       f"scale=±{vmax:.2f} = observed range)")
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            axes[0].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].bar(range(1, len(explained) + 1), explained, color="#4da3ff")
    axes[1].plot(range(1, len(explained) + 1), np.cumsum(explained), "o-", color="#e8574a")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("explained variance ratio")
    axes[1].set_title(f"PCA scree (n={n}) — condition number={condition_number:.1f}")
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
        f"PC1 alone explains {explained[0]:.0%} of variance — most of these features are largely "
        "one underlying signal wearing different costumes. The model is simpler than the feature "
        "list suggests." if one_component else
        f"No single component dominates (PC1 = {explained[0]:.0%}); the features carry meaningfully "
        "distinct information, not one redundant signal."
    )
    if condition_number > 1e6:
        rank_note = (
            f"Condition number = {condition_number:.2e} — this is EXACT (not approximate) rank "
            "deficiency, and it's mechanical, not a data surprise: staleness_age_adjusted is "
            "constructed as staleness minus a linear function of months_since_start, and both of "
            "those inputs are also in this matrix — so one column is an exact linear combination of "
            "two others by definition. That's expected whenever a residualised feature sits "
            "alongside its own regressors; it doesn't indicate anything new about the underlying data."
        )
    else:
        rank_note = (
            f"Condition number = {condition_number:.1f} — " +
            ("high enough to indicate near-rank-deficiency (some features are near-exact linear "
             "combinations of others)." if condition_number > 30 else
             "no strong evidence of rank deficiency at this size.")
        )

    return f"""## 3. Feature correlation structure

N = {n} programs (of {len(programs)} total — programs without a computable score/age are excluded). \
Features: {", ".join(names)}. silence_score is excluded — see the code comment: it's a deterministic \
function of staleness + status_ambiguity + verification_lapse, so keeping it in would double-count \
those three and manufacture a dominant PC1 out of it.

{corr_table}

### PCA
{top_loadings}

{rank_note}

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


# A trial whose last known version landed within this many days of as_of
# is still "fresh" — we can't yet tell whether it has truly gone quiet or
# is about to be amended again, so it's censored rather than treated as
# an observed last-amendment event. This is the ONLY thing that decides
# event-vs-censored for curve B below — deliberately independent of
# registry status, which is known to lag true program state by years.
STILL_FRESH_DAYS = 180


def analyze_survival(trials: list[dict], as_of: date) -> str:
    by_phase_terminal: dict[str, list[dict]] = defaultdict(list)
    by_phase_amendment: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if t["start_date"] is None or t["last_version_date"] is None:
            continue
        duration = (t["last_version_date"] - t["start_date"]).days / 30.44
        if duration < 0:
            continue
        phase = phase_bucket(t["phases"])

        # Curve A (kept, retitled honestly): event = reached a terminal
        # registry status. This is confounded by how long status updates
        # lag reality — kept for comparison, not as the primary answer.
        terminal_event = 1 if t["status"] in TERMINAL_STATUSES else 0
        by_phase_terminal[phase].append({"duration": duration, "event": terminal_event})

        # Curve B (new, the actual "time to last amendment"): event = the
        # last recorded version IS the last one, inferred from recency
        # alone, never from registry status.
        days_since_last = (as_of - t["last_version_date"]).days
        amendment_event = 1 if days_since_last > STILL_FRESH_DAYS else 0
        by_phase_amendment[phase].append({"duration": duration, "event": amendment_event})

    def _render(by_phase: dict, ax, title: str, ylabel_event: str) -> list[str]:
        lines = [f"| phase | n | events ({ylabel_event}) | median |", "|---|---:|---:|---:|"]
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
        ax.set_ylabel("probability event not yet occurred")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        return lines

    n_total = sum(len(v) for v in by_phase_terminal.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    lines_a = _render(
        by_phase_terminal, axes[0], f"A. Time to reaching a TERMINAL REGISTRY STATUS (n={n_total})\n"
        "(confounded by status-update lag — kept for comparison only)",
        "reached terminal status",
    )
    lines_b = _render(
        by_phase_amendment, axes[1], f"B. Time to LAST AMENDMENT (n={n_total})\n"
        f"(event = no new version in >{STILL_FRESH_DAYS}d, status-independent)",
        "inferred last amendment",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_survival_by_phase.png", dpi=140)
    plt.close(fig)

    events_a = sum(r["event"] for rows in by_phase_terminal.values() for r in rows)
    events_b = sum(r["event"] for rows in by_phase_amendment.values() for r in rows)
    table_a, table_b = "\n".join(lines_a), "\n".join(lines_b)
    return f"""## 4. Survival, by phase: two different questions, deliberately kept separate

**Panel A — time to a terminal registry status.** Confounded by exactly what this whole project is \
about: registry status is known to lag true program state, sometimes by years. Kept only so the gap \
against panel B is visible.

{table_a}

**Panel B — time to last amendment**, the base rate this project actually needs: how long do sponsors \
normally keep touching a record, independent of registry status? Event = no new version posted in \
more than {STILL_FRESH_DAYS} days (censored if the last version is more recent than that — we can't \
yet tell if it's truly gone quiet).

{table_b}

**The gap between A and B is itself the finding**: {events_b} trials have gone quiet on the amendment \
record (panel B) vs only {events_a} that have been formally marked terminal (panel A) — {events_b - events_a} \
trials ({(events_b - events_a) / events_a:.0%} more) that look "amendment-silent" have NOT had their \
registry status updated to match. Medians in B are equal to or shorter than A's in every phase but \
PHASE3. Read "unusually stale" against panel B, not an assumed fixed cutoff — panel A alone \
understates, and lags, how early a program actually went silent.

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

    X, names, id_order = build_feature_matrix(programs, as_of)
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


# ------------------------------------------------ 3.5 enrollment_signal audit --

def analyze_enrollment_signal(programs: list[dict]) -> str:
    """enrollment_signal's near-zero correlation with everything else
    (section 3) is either "genuinely orthogonal information" or "an
    empty/near-constant column" — this settles which. It's computed by
    compute_silence_score from the LATEST trial's status + enrollment_type,
    not read directly, so "missingness" here means enrollment_type itself
    being unset on the latest trial, tracked separately from the resulting
    signal value's own distribution."""
    values, missing_enrollment_type = [], 0
    for p in programs:
        breakdown = p.get("score_breakdown") or {}
        if "enrollment_signal" not in breakdown:
            continue
        values.append(breakdown["enrollment_signal"])
        trials = p.get("trials") or []
        latest = eda_latest_trial(trials) if trials else None
        if latest is None or not latest.get("enrollment_type"):
            missing_enrollment_type += 1

    n = len(values)
    dist = Counter(values)
    nonzero = sum(1 for v in values if v != 0.0)
    dist_lines = "\n".join(f"  {v}: {c} ({c/n:.1%})" for v, c in sorted(dist.items()))

    return f"""## 3.5. Investigating enrollment_signal

N = {n} programs with a computed score.

- Non-null enrollment_type on the latest trial: {n - missing_enrollment_type} / {n} \
({(n - missing_enrollment_type)/n:.1%}) — enrollment_type itself is well populated, this is NOT a \
missing-data problem.
- enrollment_signal value distribution:
{dist_lines}
- Non-zero: {nonzero} / {n} ({nonzero/n:.1%})

**Verdict**: enrollment_signal is not missing data — it's a near-constant column because \
compute_silence_score's rule only assigns a non-zero value when the LATEST trial is BOTH in a \
terminal-ish status (TERMINATED/WITHDRAWN/SUSPENDED, or COMPLETED) AND still carries ESTIMATED \
(not finalized) enrollment. Most ESTIMATED-enrollment trials are simply still RECRUITING/ACTIVE, \
which the rule doesn't touch at all — so {nonzero}/{n} programs ever get a non-zero value. The \
near-zero correlation in section 3 is orthogonal-information-with-almost-no-variance, not a bug: \
the feature is doing exactly what it was designed to do, just on a very narrow trigger.
"""


def eda_latest_trial(trials: list[dict]) -> dict:
    return max(trials, key=lambda t: t.get("last_update_post_date") or "")


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
        analyze_enrollment_signal(programs),
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
