# 0004 — Spend measures sponsor conviction, not asset quality

Date: 2026-09-02

## Context

The financial layer (`pharma_stats.finance`) produces cost, conviction-
ratio, and (as later components land) sponsor-distress and impairment
signals for every program. It would be easy for a downstream consumer —
or a future version of this project's own model — to treat high relative
spend as evidence of a good asset, or a spending cut as evidence of a
bad one. Neither inference is sound, for a structural reason, not a data-
quality one.

## Decision

**Spend and survival are jointly determined**, not independent
observations of one causing the other. A sponsor spends more on a
program *because* they believe in it (a resourcing decision informed by
the same internal signal — emerging efficacy, competitive position,
strategic fit — that also determines whether they keep running it), and
a program that's already succeeding attracts more spend as it advances
through phases (Phase 3 costs more than Phase 1 largely because it's a
later, more advanced trial of an asset that already cleared Phase 1/2,
not an independent input). Observing "high spend, program survived" is
consistent with "spend caused survival," "survival caused spend," and
"an unobserved third factor (genuine efficacy) caused both" — the
financial layer's data alone cannot distinguish between these.

Every feature in this layer (`synthetic_cost_index_monthly`,
`conviction_ratio_monthly`, and the sponsor-level signals still to come)
is therefore a **conviction signal**: it says something honest about how
much a sponsor is committing to a program relative to peers, which is
useful for *prediction* (a sponsor's revealed conviction is real
information about what they expect to happen next) but must never be
read as a *causal* claim about the asset's underlying scientific
quality, and must never be used to justify a causal claim in any report
this project produces downstream.

## Consequences

- Any backtest or model built on these features should be evaluated as a
  predictive tool (does conviction correlate with what happens next),
  never framed as "spending more makes programs succeed" or "cutting
  spend kills good programs" — both are the reverse-causation trap this
  decision exists to name.
- A sponsor-level distress signal (item 3 of the financial layer) killing
  programs "in batches" is a real, useful predictive pattern — a sponsor
  going bankrupt genuinely does end multiple programs at once — but it's
  a statement about the sponsor's capacity, not a judgement on any one
  asset's merit; a program dropped in a distress-driven portfolio cut can
  be scientifically sound and still die for reasons that have nothing to
  do with it.
- This applies to every event this layer emits, present and future —
  scoped at the layer level, not per-feature, because the joint-
  determination problem is structural to "spend," not a quirk of any one
  signal.
