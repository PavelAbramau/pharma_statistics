# 0001 — Scope of the current-state fetch

Date: 2026-08-30

## Context

`derivedSection.conditionBrowseModule` (the NLM MeSH terms this project's
heme/solid scope classifier reads — see `labelling/trial_scope.py`,
`discovery/mesh_categories.py`) only exists on CT.gov's **current-state**
study fetch (`GET /api/v2/studies/{nctId}`, `CtgovClient.get_study()`). It
is never present on a versioned-history body
(`GET /api/int/studies/{nctId}/history/{version}`,
`CtgovClient.get_study_version()`) — confirmed by inspecting live
responses, not assumed. `scripts/fetch_current_state.py` fetches this for
every in-universe trial and stores it through the normal snapshot store,
under the bare NCT id (the same key `pharma_stats.snapshot.get_as_of` /
`.latest` resolve).

This creates a real risk. The project's entire silence-detection premise
depends on time-cut discipline: every feature that claims to describe a
program's state must be resolvable "as of" a specific historical date
(CLAUDE.md — `snapshot.get_as_of` is the primitive the time-cut backtest
depends on). A **current-state** fetch has no "as of" — it describes the
trial as CT.gov shows it *today*, regardless of when the record last
changed. Letting that data flow into a silence/staleness feature would
leak forward-looking information into what must be a historically-honest
signal, silently invalidating any backtest built on it.

## Decision

The current-state fetch is permitted to be read for exactly three
**static, universe-membership** properties, and nothing else:

1. **Disease category** (MeSH terms, via `conditionBrowseModule`) — used
   only for scope classification (`labelling/trial_scope.classify_trial`):
   is this trial heme, solid, non-oncology, or ambiguous. This decides
   whether a program is *in the universe this project studies at all*, not
   how silent it is.
2. **Sponsor class** (industry vs. not) — used only for the
   `non_industry` scope flag.
3. **Start date** — used only for the `pre_2012` scope boundary.

None of these describe *whether a program has gone quiet* — they describe
what the program fundamentally *is*, facts that (for this project's
purposes) don't meaningfully change over a program's life. That's what
makes reading them from a current-state snapshot an acceptable, bounded
exception rather than a leak.

**Every other field — status, last-update date, enrollment,
verification date, amendment count, anything that feeds
`compute_silence_score` / `classify_archetypes` / any future model
feature — must keep coming from the time-cut versioned-history path**
(`provisional_programs._best_trial_snapshot`, which prefers the highest
indexed version and falls back to the plain snapshot only when no history
has been indexed yet). Never from a current-state-only read.

## Enforcement

Read discipline, once established, tends to erode by a thousand
reasonable-looking additions. `audit/universe.py`'s
`_current_state_read_boundary` check guards against that: it statically
scans `labelling/provisional_programs.py` for calls to
`snapshot.latest`/`snapshot.get_as_of` outside an explicit whitelist of
functions permitted to make a current-state-only read
(`_CURRENT_STATE_READ_WHITELIST` in that module). Adding a new current-
state read anywhere in that file requires explicitly widening the
whitelist — a visible, deliberate, reviewable change to the audit check
itself, not something that can happen quietly inside an unrelated
feature-computation function.

## Consequences

- `labelling/trial_scope.py`'s `_condition_browse_data` (in
  `provisional_programs.py`) is the one sanctioned current-state read
  today, and is in the whitelist.
- If a future contributor needs another current-state field, they must
  either (a) show it's a static universe-membership property like the
  three above and add it to this document + the whitelist, or (b) source
  it from the versioned-history path instead.
- This decision does not restrict what `scripts/fetch_current_state.py`
  *fetches and stores* — it stores the full record, same as any other
  snapshot. The restriction is entirely on what downstream *feature code*
  is allowed to *read* from it.
