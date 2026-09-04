"""Product B — the sourcing screener (diagrams/diagram.md: "kill-reason ·
recoverability · crowding"). This package holds the B1-B7 axes as they get
built; B0 (asset attribute derivation) and B5 (crowding/failure-density
opportunity matrix) already live in `pharma_stats.attributes` and stay
there rather than moving here.

Built so far:

- **B1 — kill reason vs. spend** (`kill_reason_spend.py`): for each
  dead_confirmed program, how much had been spent (money layer:
  `pharma_stats.finance.panel`) by the time it died, broken out by the
  stated `kill_reason`. Per `docs/decisions/0004`, this is a descriptive
  cut, never a causal one — see that decision and this module's own
  docstring before reading it any other way.
"""
