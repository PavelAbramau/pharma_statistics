# 0005 — Opportunity matrix rebuilt on payload x tumour-system axes

Date: 2026-09-04

## Context

The B5 opportunity matrix (`attributes/matrix.py`) originally crossed
target antigen (HGNC symbol, ~35 distinct values seen so far, headed
toward the low hundreds as more candidates resolve) against the specific
MeSH indication term (~100+ distinct values, one per condition name).
With the matrix's population strictly gated to confirmed
is_adc=yes/in_scope=yes programs — currently 38, and by CLAUDE.md's own
expected-universe ceiling never more than ~450 — that grid can never be
dense. ~300-450 programs spread across (tens of targets) x (hundreds of
indications) puts almost every cell at 0 or 1 programs regardless of how
much labelling happens later. That's arithmetic, not a data gap: no
amount of further Gate-2/Gate-3 review fixes a denominator problem
created by the axis choice itself.

## Decision

**Coarsen both axes before the matrix is read, not after.** New axes:

- **Payload chemotype** (`attributes/payload.py`, already a CLAUDE.md-
  controlled vocabulary, ~10 classes) instead of target antigen.
- **Tumour system** (`attributes/tumour_system.py`, new) instead of the
  specific MeSH indication term — 8 coarse organ/tissue groups (breast,
  gi_hepatobiliary, thoracic_lung, genitourinary, gynecologic,
  skin_ocular_melanoma, sarcoma_soft_tissue,
  endocrine_neuroendocrine_germcell), hand-curated on top of
  `discovery/mesh_categories.py`'s already-reviewed "solid" MeSH ID set.

~10 x ~8 = ~80 cells for ~300-450 programs is still thin, but it's thin
in the way the problem is actually thin, not thin because the axes were
chosen too fine.

**Both axes exclude unresolved values entirely, never bucket them.** A
program with no payload suffix on file yet ("undisclosed") or no
resolvable tumour system is dropped from the matrix's population, full
stop — not counted into an "undisclosed" column or "unknown" row. A
catch-all bucket would itself become the single largest cell in the
matrix (most early-stage candidates are bare dev codes with no INN
suffix yet) and would misrepresent "we don't know" as "this combination
is common." `build_matrix`'s `population_stats` return value reports the
excluded counts by reason so this stays visible, never silently thinned.

**"Minimum MeSH tree depth" is enforced the only way the data supports
it.** CT.gov's `conditionBrowseModule` returns MeSH id + term only, never
a tree number — this project has no real MeSH tree-depth data on disk,
so depth is enforced via `mesh_categories.py`'s existing "generic_basket"
category (root/umbrella terms like "Neoplasms", "Neoplasm Metastasis")
plus two additional site-agnostic histology terms within "solid" itself
("Carcinoma", "Adenocarcinoma") that name a real pathology but no body
site. None of these five get a tumour system; a program whose only MeSH
signal is one of them is excluded, never becomes its own cell, and never
gets folded into a nominal system it doesn't actually belong to.

**The graveyard cells are reported as a ranked list, not a heatmap.**
`matrix_report.py` no longer renders a colour grid — with ~80 mostly-thin
cells, a heatmap invites reading colour into cells that have one or two
programs in them. `render_graveyard_markdown` (unchanged in spirit from
the original grid version) is now the module's only output: graveyard
cells (few/no live, >= min_n dead), ranked by dead-program count, with
cells below min_n excluded from the ranking as insufficient evidence
rather than shown as either graveyard or white space.

## Consequences

- `reports/opportunity_matrix.html` (the old target x indication heatmap)
  is removed rather than regenerated on the new axes — the deliverable is
  now `reports/opportunity_matrix_graveyard.md` only.
- If OncoTree indication normalisation lands later (per CLAUDE.md's
  five-entity model), `tumour_system.py`'s 8 groups should be revisited —
  they're a documented, hand-curated stand-in for OncoTree's tissue-level
  grouping, not the real thing, same posture as 0003's peer-group
  stand-in.
- Extending either axis (a new payload suffix in `payload.py`, a new
  "solid" MeSH ID in `mesh_categories.py`) requires also deciding its
  tumour-system group in `tumour_system.py` by hand — there is no
  automatic inference step to keep honest here, by design.
