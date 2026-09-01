# 0002 — Non-antibody conjugates are not ADCs

Date: 2026-08-31

## Context

Layer 2 of the Gate 1/2 triage (see `pharma_stats.triage`) answers
"is this an antibody-drug conjugate?" The project's locked scope
(CLAUDE.md) is **ADCs only**: an antibody or antibody fragment covalently
linked to a cytotoxic payload via a chemical linker. Neighbouring
conjugate platforms share the "toxin on a targeting ligand" *shape* and
show up in the same CT.gov residue — they are not ADCs.

The motivating case is **BT5528**, a Bicycle toxin conjugate (BTC): a
constrained peptide ("Bicycle"), not an antibody or antibody fragment,
linked to a cytotoxin. Layer 2 answered `is_adc=no` from recall. That
answer is correct, but recall is the expensive, hallucination-prone
path; the rule is a scope fact, not a one-off judgement about BT5528.

Confirmed against the same false-positive trap
`discovery/patterns.py`'s `LITERAL_TERMS` already documents for bare
"conjugate": polymer-drug conjugates (etirinotecan pegol), siRNA
conjugates, vaccine conjugates. Peptide-drug conjugates (PDCs) and
small-molecule drug conjugates (SMDCs) are the same family.

## Decision

`is_adc=no` whenever the name, a synonym, or a later curated code list
identifies the molecule as a **non-antibody conjugate**:

- Bicycle toxin conjugate / BTC (e.g. BT5528, and any later Bicycle
  Therapeutics toxin conjugate that lands in the residue)
- Peptide-drug conjugate / PDC
- Small-molecule drug conjugate / SMDC
- Polymer-drug conjugate (including PEGylated cytotoxics described as
  conjugates)
- siRNA / oligonucleotide conjugates
- Vaccine conjugates (KLH, CRM197, polysaccharide)

An antibody **fragment** (Fab, scFv, nanobody, VHH) covalently linked to
a payload **is** an ADC for this project and is not excluded here.

The rule lives in `pharma_stats.triage.deterministic` as
`layer1_non_antibody_conjugate`, fires before the generic-ADC-class-label
rule (so "X peptide-drug conjugate" cannot be promoted to `is_adc=yes`
by the word "conjugate"), and is the single place this decision is
applied — not a per-candidate override in the labelling app.

## Consequences

- Extending the known-code set (new Bicycle / PDC / SMDC assets) is a
  reviewable edit to `NON_ANTIBODY_CONJUGATE_CODES` in
  `deterministic.py`, pointed at by this document. Do not special-case
  a name in the labelling UI.
- Discovery may still *surface* these as candidates (literal "conjugate"
  is a weak recall term on purpose). Layer 1 is where they leave the
  manual queue, with `decided_by=auto`.
- If a future asset is an antibody-fragment conjugate marketed with
  "bicycle-like" language, it must be *removed* from the code set
  here, not silently reclassified in a one-off gold line.
