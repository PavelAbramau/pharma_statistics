"""Cheap, high-volume Gate 1 (is_adc) / Gate 2 (in_scope) auto-triage for
provisional programs — separate from the silver status labeller
(pharma_stats.silver), which answers a completely different, higher-stakes
question (is the program dead) and must stay isolated from gold.

Token efficiency is the primary design constraint here: most candidates
should never reach a model call at all.

    Layer 1 (deterministic.py)  — no API calls. See its docstring for the
                                   five rules, in order.
    Layer 2 (layer2.py)         — batched Message Batches API calls on the
                                   Layer-1 residue only, k=3 adaptive to 5.
    Layer 3 (layer3.py)         — single web_search query, Layer-2 "unsure"
                                   residue only, capped.
    Validation (validation.py)  — blind agreement gate against real human
                                   decisions before any of this is trusted.

Every decision this package proposes is decided_by="auto" with a
triage_layer, triage_rule, and (for layers 2/3) triage_model/
triage_prompt_version — see labelling/vocab.py and labelling/store.py.
Nothing here writes to gold/labels.jsonl directly; see
scripts/run_triage.py (stages to data/triage_pending.json) and
scripts/apply_triage_decisions.py (the only place that commits, and only
after validation.check_gate() passes).
"""
