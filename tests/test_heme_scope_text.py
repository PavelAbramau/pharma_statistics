"""Tests for discovery/heme_scope_text.py — the text-based heme/tumour-
system proxy used ONLY by the matrix-only universe extension
(docs/decisions/0008), never the main opportunity matrix's MeSH pipeline."""
from __future__ import annotations

from pharma_stats.discovery import heme_scope_text as hst


def test_classifies_heme_keyword():
    assert hst.classify_condition_text(["Diffuse Large B-Cell Lymphoma"]) == "heme"


def test_classifies_solid_system():
    assert hst.classify_condition_text(["Metastatic Breast Cancer"]) == "breast"


def test_returns_none_for_generic_text():
    assert hst.classify_condition_text(["Advanced Solid Tumors", "Neoplasms"]) is None


def test_returns_none_for_empty_conditions():
    assert hst.classify_condition_text([]) is None


def test_majority_vote_across_multiple_conditions():
    conditions = ["Non-Small Cell Lung Cancer", "Lung Adenocarcinoma", "Breast Cancer"]
    assert hst.classify_condition_text(conditions) == "thoracic_lung"


def test_heme_beats_solid_by_majority():
    conditions = ["Multiple Myeloma", "Leukemia", "Breast Cancer"]
    assert hst.classify_condition_text(conditions) == "heme"


def test_does_not_false_positive_on_short_acronyms():
    # "all" as a bare acronym for ALL leukemia is deliberately not
    # keyworded -- would false-positive on unrelated text ("overall...").
    assert hst.classify_condition_text(["Overall Survival Cohort"]) is None
