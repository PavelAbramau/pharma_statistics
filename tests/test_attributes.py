"""Tests for attributes/payload.py and attributes/target.py — B0 asset
attribute derivation. Every case either resolves to a real value or
falls back to undisclosed/unresolved; nothing here should ever guess."""
from __future__ import annotations

from pharma_stats.attributes import payload as payload_attr
from pharma_stats.attributes import target as target_attr


def test_derive_payload_chemotype_from_suffix():
    assert payload_attr.derive_payload_chemotype("Trastuzumab deruxtecan") == "camptothecin_topo1"
    assert payload_attr.derive_payload_chemotype("Enfortumab vedotin") == "auristatin"
    assert payload_attr.derive_payload_chemotype("Inotuzumab ozogamicin") == "calicheamicin"


def test_derive_payload_chemotype_undisclosed_for_bare_dev_code():
    assert payload_attr.derive_payload_chemotype("XMT-1592") == "undisclosed"
    assert payload_attr.derive_payload_chemotype("SKB264", synonyms=["MK-2870"]) == "undisclosed"


def test_derive_payload_chemotype_checks_synonyms_too():
    assert payload_attr.derive_payload_chemotype("XYZ-001", synonyms=["Sacituzumab govitecan"]) == "camptothecin_topo1"


def test_derive_payload_chemotype_other_for_non_claude_md_category():
    assert payload_attr.derive_payload_chemotype("Upifitamab rilsodotin") == "other"


def test_target_from_antibody_stem_well_known_compounds():
    assert target_attr.derive_target("Trastuzumab deruxtecan") == ("ERBB2", "antibody_stem")
    assert target_attr.derive_target("Sacituzumab govitecan") == ("TACSTD2", "antibody_stem")
    assert target_attr.derive_target("Brentuximab vedotin") == ("TNFRSF8", "antibody_stem")


def test_target_from_trial_text_single_hit():
    target, source = target_attr.derive_target(
        "XMT-1592", text_snippets=["This study evaluates XMT-1592, an ADC targeting NaPi2b."],
    )
    assert (target, source) == ("SLC34A2", "trial_text")


def test_target_from_trial_text_ambiguous_multiple_hits_not_guessed():
    target, source = target_attr.derive_target(
        "XYZ-001",
        text_snippets=["A combination study of an anti-HER2 ADC and an anti-EGFR ADC."],
    )
    assert target is None
    assert source == "unresolved"


def test_target_from_name_when_text_has_nothing():
    target, source = target_attr.derive_target("Anti-Claudin18.2 ADC-009", text_snippets=[])
    assert (target, source) == ("CLDN18", "name")


def test_target_unresolved_when_nothing_matches():
    target, source = target_attr.derive_target("XL114", text_snippets=["A study of XL114 in solid tumors."])
    assert target is None
    assert source == "unresolved"


def test_antibody_stem_takes_priority_over_trial_text():
    target, source = target_attr.derive_target(
        "Trastuzumab deruxtecan", text_snippets=["Also mentions EGFR expression in exploratory biomarker analysis."],
    )
    assert (target, source) == ("ERBB2", "antibody_stem")
