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


def test_her2_full_name_does_not_false_positive_match_egfr():
    """Real bug found via TQB2102: EGFR's spelled-out name is a strict
    substring of ERBB2/HER2's own spelled-out name, so every HER2 mention
    was also hitting EGFR and manufacturing ambiguity that isn't real
    biology."""
    target, source = target_attr.derive_target(
        "TQB2102", text_snippets=[
            "TQB2102 is an antibody-drug conjugate comprised of a humanised antibody against "
            "Human Epidermal Growth Factor Receptor 2 (HER2), an enzyme-cleavable linker.",
        ],
    )
    assert (target, source) == ("ERBB2", "trial_text")


def test_mucin_16_spelled_out_alias():
    target, source = target_attr.derive_target(
        "HWK-016", synonyms=["HWK-016, MUCIN-16-targeted ADC"], text_snippets=[],
    )
    assert (target, source) == ("MUC16", "name")


def test_claudin_cldn_parenthetical_phrasing():
    target, source = target_attr.derive_target(
        "IBI343", text_snippets=["Participants with Claudin (CLDN) 18.2-Positive tumors."],
    )
    assert (target, source) == ("CLDN18", "trial_text")


def test_cd46_and_adam9_aliases():
    assert target_attr.derive_target("FOR46", text_snippets=["designed to target and bind to CD46"])[0] == "CD46"
    assert target_attr.derive_target("MGC028", text_snippets=["targeted against ADAM9"])[0] == "ADAM9"


def test_trial_text_majority_resolves_repeated_target_over_single_stray_mention():
    """RC108's own text repeatedly says c-Met-targeting; one unrelated
    combination-arm eligibility snippet mentions EGFR mutation status for
    the COMBINATION PARTNER drug, not RC108 itself. Majority (2 vs 1)
    should resolve to the real, repeated target."""
    target, source = target_attr.derive_target(
        "RC108", text_snippets=[
            "RC108 is a novel antibody-drug conjugate, with a c-Met-targeting antibody.",
            "RC108 for injection in subjects with c-Met positive advanced malignant solid tumors.",
            "RC108 in combination with Furmonertinib for treatment of EGFR mutation combined "
            "with MET-positive unresectable locally advanced or recurrent malignancy.",
        ],
    )
    assert (target, source) == ("MET", "trial_text_majority")


def test_trial_text_majority_does_not_resolve_real_tie():
    """A genuine bispecific/dual-target ADC (each target mentioned once,
    same sentence) must stay unresolved -- majority requires a real
    majority, not just "mentioned first.\""""
    target, source = target_attr.derive_target(
        "JSKN016HC", text_snippets=["a bispecific ADC targeting HER3 and TROP2"],
    )
    assert target is None
    assert source == "unresolved"
