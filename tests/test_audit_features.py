"""Tests for the merged audit/features.py — two independently-evolved
check families (the program x month panel's knowability registry + real
as-of probe, and the money-layer panel's row/knowability/NaN checks) that
must both fire, sharing one leakage-register check over the union of
their feature names."""
from __future__ import annotations

from pharma_stats.audit import features
from pharma_stats.features.knowability import FeatureKnowability


def _fake_registry():
    return {
        "feat_a": FeatureKnowability("feat_a", True, "rule a", "src.a"),
        "feat_b": FeatureKnowability("feat_b", False, "rule b", "src.b"),
    }


def test_leakage_register_fails_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "LEAKAGE_REGISTER", tmp_path / "leakage.md")
    monkeypatch.setattr(features, "PANEL_REGISTRY", _fake_registry())
    monkeypatch.setattr(features.money_panel, "FEATURE_NAMES", ("feat_c",))
    checks = features._leakage_register_check()
    assert checks[0].level == "FAIL"
    assert "missing" in checks[0].actual


def test_leakage_register_fails_when_some_features_unregistered(tmp_path, monkeypatch):
    register = tmp_path / "leakage.md"
    register.write_text("## feat_a\nsome prose about feat_a\n", encoding="utf-8")
    monkeypatch.setattr(features, "LEAKAGE_REGISTER", register)
    monkeypatch.setattr(features, "PANEL_REGISTRY", _fake_registry())
    monkeypatch.setattr(features.money_panel, "FEATURE_NAMES", ("feat_c",))
    checks = features._leakage_register_check()
    assert checks[0].level == "FAIL"
    assert "feat_b" in checks[0].detail
    assert "feat_c" in checks[0].detail
    assert "feat_a" not in checks[0].detail


def test_leakage_register_passes_when_union_fully_registered(tmp_path, monkeypatch):
    register = tmp_path / "leakage.md"
    register.write_text("feat_a ... feat_b ... feat_c ...", encoding="utf-8")
    monkeypatch.setattr(features, "LEAKAGE_REGISTER", register)
    monkeypatch.setattr(features, "PANEL_REGISTRY", _fake_registry())
    monkeypatch.setattr(features.money_panel, "FEATURE_NAMES", ("feat_c",))
    checks = features._leakage_register_check()
    assert checks[0].level == "PASS"
    assert "3 registered" in checks[0].actual


def test_asof_probe_skips_with_info_when_warehouse_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "WAREHOUSE_DB", tmp_path / "nope.duckdb")
    monkeypatch.setattr(features, "PANEL_REGISTRY", _fake_registry())
    checks = features._panel_asof_probe_checks()
    assert checks[0].level == "PASS"  # registry non-empty
    assert checks[1].level == "INFO"
    assert "SKIPPED" in checks[1].actual


def test_panel_registry_check_fails_when_registry_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "WAREHOUSE_DB", tmp_path / "nope.duckdb")
    monkeypatch.setattr(features, "PANEL_REGISTRY", {})
    checks = features._panel_asof_probe_checks()
    assert checks[0].level == "FAIL"


def test_run_reports_info_not_crash_when_money_layer_panel_empty(monkeypatch):
    """Instruction: both WAREHOUSE_DB missing and financial_events empty
    must degrade to INFO with a clear 'run script X first', never crash."""
    monkeypatch.setattr(features, "WAREHOUSE_DB", features.REPO_ROOT / "definitely-does-not-exist.duckdb")
    monkeypatch.setattr(features.money_panel, "build_money_layer_panel", lambda: [])
    checks = features.run()
    money_checks = [c for c in checks if "money-layer feature panel (conviction" in c.name]
    assert len(money_checks) == 1
    assert money_checks[0].level == "INFO"
    assert "run scripts/build_financial_layer_cost_index.py first" in money_checks[0].detail


def test_money_layer_checks_flag_knowability_date_violations():
    panel = [
        {"program_id": "p1", "as_of": "2020-01-01", "knowability_date": "2020-01-01", "conviction_ratio": 1.0, "estimated_cumulative_spend": 5.0},
        {"program_id": "p2", "as_of": "2020-01-01", "knowability_date": "2020-02-01", "conviction_ratio": None, "estimated_cumulative_spend": None},
    ]
    monkeypatch_names = ("conviction_ratio", "estimated_cumulative_spend")
    import pharma_stats.finance.panel as real_money_panel
    assert real_money_panel.FEATURE_NAMES == monkeypatch_names
    checks = features._money_layer_checks(panel)
    violation_check = next(c for c in checks if "knowability_date is later than" in c.name)
    assert violation_check.level == "FAIL"
    assert "1 violations" in violation_check.actual
    assert "p2@2020-01-01" in violation_check.detail


def test_run_includes_still_not_built_info(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "LEAKAGE_REGISTER", tmp_path / "nope.md")
    monkeypatch.setattr(features, "WAREHOUSE_DB", tmp_path / "nope.duckdb")
    monkeypatch.setattr(features.money_panel, "build_money_layer_panel", lambda: [])
    checks = features.run()
    assert any(c.name == "still not built" for c in checks)
