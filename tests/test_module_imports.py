"""Every module under pharma_stats must import cleanly.

Real incident (2026-09-04): a merge between two branches that both
touched audit/features.py dropped the `from pharma_stats.finance import
panel as money_panel` import while keeping a line that used
`money_panel.FEATURE_NAMES` — a module-level NameError that broke
`import pharma_stats.audit` entirely, so every audit stage failed
silently (no stage ran, no test caught it, because no existing test
imported the audit package as a whole). A plain "does it import"
sweep over every module would have caught this in CI before it ever
reached main. Parametrized per-module (not one bulk try/except) so a
broken module shows up as one specific failing test, not a single
opaque failure covering the whole package.
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "pharma_stats"


def _all_module_names() -> list[str]:
    names = []
    for p in sorted(SRC_ROOT.rglob("*.py")):
        rel = p.relative_to(SRC_ROOT.parent).with_suffix("")
        names.append(".".join(rel.parts))
    return names


@pytest.mark.parametrize("module_name", _all_module_names())
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)
