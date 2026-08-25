"""Renders a list of Check results to a timestamped Markdown report."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pharma_stats.audit.types import Check
from pharma_stats.config import AUDIT_DIR

_LEVEL_ORDER = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3}
_LEVEL_EMOJI = {"FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO", "PASS": "PASS"}


def render(checks: list[Check], *, stages_run: list[str]) -> str:
    now = datetime.now(timezone.utc)
    counts = {lvl: 0 for lvl in _LEVEL_ORDER}
    for c in checks:
        counts[c.level] += 1

    lines = [
        f"# Pipeline audit — {now.isoformat()}",
        "",
        f"Stages run: {', '.join(stages_run)}",
        "",
        f"**{counts['FAIL']} FAIL / {counts['WARN']} WARN / {counts['INFO']} INFO / {counts['PASS']} PASS**",
        "",
    ]

    if counts["FAIL"]:
        lines.append("## FAIL — stop and look at these first")
        lines.append("")
        for c in sorted((c for c in checks if c.level == "FAIL"), key=lambda c: c.stage):
            lines += _render_check(c)

    by_stage: dict[str, list[Check]] = {}
    for c in checks:
        by_stage.setdefault(c.stage, []).append(c)

    for stage in stages_run:
        stage_checks = by_stage.get(stage, [])
        lines.append(f"## {stage}")
        lines.append("")
        if not stage_checks:
            lines.append("_no checks ran_")
            lines.append("")
            continue
        stage_checks.sort(key=lambda c: _LEVEL_ORDER[c.level])
        for c in stage_checks:
            lines += _render_check(c)

    return "\n".join(lines)


def _render_check(c: Check) -> list[str]:
    out = [f"- **[{_LEVEL_EMOJI[c.level]}]** {c.name}"]
    out.append(f"  - expected: {c.expected}")
    out.append(f"  - actual: {c.actual}")
    if c.detail:
        out.append(f"  - detail: {c.detail}")
    out.append("")
    return out


def write(checks: list[Check], *, stages_run: list[str], out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or AUDIT_DIR  # resolved at call time, not import time, so it stays testable
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = out_dir / f"{ts}.md"
    path.write_text(render(checks, stages_run=stages_run), encoding="utf-8")
    return path
