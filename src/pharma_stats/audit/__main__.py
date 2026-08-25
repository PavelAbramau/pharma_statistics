"""python -m pharma_stats.audit --stage all|<name>

Runs the requested audit stage(s), writes a timestamped Markdown report
to audit/, prints a summary to stdout, and exits non-zero on any FAIL.
"""
from __future__ import annotations

import argparse
import sys

from pharma_stats.audit import GATING_STAGES, STAGE_ORDER, STAGE_REGISTRY, report
from pharma_stats.audit.types import Check


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pharma_stats.audit")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", *STAGE_REGISTRY.keys()],
        help="which audit stage to run (default: all)",
    )
    args = parser.parse_args(argv)

    run_all = args.stage == "all"
    stages_to_run = STAGE_ORDER if run_all else [args.stage]

    checks: list[Check] = []
    stages_run: list[str] = []
    for stage in stages_to_run:
        print(f"[{stage}] running...")
        stage_checks = STAGE_REGISTRY[stage]()
        checks.extend(stage_checks)
        stages_run.append(stage)
        for c in stage_checks:
            print(f"  [{c.level:4s}] {c.name} — expected {c.expected}; got {c.actual}")

        if run_all and stage in GATING_STAGES and any(c.level == "FAIL" for c in stage_checks):
            skipped = STAGE_ORDER[STAGE_ORDER.index(stage) + 1:]
            print(f"\n'{stage}' FAILed a gating check — halting before: {', '.join(skipped)}")
            break

    path = report.write(checks, stages_run=stages_run)
    n_fail = sum(1 for c in checks if c.level == "FAIL")
    n_warn = sum(1 for c in checks if c.level == "WARN")
    print()
    print(f"Report written to {path}")
    print(f"{n_fail} FAIL / {n_warn} WARN / "
          f"{sum(1 for c in checks if c.level == 'INFO')} INFO / "
          f"{sum(1 for c in checks if c.level == 'PASS')} PASS")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
