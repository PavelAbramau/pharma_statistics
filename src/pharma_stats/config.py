"""Shared paths for the repo. All paths are relative to the repo root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "raw"
DATA_DIR = REPO_ROOT / "data"
GOLD_DIR = REPO_ROOT / "gold"
# Sibling of GOLD_DIR, never a subdirectory of it — the silver (auto-
# labelled) track must stay structurally separate from the gold
# evaluation set. See silver/store.py.
SILVER_DIR = REPO_ROOT / "silver"
# Sibling of GOLD_DIR/SILVER_DIR — Gate 1/2 auto-triage decisions are
# staged here (pharma_stats.triage.staging), never written directly to
# gold/labels.jsonl. See src/pharma_stats/triage/ for the code.
TRIAGE_DIR = REPO_ROOT / "triage"
REPORTS_DIR = REPO_ROOT / "reports"
AUDIT_DIR = REPO_ROOT / "audit"

MANIFEST_DB = DATA_DIR / "manifest.duckdb"
WAREHOUSE_DB = DATA_DIR / "warehouse.duckdb"

for _d in (RAW_DIR, DATA_DIR, GOLD_DIR, SILVER_DIR, TRIAGE_DIR, REPORTS_DIR, AUDIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
