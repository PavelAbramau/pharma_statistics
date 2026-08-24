"""Shared paths for the repo. All paths are relative to the repo root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = REPO_ROOT / "raw"
DATA_DIR = REPO_ROOT / "data"
GOLD_DIR = REPO_ROOT / "gold"
REPORTS_DIR = REPO_ROOT / "reports"

MANIFEST_DB = DATA_DIR / "manifest.duckdb"
WAREHOUSE_DB = DATA_DIR / "warehouse.duckdb"

for _d in (RAW_DIR, DATA_DIR, GOLD_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
