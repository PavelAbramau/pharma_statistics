"""Thin, rate-limited client for the ChEMBL REST API — free, no
authentication required. Used only by triage/layer1_5.py's molecule_type
lookup.

Verified live 2026-09-01 against
https://www.ebi.ac.uk/chembl/api/data/molecule/search?q=<query>&format=json:
response envelope is {"molecules": [...], "page_meta": {...}}; each
molecule has molecule_type (observed values: "Antibody drug conjugate",
"Antibody", "Small molecule"), pref_name, molecule_chembl_id, and
molecule_synonyms (list of {"molecule_synonym", "syn_type", "synonyms"}).
Not read from memory/training data — read live before writing this, per
CLAUDE.md's "read the actual current API docs" rule.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_USER_AGENT = (
    "pharma-stats-adc-research-crawler/0.1 "
    "(non-commercial academic research; contact via CTGOV_CONTACT env var)"
)


class ChemblError(RuntimeError):
    pass


class ChemblClient:
    """No API key needed. Polite default rate limit (~3 req/s) — ChEMBL
    publishes no hard limit for this endpoint, but it's a free public
    service, not ours to hammer."""

    def __init__(
        self, *, base_url: str = BASE_URL, min_interval: float = 0.34,
        timeout: float = 15.0, session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = session or requests.Session()

        contact = os.environ.get("CTGOV_CONTACT")
        ua = DEFAULT_USER_AGENT
        if contact:
            ua = f"{ua.rstrip(')')}; contact: {contact})"
        self.session.headers["User-Agent"] = ua
        self.session.headers["Accept"] = "application/json"
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def search_molecule(self, query: str) -> list[dict]:
        """Raw `molecules` list for a free-text query — ChEMBL's search is
        fuzzy and may return several candidates or none. The caller MUST
        verify an exact name/synonym match before trusting molecule_type;
        this method does no such verification itself (see
        triage/layer1_5.py.chembl_lookup)."""
        if not query or not query.strip():
            return []
        self._throttle()
        try:
            resp = self.session.get(
                f"{self.base_url}/molecule/search",
                params={"q": query, "format": "json"}, timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ChemblError(f"ChEMBL request failed for {query!r}: {e}") from e
        finally:
            self._last_request_at = time.monotonic()
        if resp.status_code != 200:
            raise ChemblError(f"ChEMBL returned {resp.status_code} for {query!r}: {resp.text[:200]!r}")
        try:
            data = resp.json()
        except ValueError as e:
            raise ChemblError(f"ChEMBL returned non-JSON for {query!r}: {e}") from e
        return data.get("molecules") or []
