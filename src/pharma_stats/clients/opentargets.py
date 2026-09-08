"""Thin, rate-limited client for the Open Targets Platform GraphQL API —
free, no authentication required. Used only by
attributes/target_external.py's residual target-antigen resolution (the
candidates B0's offline dictionaries/trial-text extraction can't resolve
— see that module's docstring).

Verified live 2026-09-07 against
https://api.platform.opentargets.org/api/v4/graphql:
  - `search(queryString, entityNames: ["drug"]) { hits { id name entity } }`
    returns ChEMBL-ID-keyed hits (id == a ChEMBL molecule id, e.g.
    "CHEMBL4298098"), fuzzy-ranked like ChEMBL's own search — same
    "don't trust rank 1 alone" caveat applies (see chembl_lookup).
  - `drug(chemblId) { id name mechanismsOfAction { rows { mechanismOfAction
    actionType targetName targets { approvedSymbol } } } }` returns one
    row per distinct mechanism. For an ADC this is typically 2+ rows: the
    antibody's own binding target (actionType "BINDING AGENT", exactly
    one target) and the cytotoxic payload's mechanism (actionType
    "INHIBITOR" etc., often MANY targets — e.g. tubulin isoforms for an
    auristatin/maytansinoid payload). approvedSymbol is HGNC's own
    approved gene symbol -- directly usable as this project's controlled
    target vocabulary, no extra mapping step.
Not read from memory/training data — read live before writing this, per
CLAUDE.md's "read the actual current API docs" rule.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_USER_AGENT = (
    "pharma-stats-adc-research-crawler/0.1 "
    "(non-commercial academic research; contact via CTGOV_CONTACT env var)"
)


class OpenTargetsError(RuntimeError):
    pass


class OpenTargetsClient:
    """No API key needed. Polite default rate limit (~3 req/s) — Open
    Targets publishes no hard limit for this endpoint, but it's a free
    public service, not ours to hammer (same policy as ChemblClient)."""

    def __init__(
        self, *, base_url: str = BASE_URL, min_interval: float = 0.34,
        timeout: float = 20.0, session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url
        self.min_interval = min_interval
        self.timeout = timeout
        self.session = session or requests.Session()

        contact = os.environ.get("CTGOV_CONTACT")
        ua = DEFAULT_USER_AGENT
        if contact:
            ua = f"{ua.rstrip(')')}; contact: {contact})"
        self.session.headers["User-Agent"] = ua
        self.session.headers["Content-Type"] = "application/json"
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _post(self, query: str, variables: dict) -> dict:
        self._throttle()
        try:
            resp = self.session.post(
                self.base_url, json={"query": query, "variables": variables}, timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise OpenTargetsError(f"Open Targets request failed: {e}") from e
        finally:
            self._last_request_at = time.monotonic()
        if resp.status_code != 200:
            raise OpenTargetsError(f"Open Targets returned {resp.status_code}: {resp.text[:200]!r}")
        try:
            data = resp.json()
        except ValueError as e:
            raise OpenTargetsError(f"Open Targets returned non-JSON: {e}") from e
        if data.get("errors"):
            raise OpenTargetsError(f"Open Targets GraphQL error: {data['errors']}")
        return data.get("data") or {}

    def search_drug(self, query: str) -> list[dict]:
        """[{id, name, entity}] — id is a ChEMBL molecule id. Fuzzy,
        relevance-ranked; caller MUST verify name/synonym match before
        trusting a hit (same discipline as ChemblClient.search_molecule)."""
        if not query or not query.strip():
            return []
        data = self._post(
            "query($q: String!) { search(queryString: $q, entityNames: [\"drug\"]) "
            "{ hits { id name entity } } }",
            {"q": query},
        )
        return ((data.get("search") or {}).get("hits")) or []

    def drug_mechanisms(self, chembl_id: str) -> list[dict]:
        """[{mechanismOfAction, actionType, targetName, targets: [{approvedSymbol}]}]
        for a drug already identified by ChEMBL id — empty list if Open
        Targets has no record for it (common for early-phase, obscure
        dev codes not yet indexed there)."""
        if not chembl_id:
            return []
        data = self._post(
            "query($id: String!) { drug(chemblId: $id) { mechanismsOfAction { rows { "
            "mechanismOfAction actionType targetName targets { approvedSymbol } } } } }",
            {"id": chembl_id},
        )
        drug = data.get("drug")
        if not drug:
            return []
        return ((drug.get("mechanismsOfAction") or {}).get("rows")) or []
