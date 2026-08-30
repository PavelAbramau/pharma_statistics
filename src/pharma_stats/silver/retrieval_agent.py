"""public_confirmation_date retrieval helper — the one piece of this
request that's higher priority than the silver auto-labeller and doesn't
touch gold or silver storage at all, so it can't contaminate anything: it
only returns candidate dates+URLs+snippets for a human to judge. Usable
directly from the labelling app's Gate 3 (pass 2) to cut down on the
tedious "find the press release" half of the work — never the judgement
half.

SEC EDGAR full-text search is real and implemented here — it's a public,
documented-enough, no-auth REST API
(https://efts.sec.gov/LATEST/search-index). Verified against current
third-party API guides before writing this client (not assumed from
training data), since the SEC has never published this as a versioned,
guaranteed contract — treat response field names as best-effort and code
defensively around missing ones.

Sponsor press archives and conference abstracts are NOT implemented here:
neither has one public API the way EDGAR does. Press archives are
per-sponsor websites (would need per-sponsor scraping rules, fragile and
high-maintenance); conference abstracts (ASCO/ESMO/AACR) mostly sit behind
search UIs with no public API either. Both need either a general web-
search API key or hand-built per-source scrapers — a decision to make
explicitly, not to guess into existence here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC's fair-access policy expects a User-Agent identifying the requester
# (org + contact email) on every request to sec.gov / efts.sec.gov. There
# is deliberately no default here — sending a fabricated identity on the
# user's behalf is worse than refusing to run.
_NO_DEFAULT_USER_AGENT = object()


@dataclass
class CandidateConfirmation:
    url: str
    entity_name: Optional[str]
    form_type: Optional[str]
    file_date: Optional[str]
    snippet: Optional[str]


def _filing_url(hit: dict) -> Optional[str]:
    """Best-effort reconstruction from the hit's _id, which SEC's own
    front end encodes as "{accession-with-dashes}:{filename}". Falls back
    to None if the shape doesn't match — a missing URL should never be
    silently replaced with a guessed one."""
    hit_id = hit.get("_id", "")
    accession, _, filename = hit_id.partition(":")
    accession = accession.replace("-", "")
    if not accession or not filename:
        return None
    cik = (hit.get("_source", {}).get("ciks") or [None])[0]
    if not cik:
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0') or '0'}/{accession}/{filename}"


def _snippet(hit: dict) -> Optional[str]:
    highlight = hit.get("highlight") or {}
    for values in highlight.values():
        if values:
            return values[0]
    return None


def search_sec_edgar_fulltext(
    query: str, *, user_agent: str = _NO_DEFAULT_USER_AGENT,  # type: ignore[assignment]
    forms: Optional[list[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
    max_results: int = 10,
) -> list[CandidateConfirmation]:
    """Search SEC full-text filings for `query` (e.g. an asset name plus a
    discontinuation-adjacent term — "sacituzumab tirumotecan discontinued").
    `user_agent` MUST be supplied as "OrgName contact@email" per SEC's fair
    -access policy — there is no default. Returns candidate confirmations
    for a human to judge; never asserts which one, if any, is the real
    confirmation date."""
    if user_agent is _NO_DEFAULT_USER_AGENT:
        raise ValueError(
            'search_sec_edgar_fulltext requires user_agent="YourOrg your-email@example.com" '
            "(SEC's fair-access policy) — there is no default"
        )

    params: dict = {"q": query, "from": 0, "size": max_results}
    if forms:
        params["forms"] = ",".join(forms)
    if start_date:
        params["startdt"] = start_date
    if end_date:
        params["enddt"] = end_date

    resp = requests.get(
        EFTS_SEARCH_URL, params=params, headers={"User-Agent": user_agent}, timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()

    hits = ((body.get("hits") or {}).get("hits")) or []
    out = []
    for hit in hits:
        source = hit.get("_source", {})
        out.append(CandidateConfirmation(
            url=_filing_url(hit) or "",
            entity_name=source.get("entity_name") or source.get("display_names"),
            form_type=source.get("form_type") or source.get("form"),
            file_date=source.get("file_date"),
            snippet=_snippet(hit),
        ))
    return out


def search_sponsor_press_archive(*args, **kwargs):
    raise NotImplementedError(
        "no public API — needs per-sponsor scraping rules or a general web-search API key, "
        "see this module's docstring"
    )


def search_conference_abstracts(*args, **kwargs):
    raise NotImplementedError(
        "no public API for ASCO/ESMO/AACR abstract search — needs a general web-search API key "
        "or per-conference scrapers, see this module's docstring"
    )
