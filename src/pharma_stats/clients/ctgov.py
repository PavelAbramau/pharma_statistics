"""Client for the ClinicalTrials.gov v2 API.

Verified against the live API on 2026-08-19 (docs at
https://clinicaltrials.gov/data-api/api-reference are a JS-rendered SPA that
doesn't expose an OpenAPI document at a stable URL, so shapes below were
confirmed directly against https://clinicaltrials.gov/api/v2/*):

- Base URL: https://clinicaltrials.gov/api/v2
- GET /studies — search endpoint.
    - query.cond, query.term, query.intr, query.spons, query.lead, query.titles,
      query.outc, query.locn, query.id, query.patient — Essie expression-search
      fields (all optional, combine as AND).
    - filter.overallStatus, filter.geo, filter.ids, filter.advanced,
      filter.synonyms — post-search filters.
    - fields — comma-separated list of field/module names to project (reduces
      payload; e.g. "NCTId,BriefTitle,OverallStatus").
    - sort — "<field>:asc" or "<field>:desc", e.g. "LastUpdatePostDate:desc".
    - pageSize — default 10, max 1000.
    - pageToken — opaque cursor from the previous response's nextPageToken;
      absent on the first request, absent from the response once exhausted.
    - countTotal — bool; when true, adds totalCount to the response.
    - Response: {"studies": [...], "nextPageToken": "...", "totalCount": N}.
      Each study is {"protocolSection": {...}, "hasResults": bool, ...}.
- GET /studies/{nctId} — single-study retrieval. Same `fields` param.
      Response: {"protocolSection": {...}, ...} (not wrapped in "studies").
- GET /version — {"apiVersion": "...", "dataTimestamp": "..."}.
- GET /stats/size — corpus size stats.
- No API key. No documented hard rate limit and no rate-limit headers were
  observed on live responses. https://clinicaltrials.gov/robots.txt specifies
  `Crawl-delay: 1` for all crawlers, so this client throttles to >=1s between
  requests by default and backs off on 429/5xx.

Version history (investigated 2026-08-19): NOT part of the documented v2 REST
API — no /history path
and no version param exist under /api/v2/. The site's own "History of Changes"
tab is powered by a separate, undocumented internal API that the frontend calls:

- Base URL: https://clinicaltrials.gov/api/int  (note: /int, not /v2)
- GET /studies/{nctId}/history — list of every version for that study:
      {"changes": [{"version": 0, "date": "...", "status": "...",
                     "studyType": "...", "moduleLabels": [...],
                     "lastUpdateSubmitQcDate": "..."}, ...]}
  `moduleLabels` names which modules changed at that version (e.g. "Study
  Status", "Study Design", "Arms and Interventions", "Outcome Measures",
  "Eligibility", "Study Identification", "Oversight", "Study Description",
  "Conditions", "Contacts/Locations") — a free pre-filter for skipping
  full-record diffs on cosmetic-only versions (see CTGOV_HISTORY_SIGNAL_LABELS
  below).
- GET /studies/{nctId}/history/{version} — the full study record
  ({"studyVersion": N, "study": {"protocolSection": {...}, ...}}) exactly as
  it existed at that version. Same protocolSection shape as /api/v2.
- robots.txt explicitly disallows /api/ in general but carves out
  `Allow: /api/int/` with the comment "Allow access to the internal API to
  let SPA function properly" — so this is a sanctioned, if unpublished and
  unversioned, surface. Being undocumented, its shape could change without
  notice; unlike /api/v2 it carries no stability guarantee.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Iterator, Optional

import requests

from pharma_stats.history import schema_guard
from pharma_stats.snapshot import get_as_of, save_snapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://clinicaltrials.gov/api/v2"
INT_BASE_URL = "https://clinicaltrials.gov/api/int"
SOURCE = "ctgov"

# Module labels that can carry a substantive EvidenceEvent. Versions whose
# moduleLabels are a subset of the cosmetic set below (contact/address
# details, free-text description wording) can skip a full-record fetch —
# see the module docstring's "Version history" section.
HISTORY_SIGNAL_LABELS = {
    "Study Status", "Study Design", "Arms and Interventions",
    "Outcome Measures", "Eligibility", "Study Identification",
}
HISTORY_COSMETIC_LABELS = {"Contacts/Locations", "Study Description", "Conditions", "Oversight"}

DEFAULT_USER_AGENT = (
    "pharma-stats-adc-research-crawler/0.1 "
    "(non-commercial academic research; contact via CTGOV_CONTACT env var)"
)


class CtgovError(RuntimeError):
    pass


class CtgovClient:
    """Thin, cached, rate-limited client for CT.gov API v2.

    Every HTTP GET this client makes is persisted verbatim through the
    snapshot store (`pharma_stats.snapshot`), so repeated requests within
    the same day are served from disk rather than the network, and every
    fetch is provenance-tracked for the time-cut backtest.
    """

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        int_base_url: str = INT_BASE_URL,
        min_interval: float = 1.05,
        max_retries: int = 6,
        timeout: float = 30.0,
        user_agent: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.int_base_url = int_base_url.rstrip("/")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()

        contact = os.environ.get("CTGOV_CONTACT")
        ua = user_agent or DEFAULT_USER_AGENT
        if contact:
            ua = f"{ua.rstrip(')')}; contact: {contact})"
        self.session.headers["User-Agent"] = ua
        self.session.headers["Accept"] = "application/json"

        self._last_request_at = 0.0
        self._throttled_up = False  # one-way ratchet, see _note_429_and_slow_down

    # -- low-level HTTP -----------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _note_429_and_slow_down(self, *, floor: float = 2.0) -> None:
        """A 429 means the server thinks we're going too fast, even though
        we're within robots.txt's Crawl-delay. Permanently (for the rest of
        this client's life) raise the baseline throttle to at least `floor`
        seconds/request rather than retrying the one request and resuming
        the prior rate — a real crawl should extend its runtime instead of
        pushing through a rate-limit signal. One-way: never speeds back up
        automatically within a run."""
        if self.min_interval < floor:
            logger.warning(
                "ctgov returned 429 — raising baseline throttle from %.2fs to %.2fs/request "
                "for the rest of this run", self.min_interval, floor,
            )
            self.min_interval = floor
        self._throttled_up = True

    def _get(self, path: str, params: dict, *, base_url: Optional[str] = None) -> str:
        """GET path?params, with polite throttling and 429/5xx backoff.

        Returns the raw response body text (not parsed), so the snapshot
        store can persist it verbatim.
        """
        url = f"{base_url or self.base_url}{path}"
        clean_params = {k: v for k, v in params.items() if v is not None}

        attempt = 0
        while True:
            self._throttle()
            self._last_request_at = time.monotonic()
            try:
                resp = self.session.get(url, params=clean_params, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                # Transient network/DNS failures raise before a response exists at
                # all, so they need their own retry path — a 2026-08-20 crawl hit a
                # sustained DNS outage that this didn't catch, permanently failing
                # every trial attempted during the outage instead of riding it out.
                attempt += 1
                if attempt > self.max_retries:
                    raise CtgovError(
                        f"GET {url} failed after {attempt} attempts (connection error): {e}"
                    ) from e
                delay = min(2.0 ** attempt, 60.0)
                logger.warning(
                    "ctgov connection error on %s, retrying in %.1fs (attempt %d/%d): %s",
                    url, delay, attempt, self.max_retries, e,
                )
                time.sleep(delay)
                continue

            if resp.status_code == 200:
                return resp.text

            if resp.status_code == 429 or resp.status_code >= 500:
                if resp.status_code == 429:
                    self._note_429_and_slow_down()
                attempt += 1
                if attempt > self.max_retries:
                    raise CtgovError(
                        f"GET {resp.url} failed after {attempt} attempts: "
                        f"{resp.status_code} {resp.text[:500]}"
                    )
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = 2.0 ** attempt
                else:
                    delay = min(2.0 ** attempt, 60.0)
                logger.warning(
                    "ctgov %s on %s, retrying in %.1fs (attempt %d/%d)",
                    resp.status_code, resp.url, delay, attempt, self.max_retries,
                )
                time.sleep(delay)
                continue

            raise CtgovError(f"GET {resp.url} failed: {resp.status_code} {resp.text[:500]}")

    def _cache_key(self, prefix: str, params: dict) -> str:
        canonical = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return f"{prefix}:{digest}"

    def _fetch_cached(
        self, path: str, params: dict, cache_id: str, *, base_url: Optional[str] = None
    ) -> str:
        """Cache-first GET: reuse today's snapshot if one already exists,
        otherwise hit the network and persist the result."""
        today = date.today()
        cached = get_as_of(SOURCE, cache_id, today)
        if cached is not None and cached.snapshot_date == today:
            return cached.body

        full_url = requests.Request("GET", f"{base_url or self.base_url}{path}", params={
            k: v for k, v in params.items() if v is not None
        }).prepare().url

        body = self._get(path, params, base_url=base_url)
        save_snapshot(SOURCE, cache_id, full_url, body, fetched_at=datetime.now(timezone.utc))
        return body

    # -- public API -----------------------------------------------------

    def search_studies(
        self,
        *,
        cond: Optional[str] = None,
        term: Optional[str] = None,
        intr: Optional[str] = None,
        spons: Optional[str] = None,
        titles: Optional[str] = None,
        overall_status: Optional[str] = None,
        fields: Optional[list[str]] = None,
        sort: Optional[str] = None,
        page_size: int = 200,
        max_studies: Optional[int] = None,
    ) -> Iterator[dict]:
        """Paginated study search. Yields one parsed study dict per result,
        walking `nextPageToken` until exhausted (or `max_studies` is hit).

        `cond`/`term`/`intr`/`spons`/`titles` map to the API's Essie
        expression-search fields (query.cond, query.term, query.intr,
        query.spons, query.titles); `overall_status` maps to
        filter.overallStatus.
        """
        base_params = {
            "query.cond": cond,
            "query.term": term,
            "query.intr": intr,
            "query.spons": spons,
            "query.titles": titles,
            "filter.overallStatus": overall_status,
            "fields": ",".join(fields) if fields else None,
            "sort": sort,
            "pageSize": page_size,
        }

        page_token = None
        page_num = 0
        yielded = 0
        while True:
            params = dict(base_params)
            if page_token:
                params["pageToken"] = page_token

            cache_id = self._cache_key(f"search:p{page_num}", params)
            body = self._fetch_cached("/studies", params, cache_id)
            payload = json.loads(body)

            for study in payload.get("studies", []):
                yield study
                yielded += 1
                if max_studies is not None and yielded >= max_studies:
                    return

            page_token = payload.get("nextPageToken")
            page_num += 1
            if not page_token:
                return

    def get_study(self, nct_id: str, *, fields: Optional[list[str]] = None) -> dict:
        """Single-study retrieval by NCT id.

        When `fields` is omitted, the full study record is fetched and
        cached under the bare NCT id — this is the canonical snapshot that
        `pharma_stats.snapshot.get_as_of("ctgov", nct_id, ...)` resolves
        for the time-cut backtest. Field-limited fetches are cached under
        a compound key instead, so they never collide with (or shadow)
        the canonical full record for the same id/day.
        """
        params = {"fields": ",".join(fields) if fields else None}
        if fields:
            cache_id = self._cache_key(f"study:{nct_id}", params)
        else:
            cache_id = nct_id

        body = self._fetch_cached(f"/studies/{nct_id}", params, cache_id)
        return json.loads(body)

    def version(self) -> dict:
        body = self._get("/version", {})
        return json.loads(body)

    # -- version history (undocumented internal API; see module docstring) --

    def get_history(self, nct_id: str) -> list[dict]:
        """List every version of a study: [{version, date, status,
        studyType, moduleLabels, lastUpdateSubmitQcDate}, ...], oldest first.

        Cached under a compound key (never collides with the canonical
        full-record snapshot at the bare nct_id).
        """
        cache_id = self._cache_key(f"history:{nct_id}", {})
        body = self._fetch_cached(
            f"/studies/{nct_id}/history", {}, cache_id, base_url=self.int_base_url
        )
        parsed = json.loads(body)
        schema_guard.check_history_list(parsed, nct_id=nct_id)
        return parsed["changes"]

    def get_study_version(self, nct_id: str, version: int) -> dict:
        """The full study record exactly as it existed at `version`
        (from get_history's per-entry "version" field). Returns just the
        study dict (protocolSection, ...), unwrapped from {"studyVersion",
        "study"}.

        Cached under a compound key that includes the version number, so
        every historical version is its own immutable snapshot — this is
        what get_as_of resolves for the time-cut backtest.
        """
        cache_id = f"{nct_id}:v{version}"
        body = self._fetch_cached(
            f"/studies/{nct_id}/history/{version}", {}, cache_id, base_url=self.int_base_url
        )
        parsed = json.loads(body)
        schema_guard.check_history_version(parsed, nct_id=nct_id, version=version)
        return parsed["study"]
