"""Two-stage citation gate: resolve the cited source, then verify the
quoted snippet actually appears in it, verbatim. No citation backs a
silver answer without passing both stages — this is what keeps a
fabricated source from ever reaching a record (the "resolve, then verify
in-corpus" pattern this reuses)."""
from __future__ import annotations

import requests

from pharma_stats import snapshot as snap
from pharma_stats.silver.questions import Citation


class CitationError(ValueError):
    pass


def resolve(citation: Citation) -> str:
    """Stage 1: fetch the raw content the citation points at. Raises
    CitationError rather than returning anything for an unresolvable
    citation — a citation that can't be resolved is not verified, full
    stop, never treated as passing by default."""
    if citation.source_type == "raw_snapshot":
        source, _, sid = citation.locator.partition(":")
        if not source or not sid:
            raise CitationError(f"malformed raw_snapshot locator {citation.locator!r}, expected 'source:id'")
        s = snap.latest(source, sid)
        if s is None:
            raise CitationError(f"no snapshot found for {citation.locator!r}")
        return s.body
    if citation.source_type == "fetched_url":
        resp = requests.get(citation.locator, timeout=20)
        resp.raise_for_status()
        return resp.text
    raise CitationError(f"unknown source_type {citation.source_type!r}")


def verify(citation: Citation, resolved_content: str) -> bool:
    """Stage 2: the quoted snippet must appear in the resolved content
    (whitespace-normalised so trivial formatting differences don't cause a
    false failure) — the quote is never trusted on its own."""
    def normalize(s: str) -> str:
        return " ".join(s.split())
    return normalize(citation.quote) in normalize(resolved_content)


def resolve_and_verify(citation: Citation) -> bool:
    """Both stages in one call. Propagates CitationError from resolve() —
    callers should treat any exception here the same as "verification
    failed", not silently pass the citation through."""
    return verify(citation, resolve(citation))
