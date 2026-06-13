"""Fetch readable article body text from a URL - the content GDELT can't give us.

GDELT's DOC API returns titles + URLs but no body text, and the GKG/BigQuery
table has no headline at all. A forecasting model with no web access can't open a
URL, so we must fetch and extract the article text ourselves and put it IN the
prompt. That is the whole point of this module.

Ladder, cheapest-first (no paid APIs - see the project's cost constraint):
  1. plain HTTP fetch + trafilatura main-text extraction (handles most sites)
  2. Wayback Machine fallback for dead/moved/geo-blocked URLs (also free)
  (Selenium for JS-rendered stragglers is a deliberate FUTURE step - it needs a
  browser binary not present in this environment, so we degrade gracefully without
  it rather than block the whole pipeline.)

Everything here is BEST-EFFORT and FAIL-SAFE: any failure returns None so the
caller keeps the article's title (still useful signal) instead of crashing a
148-market snapshot build. Link-rot on 6–12-month-old foreign URLs is expected;
the Wayback fallback recovers a chunk of it.

No temporal leakage: fetching today the HTML of an article published before a
market's cutoff introduces no look-ahead - the content was fixed at publish time.
The CALLER is responsible for only passing URLs whose seendate < as_of.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-fetch network timeout. Short: a slow publisher must not stall the build.
_FETCH_TIMEOUT_S = 12
# Cap extracted text we keep per article. The genome's max_context_chars governs
# the final prompt budget; this is a per-article guard so one long article can't
# swallow the whole context. ~1200 chars ≈ 200 words ≈ the lede + key paras.
_MAX_TEXT_CHARS = 1200
_WAYBACK_AVAILABLE = "https://archive.org/wayback/available"


def fetch_article_text(url: str, *, wayback: bool = True) -> str | None:
    """Return cleaned main article text for `url`, or None if unavailable.

    Never raises. Tries a direct fetch first, then (optionally) the most recent
    Wayback snapshot. Returns None on any failure so the caller can fall back to
    the title alone.
    """
    if not url:
        return None
    text = _extract(url)
    if text:
        return text[:_MAX_TEXT_CHARS]
    if wayback:
        archived = _wayback_url(url)
        if archived:
            text = _extract(archived)
            if text:
                return text[:_MAX_TEXT_CHARS]
    return None


def _extract(url: str) -> str | None:
    """fetch_url + extract via trafilatura. None on any failure (incl. import)."""
    try:
        import trafilatura  # noqa: PLC0415 - lazy so the dep is optional at import

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if not text:
            return None
        # Collapse whitespace; drop trivially short extractions (nav/boilerplate
        # that slipped through - not real article content).
        cleaned = " ".join(text.split())
        return cleaned if len(cleaned) >= 200 else None
    except Exception as exc:  # noqa: BLE001 - best-effort; never break the build
        logger.debug("article extract failed for %s: %r", url, exc)
        return None


def _wayback_url(url: str) -> str | None:
    """Resolve the closest Wayback snapshot URL for `url`, or None."""
    try:
        import httpx  # noqa: PLC0415

        resp = httpx.get(_WAYBACK_AVAILABLE, params={"url": url}, timeout=_FETCH_TIMEOUT_S)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        snap = data.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available") and snap.get("url"):
            return str(snap["url"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("wayback lookup failed for %s: %r", url, exc)
    return None
