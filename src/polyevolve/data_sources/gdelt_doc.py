"""GDELT DOC 2.0 news source WITH article body text - the content-bearing source.

This replaces the gdelt_bq (BigQuery/GKG) path that caused the failure we
diagnosed: GKG is URL-keyed structured metadata (themes/tone/entities, NO title,
NO body), and we matched on country co-occurrence sorted by recency - so the model
received 32 recency-ranked URLs about a *country* (concerts, sports, celebrities)
with zero readable content. 100% of contexts were noise it couldn't even read.

DOC 2.0 fixes BOTH problems and is free (plain HTTP, no BigQuery, no key):
  - RELEVANCE: it's a full-text search engine; sort=hybridrel ranks by textual
    relevance to the QUESTION (not the country), so we get on-topic articles.
  - TITLES: artlist mode returns real headlines, not bare URLs.
  - CONTENT: we then scrape the body text of the top-K articles (scraping.py) and
    put it in the prompt, so the model actually reads what happened.
  - HISTORY: DOC 2.0 now serves ~1 year (not the old ~3 months), enough for our
    markets (148/150 reachable at a 30-day lead).

Point-in-time discipline (no leakage): server-side enddatetime is approximate at
the boundary, so we ALSO hard-filter client-side on seendate < as_of, exactly as
the legacy GdeltSource does. Live runs pass as_of=None.

Rate limit: GDELT throttles by IP at ~1 req / 5s (confirmed live: a 429 plain-text
body). We self-throttle at 6s with retry/backoff, shared across instances.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from .gdelt import (
    GDELT_BASE,
    GDELT_DT_FMT,
    _before,
    _question_to_query,
)
from .processing import dedup_articles, rank_articles
from .scraping import fetch_article_text

logger = logging.getLogger(__name__)

# DOC 2.0 now serves roughly the trailing year. Keep a guard so an out-of-range
# cutoff yields a LOUD marker, never a misleading empty result (the 0/150 lesson).
_DOC_WINDOW_DAYS = 365
# GDELT throttles by IP at ~1 req/5s; we space at 8s for headroom. If the IP gets
# into a penalty box (sustained abuse), a single 429 must trigger a LONG cooldown
# so the box clears mid-run rather than burning all retries against a hard wall.
_MIN_INTERVAL_S = 8.0
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 8.0
_RATELIMIT_BACKOFF_S = 45.0


class GdeltDocSource:
    """DOC 2.0 relevance search + body-text scraping. Source key: 'news'."""

    name = "news"

    # Shared IP throttle (GDELT limits by IP, so all instances coordinate).
    _rate_lock = threading.Lock()
    _last_call = 0.0

    def __init__(
        self,
        http: httpx.Client | None = None,
        max_records: int = 6,
        scrape_top_k: int = 4,
        min_interval_s: float = _MIN_INTERVAL_S,
        scrape: bool = True,
    ) -> None:
        self._http = http or httpx.Client(timeout=30.0)
        self._max_records = max_records
        # Only the top-K ranked articles get their body scraped (network cost +
        # publisher politeness). The rest contribute title-only signal.
        self._scrape_top_k = scrape_top_k
        self._min_interval_s = min_interval_s
        self._scrape = scrape

    def _throttle(self) -> None:
        with GdeltDocSource._rate_lock:
            wait = self._min_interval_s - (time.monotonic() - GdeltDocSource._last_call)
            if wait > 0:
                time.sleep(wait)
            GdeltDocSource._last_call = time.monotonic()

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"articles": [], "query": "", "error": "empty_question"}

        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")

        query = _question_to_query(question)
        if not query:
            return {"articles": [], "query": "", "error": "empty_query"}

        # Reachability guard: a cutoff older than the DOC window can't be served -
        # say so loudly rather than freeze a misleading empty context.
        if as_of is not None:
            now = datetime.now(UTC)
            as_of_cmp = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
            age_days = (now - as_of_cmp).days
            if age_days > _DOC_WINDOW_DAYS:
                return {
                    "articles": [],
                    "query": query,
                    "error": "out_of_window",
                    "detail": (
                        f"as_of {as_of_cmp.date()} is {age_days}d old; DOC 2.0 "
                        f"serves ~{_DOC_WINDOW_DAYS}d."
                    ),
                }

        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "maxrecords": self._max_records * 4,
            "format": "json",
            "sort": "hybridrel",  # relevance to the question, not recency
        }
        if as_of is not None:
            # Use the UTC-normalized cutoff for BOTH bounds (consistent with the
            # authoritative client-side seendate<as_of filter below).
            params["enddatetime"] = as_of_cmp.strftime(GDELT_DT_FMT)
            # A window helps the server scope; start one year before the cutoff.
            start = as_of_cmp.replace(year=as_of_cmp.year - 1)
            params["startdatetime"] = start.strftime(GDELT_DT_FMT)

        data, error = self._get_with_retry(params)
        if error is not None:
            return {"articles": [], "query": query, "error": error}

        raw = data.get("articles", [])
        if as_of is not None:
            raw = [a for a in raw if _before(a.get("seendate", ""), as_of)]

        articles = [
            {
                "url": a.get("url", ""),
                "title": a.get("title", ""),
                "domain": a.get("domain", ""),
                "language": a.get("language", ""),
                "seendate": a.get("seendate", ""),
                "sourcecountry": a.get("sourcecountry", ""),
            }
            for a in raw
        ]
        # Collapse syndicated reprints (fake corroboration) before capping, then
        # rank for source diversity. Keep enough to scrape the top-K.
        articles = rank_articles(dedup_articles(articles), max_n=self._max_records)

        # Scrape body text for the top-K (best-effort; title survives on failure).
        if self._scrape:
            for a in articles[: self._scrape_top_k]:
                body = fetch_article_text(a.get("url", ""))
                if body:
                    a["text"] = body

        return {
            "articles": articles,
            "query": query,
            "as_of": as_of.isoformat() if as_of else None,
        }

    def _get_with_retry(self, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        last_error = "http_error"
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            try:
                resp = self._http.get(GDELT_BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # 429 (or 503) => IP rate-limited / penalty box. Back off LONG so
                # the box clears, rather than spending retries against a hard wall.
                if exc.response.status_code in (429, 503):
                    last_error = "rate_limited"
                    time.sleep(_RATELIMIT_BACKOFF_S)
                else:
                    last_error = "http_error"
                    self._backoff(attempt)
                continue
            except httpx.HTTPError:
                last_error = "http_error"
                self._backoff(attempt)
                continue
            # GDELT also signals rate-limit / bad-query as HTTP 200 plain text.
            if "json" not in resp.headers.get("content-type", "").lower():
                body = resp.text.lower()
                if "limit" in body or "one every" in body:
                    last_error = "rate_limited"
                    time.sleep(_RATELIMIT_BACKOFF_S)
                else:
                    last_error = "gdelt_rejected"
                    self._backoff(attempt)
                continue
            try:
                return resp.json(), None
            except ValueError:
                last_error = "bad_json"
                self._backoff(attempt)
                continue
        return {}, last_error

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(_BACKOFF_BASE_S * (attempt + 1))

    def render(self, payload: dict[str, Any]) -> str:
        """Three distinct states: error / no-coverage / found (with body text)."""
        query = payload.get("query", "")
        as_of = payload.get("as_of")
        suffix = f", as of {as_of}" if as_of else ""

        error = payload.get("error")
        if error:
            detail = payload.get("detail", "")
            tail = f": {detail}" if detail else ""
            return f"[SOURCE ERROR] news fetch failed ({error}{tail})"

        articles = payload.get("articles", [])
        if not articles:
            return f"(News searched but found no coverage for query: {query}{suffix})"

        lines = [f"Relevant news (GDELT DOC 2.0, query={query}{suffix}):"]
        for a in articles:
            dup = a.get("dup_count", 1)
            carried = f" [carried by ~{dup} outlets]" if dup > 1 else ""
            lines.append(
                f"\n• [{a['language']}] {a['title']} ({a['domain']}, {a['seendate']}){carried}"
            )
            body = a.get("text")
            if body:
                lines.append(f"  {body}")
        return "\n".join(lines)
