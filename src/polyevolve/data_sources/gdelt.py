"""GDELT Doc API data source - free, no key, multilingual global news.

GDELT indexes news from around the world in 100+ languages and machine-
translates to English. This directly serves the v0 thesis: a research agent
that surfaces foreign-language coverage US-centric bots don't read.

Doc API: https://api.gdeltproject.org/api/v2/doc/doc

Point-in-time (`as_of`) support: for honest backtesting the agent must only see
news that existed before a market resolved. GDELT's server-side enddatetime
filter is approximate at the boundary (observed leaking ~1 day past), so we ALSO
hard-filter client-side on each article's seendate < as_of. Live runs pass
as_of=None (latest news).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from .processing import dedup_articles, rank_articles

logger = logging.getLogger(__name__)

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DT_FMT = "%Y%m%d%H%M%S"
# Article seendate format, e.g. "20260530T203000Z".
_SEENDATE_FMT = "%Y%m%dT%H%M%SZ"
# GDELT throttles by IP; over-limit returns a plain-text 200 body. The exact
# numeric limit is undocumented (the oft-quoted "1 req/5s" is folklore), so we
# self-throttle conservatively.
_MIN_INTERVAL_S = 6.0
# Transient failures (timeouts, 5xx, sporadic rate-limit text bodies) are common
# - a single market that flaked must be RETRIED, not silently emptied (that
# silent path froze 0/150 empty contexts into fp_v1).
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 6.0
# The Doc API's enddatetime filter only serves the trailing ~3 months. A cutoff
# older than this CANNOT be answered by this source - we must say so explicitly
# (a loud out-of-range marker), not return a misleading empty result. Historical
# backtests beyond this window need the BigQuery source instead.
_DOC_API_WINDOW_DAYS = 90

# Words to strip when turning a market question into a news query.
_STOPWORDS = frozenset(
    {
        "will",
        "the",
        "a",
        "an",
        "be",
        "by",
        "in",
        "on",
        "of",
        "to",
        "for",
        "is",
        "are",
        "win",
        "before",
        "after",
        "next",
        "out",
        "this",
        "that",
        "and",
        "or",
        "at",
        "as",
    }
)


def _before(seendate: str, as_of: datetime) -> bool:
    """True if the article's seendate is strictly before as_of.

    Conservative: an unparseable seendate is dropped (treated as not-before), so
    we never leak an article we can't date-verify into a point-in-time view.
    """
    if not seendate:
        return False
    try:
        seen = datetime.strptime(seendate, _SEENDATE_FMT)
    except ValueError:
        return False
    # as_of may be tz-aware; seendate is UTC-naive. Compare on naive UTC.
    cutoff = as_of.replace(tzinfo=None) if as_of.tzinfo else as_of
    return seen < cutoff


def _question_to_query(question: str, max_terms: int = 4) -> str:
    """Turn a market question into a GDELT query.

    GDELT ANDs space-separated terms, so MORE terms = FEWER (often zero) results.
    The old builder joined up to 6 content words incl. generic ones ("sworn",
    "December"), over-constraining into empty results. We now (1) prefer
    proper-noun-ish tokens (capitalized / multi-cap), which carry the entity
    signal, falling back to other content words only to fill, and (2) cap to a
    small number of terms. Proper nouns are the discriminating signal; generic
    verbs/months mostly just shrink the result set.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", question)
    proper: list[str] = []
    other: list[str] = []
    for w in words:
        if w.lower() in _STOPWORDS:
            continue
        # Capitalized mid-sentence ≈ proper noun (entity), the high-signal term.
        if w[0].isupper():
            proper.append(w)
        else:
            other.append(w)
    # Entities first, then backfill with content words up to the cap.
    terms = proper[:max_terms]
    for w in other:
        if len(terms) >= max_terms:
            break
        terms.append(w)
    return " ".join(terms)


class GdeltSource:
    name = "gdelt_news"

    # Class-level throttle: GDELT rate-limits by IP, so all instances share it.
    _rate_lock = threading.Lock()
    _last_call = 0.0

    def __init__(
        self,
        http: httpx.Client | None = None,
        max_records: int = 8,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self._http = http or httpx.Client(timeout=30.0)
        self._max_records = max_records
        self._min_interval_s = min_interval_s

    def _throttle(self) -> None:
        """Block until at least min_interval_s has passed since the last call."""
        with GdeltSource._rate_lock:
            wait = self._min_interval_s - (time.monotonic() - GdeltSource._last_call)
            if wait > 0:
                time.sleep(wait)
            GdeltSource._last_call = time.monotonic()

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"articles": [], "query": "", "error": "empty_question"}

        # Point-in-time cutoff for backtesting; None => latest news.
        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")

        # 90-day reachability guard: the Doc API's enddatetime only serves the
        # trailing ~3 months. A cutoff older than that returns nothing - but that
        # "nothing" means "this source CAN'T answer", not "no news existed". Say
        # so loudly so it's never frozen as a misleading empty result.
        if as_of is not None:
            now = datetime.now(UTC)
            as_of_cmp = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
            age_days = (now - as_of_cmp).days
            if age_days > _DOC_API_WINDOW_DAYS:
                return {
                    "articles": [],
                    "query": _question_to_query(question),
                    "error": "out_of_window",
                    "detail": (
                        f"as_of {as_of_cmp.date()} is {age_days}d old; Doc API "
                        f"only serves ~{_DOC_API_WINDOW_DAYS}d. Use BigQuery source."
                    ),
                }

        query = _question_to_query(question)
        if not query:
            return {"articles": [], "query": "", "error": "empty_query"}

        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "maxrecords": self._max_records,
            "format": "json",
            "sort": "datedesc",
        }
        if as_of is not None:
            # Server-side window is approximate at the boundary - we re-filter
            # client-side below. Over-fetch a bit so the strict filter + dedup
            # still leave enough articles.
            params["enddatetime"] = as_of.strftime(GDELT_DT_FMT)
            params["maxrecords"] = self._max_records * 4

        data, error = self._get_with_retry(params, query)
        if error is not None:
            return {"articles": [], "query": query, "error": error}

        raw = data.get("articles", [])
        if as_of is not None:
            raw = [a for a in raw if _before(a.get("seendate", ""), as_of)]

        articles = [
            {
                "title": a.get("title", ""),
                "domain": a.get("domain", ""),
                "language": a.get("language", ""),
                "seendate": a.get("seendate", ""),
                "sourcecountry": a.get("sourcecountry", ""),
            }
            for a in raw
        ]
        # Collapse syndicated wire-copies BEFORE capping, so the cap keeps
        # distinct stories rather than 8 reprints of one (fake corroboration),
        # then rank for source diversity.
        articles = rank_articles(dedup_articles(articles), max_n=self._max_records)
        return {
            "articles": articles,
            "query": query,
            "as_of": as_of.isoformat() if as_of else None,
        }

    def _get_with_retry(
        self, params: dict[str, Any], query: str
    ) -> tuple[dict[str, Any], str | None]:
        """GET with throttle + retry/backoff. Returns (json_data, error_code).

        Transient failures (network error, non-JSON rate-limit body) are retried
        with linear backoff. A persistent failure returns ({}, error_code) so the
        caller can record it LOUDLY - never a silent empty success.
        """
        last_error = "http_error"
        for attempt in range(_MAX_RETRIES):
            self._throttle()
            try:
                resp = self._http.get(GDELT_BASE, params=params)
                resp.raise_for_status()
            except httpx.HTTPError:
                last_error = "http_error"
                self._backoff(attempt)
                continue

            # GDELT signals errors as HTTP 200 with a plain-text body (rate-limit
            # notice, "query too short/long"). Non-JSON => treat as transient and
            # retry; the rate-limit case usually clears after a longer wait.
            if "json" not in resp.headers.get("content-type", "").lower():
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
        """Human/LLM-readable rendering for inclusion in the prompt.

        Three DISTINCT states, never conflated (the 0/150 bug was conflating the
        last two): articles found / a fetch error occurred / fetched cleanly but
        genuinely no coverage.
        """
        query = payload.get("query", "")
        as_of = payload.get("as_of")
        suffix = f", as of {as_of}" if as_of else ""

        error = payload.get("error")
        if error:
            detail = payload.get("detail", "")
            tail = f": {detail}" if detail else ""
            return f"[SOURCE ERROR] gdelt_news fetch failed ({error}{tail})"

        articles = payload.get("articles", [])
        if not articles:
            return f"(GDELT searched but found no coverage for query: {query}{suffix})"
        lines = [f"Recent news (GDELT, query={query}{suffix}):"]
        for a in articles:
            dup = a.get("dup_count", 1)
            carried = f" [carried by ~{dup} outlets]" if dup > 1 else ""
            lines.append(
                f"- [{a['language']}] {a['title']} ({a['domain']}, {a['seendate']}){carried}"
            )
        return "\n".join(lines)
