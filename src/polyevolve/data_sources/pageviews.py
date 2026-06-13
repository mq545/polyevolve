"""Wikipedia pageviews momentum - a point-in-time attention signal for elections.

Why this exists: in foreign-politics election markets, public ATTENTION to a
candidate/party tends to move before (or with) the vote, and Wikipedia pageviews
are the cleanest free proxy for it. They are:
  - $0 (free Wikimedia REST API, no key) and unthrottled in practice;
  - LOW leakage: the daily-pageviews history is IMMUTABLE - the count for a past
    day never changes - so a strictly-before-as_of window is genuinely
    point-in-time (unlike news search, whose relevance ranking drifts);
  - LOCAL-LANGUAGE aware: we query the country's OWN-language Wikipedia (Italy→it,
    Japan→ja, ...) where local readers actually look, plus English as a fallback.

What it does, per market (question + as_of + tags):
  1. Extract candidate/party/entity names from the question (reuse the news
     source's `_match_terms` - multi-word proper nouns like "Andrea Martella").
  2. Map the market's country (from tags) to its local Wikipedia language, and
     always also query en.wikipedia.
  3. For each (entity, lang) fetch DAILY pageviews for the 60 days ending the day
     BEFORE as_of. POINT-IN-TIME IS THE WHOLE POINT: we set the API end date to
     as_of-1d and additionally hard-filter every returned timestamp to be < as_of.
  4. Compute a momentum signal: mean daily views over the last ~7 days vs the
     prior ~30 days, and the % change (rising / falling / flat).
  5. render() a compact, LLM-readable block - distinguishing error / no-data /
     found exactly like the other sources (fail loud, never silent-empty).

Fail-soft per entity: one entity's 404 (no article) must not kill the others;
we only fail LOUD for the whole fetch (empty question, or every request erroring).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from ..config import USER_AGENT
from .gdelt_bq import tags_to_fips
from .gdelt_bq_news import _match_terms

logger = logging.getLogger(__name__)

# Wikimedia REST Pageviews API. Path template (verified live 2026-06-02):
#   .../per-article/{project}/all-access/user/{article}/daily/{start}/{end}
# {start}/{end} are YYYYMMDD; daily items come back stamped YYYYMMDD00. We use the
# "user" agent class (excludes bots/spiders) so the signal reflects real readers.
_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
_ACCESS = "all-access"
_AGENT = "user"
_DATE_FMT = "%Y%m%d"

# Wikimedia's User-Agent policy asks for a descriptive UA with contact info (see config).
_USER_AGENT = USER_AGENT

# Window: 60 days up to (but excluding) as_of. Recent = last RECENT_DAYS; baseline
# = the BASELINE_DAYS before that. 7 vs 30 is a standard short-vs-medium momentum.
_WINDOW_DAYS = 60
_RECENT_DAYS = 7
_BASELINE_DAYS = 30

_TIMEOUT_S = 15.0
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0

# Market country tag -> the country's LOCAL-LANGUAGE Wikipedia subdomain code.
# Keyed off the same FIPS map the news source uses (so coverage stays in sync):
# we map the verified FIPS country code -> wiki language. English (en) is always
# queried as a secondary, so this only needs the *local* language per country.
# Multilingual countries are mapped to the dominant Wikipedia for the polity.
_FIPS_TO_WIKI_LANG: dict[str, str] = {
    "IR": "fa",  # Iran -> Persian
    "IS": "he",  # Israel -> Hebrew
    "UP": "uk",  # Ukraine -> Ukrainian
    "HU": "hu",  # Hungary -> Hungarian
    "NL": "nl",  # Netherlands -> Dutch
    "KS": "ko",  # South Korea -> Korean
    "CA": "en",  # Canada -> English (en already secondary; kept explicit)
    "LE": "ar",  # Lebanon -> Arabic
    "EI": "en",  # Ireland -> English
    "IN": "hi",  # India -> Hindi (en also queried)
    "JA": "ja",  # Japan -> Japanese
    "VE": "es",  # Venezuela -> Spanish
    "TH": "th",  # Thailand -> Thai
    "CB": "km",  # Cambodia -> Khmer
    "IT": "it",  # Italy -> Italian
    "CY": "el",  # Cyprus -> Greek
    "AS": "en",  # Australia -> English
    "MX": "es",  # Mexico -> Spanish
    "CH": "zh",  # China -> Chinese
    "QA": "ar",  # Qatar -> Arabic
    "TW": "zh",  # Taiwan -> Chinese
    "RS": "ru",  # Russia -> Russian
    "IZ": "ar",  # Iraq -> Arabic
}

# Always query English in addition to the local wiki: many entities (smaller
# parties, regional figures) only have an en article, and en is a useful
# cross-check. Listed first so it's the stable secondary.
_FALLBACK_LANG = "en"


class WikipediaPageviewsSource:
    """Local-language Wikipedia pageview momentum. Source key: 'pageviews'."""

    name = "pageviews"

    def __init__(
        self,
        http: httpx.Client | None = None,
        max_entities: int = 4,
        window_days: int = _WINDOW_DAYS,
        recent_days: int = _RECENT_DAYS,
        baseline_days: int = _BASELINE_DAYS,
    ) -> None:
        self._http = http or httpx.Client(timeout=_TIMEOUT_S, headers={"User-Agent": _USER_AGENT})
        self._max_entities = max_entities
        self._window_days = window_days
        self._recent_days = recent_days
        self._baseline_days = baseline_days

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"entities": [], "query": "", "error": "empty_question"}

        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")
        cutoff = as_of if as_of is not None else datetime.now(UTC)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)

        terms = _match_terms(question, max_terms=self._max_entities)
        if not terms:
            return {"entities": [], "query": "", "error": "no_entities"}

        langs = _wiki_langs(context.get("tags") or [])
        query_desc = " | ".join(terms)

        # POINT-IN-TIME window: [as_of - window_days, as_of - 1 day]. We ask the
        # API only for days strictly before as_of, and re-verify below.
        end = cutoff - timedelta(days=1)
        start = cutoff - timedelta(days=self._window_days)

        results: list[dict[str, Any]] = []
        any_request_ok = False
        for term in terms:
            series, ok = self._best_series(term, langs, start, end, cutoff)
            any_request_ok = any_request_ok or ok
            if series is None:
                # No article on any wiki (or all 404) - record as no-data, not error.
                results.append({"entity": term, "lang": None, "found": False})
                continue
            lang, points = series
            signal = _momentum(points, cutoff, self._recent_days, self._baseline_days)
            results.append({"entity": term, "lang": lang, "found": True, **signal})

        # Fail LOUD only if EVERY request errored at the transport level (so we
        # can't tell "no article" from "API down"). A clean 404-everywhere is
        # legitimate no-data, not an error.
        if not any_request_ok:
            return {"entities": [], "query": query_desc, "error": "api_unreachable"}

        return {
            "entities": results,
            "query": query_desc,
            "langs": langs,
            "as_of": cutoff.isoformat(),
        }

    def _best_series(
        self,
        entity: str,
        langs: list[str],
        start: datetime,
        end: datetime,
        cutoff: datetime,
    ) -> tuple[tuple[str, list[dict[str, Any]]] | None, bool]:
        """Try the entity title on each wiki (local first, then en). Returns the
        first wiki that yields any in-window points, plus whether ANY request
        completed without a transport error (so the caller can tell down-vs-404).
        """
        any_ok = False
        for lang in langs:
            points, ok = self._fetch_article(entity, lang, start, end, cutoff)
            any_ok = any_ok or ok
            if points:
                return (lang, points), any_ok
        return None, any_ok

    def _fetch_article(
        self,
        entity: str,
        lang: str,
        start: datetime,
        end: datetime,
        cutoff: datetime,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fetch one article's daily series. Returns (in-window points, request_ok).

        request_ok is True if the HTTP call completed (incl. a clean 404 = "no
        such article"); False only on transport/5xx exhaustion. Every returned
        point is hard-filtered to timestamp < cutoff (the leakage guard).
        """
        article = _to_title(entity)
        url = (
            f"{_BASE}/{lang}.wikipedia/{_ACCESS}/{_AGENT}/{quote(article, safe='')}"
            f"/daily/{start.strftime(_DATE_FMT)}/{end.strftime(_DATE_FMT)}"
        )
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.get(url)
            except httpx.HTTPError:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            if resp.status_code == 404:
                # No such article on this wiki - a clean, expected "no data".
                return [], True
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            try:
                resp.raise_for_status()
                items = resp.json().get("items", [])
            except (httpx.HTTPError, ValueError):
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            return _filter_before(items, cutoff), True
        return [], False

    def render(self, payload: dict[str, Any]) -> str:
        query = payload.get("query", "")
        as_of = payload.get("as_of")
        suffix = f", as of {as_of}" if as_of else ""

        error = payload.get("error")
        if error:
            return f"[SOURCE ERROR] pageviews fetch failed ({error})"

        entities = payload.get("entities", [])
        found = [e for e in entities if e.get("found")]
        if not found:
            return f"(Wikipedia pageviews: no articles found for: {query}{suffix})"

        langs = "+".join(payload.get("langs", []))
        lines = [f"Wikipedia attention ({langs} wiki{suffix}):"]
        for e in found:
            lines.append(f"- {_render_entity(e)}")
        # Note any entities with no article so the model sees the gap explicitly.
        missing = [e["entity"] for e in entities if not e.get("found")]
        if missing:
            lines.append(f"(no Wikipedia article found for: {', '.join(missing)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network).
# ---------------------------------------------------------------------------


def _wiki_langs(tags: list[str]) -> list[str]:
    """Country tags -> ordered list of Wikipedia langs to query.

    The local language(s) of the market's country first (de-duplicated, in tag
    order), then English as a secondary fallback. Always non-empty: a market with
    no mapped country still gets English.
    """
    langs: list[str] = []
    for code in tags_to_fips(tags):
        lang = _FIPS_TO_WIKI_LANG.get(code)
        if lang and lang not in langs:
            langs.append(lang)
    if _FALLBACK_LANG not in langs:
        langs.append(_FALLBACK_LANG)
    return langs


def _to_title(entity: str) -> str:
    """Wikipedia article titles use underscores for spaces. The entity name is
    usually the title verbatim (e.g. "Andrea Martella" -> "Andrea_Martella"); we
    keep it simple and try it directly rather than resolving via the search API."""
    return entity.strip().replace(" ", "_")


def _parse_ts(timestamp: str) -> datetime | None:
    """Parse a pageviews timestamp (YYYYMMDD00, hourly field always '00')."""
    try:
        return datetime.strptime(timestamp[:8], _DATE_FMT).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _filter_before(items: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    """Leakage guard: keep ONLY days strictly before the cutoff (as_of).

    The pageviews stamp is the START of the day (YYYYMMDD00). A day stamped >=
    as_of would include same-day-or-later attention the forecaster could not have
    seen, so we drop it. Unparseable stamps are dropped (fail-closed)."""
    kept: list[dict[str, Any]] = []
    for it in items:
        ts = _parse_ts(str(it.get("timestamp", "")))
        if ts is None or ts >= cutoff:
            continue
        kept.append({"date": ts, "views": int(it.get("views", 0) or 0)})
    return kept


def _momentum(
    points: list[dict[str, Any]],
    cutoff: datetime,
    recent_days: int,
    baseline_days: int,
) -> dict[str, Any]:
    """Recent-vs-baseline attention momentum from a daily series (all < cutoff).

    recent  = mean daily views over the last `recent_days` before cutoff,
    baseline= mean daily views over the `baseline_days` immediately before that.
    pct_change is (recent-baseline)/baseline; trend buckets it rising/flat/falling.
    Days with no data simply don't contribute (mean over present days)."""
    recent_start = cutoff - timedelta(days=recent_days)
    baseline_start = recent_start - timedelta(days=baseline_days)

    recent = [p["views"] for p in points if recent_start <= p["date"] < cutoff]
    baseline = [p["views"] for p in points if baseline_start <= p["date"] < recent_start]

    recent_mean = sum(recent) / len(recent) if recent else 0.0
    baseline_mean = sum(baseline) / len(baseline) if baseline else 0.0

    if baseline_mean > 0:
        pct = (recent_mean - baseline_mean) / baseline_mean
    elif recent_mean > 0:
        pct = None  # new attention with no prior baseline - "new" rather than %
    else:
        pct = 0.0

    return {
        "recent_mean": recent_mean,
        "baseline_mean": baseline_mean,
        "pct_change": pct,
        "trend": _trend(pct),
        "n_days": len(points),
    }


def _trend(pct: float | None) -> str:
    """Bucket a percent change into a human label. None = new (no baseline)."""
    if pct is None:
        return "new"
    if pct >= 0.15:
        return "rising"
    if pct <= -0.15:
        return "falling"
    return "flat"


def _render_entity(e: dict[str, Any]) -> str:
    """One rendered line, e.g.
    'Andrea Martella (it): 1,240 views/day last 7d, +38% vs prior 30d (rising)'."""
    pct = e.get("pct_change")
    change = "new (no prior baseline)" if pct is None else f"{pct * 100:+.0f}% vs prior 30d"
    return (
        f"{e['entity']} ({e['lang']}): {e['recent_mean']:,.0f} views/day last 7d, "
        f"{change} ({e['trend']})"
    )
