"""Financial-market movement - a point-in-time LEADING-INDICATOR signal for politics.

Why this exists (the confirmed thesis): deep, liquid, fast markets price a
geopolitical or political shift BEFORE thin Polymarket does, whenever the question
maps to a tradable instrument. The motivating example: before Polymarket priced
"US/Israel strike Iran" at 0.20, the defense ETF was up double digits, oil up
~14%, gold up ~14% - the financial tape had already moved. So for any market that
cleanly maps to an instrument, the pre-as_of price MOVEMENT is a real leading
signal the forecaster otherwise can't see.

Design choices:
  - $0: the free Yahoo Finance chart JSON endpoint (v8), no key, no account. We
    hit the JSON directly with httpx + stdlib - no pandas/numpy/yfinance.
  - HIGH-PRECISION / LOW-RECALL: a simple, rule-based keyword/country -> ticker
    map. If a market maps to nothing, we return a CLEAN no-data - we never
    fabricate a "signal" from an instrument that isn't actually about the question.
  - POINT-IN-TIME is the whole game: daily closes are immutable history, but we
    must never include a close dated on/after as_of. We set the request's period2
    to as_of - 1 day AND hard-filter every returned bar to timestamp < as_of, then
    assert the last kept bar is strictly before as_of.
  - Fail-soft per ticker: one ticker's 429/404 must not kill the others; we only
    fail LOUD for the whole fetch (e.g. empty question). Nothing-mapped and
    all-tickers-failed are clean no-data, not errors.

Mapping rules (rule-based, no LLM at fetch time):
  1. CONFLICT/war questions (keywords below) -> defense ETF ITA, WTI oil CL=F,
     gold GC=F.
  2. ELECTION / country-political questions -> the country's main EQUITY INDEX
     and its CURRENCY vs USD (USDXXX=X), via the curated _FIPS_TO_YF map.
  3. Both conflict keywords AND a mapped country -> include both sets.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

import httpx

from ..config import CONTACT
from .gdelt_bq import tags_to_fips

logger = logging.getLogger(__name__)

# Yahoo Finance chart v8 JSON endpoint. Verified shape (human-confirmed live for
# ITA/LMT/CL=F/GC=F over 2026-01-05..02-21): a GET returns
#   {"chart": {"result": [{"timestamp": [epoch_s, ...],
#               "indicators": {"quote": [{"close": [float|null, ...]}]}}],
#             "error": null}}
# Yahoo blocks the default httpx UA, so we send a browser User-Agent.
_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    f"Chrome/120.0.0.0 Safari/537.36 polyevolve-forecaster/0.1 ({CONTACT})"
)

# Window: LOOKBACK_DAYS up to (but excluding) as_of. We compute the % change in
# close from the first in-window bar to the last in-window bar.
_LOOKBACK_DAYS = 30
_TIMEOUT_S = 15.0
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0

# --- Rule 1: conflict / war questions -> macro instruments -----------------
# Lowercased keyword -> matched as a whole word (or substring of a word) in the
# question. Kept deliberately small and high-precision: these words almost always
# denote a kinetic/geopolitical event that moves defense, oil and gold together.
_CONFLICT_KEYWORDS: tuple[str, ...] = (
    "strike",
    "war",
    "invade",
    "invasion",
    "missile",
    "nuclear",
    "ceasefire",
    "attack",
    "hormuz",
    "annex",
    "troops",
)

# (ticker, human label) for the conflict basket. ITA = iShares U.S. Aerospace &
# Defense ETF; CL=F = WTI crude front-month; GC=F = gold front-month.
_CONFLICT_INSTRUMENTS: tuple[tuple[str, str], ...] = (
    ("ITA", "defense ETF (ITA)"),
    ("CL=F", "WTI oil (CL=F)"),
    ("GC=F", "gold (GC=F)"),
)

# --- Rule 2: country-political questions -> equity index + FX vs USD -------
# GDELT FIPS country code -> (index ticker, index label, fx ticker, fx label).
# Mirrors the _FIPS_TO_TLD / _FIPS_TO_WIKI_LANG pattern so coverage stays in sync
# with the news/pageviews sources. ONLY countries with a reliable FREE Yahoo
# ticker are listed; the rest are deliberately skipped (high-precision: a market
# we can't map returns clean no-data rather than a wrong instrument).
# FX is always quoted USD->local (USDXXX=X), so a RISING value = local currency
# WEAKENING vs USD; the renderer states the raw % so the model reads direction.
_FIPS_TO_YF: dict[str, tuple[str, str, str, str]] = {
    "IT": ("FTSEMIB.MI", "Italy FTSE MIB", "USDEUR=X", "EUR/USD (USDEUR)"),
    "NL": ("^AEX", "Netherlands AEX", "USDEUR=X", "EUR/USD (USDEUR)"),
    "IN": ("^BSESN", "India BSE Sensex", "USDINR=X", "INR (USDINR)"),
    "JA": ("^N225", "Japan Nikkei 225", "USDJPY=X", "JPY (USDJPY)"),
    "HU": ("^BUX.BD", "Hungary BUX", "USDHUF=X", "HUF (USDHUF)"),
    "IS": ("^TA125.TA", "Israel TA-125", "USDILS=X", "ILS (USDILS)"),
    "MX": ("^MXX", "Mexico IPC", "USDMXN=X", "MXN (USDMXN)"),
    "CY": ("^STOXX50E", "Euro STOXX 50", "USDEUR=X", "EUR/USD (USDEUR)"),
    "CA": ("^GSPTSE", "Canada S&P/TSX", "USDCAD=X", "CAD (USDCAD)"),
}


class FinancialMarketsSource:
    """Pre-as_of financial-market movement as a leading indicator. Key: 'markets'."""

    name = "markets"

    def __init__(
        self,
        http: httpx.Client | None = None,
        lookback_days: int = _LOOKBACK_DAYS,
    ) -> None:
        self._http = http or httpx.Client(timeout=_TIMEOUT_S, headers={"User-Agent": _USER_AGENT})
        self._lookback_days = lookback_days

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"instruments": [], "as_of": None, "error": "empty_question"}

        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")
        cutoff = as_of if as_of is not None else datetime.now(UTC)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)

        mapped = route_instruments(question, context.get("tags") or [])
        if not mapped:
            # Nothing mapped - clean no-data (high precision: never fabricate).
            return {"instruments": [], "as_of": cutoff.isoformat()}

        # POINT-IN-TIME window: [as_of - lookback, as_of - 1 day]. period2 is
        # STRICTLY before as_of; every bar is re-filtered to timestamp < as_of.
        end = cutoff - timedelta(days=1)
        start = cutoff - timedelta(days=self._lookback_days)

        results: list[dict[str, Any]] = []
        for ticker, label in mapped:
            closes, ok = self._fetch_closes(ticker, start, end, cutoff)
            if not ok:
                # Transport/429/5xx exhaustion or no in-window bars -> skip this
                # ticker (fail-soft); it simply doesn't contribute a line.
                results.append({"ticker": ticker, "label": label, "found": False})
                continue
            pct = _pct_change(closes)
            if pct is None:
                results.append({"ticker": ticker, "label": label, "found": False})
                continue
            results.append(
                {
                    "ticker": ticker,
                    "label": label,
                    "found": True,
                    "pct_change": pct,
                    "n_bars": len(closes),
                }
            )

        return {
            "instruments": results,
            "as_of": cutoff.isoformat(),
            "lookback_days": self._lookback_days,
        }

    def _fetch_closes(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        cutoff: datetime,
    ) -> tuple[list[tuple[datetime, float]], bool]:
        """Fetch one ticker's daily closes in [start, end], hard-filtered < cutoff.

        Returns (in-window (date, close) bars, ok). ok is False on transport/429/5xx
        exhaustion OR when the response yields no usable in-window bars, so the
        caller can fail-soft. Every kept bar is asserted strictly before cutoff.
        """
        p1 = int(start.timestamp())
        # period2 is EXCLUSIVE-ish on Yahoo's side, but we don't rely on that: we
        # request up to end (as_of - 1 day) and then hard-filter < cutoff below.
        p2 = int(end.timestamp())
        url = f"{_BASE}/{quote(ticker, safe='')}?period1={p1}&period2={p2}&interval=1d"
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.get(url)
            except httpx.HTTPError:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            if resp.status_code == 404:
                # Unknown ticker - clean miss, no point retrying.
                return [], False
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            try:
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError):
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            bars = _parse_closes(payload, cutoff)
            return bars, bool(bars)
        return [], False

    def render(self, payload: dict[str, Any]) -> str:
        error = payload.get("error")
        if error:
            return f"[SOURCE ERROR] markets fetch failed ({error})"

        instruments = payload.get("instruments", [])
        found = [i for i in instruments if i.get("found")]
        if not found:
            return "(No mapped financial instrument for this market)"

        as_of = payload.get("as_of")
        date_str = (as_of or "")[:10]
        lookback = payload.get("lookback_days", _LOOKBACK_DAYS)
        header = f"Financial-market signal (pre-{date_str}, {lookback}d): "
        parts = [f"{i['label']} {i['pct_change'] * 100:+.1f}%" for i in found]
        return header + ", ".join(parts)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network).
# ---------------------------------------------------------------------------


def route_instruments(question: str, tags: list[str]) -> list[tuple[str, str]]:
    """Rule-based map: question + country tags -> ordered [(ticker, label)].

    Rule 1: conflict keywords -> defense/oil/gold basket.
    Rule 2: a mapped country -> its equity index + currency vs USD.
    Rule 3: both -> both sets (conflict basket first, then per-country).
    De-duplicated by ticker, order preserved. Empty list = nothing mapped (the
    high-precision no-data case).
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(ticker: str, label: str) -> None:
        if ticker not in seen:
            seen.add(ticker)
            out.append((ticker, label))

    if _has_conflict_keyword(question):
        for ticker, label in _CONFLICT_INSTRUMENTS:
            add(ticker, label)

    for code in tags_to_fips(tags):
        mapping = _FIPS_TO_YF.get(code)
        if mapping:
            idx_ticker, idx_label, fx_ticker, fx_label = mapping
            add(idx_ticker, idx_label)
            add(fx_ticker, fx_label)

    return out


def _has_conflict_keyword(question: str) -> bool:
    """True if any conflict keyword appears as a token in the question.

    Token-based (split on non-letters) so 'strike' matches 'strikes'/'striking'
    via prefix, but we keep it simple: a keyword matches if it is a substring of
    any alphabetic token. High-precision keywords chosen so false positives are rare.
    """
    import re  # noqa: PLC0415

    toks = re.findall(r"[a-z]+", question.lower())
    return any(any(kw in tok for tok in toks) for kw in _CONFLICT_KEYWORDS)


def _parse_closes(payload: dict[str, Any], cutoff: datetime) -> list[tuple[datetime, float]]:
    """Extract (date, close) bars from a Yahoo chart payload, filtered < cutoff.

    LEAKAGE GUARD: every bar whose timestamp is on/after the cutoff (as_of) is
    dropped - the forecaster could not have seen that close. Null closes (Yahoo
    pads non-trading sessions with null) are dropped too. Bars are returned in
    timestamp order; the last one is guaranteed strictly before cutoff.
    """
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        return []
    res = results[0]
    timestamps = res.get("timestamp") or []
    quote_blocks = (res.get("indicators") or {}).get("quote") or []
    if not quote_blocks:
        return []
    closes = quote_blocks[0].get("close") or []

    bars: list[tuple[datetime, float]] = []
    for ts, close in zip(timestamps, closes, strict=False):
        if close is None:
            continue
        try:
            dt = datetime.fromtimestamp(int(ts), tz=UTC)
        except (ValueError, OverflowError, OSError, TypeError):
            continue
        if dt >= cutoff:  # leakage: on/after as_of must never enter the window
            continue
        bars.append((dt, float(close)))
    bars.sort(key=lambda b: b[0])
    return bars


def _pct_change(closes: list[tuple[datetime, float]]) -> float | None:
    """Percent change from the first to the last in-window close (as a fraction).

    Needs >= 2 bars and a non-zero first close. Returns None otherwise (the
    caller treats that as no-data for the ticker)."""
    if len(closes) < 2:
        return None
    first = closes[0][1]
    last = closes[-1][1]
    if first == 0:
        return None
    return (last - first) / first
