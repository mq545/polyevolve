"""Kalshi MarketSource plugin - adapts Kalshi's public read-only API to core.

Kalshi's trade-api v2 read endpoints (markets list, single market, orderbook)
are public - no RSA-signed auth is required for the discovery/resolution/book
calls this plugin makes (only trading needs signed requests). So unlike the old
`market_sources/kalshi.py` placeholder, this is a real, working source.

Endpoints used (base https://api.elections.kalshi.com/trade-api/v2):
  GET /markets?status=open&cursor=...   - paginated market discovery
  GET /markets/{ticker}                 - single market (for resolution)
  GET /markets/{ticker}/orderbook       - top-of-book depth

Kalshi prices are YES-share prices in dollars [0, 1]. The orderbook gives
resting bids on each side: `yes_dollars` are YES bids; `no_dollars` are NO bids,
each equivalent to a YES *ask* at (1 - price). We normalize to the OrderBook
convention (bids best-first highest, asks best-first lowest).

Self-registers as @register_market("kalshi"). Core never imports this file; the
registry's discover() does, which is what fires the decorator.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from polyevolve.core.registry import register_market
from polyevolve.core.types import Market, MarketFilter, OrderBook, Resolution

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi caps page size at 1000; 100 keeps responses small and is plenty per page.
_PAGE_LIMIT = 100
# Safety bound on pagination so an empty/lenient filter can't loop the whole venue.
_MAX_PAGES = 20


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@register_market("kalshi")
class KalshiMarket:
    """Kalshi via its public trade-api v2. Plugin key: 'kalshi'."""

    key = "kalshi"

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(base_url=KALSHI_BASE, timeout=30.0)

    def list_markets(self, filt: MarketFilter) -> Iterable[Market]:
        # open_only maps to Kalshi's status filter; omitting status returns all.
        status = "open" if filt.open_only else None
        cutoff: datetime | None = None
        if filt.resolves_within_days is not None:
            cutoff = datetime.now(UTC) + timedelta(days=filt.resolves_within_days)

        for raw in self._iter_raw_markets(status):
            close = _parse_dt(raw.get("close_time"))
            if cutoff is not None and (close is None or close > cutoff):
                continue
            market = _to_core_market(raw, close, filt.category)
            if filt.category is not None and market.category != filt.category:
                continue
            if filt.tags and not (set(filt.tags) & set(market.tags)):
                continue
            yield market

    def get_resolution(self, external_id: str) -> Resolution | None:
        try:
            resp = self._http.get(f"/markets/{external_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            row = resp.json().get("market") or {}
        except (httpx.HTTPError, ValueError):
            return None

        result = str(row.get("result") or "").lower()
        if result not in ("yes", "no"):
            # "" (still open), "void"/"" cancelled, or anything non-binary → unresolved.
            return None

        resolved_at = (
            _parse_dt(row.get("close_time"))
            or _parse_dt(row.get("expiration_time"))
            or datetime.now(UTC)
        )
        return Resolution(
            external_id=external_id,
            outcome="YES" if result == "yes" else "NO",
            resolved_at=resolved_at,
        )

    def order_book(self, external_id: str) -> OrderBook | None:
        """Top-of-book for the YES side via the public orderbook endpoint.

        Kalshi returns resting bids per side under `orderbook_fp`:
          yes_dollars: [[price, size], ...]  → YES bids as-is
          no_dollars:  [[price, size], ...]  → NO bids; a NO bid at p is a YES
                                               ask at (1 - p) of the same size
        Returns None on any miss (404, malformed, empty book) rather than raising.
        """
        try:
            resp = self._http.get(f"/markets/{external_id}/orderbook")
            if resp.status_code != 200:
                return None
            book = resp.json().get("orderbook_fp") or {}
        except (httpx.HTTPError, ValueError):
            return None

        yes_bids = _levels(book.get("yes_dollars"))
        # NO bid at price p == YES ask at (1 - p); same resting size.
        no_bids = _levels(book.get("no_dollars"))
        yes_asks = tuple((round(1.0 - price, 6), size) for price, size in no_bids)

        bids = tuple(sorted(yes_bids, key=lambda lvl: lvl[0], reverse=True))
        asks = tuple(sorted(yes_asks, key=lambda lvl: lvl[0]))
        if not bids and not asks:
            return None
        return OrderBook(bids=bids, asks=asks)

    def _iter_raw_markets(self, status: str | None) -> Iterator[dict[str, Any]]:
        """Page through /markets, following Kalshi's `cursor` until exhausted."""
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            params: dict[str, Any] = {"limit": _PAGE_LIMIT}
            if status is not None:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            try:
                resp = self._http.get("/markets", params=params)
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError):
                return
            markets = data.get("markets") or []
            yield from markets
            cursor = data.get("cursor") or None
            if not cursor or not markets:
                return


def _to_core_market(row: dict[str, Any], close: datetime | None, category: str | None) -> Market:
    """Map a raw Kalshi market dict to the core `Market` shape.

    Kalshi's own `category` is often null on markets; fall back to the explicit
    filter category, then "general". Tags carry the event ticker so connectors
    can group sibling markets.
    """
    kalshi_cat = row.get("category") or None
    event_ticker = str(row.get("event_ticker") or "")
    tags = tuple(t for t in (kalshi_cat, event_ticker) if t)
    return Market(
        external_id=str(row.get("ticker") or ""),
        question=str(row.get("title") or ""),
        category=kalshi_cat or category or "general",
        tags=tags,
        resolution_criteria=str(row.get("rules_primary") or ""),
        end_date=close,
        metadata={
            "event_ticker": event_ticker or None,
            "subtitle": row.get("yes_sub_title") or row.get("subtitle"),
            "status": row.get("status"),
            "yes_bid": row.get("yes_bid_dollars"),
            "yes_ask": row.get("yes_ask_dollars"),
            "last_price": row.get("last_price_dollars"),
            "volume": row.get("volume_fp"),
            "liquidity": row.get("liquidity_dollars"),
            "open_time": row.get("open_time"),
            "close_time": row.get("close_time"),
            "rules_secondary": (row.get("rules_secondary") or "")[:2000],
        },
    )


def _levels(raw: Any) -> tuple[tuple[float, float], ...]:
    """Parse Kalshi book levels ([[price, size], ...], strings) into float tuples.

    Malformed levels are skipped. Ordering is normalized by the caller.
    """
    if not isinstance(raw, list):
        return ()
    levels: list[tuple[float, float]] = []
    for lvl in raw:
        if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
            continue
        try:
            price = float(lvl[0])
            size = float(lvl[1])
        except (ValueError, TypeError):
            continue
        levels.append((price, size))
    return tuple(levels)
