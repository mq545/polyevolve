"""Polymarket MarketSource plugin - the single source of truth for the live path.

Implements the proven Polymarket Gamma client (tag-filtered event discovery +
clean-settlement resolution) directly against the `core.interfaces.MarketSource`
contract, plus the `order_book()` method the executability check needs (CLOB
`/book`). This module owns the live discovery/resolution logic; the legacy
`market_sources.polymarket` module now re-uses these helpers and only adds the
backtest-only machinery (`ResolvedMarket`/`list_resolved_markets`/`price_at`)
that the offline snapshot/backtest harness consumes.

Discovery uses the /events endpoint because tags live on events, not markets,
and /events supports server-side tag_slug filtering. Each event contains nested
markets; we flatten them and attach the event's tags to each market's metadata.

Self-registers as @register_market("polymarket"). Core never imports this file;
the registry's discover() does, which is what fires the decorator.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from polyevolve.core.registry import register_market
from polyevolve.core.types import Market, MarketFilter, OrderBook, Outcome, Resolution

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Default discovery tags when a filter names no tags - the political set the
# client has always used, which is where our experiments live.
_DEFAULT_TAGS = ("politics", "geopolitics", "world", "elections")


def parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (with a trailing 'Z') into an aware datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_price_list(value: Any) -> list[float] | None:
    """outcomePrices is JSON-encoded as a string, e.g. '["0.04", "0.96"]'."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except (ValueError, TypeError):
            return None
    return None


@register_market("polymarket")
class PolymarketMarket:
    """Polymarket via the Gamma API. Plugin key: 'polymarket'."""

    key = "polymarket"
    # Venue name carried on metadata so legacy consumers and the ledger agree.
    name = "polymarket"

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(base_url=GAMMA_BASE, timeout=30.0)
        # A separate client for the CLOB (different base URL) used by order_book.
        self._clob = httpx.Client(timeout=30.0)

    def list_markets(self, filt: MarketFilter) -> Iterable[Market]:
        """Fetch markets via tag-filtered events, mapped to the core shape.

        Honors `filt`: tags (default political set), open_only, and
        resolves_within_days (bound on end_date relative to now).
        """
        tags = list(filt.tags) if filt.tags else list(_DEFAULT_TAGS)
        limit_per_tag = 100
        cutoff: datetime | None = None
        if filt.resolves_within_days is not None:
            cutoff = datetime.now(UTC) + timedelta(days=filt.resolves_within_days)

        seen: set[str] = set()
        for tag in tags:
            for raw, event_tags, event_title in self._raw_for_tag(tag, limit_per_tag, seen):
                status = "resolved" if raw.get("closed") else "active"
                if filt.open_only and status != "active":
                    continue
                close = parse_dt(raw.get("endDate"))
                if cutoff is not None and (close is None or close > cutoff):
                    continue
                yield _to_core_market(raw, event_tags, event_title, close, filt.category)

    def _raw_for_tag(
        self, tag: str, limit: int, seen: set[str]
    ) -> Iterator[tuple[dict[str, Any], list[str], str | None]]:
        """Yield (raw_market, event_tags, event_title) for one tag's open events."""
        resp = self._http.get(
            "/events",
            params={"limit": limit, "closed": "false", "active": "true", "tag_slug": tag},
        )
        resp.raise_for_status()
        for event in resp.json():
            event_tags = _event_tags(event)
            for raw_market in event.get("markets") or []:
                ext_id = str(raw_market.get("id", ""))
                if not ext_id or ext_id in seen:
                    continue
                if raw_market.get("closed"):
                    continue
                seen.add(ext_id)
                yield raw_market, event_tags, event.get("title")

    def get_resolution(self, external_id: str) -> Resolution | None:
        """Resolution of a settled binary market, or None if open/ambiguous.

        Used by the forward-ledger auto-grader. A resolved binary settles to
        [1, 0] or [0, 1]; a wide threshold tolerates near-settled prices and
        skips closed-but-disputed markets.
        """
        resp = self._http.get(f"/markets/{external_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        row = resp.json()

        if not row.get("closed"):
            return None
        prices = parse_price_list(row.get("outcomePrices"))
        if not prices:
            return None
        yes_price = prices[0]  # Convention: outcomes[0] is "Yes".
        if 0.05 < yes_price < 0.95:
            return None  # closed but ambiguous / disputed - skip.
        outcome: Outcome = "YES" if yes_price >= 0.95 else "NO"
        resolved_at = (
            parse_dt(row.get("closedTime")) or parse_dt(row.get("endDate")) or datetime.now(UTC)
        )
        return Resolution(
            external_id=external_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

    def order_book(self, external_id: str) -> OrderBook | None:
        """Top-of-book for the YES token, via the CLOB `/book` endpoint.

        `external_id` is a Gamma market id; the CLOB is keyed by token id, so we
        first resolve the market's YES clobTokenId from Gamma, then fetch its
        book. Returns None on any miss (no token, 404, malformed) rather than
        raising - the executability check treats absence as "not executable".
        """
        token_id = self._yes_token_id(external_id)
        if token_id is None:
            return None
        try:
            resp = self._clob.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        # OrderBook convention: bids best-first (highest price), asks best-first
        # (lowest price) - the order needed to walk the book at size.
        bids = tuple(sorted(_levels(data.get("bids")), key=lambda lvl: lvl[0], reverse=True))
        asks = tuple(sorted(_levels(data.get("asks")), key=lambda lvl: lvl[0]))
        if not bids and not asks:
            return None
        return OrderBook(bids=bids, asks=asks)

    def _yes_token_id(self, external_id: str) -> str | None:
        try:
            resp = self._http.get(f"/markets/{external_id}")
            if resp.status_code != 200:
                return None
            row = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        clob_ids = row.get("clobTokenIds")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                return None
        if isinstance(clob_ids, list) and clob_ids:
            return str(clob_ids[0])
        return None


def _event_tags(event: dict[str, Any]) -> list[str]:
    return [
        str(t["slug"]) for t in (event.get("tags") or []) if isinstance(t, dict) and t.get("slug")
    ]


def _to_core_market(
    row: dict[str, Any],
    event_tags: list[str],
    event_title: str | None,
    close: datetime | None,
    category: str | None,
) -> Market:
    """Map a raw Gamma market dict to the core `Market` shape."""
    return Market(
        external_id=str(row["id"]),
        question=row.get("question", ""),
        category=category or "politics",
        tags=tuple(event_tags),
        resolution_criteria=(row.get("description") or "")[:2000],
        end_date=close,
        metadata={
            "slug": row.get("slug"),
            "event_title": event_title,
            "tags": event_tags,
            "volume": row.get("volume"),
            "liquidity": row.get("liquidity"),
            "outcomePrices": row.get("outcomePrices"),
            "outcomes": row.get("outcomes"),
            "description": (row.get("description") or "")[:2000],
            "endDate": row.get("endDate"),
        },
    )


def _levels(raw: Any) -> tuple[tuple[float, float], ...]:
    """Parse CLOB book levels (list of {price, size}) into (price, size) tuples.

    CLOB returns prices/sizes as strings. Malformed levels are skipped. Ordering
    is normalized by the caller (order_book) to the OrderBook convention.
    """
    if not isinstance(raw, list):
        return ()
    levels: list[tuple[float, float]] = []
    for lvl in raw:
        if not isinstance(lvl, dict):
            continue
        try:
            price = float(lvl["price"])
            size = float(lvl["size"])
        except (KeyError, ValueError, TypeError):
            continue
        levels.append((price, size))
    return tuple(levels)
