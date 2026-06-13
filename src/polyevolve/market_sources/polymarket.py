"""Polymarket Gamma client - backtest/snapshot data path (contracts-shaped).

DEPRECATED for the LIVE path: the plugin `polyevolve.markets.polymarket`
(`PolymarketMarket`, registered as `@register_market("polymarket")`) is the
single source of truth for live discovery/resolution/order-book and produces the
`core.types` shapes the harness and ledger use. New code should use the plugin
via the registry, not this class.

This module remains because the OFFLINE snapshot/backtest harness
(`orchestration/{snapshot,backtest}.py`, `scripts/build_race_expansion.py`)
depends on machinery that is NOT part of the 3-method plugin contract:
`ResolvedMarket`, `list_resolved_markets` (past-resolved markets for
backtesting), and `price_at` (historical CLOB price lookup). Those, plus the
older `contracts.Market`/`contracts.Resolution` shapes the orchestration layer
and Postgres upsert path were built against, live here. The low-level parsing
and clean-settlement rules are shared with the plugin (imported below) so there
is exactly one copy of that logic.

Discovery uses the /events endpoint because tags live on events, not markets,
and /events supports server-side tag_slug filtering. Each event contains nested
markets; we flatten them and attach the event's tags to each market's metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from polyevolve.contracts import Market, MarketSource, Resolution

# Shared low-level helpers + endpoints live with the plugin (single source of
# truth for parsing and settlement rules). Re-exported here for back-compat.
from polyevolve.markets.polymarket import CLOB_BASE, GAMMA_BASE
from polyevolve.markets.polymarket import parse_dt as _parse_dt
from polyevolve.markets.polymarket import parse_price_list as _parse_price_list

__all__ = [
    "CLOB_BASE",
    "GAMMA_BASE",
    "PolymarketSource",
    "ResolvedMarket",
]

# prices-history rejects windows longer than this; keep requests under it.
_MAX_PRICE_WINDOW_S = 7 * 86400


@dataclass(frozen=True)
class ResolvedMarket:
    """A genuinely past-resolved market, for backtesting.

    `resolved_at` is the ACTUAL resolution time (closedTime), not the scheduled
    `endDate` - these diverge sharply for markets that resolve early (e.g. an
    event happens months before the deadline). Using closedTime is what makes
    point-in-time price lookup and contamination tagging correct.
    `created_at` bounds how early we can place an as_of (can't predict a market
    before it existed).
    """

    market: Market
    outcome: str  # YES | NO
    resolved_at: datetime  # closedTime - actual resolution
    created_at: datetime | None  # market creation; as_of must be >= this
    yes_token_id: str | None  # CLOB token for historical price lookup


class PolymarketSource:
    name = "polymarket"

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(base_url=GAMMA_BASE, timeout=30.0)

    def list_markets(self, filters: dict[str, Any]) -> Iterable[Market]:
        """Fetch markets via tag-filtered events.

        filters:
          tags: list[str] of event tag_slugs to pull (default: politics + geopolitics + world)
          limit_per_tag: events per tag (default 100)
        """
        tags = filters.get("tags", ["politics", "geopolitics", "world", "elections"])
        limit_per_tag = filters.get("limit_per_tag", 100)

        seen: set[str] = set()
        for tag in tags:
            yield from self._markets_for_tag(tag, limit_per_tag, seen)

    def _markets_for_tag(self, tag: str, limit: int, seen: set[str]) -> Iterator[Market]:
        resp = self._http.get(
            "/events",
            params={
                "limit": limit,
                "closed": "false",
                "active": "true",
                "tag_slug": tag,
            },
        )
        resp.raise_for_status()
        events = resp.json()

        for event in events:
            event_tags: list[str] = [
                str(t["slug"])
                for t in (event.get("tags") or [])
                if isinstance(t, dict) and t.get("slug")
            ]
            for raw_market in event.get("markets") or []:
                ext_id = str(raw_market.get("id", ""))
                if not ext_id or ext_id in seen:
                    continue
                if raw_market.get("closed"):
                    continue
                seen.add(ext_id)
                yield self._to_market(raw_market, event_tags, event.get("title"))

    def get_resolution(self, external_id: str) -> Resolution | None:
        resp = self._http.get(f"/markets/{external_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        row = resp.json()

        if not row.get("closed"):
            return None

        prices = _parse_price_list(row.get("outcomePrices"))
        if not prices:
            return None

        # Convention: outcomes[0] is "Yes". A resolved binary market settles to
        # [1, 0] or [0, 1]. Use a wide threshold to tolerate near-settled prices.
        yes_price = prices[0]
        if 0.05 < yes_price < 0.95:
            # Not actually settled (closed but ambiguous / disputed) - skip.
            return None
        outcome = "YES" if yes_price >= 0.95 else "NO"

        resolved_at = (
            _parse_dt(row.get("closedTime")) or _parse_dt(row.get("endDate")) or datetime.now(UTC)
        )

        return Resolution(
            venue=self.name,
            external_id=external_id,
            outcome=outcome,
            resolved_at=resolved_at,
        )

    def list_resolved_markets(self, filters: dict[str, Any]) -> Iterable[ResolvedMarket]:
        """Yield genuinely past-resolved markets for backtesting.

        Only markets whose endDate is strictly before `now` and that settled
        cleanly (yes price ~0 or ~1) are returned - this excludes future-dated
        placeholder markets that the API also flags `closed`.

        filters:
          tags: event tag_slugs (default political set)
          limit_per_tag, pages_per_tag: pagination control
          now: datetime treated as "present" (defaults to caller-supplied; required)
        """
        tags = filters.get("tags", ["politics", "geopolitics", "world", "elections"])
        limit = filters.get("limit_per_tag", 100)
        pages = filters.get("pages_per_tag", 3)
        now = filters["now"]

        seen: set[str] = set()
        for tag in tags:
            for page in range(pages):
                resp = self._http.get(
                    "/events",
                    params={
                        "limit": limit,
                        "offset": page * limit,
                        "closed": "true",
                        "tag_slug": tag,
                        "order": "endDate",
                        "ascending": "false",
                    },
                )
                resp.raise_for_status()
                events = resp.json()
                if not events:
                    break
                for event in events:
                    event_tags = [
                        str(t["slug"])
                        for t in (event.get("tags") or [])
                        if isinstance(t, dict) and t.get("slug")
                    ]
                    for raw in event.get("markets") or []:
                        rm = self._to_resolved(raw, event_tags, event.get("title"), now)
                        if rm is None or rm.market.external_id in seen:
                            continue
                        seen.add(rm.market.external_id)
                        yield rm

    def price_at(self, clob_token_id: str, ts: datetime) -> float | None:
        """Historical YES price at (or just before) `ts` from the CLOB.

        Uses a short window ending at ts (the endpoint rejects long windows),
        returns the price of the closest point at/before ts, or None.
        """
        end = int(ts.timestamp())
        start = end - _MAX_PRICE_WINDOW_S
        try:
            resp = self._http.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": clob_token_id,
                    "startTs": start,
                    "endTs": end,
                    "fidelity": 60,
                },
            )
            if resp.status_code != 200:
                return None
            history = resp.json().get("history", [])
        except (httpx.HTTPError, ValueError):
            return None
        if not history:
            return None
        # closest point at or before ts; fall back to overall closest
        at_or_before = [h for h in history if h.get("t", 0) <= end]
        pool = at_or_before or history
        closest = min(pool, key=lambda h: abs(h.get("t", 0) - end))
        try:
            return float(closest["p"])
        except (KeyError, ValueError, TypeError):
            return None

    def _to_resolved(
        self,
        row: dict[str, Any],
        event_tags: list[str],
        event_title: str | None,
        now: datetime,
    ) -> ResolvedMarket | None:
        if not row.get("closed"):
            return None
        # ACTUAL resolution time: closedTime, NOT the scheduled endDate. These
        # diverge hard for early-resolving markets (event happens months before
        # deadline). endDate is only a fallback when closedTime is absent.
        resolved_at = _parse_dt(row.get("closedTime")) or _parse_dt(row.get("endDate"))
        if resolved_at is None or resolved_at >= now:
            return None  # not actually resolved in the past
        prices = _parse_price_list(row.get("outcomePrices"))
        if not prices:
            return None
        yes = prices[0]
        if not (yes <= 0.05 or yes >= 0.95):
            return None  # closed but not cleanly settled (dispute/ambiguous)
        outcome = "YES" if yes >= 0.95 else "NO"

        clob_ids = row.get("clobTokenIds")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                clob_ids = None
        yes_token = clob_ids[0] if isinstance(clob_ids, list) and clob_ids else None

        market = self._to_market(row, event_tags, event_title)
        return ResolvedMarket(
            market=market,
            outcome=outcome,
            resolved_at=resolved_at,
            created_at=_parse_dt(row.get("createdAt")),
            yes_token_id=yes_token,
        )

    def _to_market(
        self, row: dict[str, Any], event_tags: list[str], event_title: str | None
    ) -> Market:
        status = "resolved" if row.get("closed") else "active"
        return Market(
            venue=self.name,
            external_id=str(row["id"]),
            cross_venue_id=None,
            question=row.get("question", ""),
            close_time=_parse_dt(row.get("endDate")),
            status=status,
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


_: type[MarketSource] = PolymarketSource  # protocol satisfaction check
