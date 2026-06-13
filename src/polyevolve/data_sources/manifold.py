"""Manifold Markets connector - a DIFFERENT free crowd's forecast, point-in-time.

The post-attention hunt (see project_polymarket_crossvenue_manifold): news/trends are
attention proxies the Polymarket crowd already prices, so the unpriced candidate is a
SEPARATE crowd. Manifold's globally-aware play-money forecasters cover obscure foreign
races Polymarket's US traders may misprice. Public API, no key.

Leakage-safe by construction: `point_in_time_prob` reconstructs the market's probability
AS OF a cutoff from the bet stream (each bet carries `createdTime` + `probAfter`), so a
historical backtest never sees post-cutoff (or post-resolution) information. Quality fields
(volume, unique bettors, bet count) are surfaced so callers can filter out the thin,
sometimes mis-resolved play-money markets - the known crux risk.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

__all__ = ["MarketLite", "market_bets", "point_in_time_prob", "search_markets"]

_BASE = "https://api.manifold.markets/v0"
_UA = {"User-Agent": "polyevolve-research/0.1"}


def _get(path: str, params: dict[str, Any] | None = None, *, timeout: float = 20.0) -> Any:
    url = f"{_BASE}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as fh:  # noqa: S310 - fixed https host
        return json.load(fh)


class MarketLite:
    """The fields we use from a Manifold market search hit."""

    __slots__ = (
        "id",
        "question",
        "prob",
        "is_resolved",
        "resolution",
        "volume",
        "n_bettors",
        "created_ms",
        "close_ms",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.id: str = raw.get("id", "")
        self.question: str = raw.get("question", "")
        self.prob: float | None = raw.get("probability")
        self.is_resolved: bool = bool(raw.get("isResolved", False))
        self.resolution: str | None = raw.get("resolution")  # "YES"/"NO"/"MKT"/"CANCEL"/...
        self.volume: float = float(raw.get("volume", 0.0) or 0.0)
        self.n_bettors: int = int(raw.get("uniqueBettorCount", 0) or 0)
        self.created_ms: int = int(raw.get("createdTime", 0) or 0)
        self.close_ms: int | None = raw.get("closeTime")

    def __repr__(self) -> str:
        return f"MarketLite({self.id} {self.question[:40]!r} res={self.resolution})"


def search_markets(term: str, *, limit: int = 8) -> list[MarketLite]:
    """Full-text market search. Binary markets only, most relevant first."""
    raw = _get("search-markets", {"term": term, "limit": limit})
    out = []
    for m in raw if isinstance(raw, list) else []:
        if m.get("outcomeType") in (None, "BINARY"):  # skip MULTIPLE_CHOICE/NUMERIC
            out.append(MarketLite(m))
    return out


def market_bets(contract_id: str, *, max_bets: int = 4000) -> list[dict[str, Any]]:
    """All bets for a market, oldest-first. Pages backward via the `before` cursor."""
    bets: list[dict[str, Any]] = []
    before: str | None = None
    while len(bets) < max_bets:
        params: dict[str, Any] = {"contractId": contract_id, "limit": 1000}
        if before:
            params["before"] = before
        page = _get("bets", params)
        if not page:
            break
        bets.extend(page)
        if len(page) < 1000:
            break
        before = page[-1].get("id")
    bets.sort(key=lambda b: b.get("createdTime", 0))
    return bets


def point_in_time_prob(
    contract_id: str, as_of: datetime, *, bets: list[dict[str, Any]] | None = None
) -> float | None:
    """Market probability as of `as_of` - the `probAfter` of the last bet at/before it.

    Returns None if the market had no bet at/before `as_of` (it did not exist or was
    untraded at our decision time, so it carries no usable signal). Pass pre-fetched
    `bets` to avoid re-hitting the API.
    """
    cutoff_ms = int(as_of.timestamp() * 1000)
    stream = bets if bets is not None else market_bets(contract_id)
    prob: float | None = None
    for b in stream:  # oldest-first; keep the latest probAfter <= cutoff
        if b.get("createdTime", 0) <= cutoff_ms and b.get("probAfter") is not None:
            prob = float(b["probAfter"])
        elif b.get("createdTime", 0) > cutoff_ms:
            break
    return prob
