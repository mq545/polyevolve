"""Polymarket historical price lookup - market-implied probability at a past date.

Needed to build longer-lead backtest snapshots. The fp_v1/fp_recent snapshots
priced markets only 7 days before resolution, where the market is near-omniscient
(brier 0.11, 45% of prices already pinned to extremes) - no edge is possible and
the local-news thesis is untestable. We need the price 30/60/90 days out, where
the market is genuinely uncertain.

Two public, keyless APIs (verified live 2026-06-01):
  - Gamma (gamma-api.polymarket.com/markets/{id}): metadata. Maps our numeric
    market id -> conditionId, outcomes[], clobTokenIds[] (positional, 1:1).
  - CLOB (clob.polymarket.com/prices-history): the YES token's price time-series.

CRITICAL granularity trap (GitHub issues #189/#216): for RESOLVED markets,
interval=max returns only ~12h points and fine fidelity returns empty. Querying
with an explicit startTs/endTs WINDOW works at fine fidelity even on resolved
markets - so we always pass a window, never interval=max.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
_TIMEOUT_S = 25
# Half-window around the target instant to pull samples from, then snap to the
# nearest one. 6h tolerates sparse trading on quiet days without drifting far.
_WINDOW_S = 6 * 3600


def yes_token_id(market_id: str, http: httpx.Client) -> str | None:
    """Resolve the YES outcome token id for a numeric Polymarket market id."""
    try:
        resp = http.get(f"{GAMMA}/markets/{market_id}", timeout=_TIMEOUT_S)
        resp.raise_for_status()
        m: dict[str, Any] = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("gamma lookup failed for market %s: %r", market_id, exc)
        return None
    outs_raw, toks_raw = m.get("outcomes"), m.get("clobTokenIds")
    if not outs_raw or not toks_raw:
        return None
    try:
        outcomes = json.loads(outs_raw)  # Gamma returns these JSON-encoded
        tokens = json.loads(toks_raw)
    except (ValueError, TypeError):
        return None
    # Match the "Yes" outcome by name; don't blindly trust index 0.
    idx = outcomes.index("Yes") if "Yes" in outcomes else 0
    return str(tokens[idx]) if idx < len(tokens) else None


def yes_price_at(
    market_id: str, target: datetime, http: httpx.Client | None = None
) -> float | None:
    """YES-implied probability for `market_id` nearest `target` instant.

    Returns None if the market has no CLOB history at that time (e.g. the market
    didn't exist yet - created < lead days before resolution, or a pre-CLOB AMM
    market). The caller should skip such markets for a clean fixed-lead snapshot.
    """
    own = http is None
    http = http or httpx.Client()
    try:
        tok = yes_token_id(market_id, http)
        if not tok:
            return None
        ts = int(target.replace(tzinfo=UTC).timestamp())
        try:
            resp = http.get(
                f"{CLOB}/prices-history",
                params={
                    "market": tok,
                    "startTs": ts - _WINDOW_S,
                    "endTs": ts + _WINDOW_S,
                    "fidelity": 60,  # 60-min samples; window makes this work on resolved mkts
                },
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            history = resp.json().get("history", [])
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("clob price-history failed for market %s: %r", market_id, exc)
            return None
        if not history:
            return None
        nearest = min(history, key=lambda x: abs(x["t"] - ts))
        return float(nearest["p"])
    finally:
        if own:
            http.close()
