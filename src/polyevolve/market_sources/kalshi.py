"""Kalshi market source - placeholder.

Kalshi auth uses RSA-signed requests (key ID + private key PEM).
Implementation deferred to v0.1 once Polymarket-only loop is validated.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from polyevolve.contracts import Market, MarketSource, Resolution


class KalshiSource:
    name = "kalshi"

    def list_markets(self, filters: dict[str, Any]) -> Iterable[Market]:
        raise NotImplementedError("Kalshi source not yet implemented")

    def get_resolution(self, external_id: str) -> Resolution | None:
        raise NotImplementedError("Kalshi source not yet implemented")


_: type[MarketSource] = KalshiSource  # protocol satisfaction check
