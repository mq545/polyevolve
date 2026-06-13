"""The three plugin contracts the whole platform builds against.

These are `typing.Protocol` classes - duck-typed, no inheritance required. A
plugin satisfies a contract simply by having the right attributes/methods; it
does not (and must not) import the protocol to subclass it. Core depends only on
these shapes; concrete plugins live under markets/ connectors/ forecasters/ and
self-register via the decorators in registry.py.

Keep these signatures stable - downstream devs build against them verbatim.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from .types import Market, MarketFilter, OrderBook, Prediction, ResearchContext, Resolution


@runtime_checkable
class MarketSource(Protocol):
    """A venue that supplies markets, resolutions, and order books.

    `key` is the stable plugin id (e.g. "polymarket") used for registration and
    in global market ids. Methods return None when the venue has no answer
    (unknown id, not yet resolved, no book) rather than raising.
    """

    key: str

    def list_markets(self, filt: MarketFilter) -> Iterable[Market]: ...

    def get_resolution(self, external_id: str) -> Resolution | None: ...

    def order_book(self, external_id: str) -> OrderBook | None: ...


@runtime_checkable
class ResearchConnector(Protocol):
    """A point-in-time, price-free research signal source.

    `categories` declares which market categories this connector applies to;
    ("*",) means "all". `fetch` returns a structured, leakage-guarded payload;
    `render` turns that payload into prompt text (or "" for no-data, so the
    forecaster sees the gap explicitly rather than a silent empty string).
    """

    key: str
    categories: tuple[str, ...]

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]: ...

    def render(self, payload: dict[str, Any]) -> str: ...


@runtime_checkable
class Forecaster(Protocol):
    """A model/rule that turns (market + assembled research text) into a P(YES).

    `context` is the rendered, price-free research block. The forecaster is never
    shown the market price - only the question, resolution criteria, and context.
    """

    key: str

    def predict(self, market: Market, context: str) -> Prediction: ...
