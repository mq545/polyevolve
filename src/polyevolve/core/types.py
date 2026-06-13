"""Core value types for the market-experiment platform.

All types are frozen dataclasses - immutable, hashable where their fields allow,
and trivially shareable across plugins. Plugins import these; core never imports
a plugin (see registry.py / ARCHITECTURE.md). Keep this module dependency-free
(stdlib only) so every plugin can rely on it without a heavy import graph.

The shapes are the contract the whole funnel builds against:
  Market           - a normalized, venue-agnostic question + metadata.
  MarketFilter     - declarative query passed to a MarketSource.list_markets.
  Resolution       - the graded outcome of a market (YES/NO + when).
  OrderBook        - top-of-book depth, for the executability (book-walk) check.
  ResearchContext  - point-in-time, PRICE-FREE inputs handed to a connector.
  Prediction       - a forecaster's calibrated P(YES) + confidence + reasoning.
  Bet              - a logged forward paper bet (our fair vs the crowd price).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

# Binary outcome convention used across the platform.
Outcome = Literal["YES", "NO"]
# Coarse confidence band a forecaster reports alongside its probability.
Confidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Market:
    """A normalized prediction-market question, venue-agnostic.

    `external_id` is unique within a venue; combine with the source key for a
    global id. `tags` carry venue taxonomy (e.g. country/category slugs) used by
    connectors to decide applicability. `metadata` holds the raw venue-specific
    extras (slug, prices, volume, ...) that core does not model explicitly.
    """

    external_id: str
    question: str
    category: str
    tags: tuple[str, ...]
    resolution_criteria: str
    end_date: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketFilter:
    """Declarative query for MarketSource.list_markets.

    All fields are optional/narrowing: an empty filter means "everything the
    source will give". `tags` matches markets carrying ANY of the listed tags.
    `resolves_within_days` bounds end_date relative to the source's notion of now.
    """

    category: str | None = None
    tags: tuple[str, ...] = ()
    open_only: bool = True
    resolves_within_days: int | None = None


@dataclass(frozen=True)
class Resolution:
    """The settled outcome of a market."""

    external_id: str
    outcome: Outcome
    resolved_at: datetime


@dataclass(frozen=True)
class OrderBook:
    """Top-of-book depth for one market side each.

    `bids`/`asks` are (price, size) levels. Convention: bids sorted best (highest
    price) first, asks best (lowest price) first - the order needed to walk the
    book for the executability check. Prices are YES-share prices in [0, 1].
    """

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ResearchContext:
    """Point-in-time, PRICE-FREE inputs assembled for a connector / forecaster.

    `as_of` is the leakage cutoff: connectors must only return data strictly
    before it. `market` is the full Market for connectors that need its metadata;
    `question`/`tags`/`category` are surfaced directly for the common case.
    The market PRICE is deliberately absent - the forecaster is never shown it.
    """

    question: str
    as_of: datetime
    tags: tuple[str, ...]
    category: str
    market: Market


@dataclass(frozen=True)
class Prediction:
    """A forecaster's output: calibrated P(YES) + confidence + reasoning trace.

    `prob_yes` is in [0, 1]. `reasoning` is stored and audited (the observation
    review step reads these), so it should be the real rationale, not a label.
    """

    prob_yes: float
    confidence: Confidence
    reasoning: str


@dataclass(frozen=True)
class Bet:
    """A forward paper bet: our fair probability vs the crowd price, logged now
    and graded at resolution. The un-foolable unit of belief (see ARCHITECTURE).
    """

    external_id: str
    question: str
    category: str
    crowd_prob: float
    fair_prob: float
    confidence: Confidence
    reasoning: str
    logged_at: datetime
    resolution_date: datetime | None = None
