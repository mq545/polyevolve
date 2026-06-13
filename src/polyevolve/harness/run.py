"""The one harness path every experiment takes: pull -> research -> predict -> score.

``run_experiment`` is the EXPLORATION-zone runner from ARCHITECTURE.md. It is a
THIN, registry-driven function - it names plugins by *key* and resolves them through
``core.registry``; it hardcodes no concrete market, connector, or forecaster. The
load-bearing discipline lives here:

  * Point-in-time. The :class:`~polyevolve.core.types.ResearchContext` carries an
    ``as_of`` cutoff = ``end_date - lead_days``; connectors must only return data
    strictly before it. ``lead_days`` is how far ahead of resolution we are
    forecasting (a longer lead = a harder, less-priced problem; see MEMORY).
  * Price-FREE. The market's crowd price is read ONLY to score edge after the
    forecast. It is never put in the ResearchContext or handed to the forecaster -
    the forecaster sees question + resolution criteria + rendered research, nothing
    about the price.
  * Connectors are filtered by category. A connector applies if its ``categories``
    contains the market's category or ``"*"``.

The result is structured (:class:`ExperimentResults` / :class:`MarketResult`), ready
for the rubric (``harness.rubric.evaluate``) and, separately, the forward ledger.
Computing edge here is in-sample scoring - it earns NO belief (that is the ledger's
job); the rubric will fail the forward/OOS checks on a raw run, which is correct.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from polyevolve.core.registry import discover, get_connector, get_forecaster, get_market
from polyevolve.core.types import (
    Confidence,
    Market,
    MarketFilter,
    Prediction,
    ResearchContext,
)


@dataclass(frozen=True)
class MarketResult:
    """The harness's outcome for ONE market: forecast vs the crowd price.

    ``crowd_prob`` is the market-implied P(YES) read AFTER forecasting, used only
    for scoring (``edge``). ``edge`` is positive when our forecast is closer to
    eventual reality than the crowd - but with no resolution yet it is the *signed
    distance from the crowd* (``|fair-crowd|`` with sign = direction of our bet),
    a pre-resolution divergence, not a graded Brier improvement.
    """

    external_id: str
    question: str
    category: str
    crowd_prob: float | None
    fair_prob: float
    confidence: Confidence
    reasoning: str
    research_text: str
    connectors_used: tuple[str, ...]
    as_of: datetime
    end_date: datetime | None
    # Crowd-divergence: signed gap our forecast takes vs the crowd (None if no price).
    divergence: float | None


@dataclass(frozen=True)
class ExperimentResults:
    """Everything one ``run_experiment`` produced, plus the rubric-feeding metadata.

    The per-market forecasts live in ``markets``. The remaining fields are the facts
    the rubric (``harness.rubric.evaluate``) consumes. The CONFIRMATION-gate fields
    (``forward_or_oos``, ``forward_confirmed``, ``observation_reviewed``) default to
    False - a raw EXPLORATION run has not earned them, so the rubric will (correctly)
    fail those checks until the forward ledger says otherwise.
    """

    market_source_key: str
    connector_keys: tuple[str, ...]
    forecaster_key: str
    lead_days: int
    markets: tuple[MarketResult, ...]

    # ---- rubric inputs (derivable from the run) ----
    n_resolved: int = 0
    edge: float = 0.0
    edge_se: float = 0.0
    edge_type: str | None = None
    # Order-book walk inputs for the executability check (best-level-first).
    executable_book: tuple[tuple[float, float], ...] = ()
    executable_side: str = "YES"
    executable_fair_prob: float = 0.5
    executable_size: float = 0.0
    # category-fit scores in [0, 1]
    inefficiency: float = 0.0
    advantage: float = 0.0
    data_machine_readable: bool = True
    n_traces: int = 0

    # ---- CONFIRMATION-gate flags (earned later; default False) ----
    forward_or_oos: bool = False
    forward_confirmed: bool = False
    fdr_aware: bool = False
    observation_reviewed: bool = False
    data_no_leakage: bool = True

    extras: dict[str, Any] = field(default_factory=dict)


def run_experiment(
    market_source_key: str,
    market_filter: MarketFilter,
    connector_keys: list[str],
    forecaster_key: str,
    lead_days: int,
    *,
    limit: int | None = None,
    edge_type: str | None = None,
) -> ExperimentResults:
    """Run one experiment end-to-end through the registry and return structured results.

    Args:
        market_source_key: registered MarketSource key (e.g. ``"polymarket"``).
        market_filter: declarative query handed to ``list_markets``.
        connector_keys: registered ResearchConnector keys to gather, in order. A
            connector is skipped for a market whose category it does not declare.
        forecaster_key: registered Forecaster key. Called PRICE-FREE.
        lead_days: forecast horizon; ``as_of = end_date - lead_days`` is the
            point-in-time leakage cutoff handed to connectors.
        limit: cap on markets processed (None = all the filter yields).
        edge_type: the claimed edge kind for rubric check 5 (predictive/structural/
            latency/calibration/resolution-artifact); None until the experiment
            declares one.

    Returns:
        ExperimentResults with one MarketResult per market and the rubric-feeding
        aggregate metadata.
    """
    discover()  # idempotent - ensures plugins are registered before lookup.

    source = get_market(market_source_key)()
    forecaster = get_forecaster(forecaster_key)()
    connectors = [(key, get_connector(key)()) for key in connector_keys]

    results: list[MarketResult] = []
    divergences: list[float] = []

    for i, market in enumerate(source.list_markets(market_filter)):
        if limit is not None and i >= limit:
            break

        as_of = _as_of_for(market, lead_days)
        ctx = _context_for(market, as_of)

        research_text, used = _gather(connectors, ctx, market.category)

        # PRICE-FREE: the forecaster sees only question + criteria + research.
        prediction: Prediction = forecaster.predict(market, research_text)

        # Crowd price is read AFTER the forecast, for scoring ONLY.
        crowd = _crowd_prob(market)
        divergence = None if crowd is None else prediction.prob_yes - crowd
        if divergence is not None:
            divergences.append(divergence)

        results.append(
            MarketResult(
                external_id=market.external_id,
                question=market.question,
                category=market.category,
                crowd_prob=crowd,
                fair_prob=prediction.prob_yes,
                confidence=prediction.confidence,
                reasoning=prediction.reasoning,
                research_text=research_text,
                connectors_used=used,
                as_of=as_of,
                end_date=market.end_date,
                divergence=divergence,
            )
        )

    edge, edge_se = _aggregate_divergence(divergences)

    return ExperimentResults(
        market_source_key=market_source_key,
        connector_keys=tuple(connector_keys),
        forecaster_key=forecaster_key,
        lead_days=lead_days,
        markets=tuple(results),
        n_resolved=len(divergences),
        edge=edge,
        edge_se=edge_se,
        edge_type=edge_type,
        n_traces=len(results),
    )


# --------------------------------------------------------------------------- #
# helpers - all pure / side-effect-free except registry-driven plugin calls.
# --------------------------------------------------------------------------- #
def _as_of_for(market: Market, lead_days: int) -> datetime:
    """Point-in-time cutoff = end_date - lead_days. Falls back to now-lead_days
    when the market has no end_date so connectors still get a sane horizon.
    """
    base = market.end_date or datetime.now(UTC)
    return base - timedelta(days=lead_days)


def _context_for(market: Market, as_of: datetime) -> ResearchContext:
    """Assemble the PRICE-FREE ResearchContext handed to connectors/forecaster."""
    return ResearchContext(
        question=market.question,
        as_of=as_of,
        tags=market.tags,
        category=market.category,
        market=market,
    )


def _applies(categories: tuple[str, ...], category: str) -> bool:
    """A connector applies to a market category if it lists it or the wildcard."""
    return "*" in categories or category in categories


def _gather(
    connectors: list[tuple[str, Any]],
    ctx: ResearchContext,
    category: str,
) -> tuple[str, tuple[str, ...]]:
    """Fetch + render each applicable connector, point-in-time, and join the text.

    Returns the assembled research block and the keys that actually contributed
    (non-empty render). A connector whose ``render`` returns "" is recorded as
    no-data and omitted from the block, so the forecaster sees the gap as absence
    rather than noise.
    """
    blocks: list[str] = []
    used: list[str] = []
    for key, conn in connectors:
        if not _applies(getattr(conn, "categories", ()), category):
            continue
        payload = conn.fetch(ctx)
        text = conn.render(payload)
        if text:
            blocks.append(f"## {key}\n{text}")
            used.append(key)
    return "\n\n".join(blocks), tuple(used)


def _crowd_prob(market: Market) -> float | None:
    """Extract the crowd-implied P(YES) from venue metadata, for SCORING only.

    Polymarket stores ``outcomePrices`` as a JSON-encoded string ``'["y","n"]'``;
    the first element is the YES price. Returns None when no usable price is
    present so divergence is simply not computed for that market.
    """
    raw = market.metadata.get("outcomePrices")
    prices = _parse_price_list(raw)
    if not prices:
        return None
    yes = prices[0]
    if not math.isfinite(yes) or not (0.0 <= yes <= 1.0):
        return None
    return yes


def _parse_price_list(value: Any) -> list[float] | None:
    """Coerce an outcomePrices value (JSON string or list) into floats, or None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for v in value:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


def _aggregate_divergence(divergences: list[float]) -> tuple[float, float]:
    """Mean signed crowd-divergence and the standard error of that mean.

    This is an in-sample pre-resolution statistic (we have no graded outcomes in a
    raw run), fed to the rubric's power check so it can refuse a verdict on a thin
    sample. With <2 points the SE is undefined and returned as 0.0, which the
    power check reads as "no usable standard error" => underpowered.
    """
    n = len(divergences)
    if n == 0:
        return 0.0, 0.0
    mean = sum(divergences) / n
    if n < 2:
        return mean, 0.0
    var = sum((d - mean) ** 2 for d in divergences) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se
