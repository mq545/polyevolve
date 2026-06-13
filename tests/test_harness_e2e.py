"""End-to-end harness test: a fake plugin set, discovered through the registry,
run through ``harness.run_experiment`` with the baseline forecaster - NO LLM, no
network, no DB. This is the integration check for the platform's core seam:

  register (decorator) -> discover() -> run_experiment() -> ExperimentResults
  -> rubric.evaluate().

It asserts the load-bearing guarantees: plugins self-register on import, the
real on-disk plugins are discovered, the forecaster is called PRICE-FREE and
point-in-time, no-data connectors are omitted, the crowd price is read only for
scoring, and the rubric runs over the result.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from polyevolve.core import registry
from polyevolve.core.interfaces import Forecaster, MarketSource, ResearchConnector
from polyevolve.core.registry import (
    register_connector,
    register_forecaster,
    register_market,
)
from polyevolve.core.types import (
    Market,
    MarketFilter,
    OrderBook,
    Prediction,
    ResearchContext,
    Resolution,
)
from polyevolve.harness.rubric import evaluate
from polyevolve.harness.run import run_experiment

# A market whose crowd price (outcomePrices) is present so divergence is scored.
_FAKE_MARKET = Market(
    external_id="fake-1",
    question="Will the fake candidate win the fake race?",
    category="politics",
    tags=("politics", "fake"),
    resolution_criteria="Resolves YES if the fake candidate wins.",
    end_date=datetime(2026, 12, 31, tzinfo=UTC),
    metadata={"outcomePrices": '["0.40", "0.60"]'},
)


@register_market("_fake_market")
class _FakeMarket:
    key = "_fake_market"

    def list_markets(self, filt: MarketFilter):  # type: ignore[no-untyped-def]
        return [_FAKE_MARKET]

    def get_resolution(self, external_id: str) -> Resolution | None:
        return None

    def order_book(self, external_id: str) -> OrderBook | None:
        return None


@register_connector("_fake_connector")
class _FakeConnector:
    key = "_fake_connector"
    categories = ("politics",)
    # Records the as_of it was handed, to prove point-in-time wiring.
    last_as_of: datetime | None = None

    def fetch(self, ctx: ResearchContext) -> dict:  # type: ignore[type-arg]
        type(self).last_as_of = ctx.as_of
        # Leak guard: the price must NOT be visible to research in a way the
        # forecaster sees. We assert separately that it never reaches the text.
        return {"note": "fake research", "as_of": ctx.as_of.isoformat()}

    def render(self, payload: dict) -> str:  # type: ignore[type-arg]
        return f"FAKE RESEARCH as_of={payload['as_of']}"


@register_connector("_fake_empty_connector")
class _FakeEmptyConnector:
    key = "_fake_empty_connector"
    categories = ("*",)

    def fetch(self, ctx: ResearchContext) -> dict:  # type: ignore[type-arg]
        return {}

    def render(self, payload: dict) -> str:  # type: ignore[type-arg]
        return ""  # no-data => must be omitted from the assembled block


@register_forecaster("_fake_spy_forecaster")
class _SpyForecaster:
    """Records the context text it was handed so the test can assert price-freedom."""

    key = "_fake_spy_forecaster"
    last_context: str | None = None

    def predict(self, market: Market, context: str) -> Prediction:
        type(self).last_context = context
        return Prediction(prob_yes=0.7, confidence="medium", reasoning="spy")


def test_fake_plugins_register_and_satisfy_protocols() -> None:
    assert isinstance(_FakeMarket(), MarketSource)
    assert isinstance(_FakeConnector(), ResearchConnector)
    assert isinstance(_SpyForecaster(), Forecaster)


def test_discover_finds_real_and_fake_plugins() -> None:
    registry.discover()  # idempotent; imports every on-disk plugin module
    markets = registry.all_markets()
    connectors = registry.all_connectors()
    forecasters = registry.all_forecasters()
    # Real plugins from the package show up...
    assert "polymarket" in markets
    assert "baseline" in forecasters
    assert "news" in connectors
    # ...and the fakes registered in this module are present too.
    assert "_fake_market" in markets
    assert "_fake_connector" in connectors
    assert "_fake_spy_forecaster" in forecasters


def test_run_experiment_end_to_end_baseline_no_llm() -> None:
    """The whole harness path with the BASELINE forecaster - no LLM, no network."""
    results = run_experiment(
        market_source_key="_fake_market",
        market_filter=MarketFilter(category="politics"),
        connector_keys=["_fake_connector", "_fake_empty_connector"],
        forecaster_key="baseline",
        lead_days=30,
        limit=1,
        edge_type="predictive",
    )

    assert len(results.markets) == 1
    m = results.markets[0]
    assert m.external_id == "fake-1"
    assert m.fair_prob == 0.5  # baseline is a flat 0.5
    # Crowd price (0.40) read AFTER the forecast, only for scoring.
    assert m.crowd_prob == pytest.approx(0.40)
    assert m.divergence == pytest.approx(0.10)  # 0.50 - 0.40
    # No-data connector omitted; only the real one contributed.
    assert m.connectors_used == ("_fake_connector",)
    assert "FAKE RESEARCH" in m.research_text
    assert "_fake_empty_connector" not in m.research_text

    # Point-in-time: as_of = end_date - lead_days.
    assert m.as_of == datetime(2026, 12, 1, tzinfo=UTC)
    assert _FakeConnector.last_as_of == datetime(2026, 12, 1, tzinfo=UTC)


def test_forecaster_is_called_price_free() -> None:
    """The forecaster's context must never contain the crowd price string."""
    run_experiment(
        market_source_key="_fake_market",
        market_filter=MarketFilter(category="politics"),
        connector_keys=["_fake_connector"],
        forecaster_key="_fake_spy_forecaster",
        lead_days=30,
        limit=1,
    )
    ctx = _SpyForecaster.last_context
    assert ctx is not None
    # The price components and outcomePrices key must NOT leak into the prompt text.
    assert "0.40" not in ctx
    assert "0.60" not in ctx
    assert "outcomePrices" not in ctx


def test_rubric_runs_over_harness_result() -> None:
    """A raw EXPLORATION run is correctly FAILED by the CONFIRMATION-gate checks."""
    results = run_experiment(
        market_source_key="_fake_market",
        market_filter=MarketFilter(category="politics"),
        connector_keys=["_fake_connector"],
        forecaster_key="baseline",
        lead_days=30,
        limit=1,
    )
    report = evaluate(results)
    names = {c.name for c in report.checks}
    # All 8 architecture checks are present.
    assert len(report.checks) == 8
    assert {"power", "out_of_sample", "executable", "multiple_testing"} <= names
    # A single-market in-sample run cannot clear the gate.
    assert not report.passed
    failed = {c.name for c in report.failures}
    assert "out_of_sample" in failed  # forward-only flag is False by default
    assert "power" in failed  # n=1 is underpowered
