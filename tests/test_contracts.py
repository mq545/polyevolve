"""Smoke tests for the contracts module."""

from datetime import UTC, datetime

from polyevolve.contracts import Market, Prediction, Resolution


def test_market_constructs() -> None:
    m = Market(
        venue="polymarket",
        external_id="abc123",
        cross_venue_id=None,
        question="Will X happen?",
        close_time=datetime(2026, 1, 1, tzinfo=UTC),
        status="active",
        metadata={"tag": "test"},
    )
    assert m.venue == "polymarket"
    assert m.external_id == "abc123"


def test_resolution_constructs() -> None:
    r = Resolution(
        venue="polymarket",
        external_id="abc123",
        outcome="YES",
        resolved_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert r.outcome == "YES"


def test_prediction_constructs() -> None:
    p = Prediction(
        market_venue="polymarket",
        market_external_id="abc123",
        agent_name="test_agent",
        model_name="claude-sonnet-4-6",
        probability_yes=0.65,
        confidence=0.7,
        reasoning="base rate 60%, slight update on recent polls",
        key_factors=["incumbent advantage"],
        uncertainty_drivers=["polling sparsity"],
    )
    assert 0 <= p.probability_yes <= 1
    assert p.created_at is not None
