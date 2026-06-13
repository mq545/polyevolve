"""Tests for the model cutoff registry + backtest-cleanliness logic."""

from datetime import UTC, datetime

from polyevolve.models.cutoffs import get_cutoff, is_clean_for_backtest


def test_known_model_resolves() -> None:
    c = get_cutoff("ollama/qwen2.5:14b")
    assert c is not None
    assert c.confidence == "inferred"


def test_tag_drift_resolves_to_base() -> None:
    # a differently-tagged qwen2.5 still resolves
    assert get_cutoff("ollama/qwen2.5:14b-instruct-q4") is not None


def test_unknown_model_returns_none() -> None:
    assert get_cutoff("some/unknown-model") is None


def test_market_after_cutoff_plus_margin_is_clean() -> None:
    # qwen2.5 cutoff 2023-12-31 + 90d margin -> well after = clean
    assert is_clean_for_backtest("ollama/qwen2.5:14b", datetime(2024, 6, 1, tzinfo=UTC))


def test_market_before_cutoff_is_contaminated() -> None:
    assert not is_clean_for_backtest("ollama/qwen2.5:14b", datetime(2023, 5, 1, tzinfo=UTC))


def test_market_inside_safety_margin_is_contaminated() -> None:
    # 2024-02-01 is after 2023-12-31 but inside the 90d margin -> not clean
    assert not is_clean_for_backtest("ollama/qwen2.5:14b", datetime(2024, 2, 1, tzinfo=UTC))


def test_unknown_model_fails_closed() -> None:
    assert not is_clean_for_backtest("some/unknown-model", datetime(2099, 1, 1, tzinfo=UTC))
