"""Unit tests for the scout efficiency map (pure helpers, no network/DB).

Covers the squashing/thinness signals, per-market and per-category aggregation,
ranking, the raw-Gamma-market parsing, and a mocked end-to-end efficiency_map()
over a stubbed httpx client so the live scan path is exercised offline."""

from __future__ import annotations

import httpx
import pytest

from polyevolve.scout.efficiency import (
    MarketThinness,
    aggregate_category,
    efficiency_map,
    format_table,
    liquidity_thinness,
    market_thinness_from_raw,
    rank_categories,
    spread_thinness,
    volume_thinness,
)


# --- thinness signals -------------------------------------------------------
def test_liquidity_thinness_monotone() -> None:
    # More liquidity -> LESS thin (lower score), bounded in [0, 1].
    deep = liquidity_thinness(1_000_000.0)
    thin = liquidity_thinness(100.0)
    assert deep is not None and thin is not None
    assert 0.0 <= deep < thin <= 1.0


def test_volume_thinness_monotone() -> None:
    deep = volume_thinness(5_000_000.0)
    thin = volume_thinness(500.0)
    assert deep is not None and thin is not None
    assert deep < thin


def test_spread_thinness_scales_and_clamps() -> None:
    assert spread_thinness(0.0) == pytest.approx(0.0)
    assert spread_thinness(0.05) == pytest.approx(0.5)  # half of the 0.10 cap
    assert spread_thinness(1.0) == pytest.approx(1.0)  # clamped


def test_signals_none_passthrough() -> None:
    assert liquidity_thinness(None) is None
    assert volume_thinness(None) is None
    assert spread_thinness(None) is None
    assert spread_thinness(-1.0) is None  # negative spread is unusable


# --- per-market blend -------------------------------------------------------
def test_market_thinness_blend_is_mean_of_present_signals() -> None:
    m = MarketThinness.of("1", "q", liquidity=100.0, volume=500.0, spread=0.05)
    assert m.thinness is not None
    parts = [m.liquidity_thinness, m.volume_thinness, m.spread_thinness]
    assert all(p is not None for p in parts)
    assert m.thinness == pytest.approx(sum(p for p in parts if p is not None) / 3.0)


def test_market_thinness_skips_missing_signals() -> None:
    # only a spread present -> thinness equals the spread signal alone
    m = MarketThinness.of("1", "q", liquidity=None, volume=None, spread=0.05)
    assert m.liquidity_thinness is None and m.volume_thinness is None
    assert m.thinness == pytest.approx(0.5)


def test_market_thinness_all_missing_is_none() -> None:
    m = MarketThinness.of("1", "q", liquidity=None, volume=None, spread=None)
    assert m.thinness is None


# --- raw Gamma parsing ------------------------------------------------------
def test_market_thinness_from_raw_open() -> None:
    raw = {
        "id": "42",
        "question": "X by date?",
        "liquidityNum": 1000.0,
        "volumeNum": 2000.0,
        "bestBid": 0.30,
        "bestAsk": 0.34,
        "closed": False,
    }
    m = market_thinness_from_raw(raw)
    assert m is not None
    assert m.external_id == "42"
    assert m.spread == pytest.approx(0.04)  # ask - bid


def test_market_thinness_from_raw_skips_closed_and_idless() -> None:
    assert market_thinness_from_raw({"id": "1", "closed": True}) is None
    assert market_thinness_from_raw({"closed": False}) is None


def test_market_thinness_from_raw_string_numbers_and_spread_fallback() -> None:
    # Gamma sometimes serializes numbers as strings; bestBid/Ask absent -> use spread.
    raw = {"id": "7", "liquidity": "500", "volume": "800", "spread": "0.02", "closed": False}
    m = market_thinness_from_raw(raw)
    assert m is not None
    assert m.liquidity == pytest.approx(500.0)
    assert m.volume == pytest.approx(800.0)
    assert m.spread == pytest.approx(0.02)


# --- category aggregation + ranking ----------------------------------------
def test_aggregate_category_drops_unscored_and_keeps_leads() -> None:
    ms = [
        MarketThinness.of("a", "qa", 100.0, 500.0, 0.08),  # thin
        MarketThinness.of("b", "qb", 1_000_000.0, 5_000_000.0, 0.001),  # deep
        MarketThinness.of("c", "qc", None, None, None),  # unscored -> dropped
    ]
    row = aggregate_category("politics", ms, top_n=2)
    assert row.n_markets == 2  # c dropped
    assert row.thinness is not None
    # thinnest-first leads: a before b
    assert [m.external_id for m in row.thinnest] == ["a", "b"]


def test_rank_categories_thinnest_first_empty_last() -> None:
    deep = aggregate_category("crypto", [MarketThinness.of("x", "q", 1e7, 1e7, 0.001)])
    thin = aggregate_category("local", [MarketThinness.of("y", "q", 50.0, 50.0, 0.09)])
    empty = aggregate_category("dead", [MarketThinness.of("z", "q", None, None, None)])
    ranked = rank_categories([deep, empty, thin])
    assert ranked[0].category == "local"  # thinnest first
    assert ranked[-1].category == "dead"  # empty (None thinness) last


# --- end-to-end over a stubbed transport ------------------------------------
def _stub_client() -> httpx.Client:
    """A client whose /events returns one event with two markets, regardless of
    tag, so efficiency_map() runs fully offline."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "markets": [
                        {
                            "id": "m1",
                            "question": "thin one",
                            "liquidityNum": 100.0,
                            "volumeNum": 200.0,
                            "spread": 0.09,
                            "closed": False,
                        },
                        {
                            "id": "m2",
                            "question": "deep one",
                            "liquidityNum": 1_000_000.0,
                            "volumeNum": 1_000_000.0,
                            "spread": 0.001,
                            "closed": False,
                        },
                    ]
                }
            ],
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_efficiency_map_end_to_end_dedupes_across_tags() -> None:
    with _stub_client() as client:
        rows = efficiency_map(["politics", "geopolitics"], http=client)
    assert [r.category for r in rows] == ["politics", "geopolitics"]
    # markets are deduped by id across tags: politics claims both, geopolitics 0.
    politics = next(r for r in rows if r.category == "politics")
    geopolitics = next(r for r in rows if r.category == "geopolitics")
    assert politics.n_markets == 2
    assert geopolitics.n_markets == 0
    assert politics.thinness is not None


def test_format_table_smoke() -> None:
    with _stub_client() as client:
        rows = efficiency_map(["politics"], http=client)
    out = format_table(rows)
    assert "POLYMARKET SCOUT" in out
    assert "politics" in out
    assert "thin one" in out  # a concrete lead is rendered
