"""Tests for the financial-market movement source (leading indicator for politics).

Mostly pure-function tests (no network): keyword->instrument routing, country->
ticker mapping, the %-change calc, the as_of leakage filter, and render states.
One mocked fetch() proves point-in-time end-to-end. One live test hits the real
Yahoo chart endpoint for a single known ticker+window (skipped offline)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from polyevolve.data_sources.finmarkets import (
    FinancialMarketsSource,
    _has_conflict_keyword,
    _parse_closes,
    _pct_change,
    route_instruments,
)

CUTOFF = datetime(2026, 2, 21, tzinfo=UTC)


def _chart(bars: list[tuple[str, float | None]]) -> dict[str, Any]:
    """Build a Yahoo chart payload from (YYYY-MM-DD, close) tuples."""
    timestamps = [int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp()) for d, _ in bars]
    closes = [c for _, c in bars]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {"quote": [{"close": closes}]},
                }
            ],
            "error": None,
        }
    }


# --- conflict keyword routing ----------------------------------------------


def test_conflict_keywords_route_to_macro_basket() -> None:
    routed = route_instruments("Will the US strike Iran before July?", [])
    tickers = [t for t, _ in routed]
    assert tickers == ["ITA", "CL=F", "GC=F"]


def test_conflict_keyword_variants_detected() -> None:
    assert _has_conflict_keyword("Will Israel invade Lebanon?")  # invade
    assert _has_conflict_keyword("missile strikes on the capital")  # missile/strike
    assert _has_conflict_keyword("Will there be a ceasefire?")  # ceasefire
    assert _has_conflict_keyword("Hormuz closed to shipping?")  # hormuz
    assert not _has_conflict_keyword("Who wins the mayoral election?")


def test_no_conflict_no_country_is_no_data() -> None:
    # High-precision/low-recall: an unmappable market maps to nothing.
    assert route_instruments("Will the budget pass committee?", ["politics"]) == []


# --- country -> ticker mapping ---------------------------------------------


def test_country_maps_to_index_and_fx() -> None:
    routed = route_instruments("Who wins the Italian general election?", ["italy"])
    tickers = [t for t, _ in routed]
    assert tickers == ["FTSEMIB.MI", "USDEUR=X"]


def test_country_mapping_covers_curated_set() -> None:
    cases = {
        "netherlands": ("^AEX", "USDEUR=X"),
        "india": ("^BSESN", "USDINR=X"),
        "japan": ("^N225", "USDJPY=X"),
        "hungary": ("^BUX.BD", "USDHUF=X"),
        "israel": ("^TA125.TA", "USDILS=X"),
        "mexico": ("^MXX", "USDMXN=X"),
        "cyprus": ("^STOXX50E", "USDEUR=X"),
        "canada": ("^GSPTSE", "USDCAD=X"),
    }
    for tag, (idx, fx) in cases.items():
        tickers = [t for t, _ in route_instruments("Who wins the election?", [tag])]
        assert tickers == [idx, fx], tag


def test_unmapped_country_skipped() -> None:
    # A country with no reliable free ticker (e.g. cambodia) maps to nothing.
    assert route_instruments("Who wins the Cambodian election?", ["cambodia"]) == []


def test_conflict_and_country_combined_and_deduped() -> None:
    # Rule 3: both sets, conflict first, then per-country; no duplicate tickers.
    routed = route_instruments("Will Israel strike Iran?", ["israel"])
    tickers = [t for t, _ in routed]
    assert tickers[:3] == ["ITA", "CL=F", "GC=F"]
    assert "^TA125.TA" in tickers and "USDILS=X" in tickers
    assert len(tickers) == len(set(tickers))


# --- %-change calc ---------------------------------------------------------


def test_pct_change_basic() -> None:
    bars = [
        (datetime(2026, 1, 5, tzinfo=UTC), 100.0),
        (datetime(2026, 1, 6, tzinfo=UTC), 110.0),
        (datetime(2026, 1, 7, tzinfo=UTC), 107.6),
    ]
    assert _pct_change(bars) == pytest.approx(0.076)


def test_pct_change_needs_two_bars() -> None:
    assert _pct_change([(datetime(2026, 1, 5, tzinfo=UTC), 100.0)]) is None
    assert _pct_change([]) is None


def test_pct_change_zero_first_close() -> None:
    bars = [(datetime(2026, 1, 5, tzinfo=UTC), 0.0), (datetime(2026, 1, 6, tzinfo=UTC), 5.0)]
    assert _pct_change(bars) is None


# --- as_of leakage filter --------------------------------------------------


def test_parse_closes_excludes_bar_at_or_after_as_of() -> None:
    payload = _chart(
        [
            ("2026-02-19", 100.0),  # before as_of -> kept
            ("2026-02-20", 110.0),  # before as_of -> kept
            ("2026-02-21", 999.0),  # == as_of -> EXCLUDED (leakage)
            ("2026-02-22", 999.0),  # > as_of  -> EXCLUDED
        ]
    )
    bars = _parse_closes(payload, CUTOFF)
    assert all(dt < CUTOFF for dt, _ in bars)
    assert [c for _, c in bars] == [100.0, 110.0]


def test_parse_closes_drops_null_closes() -> None:
    payload = _chart([("2026-02-18", None), ("2026-02-19", 50.0), ("2026-02-20", 55.0)])
    bars = _parse_closes(payload, CUTOFF)
    assert [c for _, c in bars] == [50.0, 55.0]


def test_parse_closes_empty_result() -> None:
    assert _parse_closes({"chart": {"result": []}}, CUTOFF) == []
    assert _parse_closes({"chart": {"result": None}}, CUTOFF) == []


# --- render states ---------------------------------------------------------


def test_render_error_state() -> None:
    src = FinancialMarketsSource(http=_StubClient({}))
    out = src.render({"error": "empty_question"})
    assert out.startswith("[SOURCE ERROR] markets fetch failed")


def test_render_no_data_state() -> None:
    src = FinancialMarketsSource(http=_StubClient({}))
    assert src.render({"instruments": [], "as_of": CUTOFF.isoformat()}) == (
        "(No mapped financial instrument for this market)"
    )
    # All-failed also renders no-data.
    out = src.render(
        {"instruments": [{"ticker": "ITA", "label": "x", "found": False}], "as_of": None}
    )
    assert out == "(No mapped financial instrument for this market)"


def test_render_found_state() -> None:
    src = FinancialMarketsSource(http=_StubClient({}))
    out = src.render(
        {
            "instruments": [
                {"label": "defense ETF (ITA)", "found": True, "pct_change": 0.076},
                {"label": "WTI oil (CL=F)", "found": True, "pct_change": 0.138},
                {"label": "gold (GC=F)", "found": True, "pct_change": 0.140},
                {"label": "skip", "found": False},
            ],
            "as_of": "2026-02-21T00:00:00+00:00",
            "lookback_days": 30,
        }
    )
    assert out == (
        "Financial-market signal (pre-2026-02-21, 30d): "
        "defense ETF (ITA) +7.6%, WTI oil (CL=F) +13.8%, gold (GC=F) +14.0%"
    )


# --- mocked fetch: point-in-time end to end --------------------------------


class _StubClient:
    """Minimal httpx.Client stand-in. Maps ticker -> Yahoo chart payload.

    A ticker not in the map returns 404 (unknown symbol), mirroring the live API.
    """

    def __init__(self, by_ticker: dict[str, dict[str, Any]]) -> None:
        self._by_ticker = by_ticker
        self.requested_urls: list[str] = []

    def get(self, url: str) -> httpx.Response:
        self.requested_urls.append(url)
        req = httpx.Request("GET", url)
        for ticker, payload in self._by_ticker.items():
            from urllib.parse import quote

            if f"/chart/{quote(ticker, safe='')}?" in url:
                return httpx.Response(200, json=payload, request=req)
        return httpx.Response(404, json={"chart": {"result": None}}, request=req)


def test_fetch_is_point_in_time_and_computes_change() -> None:
    # Series ends with a bar AT as_of (huge spike) that MUST be excluded, so the
    # last close used is the pre-as_of one.
    ita = _chart(
        [
            ("2026-01-22", 100.0),
            ("2026-02-19", 105.0),
            ("2026-02-20", 107.6),
            ("2026-02-21", 999.0),  # == as_of, MUST be excluded
        ]
    )
    src = FinancialMarketsSource(
        http=_StubClient(
            {"ITA": ita, "CL=F": _chart([("2026-01-22", 70.0), ("2026-02-20", 79.66)])}
        )
    )
    payload = src.fetch({"question": "Will the US strike Iran?", "as_of": CUTOFF, "tags": []})
    found = {i["label"]: i for i in payload["instruments"] if i.get("found")}
    assert found["defense ETF (ITA)"]["pct_change"] == pytest.approx(0.076)
    # gold (GC=F) had no stubbed data -> 404 -> not found (fail-soft, others survive)
    assert any(not i["found"] for i in payload["instruments"])
    out = src.render(payload)
    assert "defense ETF (ITA) +7.6%" in out


def test_fetch_no_mapping_is_clean_no_data() -> None:
    src = FinancialMarketsSource(http=_StubClient({}))
    payload = src.fetch({"question": "Will the budget pass?", "as_of": CUTOFF, "tags": []})
    assert "error" not in payload
    assert payload["instruments"] == []
    assert src.render(payload) == "(No mapped financial instrument for this market)"


def test_fetch_empty_question_errors() -> None:
    src = FinancialMarketsSource(http=_StubClient({}))
    assert src.fetch({"question": "", "as_of": CUTOFF, "tags": []})["error"] == "empty_question"


def test_fetch_period2_strictly_before_as_of() -> None:
    # The requested period2 must be < as_of (we request as_of - 1 day).
    stub = _StubClient({"ITA": _chart([("2026-02-19", 1.0)])})
    src = FinancialMarketsSource(http=stub)
    src.fetch({"question": "Will the US strike Iran?", "as_of": CUTOFF, "tags": []})
    url = next(u for u in stub.requested_urls if "/chart/ITA?" in u)
    import re

    p2 = int(re.search(r"period2=(\d+)", url).group(1))  # type: ignore[union-attr]
    assert p2 < int(CUTOFF.timestamp())


# --- live sanity check (one known ticker) ----------------------------------


@pytest.mark.skipif(
    os.environ.get("POLYEVOLVE_LIVE_TESTS") != "1",
    reason="live Yahoo Finance API test; set POLYEVOLVE_LIVE_TESTS=1 to run",
)
def test_live_single_ticker() -> None:
    # Human-verified: ITA +7.6% over 2026-01-05..02-21. We fetch with as_of just
    # after the window and confirm we get closes, all strictly before as_of.
    src = FinancialMarketsSource(lookback_days=60)
    payload = src.fetch(
        {
            "question": "Will the US strike Iran?",
            "as_of": datetime(2026, 2, 22, tzinfo=UTC),
            "tags": [],
        }
    )
    assert "error" not in payload
    found = [i for i in payload["instruments"] if i.get("found")]
    assert found, "expected at least one resolved instrument (ITA/oil/gold)"
    ita = next((i for i in found if i["ticker"] == "ITA"), None)
    assert ita is not None and ita["n_bars"] >= 2
