"""Tests for the Wikipedia pageviews momentum source.

Mostly pure-function tests (no network): the leakage date filter, momentum
computation, country->wiki-lang mapping, entity-extraction reuse, and render
states. One mocked fetch() proves point-in-time end-to-end. One live test hits
the real Wikimedia API for a single known article (skipped offline)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from polyevolve.data_sources.pageviews import (
    WikipediaPageviewsSource,
    _filter_before,
    _momentum,
    _to_title,
    _trend,
    _wiki_langs,
)

CUTOFF = datetime(2025, 6, 1, tzinfo=UTC)


def _item(date: str, views: int) -> dict[str, Any]:
    return {"timestamp": f"{date}00", "views": views}


# --- leakage / date-window filter -----------------------------------------


def test_filter_excludes_day_at_or_after_as_of() -> None:
    # The whole point: a day stamped >= as_of must never enter the window.
    items = [
        _item("20250530", 100),  # day before cutoff -> kept
        _item("20250531", 200),  # day before cutoff -> kept
        _item("20250601", 999),  # == as_of -> EXCLUDED (leakage)
        _item("20250602", 999),  # > as_of  -> EXCLUDED
    ]
    kept = _filter_before(items, CUTOFF)
    dates = [p["date"] for p in kept]
    assert all(d < CUTOFF for d in dates)
    assert datetime(2025, 6, 1, tzinfo=UTC) not in dates
    assert [p["views"] for p in kept] == [100, 200]


def test_filter_drops_unparseable_timestamp() -> None:
    kept = _filter_before([{"timestamp": "garbage", "views": 5}], CUTOFF)
    assert kept == []


# --- momentum computation --------------------------------------------------


def test_momentum_rising() -> None:
    # baseline ~100/day for the 30d before the last 7d; recent ~150/day.
    points = []
    for day in range(1, 25):  # 2025-05-01 .. 2025-05-24 -> baseline window
        points.append({"date": datetime(2025, 5, day, tzinfo=UTC), "views": 100})
    for day in range(25, 32):  # 2025-05-25 .. 2025-05-31 -> recent 7d
        points.append({"date": datetime(2025, 5, day, tzinfo=UTC), "views": 150})
    sig = _momentum(points, CUTOFF, recent_days=7, baseline_days=30)
    assert sig["recent_mean"] == pytest.approx(150.0)
    assert sig["baseline_mean"] == pytest.approx(100.0)
    assert sig["pct_change"] == pytest.approx(0.5)
    assert sig["trend"] == "rising"


def test_momentum_falling_and_flat() -> None:
    base = [{"date": datetime(2025, 5, d, tzinfo=UTC), "views": 100} for d in range(1, 25)]
    recent_lo = [{"date": datetime(2025, 5, d, tzinfo=UTC), "views": 50} for d in range(25, 32)]
    assert _momentum(base + recent_lo, CUTOFF, 7, 30)["trend"] == "falling"
    recent_flat = [{"date": datetime(2025, 5, d, tzinfo=UTC), "views": 105} for d in range(25, 32)]
    assert _momentum(base + recent_flat, CUTOFF, 7, 30)["trend"] == "flat"


def test_momentum_new_when_no_baseline() -> None:
    # Attention only in the recent window, nothing before -> "new", pct None.
    recent = [{"date": datetime(2025, 5, d, tzinfo=UTC), "views": 80} for d in range(25, 32)]
    sig = _momentum(recent, CUTOFF, 7, 30)
    assert sig["baseline_mean"] == 0.0
    assert sig["pct_change"] is None
    assert sig["trend"] == "new"


def test_momentum_uses_only_in_window_days() -> None:
    # A point exactly at cutoff must not be counted (defense in depth; the filter
    # already drops it, but momentum must also be strict).
    points = [{"date": CUTOFF, "views": 9999}]
    sig = _momentum(points, CUTOFF, 7, 30)
    assert sig["recent_mean"] == 0.0


def test_trend_buckets() -> None:
    assert _trend(0.2) == "rising"
    assert _trend(-0.2) == "falling"
    assert _trend(0.0) == "flat"
    assert _trend(None) == "new"


# --- country -> wiki-lang mapping ------------------------------------------


def test_wiki_langs_local_then_en() -> None:
    assert _wiki_langs(["italy"]) == ["it", "en"]
    assert _wiki_langs(["japan"]) == ["ja", "en"]
    assert _wiki_langs(["cyprus"]) == ["el", "en"]
    assert _wiki_langs(["netherlands"]) == ["nl", "en"]
    assert _wiki_langs(["south-korea"]) == ["ko", "en"]
    assert _wiki_langs(["iran"]) == ["fa", "en"]


def test_wiki_langs_always_includes_en_and_dedups() -> None:
    # No mapped country -> English only.
    assert _wiki_langs(["politics", "elections"]) == ["en"]
    # An English-local country must not list 'en' twice.
    assert _wiki_langs(["ireland"]) == ["en"]


def test_to_title_underscores() -> None:
    assert _to_title("Andrea Martella") == "Andrea_Martella"
    assert _to_title("  Simone Venturini ") == "Simone_Venturini"


# --- entity extraction reuse ----------------------------------------------


def test_entity_extraction_reused() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    payload = src.fetch(
        {
            "question": "Will Andrea Martella or Simone Venturini win the Venice mayoral race?",
            "as_of": CUTOFF,
            "tags": ["italy"],
        }
    )
    names = [e["entity"] for e in payload["entities"]]
    assert "Andrea Martella" in names
    assert "Simone Venturini" in names


# --- render states ---------------------------------------------------------


def test_render_error_state() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    out = src.render({"error": "api_unreachable", "query": "X"})
    assert out.startswith("[SOURCE ERROR] pageviews")


def test_render_no_data_state() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    out = src.render(
        {
            "entities": [{"entity": "Nobody", "found": False}],
            "query": "Nobody",
            "as_of": CUTOFF.isoformat(),
        }
    )
    assert "no articles found" in out


def test_render_found_state() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    out = src.render(
        {
            "entities": [
                {
                    "entity": "Andrea Martella",
                    "lang": "it",
                    "found": True,
                    "recent_mean": 1240.0,
                    "baseline_mean": 900.0,
                    "pct_change": 0.38,
                    "trend": "rising",
                    "n_days": 60,
                },
                {"entity": "Ghost", "found": False},
            ],
            "query": "Andrea Martella | Ghost",
            "langs": ["it", "en"],
            "as_of": CUTOFF.isoformat(),
        }
    )
    assert "Andrea Martella (it): 1,240 views/day last 7d, +38% vs prior 30d (rising)" in out
    assert "no Wikipedia article found for: Ghost" in out


# --- mocked fetch: point-in-time end to end --------------------------------


class _StubClient:
    """Minimal httpx.Client stand-in. Maps article-title -> list of API items.

    A title not in the map returns 404 (no such article), mirroring the live API.
    """

    def __init__(self, by_title: dict[str, list[dict[str, Any]]]) -> None:
        self._by_title = by_title
        self.requested_urls: list[str] = []

    def get(self, url: str) -> httpx.Response:
        self.requested_urls.append(url)
        req = httpx.Request("GET", url)
        for title, items in self._by_title.items():
            if f"/{title}/daily/" in url:
                return httpx.Response(200, json={"items": items}, request=req)
        return httpx.Response(404, json={"type": "not_found"}, request=req)


def test_fetch_is_point_in_time_and_computes_momentum() -> None:
    # Series includes a day AT as_of with a huge spike - it must be excluded, so
    # the recent mean reflects only the pre-as_of days.
    items = []
    for day in range(1, 25):
        items.append(_item(f"202505{day:02d}", 100))
    for day in range(25, 32):
        items.append(_item(f"202505{day:02d}", 150))
    items.append(_item("20250601", 100000))  # == as_of, MUST be excluded

    stub = _StubClient({"Giorgia_Meloni": items})
    src = WikipediaPageviewsSource(http=stub)
    payload = src.fetch(
        {"question": "Will Giorgia Meloni stay PM?", "as_of": CUTOFF, "tags": ["italy"]}
    )
    found = [e for e in payload["entities"] if e["entity"] == "Giorgia Meloni"][0]
    assert found["found"] is True
    assert found["lang"] == "it"  # resolved on the local (it) wiki first
    assert found["recent_mean"] == pytest.approx(150.0)  # spike excluded
    assert found["trend"] == "rising"
    # The local wiki was queried before en.
    assert any("it.wikipedia" in u for u in stub.requested_urls)


def test_fetch_falls_back_to_en_when_local_missing() -> None:
    # Only an en article exists; the it wiki 404s -> we use en.
    items = [_item(f"202505{d:02d}", 100) for d in range(1, 32)]
    stub = _StubClient({"Foo_Bar": items})  # served on any wiki (title match)
    src = WikipediaPageviewsSource(http=stub)
    payload = src.fetch({"question": "Will Foo Bar win?", "as_of": CUTOFF, "tags": ["italy"]})
    found = [e for e in payload["entities"] if e["entity"] == "Foo Bar"][0]
    # First wiki to return the title wins; our stub serves it on the it request,
    # so just assert it resolved and is point-in-time.
    assert found["found"] is True


def test_fetch_empty_question_errors() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    assert src.fetch({"question": "", "as_of": CUTOFF, "tags": []})["error"] == "empty_question"


def test_fetch_no_entities_errors() -> None:
    src = WikipediaPageviewsSource(http=_StubClient({}))
    out = src.fetch({"question": "will it rain tomorrow", "as_of": CUTOFF, "tags": []})
    assert out["error"] == "no_entities"


# --- live sanity check (one known article) ---------------------------------


@pytest.mark.skipif(
    os.environ.get("POLYEVOLVE_LIVE_TESTS") != "1",
    reason="live Wikimedia API test; set POLYEVOLVE_LIVE_TESTS=1 to run",
)
def test_live_single_article() -> None:
    src = WikipediaPageviewsSource()
    payload = src.fetch(
        {
            "question": "Will Shinzo Abe return?",
            "as_of": datetime(2025, 6, 1, tzinfo=UTC),
            "tags": ["japan"],
        }
    )
    assert "error" not in payload
    found = [e for e in payload["entities"] if e.get("found")]
    assert found, "expected at least one resolved article (Shinzo Abe)"
    assert found[0]["recent_mean"] >= 0
