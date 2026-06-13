"""Tests for market filtering + domain selection."""

import pytest

from polyevolve.contracts import Market
from polyevolve.market_sources.filters import domain, is_foreign_politics


def _market(tags: list[str]) -> Market:
    return Market(
        venue="polymarket",
        external_id="x",
        cross_venue_id=None,
        question="q",
        close_time=None,
        status="active",
        metadata={"tags": tags},
    )


def test_keeps_foreign_political_market() -> None:
    assert is_foreign_politics(_market(["france", "politics", "macron", "world"]))


def test_keeps_geopolitics() -> None:
    assert is_foreign_politics(_market(["china", "geopolitics", "world"]))


def test_drops_us_headline() -> None:
    assert not is_foreign_politics(_market(["politics", "trump", "us-politics"]))


def test_drops_uk_politics() -> None:
    assert not is_foreign_politics(_market(["politics", "uk", "starmer"]))


def test_drops_sports_even_if_political_tag_present() -> None:
    assert not is_foreign_politics(_market(["politics", "sports", "soccer"]))


def test_drops_nonpolitical() -> None:
    assert not is_foreign_politics(_market(["crypto", "bitcoin"]))


def test_drops_generic_politics_without_country() -> None:
    # political but no foreign-country / geopolitics marker → excluded
    assert not is_foreign_politics(_market(["politics", "2025-predictions"]))


# --- domain selection (cast the net to other markets) ---
def test_domain_foreign_politics_matches_legacy_filter() -> None:
    dom = domain("foreign_politics")
    assert dom.keep is is_foreign_politics
    assert "politics" in dom.tags


def test_domain_sports_keeps_sports_drops_politics() -> None:
    dom = domain("sports")
    assert dom.keep(_market(["nba"]))
    assert not dom.keep(_market(["politics", "france"]))
    assert "nba" in dom.tags


def test_domain_crypto_keeps_crypto() -> None:
    assert domain("crypto").keep(_market(["crypto", "bitcoin"]))
    assert not domain("crypto").keep(_market(["nba"]))


def test_domain_all_keeps_everything() -> None:
    dom = domain("all")
    assert dom.keep(_market(["nba"]))
    assert dom.keep(_market(["crypto"]))
    assert dom.keep(_market(["politics", "france"]))
    assert {"politics", "nba", "crypto"} <= set(dom.tags)


def test_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        domain("nonsense")
