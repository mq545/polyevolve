"""Tests for article dedup + ranking (data_sources/processing.py)."""

from polyevolve.data_sources.processing import (
    dedup_articles,
    rank_articles,
)


def _art(title: str, domain: str = "x.com") -> dict:
    return {"title": title, "domain": domain}


def test_dedup_collapses_identical_titles() -> None:
    arts = [_art("Maduro flees Venezuela"), _art("Maduro flees Venezuela")]
    out = dedup_articles(arts)
    assert len(out) == 1
    assert out[0]["dup_count"] == 2


def test_dedup_collapses_near_duplicates() -> None:
    # Same story, trivially reworded - high token overlap.
    arts = [
        _art("Maduro flees Venezuela amid unrest"),
        _art("Maduro flees Venezuela amid the unrest"),
    ]
    out = dedup_articles(arts)
    assert len(out) == 1
    assert out[0]["dup_count"] == 2


def test_dedup_keeps_distinct_stories() -> None:
    arts = [_art("Maduro flees Venezuela"), _art("Iran closes Strait of Hormuz")]
    out = dedup_articles(arts)
    assert len(out) == 2
    assert all(a["dup_count"] == 1 for a in out)


def test_dedup_untitled_kept_separately() -> None:
    arts = [_art(""), _art("")]
    out = dedup_articles(arts)
    # Can't reliably dedup untitled; both kept.
    assert len(out) == 2


def test_rank_caps_to_max_n() -> None:
    arts = [_art(f"story {i}", domain=f"d{i}.com") for i in range(10)]
    out = rank_articles(dedup_articles(arts), max_n=3)
    assert len(out) == 3


def test_rank_prefers_domain_diversity() -> None:
    arts = [
        _art("a", domain="same.com"),
        _art("b", domain="same.com"),
        _art("c", domain="other.com"),
    ]
    out = rank_articles(dedup_articles(arts), max_n=2)
    domains = {a["domain"] for a in out}
    # With diversity-greedy selection, two slots => two distinct domains.
    assert domains == {"same.com", "other.com"}


def test_rank_backfills_when_diversity_short() -> None:
    # Only one domain but max_n=2 => must backfill the second slot.
    arts = [_art("a", domain="same.com"), _art("b", domain="same.com")]
    out = rank_articles(dedup_articles(arts), max_n=2)
    assert len(out) == 2


def test_rank_empty_and_zero() -> None:
    assert rank_articles([], max_n=5) == []
    assert rank_articles([_art("a")], max_n=0) == []


def test_more_syndicated_story_floats_up() -> None:
    # A widely-carried story (higher dup_count) should be preferred when capping.
    arts = [_art("wire story", domain=f"a{i}.com") for i in range(3)] + [
        _art("niche story", domain="niche.com")
    ]
    deduped = dedup_articles(arts)
    # wire story now dup_count=3, niche dup_count=1
    out = rank_articles(deduped, max_n=1)
    assert out[0]["title"] == "wire story"
