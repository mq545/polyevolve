"""Tests for GDELT point-in-time filtering (no network - tests the date logic)."""

from datetime import UTC, datetime

from polyevolve.data_sources.gdelt import _before


def test_before_true_for_earlier_article() -> None:
    assert _before("20250101T120000Z", datetime(2025, 6, 1, tzinfo=UTC))


def test_before_false_for_later_article() -> None:
    assert not _before("20250701T120000Z", datetime(2025, 6, 1, tzinfo=UTC))


def test_before_false_at_exact_boundary() -> None:
    # strictly-before: an article stamped exactly at cutoff is excluded
    assert not _before("20250601T000000Z", datetime(2025, 6, 1, tzinfo=UTC))


def test_unparseable_seendate_dropped() -> None:
    # fail-closed: can't date it => treat as not-before (don't leak it)
    assert not _before("", datetime(2025, 6, 1, tzinfo=UTC))
    assert not _before("garbage", datetime(2025, 6, 1, tzinfo=UTC))


def test_naive_cutoff_handled() -> None:
    assert _before("20250101T120000Z", datetime(2025, 6, 1))
