"""Tests for backtest as_of computation (lead time + clamping + drop)."""

from datetime import UTC, datetime, timedelta

from polyevolve.orchestration.backtest import compute_as_of


def test_normal_lead() -> None:
    resolved = datetime(2026, 3, 1, tzinfo=UTC)
    created = datetime(2026, 1, 1, tzinfo=UTC)
    as_of = compute_as_of(resolved, created, lead_days=7)
    assert as_of == resolved - timedelta(days=7)


def test_clamps_to_created_at() -> None:
    # market created only 4 days before resolution; 7-day lead would predate it
    resolved = datetime(2026, 3, 1, tzinfo=UTC)
    created = datetime(2026, 2, 25, tzinfo=UTC)  # 4 days before
    as_of = compute_as_of(resolved, created, lead_days=7)
    assert as_of == created  # clamped, and 4 days >= min_lead 3 => kept


def test_drops_too_short_lived() -> None:
    # market lived only 2 days - below min_lead_days=3 => dropped
    resolved = datetime(2026, 3, 1, tzinfo=UTC)
    created = datetime(2026, 2, 27, tzinfo=UTC)  # 2 days before
    assert compute_as_of(resolved, created, lead_days=7) is None


def test_no_created_at_uses_lead() -> None:
    resolved = datetime(2026, 3, 1, tzinfo=UTC)
    as_of = compute_as_of(resolved, None, lead_days=14)
    assert as_of == resolved - timedelta(days=14)


def test_as_of_always_before_resolution() -> None:
    resolved = datetime(2026, 3, 1, tzinfo=UTC)
    created = datetime(2026, 1, 1, tzinfo=UTC)
    as_of = compute_as_of(resolved, created, lead_days=7)
    assert as_of is not None
    assert as_of < resolved
