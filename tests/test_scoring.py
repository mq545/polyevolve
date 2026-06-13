"""Tests for backtest scoring + deterministic splits."""

from polyevolve.orchestration.scoring import assign_splits, brier


def test_brier_perfect_yes() -> None:
    assert brier(1.0, "YES") == 0.0


def test_brier_perfect_no() -> None:
    assert brier(0.0, "NO") == 0.0


def test_brier_worst_case() -> None:
    assert brier(1.0, "NO") == 1.0
    assert brier(0.0, "YES") == 1.0


def test_brier_midpoint() -> None:
    assert brier(0.5, "YES") == 0.25
    assert brier(0.5, "NO") == 0.25


def test_splits_are_deterministic() -> None:
    ids = [str(i) for i in range(200)]
    a = assign_splits(ids)
    b = assign_splits(ids)
    assert a == b  # same input -> same split, every time


def test_splits_roughly_match_fraction() -> None:
    ids = [f"market-{i}" for i in range(1000)]
    splits = assign_splits(ids, holdout_frac=0.3)
    holdout = sum(1 for s in splits.values() if s == "holdout")
    # within a reasonable tolerance of 30%
    assert 0.2 < holdout / len(ids) < 0.4


def test_splits_only_train_or_holdout() -> None:
    splits = assign_splits([f"m{i}" for i in range(50)])
    assert set(splits.values()) <= {"train", "holdout"}


def test_default_test_frac_is_zero_no_test_bucket() -> None:
    # Backward compat: without opting in, no market lands in "test".
    splits = assign_splits([f"m{i}" for i in range(500)])
    assert "test" not in set(splits.values())


def test_three_way_split_disjoint_and_stable() -> None:
    ids = [f"market-{i}" for i in range(2000)]
    a = assign_splits(ids, holdout_frac=0.25, test_frac=0.25)
    b = assign_splits(ids, holdout_frac=0.25, test_frac=0.25)
    assert a == b  # deterministic across calls
    assert set(a.values()) == {"train", "holdout", "test"}
    train = {m for m, s in a.items() if s == "train"}
    holdout = {m for m, s in a.items() if s == "holdout"}
    test = {m for m, s in a.items() if s == "test"}
    # the three sets must be mutually disjoint - no market in two splits
    assert train & holdout == set()
    assert train & test == set()
    assert holdout & test == set()
    assert len(train) + len(holdout) + len(test) == len(ids)
    # fractions roughly honoured
    assert 0.18 < len(test) / len(ids) < 0.32
    assert 0.18 < len(holdout) / len(ids) < 0.32


def test_test_membership_independent_of_holdout_frac() -> None:
    # The pristine test set must be carved from the SAME low buckets regardless
    # of holdout size, so changing holdout_frac can never move a market into or
    # out of test (which would leak test identity across configs).
    ids = [f"m{i}" for i in range(2000)]
    a = assign_splits(ids, holdout_frac=0.25, test_frac=0.25)
    b = assign_splits(ids, holdout_frac=0.40, test_frac=0.25)
    test_a = {m for m, s in a.items() if s == "test"}
    test_b = {m for m, s in b.items() if s == "test"}
    assert test_a == test_b
