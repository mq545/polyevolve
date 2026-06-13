"""Tests for evaluator aggregation (pure scoring, no DB/network)."""

from polyevolve.evolution.evaluator import _aggregate


def _m(outcome: str, prob: float, price: float | None, clean: bool, split: str | None) -> dict:
    from polyevolve.orchestration.scoring import brier

    return {
        "market_id": "x",
        "outcome": outcome,
        "prob": prob,
        "price": price,
        "is_clean": clean,
        "split": split,
        "brier_agent": brier(prob, outcome),
        "brier_market": brier(price, outcome) if price is not None else None,
    }


def test_only_clean_counted_in_metrics() -> None:
    per = [
        _m("YES", 0.9, 0.8, clean=True, split="train"),
        _m("NO", 0.1, 0.2, clean=False, split=None),  # contaminated - excluded
    ]
    r = _aggregate("m", "h", [], per, hits=0, misses=2)
    assert r.n_total == 2
    assert r.n_clean == 1


def test_holdout_brier_isolated_from_train() -> None:
    per = [
        _m("YES", 1.0, 0.5, clean=True, split="train"),  # brier 0
        _m("NO", 1.0, 0.5, clean=True, split="holdout"),  # brier 1
    ]
    r = _aggregate("m", "h", [], per, hits=0, misses=0)
    assert r.brier_train == 0.0
    assert r.brier_holdout == 1.0


def test_edge_positive_when_agent_beats_market() -> None:
    # agent perfect (brier 0), market wrong (brier 1) -> edge +1
    per = [_m("YES", 1.0, 0.0, clean=True, split="holdout")]
    r = _aggregate("m", "h", [], per, hits=0, misses=0)
    assert r.edge_holdout == 1.0


def test_edge_none_without_price() -> None:
    per = [_m("YES", 0.7, None, clean=True, split="holdout")]
    r = _aggregate("m", "h", [], per, hits=0, misses=0)
    assert r.edge_holdout is None
    assert r.n_priced_clean == 0


def test_combined_score_uses_cv_over_all_nontest() -> None:
    # Fitness validates on ALL non-test markets (train+holdout), not just holdout -
    # the denoising fix. The pristine TEST split must NOT influence the score.
    per = [
        _m("YES", 1.0, 0.5, clean=True, split="train"),  # brier 0
        _m("NO", 1.0, 0.5, clean=True, split="holdout"),  # brier 1
        _m("YES", 0.0, 0.5, clean=True, split="test"),  # brier 1 - must be ignored
    ]
    r = _aggregate("m", "h", [], per, hits=0, misses=0)
    # brier_cv = mean(train,holdout) = 0.5 ; genome_chars=0 → no penalty.
    assert r.brier_cv == 0.5
    # -0.5, not -1.0 (holdout-only) and not -0.667 (would include the test split):
    assert r.combined_score == -0.5


def test_complexity_penalty_lowers_score() -> None:
    per = [_m("YES", 1.0, 0.5, clean=True, split="holdout")]  # brier 0
    lean = _aggregate("m", "h", [], per, hits=0, misses=0, genome_chars=2000)
    fat = _aggregate("m", "h", [], per, hits=0, misses=0, genome_chars=12000)
    assert fat.combined_score < lean.combined_score < 0  # baroque genome penalized
