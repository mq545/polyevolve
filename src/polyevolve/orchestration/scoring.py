"""Scoring helpers for backtest + calibration."""

from __future__ import annotations

import hashlib


def brier(probability_yes: float, outcome: str) -> float:
    """Brier score for a single binary prediction. Lower is better; range [0, 1]."""
    actual = 1.0 if outcome == "YES" else 0.0
    return (probability_yes - actual) ** 2


def assign_splits(
    market_ids: list[str], holdout_frac: float = 0.3, test_frac: float = 0.0
) -> dict[str, str]:
    """Deterministic train / holdout / test split by stable hash of market id.

    Deterministic (not random) so re-runs are reproducible and the splits are
    stable across runs AND across genomes - critical so evolution can't leak
    markets between sets between generations.

    Three-way split for honest evolution (see evolve_task/evaluate.py):
      - "test"    bucket in [0, test_frac):  NEVER used for selection. Scored
                  once on the final champion - the only trustworthy "did we beat
                  the market" number.
      - "holdout" bucket in [test_frac, test_frac+holdout_frac): the VALIDATION
                  set that drives combined_score (evolution selects on it).
      - "train"   the rest: metrics shown to the mutator.

    test_frac defaults to 0.0, which reproduces the original two-way behaviour
    exactly - callers that don't opt in are unaffected.
    """
    splits: dict[str, str] = {}
    for mid in market_ids:
        # md5 gives a well-distributed bucket even when ids share a prefix;
        # deterministic across runs (unlike Python's salted hash()).
        digest = hashlib.md5(mid.encode()).hexdigest()  # noqa: S324 - non-crypto use
        bucket = (int(digest[:8], 16) % 10000) / 10000.0
        if bucket < test_frac:
            splits[mid] = "test"
        elif bucket < test_frac + holdout_frac:
            splits[mid] = "holdout"
        else:
            splits[mid] = "train"
    return splits
