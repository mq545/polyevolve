"""Calibration scoring primitives.

A `Pair` is `(p_yes, outcome_bool)`: the genome's probability for YES and the
realized binary outcome. All scorers operate on `list[Pair]`.

- `brier`  : mean squared error of the probability vs the 0/1 outcome (lower is
  better; a constant 0.5 forecaster scores 0.25).
- `ece`    : expected calibration error - the bin-weighted gap between mean
  predicted probability and observed frequency.
- `calibration_curve`: per-bin (mean_pred, observed_freq, count) for plotting /
  diagnosing over/under-confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["Pair", "brier", "ece", "calibration_curve", "Bin"]

Pair = tuple[float, bool]


class Bin:
    """One calibration bin: predicted-mean vs observed-frequency over `count` items."""

    __slots__ = ("lo", "hi", "mean_pred", "observed", "count")

    def __init__(self, lo: float, hi: float, mean_pred: float, observed: float, count: int):
        self.lo = lo
        self.hi = hi
        self.mean_pred = mean_pred
        self.observed = observed
        self.count = count

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Bin([{self.lo:.2f},{self.hi:.2f}) n={self.count} "
            f"pred={self.mean_pred:.3f} obs={self.observed:.3f})"
        )


def _split(pairs: Sequence[Pair]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty(0), np.empty(0)
    p = np.array([float(a) for a, _ in pairs], dtype=float)
    y = np.array([1.0 if b else 0.0 for _, b in pairs], dtype=float)
    return p, y


def brier(pairs: Sequence[Pair]) -> float:
    """Mean (p - outcome)^2. Returns NaN on an empty input."""
    p, y = _split(pairs)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def calibration_curve(pairs: Sequence[Pair], bins: int = 10) -> list[Bin]:
    """Equal-width reliability bins over [0, 1]. Empty bins are omitted."""
    p, y = _split(pairs)
    out: list[Bin] = []
    if p.size == 0:
        return out
    edges = np.linspace(0.0, 1.0, bins + 1)
    # rightmost edge inclusive so p == 1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            Bin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                mean_pred=float(p[mask].mean()),
                observed=float(y[mask].mean()),
                count=n,
            )
        )
    return out


def ece(pairs: Sequence[Pair], bins: int = 10) -> float:
    """Expected calibration error: sum_b (count_b/N) * |mean_pred_b - observed_b|."""
    p, _ = _split(pairs)
    if p.size == 0:
        return float("nan")
    total = float(p.size)
    curve = calibration_curve(pairs, bins)
    return float(sum((b.count / total) * abs(b.mean_pred - b.observed) for b in curve))
