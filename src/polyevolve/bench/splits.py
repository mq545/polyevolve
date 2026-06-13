"""Dataset splits for the fitness bench.

`temporal_split` is the anti-leakage workhorse: it orders questions by their
point-in-time cutoff (`as_of`) and slices the EARLIEST into train, the next into
val, and the LATEST into test - so a genome selected on the past is always
evaluated on its future (mirrors how it would trade live).

`event_cluster` is a stub for later: many Manifold/Polymarket questions share an
underlying event (e.g. one election spawns a dozen threshold markets), and naive
random/temporal splits leak correlated outcomes across train/test. The eventual
implementation will group by a caller-supplied key and keep whole clusters on one
side of the split.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from polyevolve.reason.dsl import Question

__all__ = ["Split", "temporal_split", "event_cluster"]


class Split:
    """Three disjoint, time-ordered question lists."""

    __slots__ = ("train", "val", "test")

    def __init__(self, train: list[Question], val: list[Question], test: list[Question]):
        self.train = train
        self.val = val
        self.test = test

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Split(train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"


def temporal_split(
    questions: Sequence[Question],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Split:
    """Order by `as_of` (oldest first) and slice into train / val / test.

    `test_frac` is the remainder (`1 - train_frac - val_frac`). A question with
    a later cutoff never appears before an earlier one, so selection on
    train+val never sees test-era information.
    """
    if not 0.0 <= train_frac <= 1.0 or not 0.0 <= val_frac <= 1.0:
        raise ValueError("fractions must be in [0, 1]")
    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    ordered = sorted(questions, key=lambda q: q.as_of)
    n = len(ordered)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return Split(
        train=ordered[:n_train],
        val=ordered[n_train : n_train + n_val],
        test=ordered[n_train + n_val :],
    )


def event_cluster(
    questions: Sequence[Question],
    keyfn: Callable[[Question], str],
) -> dict[str, list[Question]]:
    """STUB: group questions by a shared-event key.

    Future use: build leakage-safe splits by keeping whole event clusters on one
    side, and to down-weight correlated outcomes in scoring (e.g. one election ->
    many threshold markets should not count as N independent samples).

    The grouping itself is implemented (it is trivial and useful now); the
    split-aware consumption of these clusters is the part deferred.
    """
    clusters: dict[str, list[Question]] = {}
    for q in questions:
        clusters.setdefault(keyfn(q), []).append(q)
    return clusters
