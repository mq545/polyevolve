"""Event-grouped coherence - the fix for cross-market (sibling) incoherence.

The genome forecasts each market independently, so mutually-exclusive siblings in one event
(e.g. every candidate's "will X win?" market) get probabilities that don't respect the
constraint that exactly one happens - they can sum to far more than 1 (the Lula 0.86 +
Flavio 0.99 bug). This module renormalizes a group of sibling probabilities AFTER inference.

Two modes, by group type:
  - ``sum_to_one``  : EXHAUSTIVE mutually-exclusive group (all candidates/outcomes listed,
                      exactly one is YES) -> divide by the sum so they total 1.
  - ``deoverround`` : PARTIAL group (a subset of outcomes, e.g. some seat bands) -> only scale
                      DOWN when the sum exceeds 1 (remove the incoherent overround) and never
                      scale up (partial coverage legitimately sums to < 1).

`deoverround` is the safe default - it can only fix over-confidence, never invent it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

__all__ = ["normalize_sum_to_one", "deoverround", "normalize_by_group"]


def normalize_sum_to_one(ps: Sequence[float]) -> list[float]:
    """Scale probabilities to sum to 1 (exhaustive mutually-exclusive). Sum<=0 -> unchanged."""
    s = float(sum(ps))
    if s <= 0.0:
        return [float(p) for p in ps]
    return [float(p) / s for p in ps]


def deoverround(ps: Sequence[float]) -> list[float]:
    """Remove an over-round: scale down only when the group sums to > 1; never scale up."""
    s = float(sum(ps))
    if s <= 1.0:
        return [float(p) for p in ps]
    return [float(p) / s for p in ps]


def normalize_by_group(
    keys: Sequence[object], ps: Sequence[float], *, mode: str = "deoverround"
) -> list[float]:
    """Renormalize ``ps`` within each group defined by ``keys`` (aligned 1:1), preserving order.

    ``mode`` is ``deoverround`` (safe; scale down over-rounds only) or ``sum_to_one``
    (exhaustive mutually-exclusive groups). Items whose key is falsy (no event) are passed
    through unchanged - a lone market has nothing to be coherent with.
    """
    if len(keys) != len(ps):
        raise ValueError("keys and ps must align 1:1")
    fn = normalize_sum_to_one if mode == "sum_to_one" else deoverround
    if mode not in ("sum_to_one", "deoverround"):
        raise ValueError(f"normalize_by_group: unknown mode {mode!r}")

    idx_by_group: dict[object, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        idx_by_group[k].append(i)

    out = [float(p) for p in ps]
    for k, idxs in idx_by_group.items():
        if not k or len(idxs) < 2:  # lone market or no event id -> nothing to normalize
            continue
        adj = fn([out[i] for i in idxs])
        for i, v in zip(idxs, adj, strict=True):
            out[i] = v
    return out
