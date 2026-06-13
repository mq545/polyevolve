"""FITNESS - the scalar an optimizer maximizes.

The evolution loop selects genomes by a single number: higher is better. v0 fitness is
**calibration-first**: ``fitness = -brier`` over a held set of resolved questions, so the
search drives the Brier score down (a constant-0.5 forecaster scores brier 0.25 -> fitness
-0.25; a perfect one scores 0.0).

This module is deliberately *pluggable*. A future RETURN-based fitness (net-of-spread PnL,
backed by the adversarial market simulator - see polyevolve.bench HOOK) can be dropped in
behind the same `FitnessFn` signature without touching the optimizer:

    fitness: FitnessFn  # (Genome, Sequence[Question]) -> float

`make_calibration_fitness` is the only constructor wired today; `make_return_fitness`
is a clearly-marked stub so the swap point is obvious.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from polyevolve.bench import evaluate_calibration
from polyevolve.reason.dsl import EvidencePool, Genome, Question

__all__ = [
    "FitnessFn",
    "fitness",
    "make_calibration_fitness",
    "make_return_fitness",
    "WORST_FITNESS",
]

# Assigned when a genome produces no scorable questions (empty split, or every
# question errored / lacked an outcome). Worse than the constant-0.5 baseline
# (-0.25) so degenerate genomes never win selection.
WORST_FITNESS = -1.0


class FitnessFn(Protocol):
    """Any callable mapping a genome + questions to a scalar fitness (higher is better)."""

    def __call__(self, genome: Genome, questions: Sequence[Question]) -> float: ...


def fitness(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
) -> float:
    """Calibration-first fitness = ``-brier`` over scorable questions.

    Thin module-level convenience equivalent to ``make_calibration_fitness()`` with default
    pools. Returns ``WORST_FITNESS`` when nothing could be scored (so an empty / all-error
    split cannot masquerade as a good genome).
    """
    result = evaluate_calibration(genome, questions, pools)
    if result.get("n", 0.0) <= 0.0:
        return WORST_FITNESS
    brier = result["brier"]
    if not math.isfinite(brier):
        return WORST_FITNESS
    return -float(brier)


def make_calibration_fitness(
    pools: Sequence[EvidencePool] | None = None,
) -> FitnessFn:
    """Build a `FitnessFn` that scores ``-brier`` (optionally over fixed evidence pools).

    ``pools`` (if given) must align 1:1 with the questions later passed to the returned
    function; this lets the optimizer evolve over a FROZEN evidence corpus.
    """

    def _fn(genome: Genome, questions: Sequence[Question]) -> float:
        return fitness(genome, questions, pools)

    return _fn


def make_return_fitness(
    pools: Sequence[EvidencePool] | None = None,
    *,
    edge_threshold: float = 0.05,
    metric: str = "mean_event_roi",
) -> FitnessFn:
    """Build a `FitnessFn` that scores NET-OF-SPREAD RETURN on betting markets.

    The genome supplies the signal; ``bench.returns.evaluate_return`` runs it through the
    adversarial sim (never mid-fill, cross the spread, walk the book, per-event Kelly,
    event-clustered) and returns honest aggregates. Fitness = the chosen ``metric`` (default
    event-aggregated ROI - the honest n). A genome that finds no tradeable edge abstains and
    scores ~0, so this fitness rewards *selective, net-positive* policies, not calibration.
    """
    from polyevolve.bench.returns import evaluate_return

    def _fn(genome: Genome, questions: Sequence[Question]) -> float:
        res = evaluate_return(genome, questions, pools, edge_threshold=edge_threshold)
        if res.get("n_events", 0.0) <= 0.0:
            return 0.0  # abstained on everything -> flat, not "worst" (no trade is valid)
        val = res.get(metric, 0.0)
        return float(val) if math.isfinite(val) else WORST_FITNESS

    return _fn
