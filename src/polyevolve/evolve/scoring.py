"""The portable scoring contract - ONE dict, ``combined_score`` is the maximized fitness.

Both LLM program-evolvers we target (ShinkaEvolve, OpenEvolve) want an evaluator that
returns a metrics dict keyed on ``combined_score`` (higher = better). Our in-process loop
wants a float. So this module is the single source of truth: ``evaluate_genome`` returns the
dict; every optimizer is a thin projection over it (see ``make_fitness``). This makes the
SAME seed genome portable across our loop + ShinkaEvolve + OpenEvolve with one-line glue.

    combined_score = -brier        (objective="calibration")
    combined_score = mean_event_roi (objective="return")
"""

from __future__ import annotations

from collections.abc import Sequence

from polyevolve.bench import evaluate_calibration
from polyevolve.bench.returns import evaluate_return
from polyevolve.evolve.fitness import WORST_FITNESS, FitnessFn
from polyevolve.reason.dsl import EvidencePool, Genome, Question

__all__ = ["evaluate_genome", "make_fitness", "Objective"]

Objective = str  # "calibration" | "return"


def evaluate_genome(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
    *,
    objective: Objective = "calibration",
    edge_threshold: float = 0.05,
) -> dict[str, float]:
    """Score a genome and return a metrics dict. ``combined_score`` (maximized) is the
    fitness; the remaining keys are diagnostics / optimizer feature dimensions.

    This is the exact dict ShinkaEvolve's ``aggregate_metrics_fn`` and OpenEvolve's
    ``evaluate`` should emit, and the float our ``FitnessFn`` projects from.
    """
    if objective == "return":
        r = evaluate_return(genome, questions, pools, edge_threshold=edge_threshold)
        combined = r["mean_event_roi"] if r.get("n_events", 0.0) > 0.0 else 0.0
        return {"combined_score": float(combined), **r}

    # default: calibration
    c = evaluate_calibration(genome, questions, pools)
    combined = -c["brier"] if c.get("n", 0.0) > 0.0 else WORST_FITNESS
    return {"combined_score": float(combined), **c}


def make_fitness(
    pools: Sequence[EvidencePool] | None = None,
    *,
    objective: Objective = "calibration",
    edge_threshold: float = 0.05,
) -> FitnessFn:
    """A `FitnessFn` (float) that projects ``combined_score`` out of `evaluate_genome` -
    the single knob (``objective``) selects calibration vs net-of-spread return."""

    def _fn(genome: Genome, questions: Sequence[Question]) -> float:
        d = evaluate_genome(
            genome, questions, pools, objective=objective, edge_threshold=edge_threshold
        )
        return float(d["combined_score"])

    return _fn
