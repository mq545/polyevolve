"""OPTIMIZER - propose mutations of the seed knobs, select on TRAIN, report VAL.

The contract is one small Protocol::

    class Optimizer(Protocol):
        def optimize(self, seed_knobs, train_qs, val_qs) -> Result: ...

`run_evolution` is the MINIMAL built-in implementation: a (mu + lambda)-style loop that
mutates `SeedKnobs` two ways -

  * numeric/flag JITTER (Gaussian on scalars, bounded; coin-flip on switches), and
  * an LLM PROMPT-MUTATION operator that asks a model (via `polyevolve.models.build_model`)
    to rewrite the forecaster `system_prompt` into a better-calibrated variant.

Selection is **on TRAIN only**; the chosen champion is then *reported* on a held-out VAL
split (the generalization guard) - we never pick the genome that merely memorised val.
A complexity-penalty hook (`complexity_lambda` x a knob-complexity proxy) can be switched
on to break ties toward simpler genomes.

The whole loop runs WITHOUT ShinkaEvolve. For full-program (composition-level) search,
`polyevolve.evolve.shinka.ShinkaEvolveOptimizer` implements the same `Optimizer` interface
by delegating to Sakana's ShinkaEvolve in a separate venv.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from polyevolve.evolve.fitness import WORST_FITNESS, FitnessFn, make_calibration_fitness
from polyevolve.reason.dsl import EvidencePool, Question
from polyevolve.reason.seed import SeedKnobs, make_seed_genome

__all__ = [
    "Optimizer",
    "Result",
    "Individual",
    "run_evolution",
    "EvolutionOptimizer",
    "knob_complexity",
]


# --------------------------------------------------------------------------------------
# results / interface
# --------------------------------------------------------------------------------------


@dataclass
class Individual:
    """One evaluated candidate: its knobs and its train/val scores."""

    knobs: SeedKnobs
    train_fitness: float
    val_fitness: float = WORST_FITNESS
    generation: int = 0


@dataclass
class Result:
    """What an optimizer returns: the champion + the lineage for inspection."""

    best_knobs: SeedKnobs
    # champion's score on the split it was SELECTED on (train) and the held-out report (val).
    best_train_fitness: float
    best_val_fitness: float
    seed_train_fitness: float
    seed_val_fitness: float
    history: list[Individual] = field(default_factory=list)
    # full-program optimizers (ShinkaEvolve) evolve CODE, not knobs: the champion's evolved
    # source and its path. None for the knob-level built-in loop.
    champion_source: str | None = None
    champion_path: str | None = None

    @property
    def improved(self) -> bool:
        """True if the champion is at least as good as the seed on the SELECTION split."""
        return self.best_train_fitness >= self.seed_train_fitness


class Optimizer(Protocol):
    """Anything that searches knob-space: select on train, report val."""

    def optimize(
        self,
        seed_knobs: SeedKnobs,
        train_qs: Sequence[Question],
        val_qs: Sequence[Question],
    ) -> Result: ...


# --------------------------------------------------------------------------------------
# complexity penalty hook
# --------------------------------------------------------------------------------------


def knob_complexity(knobs: SeedKnobs) -> float:
    """A cheap, monotone complexity proxy for the generalization penalty.

    Larger evidence budgets, ensembling, decomposition, and a longer prompt all add
    capacity that can overfit a small train split. Subtracting ``lambda * this`` from
    fitness biases selection toward the simplest genome that ties.
    """
    return (
        0.02 * float(knobs.select_k)
        + (0.5 * float(knobs.ensemble_k) if knobs.use_ensemble else 0.0)
        + (0.5 if knobs.use_decompose else 0.0)
        + 0.0005 * float(len(knobs.system_prompt))
    )


# --------------------------------------------------------------------------------------
# mutation operators
# --------------------------------------------------------------------------------------

# Bounds the numeric jitter must respect (mirrors SeedKnobs.__post_init__ validity).
_BOUNDS: dict[str, tuple[float, float]] = {
    "calibrate_coeff": (0.2, 4.0),
    "abstain_min_conf": (0.0, 1.0),
    "abstain_min_div": (0.0, 1.0),
    "kelly_frac": (0.0, 1.0),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _jitter_numeric(knobs: SeedKnobs, rng: random.Random) -> dict[str, Any]:
    """Gaussian jitter on the scalar knobs + coin-flips on the switches."""
    updates: dict[str, Any] = {}
    for name, (lo, hi) in _BOUNDS.items():
        cur = float(getattr(knobs, name))
        scale = 0.15 * (hi - lo)
        updates[name] = round(_clamp(cur + rng.gauss(0.0, scale), lo, hi), 4)
    # integer evidence budget: +-2, kept >= 1.
    updates["select_k"] = max(1, knobs.select_k + rng.choice([-2, -1, 0, 1, 2]))
    # composition switches flip occasionally.
    if rng.random() < 0.2:
        updates["use_decompose"] = not knobs.use_decompose
    if rng.random() < 0.2:
        updates["use_ensemble"] = not knobs.use_ensemble
    if knobs.use_ensemble and rng.random() < 0.3:
        updates["ensemble_k"] = max(1, knobs.ensemble_k + rng.choice([-1, 1]))
    return updates


_PROMPT_TOOL: dict[str, Any] = {
    "name": "submit_prompt",
    "description": "Submit one improved forecaster system-prompt variant.",
    "input_schema": {
        "type": "object",
        "properties": {
            "system_prompt": {
                "type": "string",
                "description": (
                    "A rewritten forecaster system prompt that should yield BETTER-CALIBRATED "
                    "probabilities (lower Brier) than the original. Keep the submit_prediction "
                    "instruction; emphasise base rates, reliability-weighting, and humility."
                ),
            }
        },
        "required": ["system_prompt"],
    },
}

_PROMPT_MUTATOR_SYS = (
    "You are optimizing a forecaster's system prompt for CALIBRATION (low Brier score). "
    "Given the current prompt, propose a single improved variant. Call submit_prompt once."
)


def _mutate_prompt_via_llm(
    knobs: SeedKnobs,
    *,
    build_model: Any,
) -> str | None:
    """Ask a model to propose a better `system_prompt`. Returns None on any failure.

    `build_model` is injected (defaults to `polyevolve.models.build_model` in `run_evolution`)
    so tests pass a mock and hit no network. Mirrors the complete_with_tool pattern in
    scripts/predict_margin.py.
    """
    try:
        model = build_model(model_id=knobs.model_id, anthropic_api_key=knobs.anthropic_api_key)
        res = model.complete_with_tool(
            cached_system_blocks=[_PROMPT_MUTATOR_SYS],
            user_content=f"CURRENT PROMPT:\n{knobs.system_prompt}\n\nPropose an improved variant.",
            tool=_PROMPT_TOOL,
            metadata={"op": "mutate_prompt"},
        )
        new_prompt = str(res["input"]["system_prompt"]).strip()
        return new_prompt or None
    except Exception:  # noqa: BLE001 - mutation is best-effort; fall back to numeric-only
        return None


def _mutate(
    knobs: SeedKnobs,
    rng: random.Random,
    *,
    propose_prompt: bool,
    build_model: Any,
) -> SeedKnobs:
    """Produce one child knob-set from a parent (numeric jitter + optional prompt rewrite)."""
    updates = _jitter_numeric(knobs, rng)
    if propose_prompt and build_model is not None and rng.random() < 0.5:
        new_prompt = _mutate_prompt_via_llm(knobs, build_model=build_model)
        if new_prompt:
            updates["system_prompt"] = new_prompt
    return dataclasses.replace(copy.deepcopy(knobs), **updates)


# --------------------------------------------------------------------------------------
# the built-in evolutionary loop
# --------------------------------------------------------------------------------------


def run_evolution(
    seed_knobs: SeedKnobs,
    train_qs: Sequence[Question],
    val_qs: Sequence[Question],
    *,
    generations: int = 5,
    pop: int = 6,
    fitness_fn: FitnessFn | None = None,
    fitness_factory: Callable[[Sequence[EvidencePool] | None], FitnessFn] | None = None,
    train_pools: Sequence[EvidencePool] | None = None,
    val_pools: Sequence[EvidencePool] | None = None,
    complexity_lambda: float = 0.0,
    propose_prompt: bool = True,
    build_model: Any | None = None,
    seed: int = 0,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> Result:
    """Minimal (mu + lambda) evolution over `SeedKnobs`. No ShinkaEvolve required.

    Selection is on TRAIN; the champion is reported on VAL (generalization guard). The seed
    itself is always evaluated and is a valid champion, so the result can only *match or
    beat* the seed on train - never regress.

    Args:
        seed_knobs: the starting genome's knobs.
        train_qs / val_qs: leakage-safe splits (see polyevolve.bench.temporal_split).
        generations / pop: search budget.
        fitness_fn: scorer (default: calibration ``-brier`` over `train_pools`).
        train_pools / val_pools: frozen evidence aligned 1:1 with the questions.
        complexity_lambda: weight on ``knob_complexity`` subtracted from SELECTION fitness
            (0 disables; the penalty never affects the *reported* raw fitness).
        propose_prompt: enable the LLM prompt-mutation operator.
        build_model: injected model factory for prompt mutation (defaults to
            polyevolve.models.build_model; pass a mock in tests).
        seed: RNG seed for reproducibility.
    """
    rng = random.Random(seed)

    if build_model is None and propose_prompt:
        from polyevolve.models import build_model as _bm  # noqa: PLC0415 - lazy; keeps import cheap

        build_model = _bm

    # Per-split fitness so each split scores against its OWN frozen pools. Precedence:
    # fitness_factory (pools-aware, the portable path) > fitness_fn (baked) > calibration.
    if fitness_factory is not None:
        train_fit: FitnessFn = fitness_factory(train_pools)
        val_fit: FitnessFn = fitness_factory(val_pools)
    else:
        train_fit = fitness_fn or make_calibration_fitness(train_pools)
        val_fit = fitness_fn or make_calibration_fitness(val_pools)

    def selection_score(knobs: SeedKnobs, raw_train: float) -> float:
        return raw_train - complexity_lambda * knob_complexity(knobs)

    def evaluate(knobs: SeedKnobs, generation: int) -> Individual:
        genome = make_seed_genome(knobs)
        return Individual(
            knobs=knobs,
            train_fitness=train_fit(genome, train_qs),
            val_fitness=val_fit(genome, val_qs),
            generation=generation,
        )

    seed_ind = evaluate(seed_knobs, 0)
    history: list[Individual] = [seed_ind]
    # the population we carry forward, always including the seed (elitism on the seed).
    population: list[Individual] = [seed_ind]

    for gen in range(1, generations + 1):
        parents = population[: max(1, pop // 2)]
        children: list[Individual] = []
        for _ in range(pop):
            parent = rng.choice(parents)
            child_knobs = _mutate(
                parent.knobs, rng, propose_prompt=propose_prompt, build_model=build_model
            )
            child = evaluate(child_knobs, gen)
            children.append(child)
            history.append(child)
        # (mu + lambda): keep the best of parents+children by SELECTION score on train.
        population = sorted(
            population + children,
            key=lambda ind: selection_score(ind.knobs, ind.train_fitness),
            reverse=True,
        )[:pop]
        if progress is not None:
            best = population[0]
            progress(gen, generations, best.train_fitness, best.val_fitness)

    champion = max(population, key=lambda ind: selection_score(ind.knobs, ind.train_fitness))
    return Result(
        best_knobs=champion.knobs,
        best_train_fitness=champion.train_fitness,
        best_val_fitness=champion.val_fitness,
        seed_train_fitness=seed_ind.train_fitness,
        seed_val_fitness=seed_ind.val_fitness,
        history=history,
    )


class EvolutionOptimizer:
    """The built-in `Optimizer` (thin OO wrapper over `run_evolution`)."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def optimize(
        self,
        seed_knobs: SeedKnobs,
        train_qs: Sequence[Question],
        val_qs: Sequence[Question],
    ) -> Result:
        return run_evolution(seed_knobs, train_qs, val_qs, **self._kwargs)


# ShinkaEvolve full-program optimizer lives in `polyevolve.evolve.shinka` (it shells out to
# a separate venv, so it is kept out of this module to avoid the import surface).
