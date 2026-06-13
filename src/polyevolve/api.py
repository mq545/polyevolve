"""PolyEvolve - the public command surface.

Six composable verbs, each a thin wrapper over an internal module, so the whole
platform reads as a pipeline you can run in a notebook, a CLI, or a skill:

    import polyevolve.api as pe

    qs    = pe.markets(source="manifold", path="data/fbench.jsonl")  # 1. resolved questions
    pools = pe.gather(qs, cache_path="data/pools.jsonl")             # 2. frozen evidence
    g     = pe.seed(calibrate_coeff=1.3, use_ensemble=True)          # 3. a genome
    score = pe.score(g, qs, pools, objective="calibration")          # 4. measure it
    best  = pe.evolve(qs, pools, objective="return", generations=5)  # 5. search for edge
    fc    = pe.forecast(best.genome, qs[0], pools[0])                # 6. one prediction

Design rules: each verb does ONE thing, takes plain values, and returns plain
values (lists, dicts, dataclasses) - no hidden global state, no surprise network
calls except inside `gather` (which caches). This is the surface we open-source and
the surface ShinkaEvolve/OpenEvolve adapters call into.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from polyevolve.bench.datasets import load_manifold, load_polymarket_resolved
from polyevolve.bench.pools import gather_pools
from polyevolve.evolve.optimizer import Result, run_evolution
from polyevolve.evolve.scoring import Objective, evaluate_genome, make_fitness
from polyevolve.reason.dsl import EvidencePool, Forecast, Genome, Question
from polyevolve.reason.seed import SeedKnobs, make_seed_genome, run_genome

__all__ = [
    "markets",
    "gather",
    "seed",
    "score",
    "evolve",
    "forecast",
    "EvolveResult",
    "SeedKnobs",
    "Question",
    "EvidencePool",
    "Forecast",
    "Genome",
]


# --------------------------------------------------------------------------------------
# 1. markets - load resolved, point-in-time forecasting questions
# --------------------------------------------------------------------------------------
def markets(
    source: str = "manifold",
    *,
    path: str | Path | None = None,
    db_url: str | None = None,
    snapshot_set: str | None = None,
    limit: int | None = None,
) -> list[Question]:
    """Load resolved questions to forecast/evolve against.

    ``source="manifold"`` reads a frozen fbench jsonl (``path`` required) - the
    calibration corpus. ``source="polymarket"`` reads resolved betting markets from
    Postgres (``eval_snapshots``) carrying price + outcome + execution metadata -
    the corpus the net-of-spread RETURN fitness needs. Ground truth rides on each
    `Question` but is never visible to a genome (only the bench reads it).
    """
    if source == "manifold":
        if path is None:
            raise ValueError("markets(source='manifold') requires path=<fbench.jsonl>")
        return load_manifold(path)
    if source == "polymarket":
        return load_polymarket_resolved(db_url=db_url, limit=limit, snapshot_set=snapshot_set)
    raise ValueError(f"unknown markets source {source!r} (use 'manifold' or 'polymarket')")


# --------------------------------------------------------------------------------------
# 2. gather - build the frozen, leakage-safe evidence pools (see bench.pools)
# --------------------------------------------------------------------------------------
def gather(
    questions: Sequence[Question],
    *,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> list[EvidencePool]:
    """Gather one frozen `EvidencePool` per question (aligned 1:1), cached to disk.

    Each pool is point-in-time (``<= as_of``) and leakage-audited. Thin wrapper over
    `bench.pools.gather_pools`; the production `DataRegistry` is used by default.
    """
    return gather_pools(questions, cache_path=cache_path, refresh=refresh)


# --------------------------------------------------------------------------------------
# 3. seed - construct a genome from knobs
# --------------------------------------------------------------------------------------
def seed(**knobs: object) -> Genome:
    """Build a seed `Genome` from `SeedKnobs` overrides (any unspecified knob defaults).

    e.g. ``pe.seed(use_ensemble=True, ensemble_k=5, calibrate_coeff=1.5)``. The
    returned genome is a plain ``(Question, EvidencePool) -> Forecast`` callable.
    """
    return make_seed_genome(SeedKnobs(**knobs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# 4. score - measure a genome (the portable combined_score dict)
# --------------------------------------------------------------------------------------
def score(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
    *,
    objective: Objective = "calibration",
    edge_threshold: float = 0.05,
) -> dict[str, float]:
    """Score a genome and return the metrics dict (``combined_score`` is the fitness).

    ``objective="calibration"`` -> ``combined_score = -brier``; ``objective="return"``
    -> ``combined_score = mean_event_roi`` (net-of-spread, adversarial sim). This is
    the exact dict ShinkaEvolve/OpenEvolve consume - see `evolve.scoring`.
    """
    return evaluate_genome(
        genome, questions, pools, objective=objective, edge_threshold=edge_threshold
    )


# --------------------------------------------------------------------------------------
# 5. evolve - search knob-space for the best genome
# --------------------------------------------------------------------------------------
@dataclass
class EvolveResult:
    """The champion genome + the raw optimizer `Result` (lineage, seed vs best scores)."""

    genome: Genome
    knobs: SeedKnobs
    train_fitness: float
    val_fitness: float
    seed_train_fitness: float
    seed_val_fitness: float
    raw: Result

    @property
    def improved(self) -> bool:
        return self.raw.improved


def evolve(
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
    *,
    objective: Objective = "calibration",
    edge_threshold: float = 0.05,
    val_questions: Sequence[Question] | None = None,
    val_pools: Sequence[EvidencePool] | None = None,
    start: SeedKnobs | None = None,
    generations: int = 5,
    pop: int = 6,
    complexity_lambda: float = 0.0,
    seed_value: int = 0,
    progress: Callable[[int, int, float, float], None] | None = None,
    optimizer: str = "builtin",
    mutator: str = "local/qwen3:30b-a3b-instruct-2507-q4_K_M@http://localhost:11434/v1",
    num_islands: int = 2,
    archive_size: int = 20,
) -> EvolveResult:
    """Evolve a genome to maximize ``combined_score`` for the chosen objective.

    Selects on ``questions`` (train) and reports held-out fitness on
    ``val_questions`` if given (else val == train). Pools, if supplied, must align
    1:1 with their questions and freeze the evidence the search runs over. Returns
    an `EvolveResult` carrying the ready-to-use champion genome.
    """
    train_qs = list(questions)
    val_qs = list(val_questions) if val_questions is not None else train_qs
    train_pools = list(pools) if pools is not None else None
    vpools = (
        list(val_pools)
        if val_pools is not None
        else (train_pools if val_questions is None else None)
    )

    # Full-program search: delegate the EVOLVE-BLOCK rewrite to ShinkaEvolve (out-of-process,
    # separate venv). Same EvolveResult shape; the champion is evolved CODE, loaded back here.
    if optimizer == "shinka":
        from polyevolve.evolve.shinka import ShinkaEvolveOptimizer, load_program_genome

        opt = ShinkaEvolveOptimizer(
            objective=objective,
            generations=generations,
            mutator=mutator,
            num_islands=num_islands,
            archive_size=archive_size,
        )
        res = opt.optimize(
            start or SeedKnobs(), train_qs, val_qs, train_pools=train_pools, val_pools=vpools
        )
        champ = (
            load_program_genome(res.champion_path)
            if res.champion_path
            else make_seed_genome(res.best_knobs)
        )
        return EvolveResult(
            genome=champ,
            knobs=res.best_knobs,
            train_fitness=res.best_train_fitness,
            val_fitness=res.best_val_fitness,
            seed_train_fitness=res.seed_train_fitness,
            seed_val_fitness=res.seed_val_fitness,
            raw=res,
        )
    if optimizer != "builtin":
        raise ValueError(f"unknown optimizer {optimizer!r} (use 'builtin' or 'shinka')")

    # Pools-aware factory: the optimizer builds the SAME objective for each split but
    # bound to that split's frozen pools, so val never scores against train evidence.
    def _factory(p: Sequence[EvidencePool] | None) -> object:
        return make_fitness(p, objective=objective, edge_threshold=edge_threshold)

    result = run_evolution(
        start or SeedKnobs(),
        train_qs,
        val_qs,
        generations=generations,
        pop=pop,
        fitness_factory=_factory,  # type: ignore[arg-type]
        train_pools=train_pools,
        val_pools=vpools,
        complexity_lambda=complexity_lambda,
        seed=seed_value,
        progress=progress,
    )
    return EvolveResult(
        genome=make_seed_genome(result.best_knobs),
        knobs=result.best_knobs,
        train_fitness=result.best_train_fitness,
        val_fitness=result.best_val_fitness,
        seed_train_fitness=result.seed_train_fitness,
        seed_val_fitness=result.seed_val_fitness,
        raw=result,
    )


# --------------------------------------------------------------------------------------
# 6. forecast - run a genome on one question
# --------------------------------------------------------------------------------------
def forecast(
    genome: Genome,
    question: Question,
    pool: EvidencePool | None = None,
) -> Forecast:
    """Produce one `Forecast` (p_yes, size, confidence, rationale) for a question."""
    return run_genome(genome, question, pool or EvidencePool(items=[]))
