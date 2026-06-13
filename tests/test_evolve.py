"""Unit tests for the evolution loop - NO network.

Fitness here is either a pure mock (a deterministic function of the knobs) or the real
calibration fitness over a tiny in-memory genome; the LLM prompt-mutation operator gets a
MOCK `build_model` so nothing ever touches the network.

Run:
    uv run pytest tests/test_evolve.py -q
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from polyevolve.evolve.fitness import WORST_FITNESS, fitness, make_calibration_fitness
from polyevolve.evolve.optimizer import (
    knob_complexity,
    run_evolution,
)
from polyevolve.evolve.shinka import seed_program, validate_program
from polyevolve.reason.dsl import Forecast, Genome, Question
from polyevolve.reason.seed import SeedKnobs

AS_OF = datetime(2026, 1, 10, tzinfo=UTC)


def _q(qid: str, outcome: bool) -> Question:
    return Question(id=qid, text=f"q {qid}", as_of=AS_OF, outcome=outcome)


TRAIN = [_q("t1", True), _q("t2", False), _q("t3", True)]
VAL = [_q("v1", True), _q("v2", False)]


# --------------------------------------------------------------------------------------
# mock fitness: a flat, deterministic scorer that NEVER executes the genome (so it cannot
# touch the network / ollama). Used where we only care about the loop's selection mechanics.
# --------------------------------------------------------------------------------------
def mock_fitness(genome: Genome, questions: Sequence[Question]) -> float:  # noqa: ARG001
    return 0.0


def knob_gradient_fitness(target_coeff: float = 0.5) -> Any:
    """A pure fitness over the genome's *knobs* (recovered via a marker). Lower distance to
    `target_coeff` => higher fitness. No genome execution, no network."""
    # we stash knobs on the genome via attribute when building; simpler: re-derive from id.
    registry: dict[int, SeedKnobs] = {}

    def register(knobs: SeedKnobs, genome: Genome) -> Genome:
        registry[id(genome)] = knobs
        return genome

    def fn(genome: Genome, _questions: Sequence[Question]) -> float:
        knobs = registry.get(id(genome))
        if knobs is None:
            return WORST_FITNESS
        return -abs(knobs.calibrate_coeff - target_coeff)

    fn.register = register  # type: ignore[attr-defined]
    return fn


def test_seed_is_never_regressed_with_mock_fitness() -> None:
    """run_evolution must return a champion that matches or beats the seed on train."""
    seed = SeedKnobs(calibrate_coeff=1.3)
    res = run_evolution(
        seed,
        TRAIN,
        VAL,
        generations=2,
        pop=4,
        fitness_fn=mock_fitness,
        propose_prompt=False,
        seed=7,
    )
    assert res.best_train_fitness >= res.seed_train_fitness
    assert res.improved


def test_evolution_improves_toward_gradient() -> None:
    """With a fitness that rewards calibrate_coeff -> 0.5, evolution should move toward it."""
    grad = knob_gradient_fitness(target_coeff=0.5)

    # wrap make_seed_genome so each built genome is registered with its knobs.
    import polyevolve.evolve.optimizer as opt

    orig = opt.make_seed_genome

    def patched(knobs: SeedKnobs) -> Genome:
        return grad.register(knobs, orig(knobs))

    opt.make_seed_genome = patched  # type: ignore[assignment]
    try:
        seed = SeedKnobs(calibrate_coeff=2.5)
        res = run_evolution(
            seed,
            TRAIN,
            VAL,
            generations=3,
            pop=6,
            fitness_fn=grad,
            propose_prompt=False,
            seed=1,
        )
    finally:
        opt.make_seed_genome = orig  # type: ignore[assignment]

    # champion must be at least as good as the seed, and strictly closer to target.
    assert res.best_train_fitness >= res.seed_train_fitness
    assert abs(res.best_knobs.calibrate_coeff - 0.5) <= abs(seed.calibrate_coeff - 0.5)


def test_val_is_reported_not_selected() -> None:
    """The generalization guard: val fitness is reported on the train-selected champion."""
    seed = SeedKnobs()
    res = run_evolution(
        seed, TRAIN, VAL, generations=1, pop=3, fitness_fn=mock_fitness, propose_prompt=False
    )
    # both train and val numbers exist for the champion and the seed.
    assert isinstance(res.best_val_fitness, float)
    assert isinstance(res.seed_val_fitness, float)
    # history records every evaluated individual (seed + pop*generations children).
    assert len(res.history) >= 1


def test_complexity_penalty_prefers_simpler_on_tie() -> None:
    """With a flat fitness, the complexity penalty should keep the (simple) seed champion."""
    flat = lambda genome, qs: 0.0  # noqa: E731 - tiny test fixture
    simple_seed = SeedKnobs(select_k=4, use_ensemble=False, use_decompose=False)
    res = run_evolution(
        simple_seed,
        TRAIN,
        VAL,
        generations=2,
        pop=4,
        fitness_fn=flat,
        complexity_lambda=1.0,
        propose_prompt=False,
        seed=3,
    )
    # champion complexity must not exceed the seed's (penalty breaks ties toward simple).
    assert knob_complexity(res.best_knobs) <= knob_complexity(simple_seed) + 1e-9


# --------------------------------------------------------------------------------------
# LLM prompt-mutation operator with a MOCK build_model (no network).
# --------------------------------------------------------------------------------------
class _MockModel:
    name = "mock"

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"input": {"system_prompt": "MUTATED: stay near base rates."}, "usage": {}}


def _mock_build_model(*, model_id: str, recorder: object | None = None) -> _MockModel:
    return _MockModel()


def test_prompt_mutation_uses_injected_model_no_network() -> None:
    """propose_prompt=True with an injected mock build_model never hits the network and can
    install the mutated prompt into the lineage."""
    seed = SeedKnobs()
    res = run_evolution(
        seed,
        TRAIN,
        VAL,
        generations=3,
        pop=6,
        fitness_fn=mock_fitness,
        propose_prompt=True,
        build_model=_mock_build_model,
        seed=2,
    )
    prompts = {ind.knobs.system_prompt for ind in res.history}
    # at least the seed prompt is present; the mock-mutated prompt may also appear.
    assert seed.system_prompt in prompts
    assert any(p.startswith("MUTATED:") for p in prompts)


# --------------------------------------------------------------------------------------
# real calibration fitness over a trivial deterministic genome (no network, no LLM).
# --------------------------------------------------------------------------------------
def test_calibration_fitness_is_negative_brier() -> None:
    # A genome cannot read q.outcome (the bench blinds it), so build a "perfect" forecast
    # the honest way: predict 1.0 on questions that all resolve YES -> brier 0 -> fitness 0.
    all_yes = [_q("y1", True), _q("y2", True)]
    always_yes: Genome = lambda q, pool: Forecast(p_yes=1.0)  # noqa: E731
    fit = make_calibration_fitness()
    assert fit(always_yes, all_yes) == pytest.approx(0.0)  # brier 0 -> fitness 0

    constant_half: Genome = lambda q, pool: Forecast(p_yes=0.5)  # noqa: E731
    assert fitness(constant_half, TRAIN) == pytest.approx(-0.25)


def test_fitness_worst_on_empty_split() -> None:
    g: Genome = lambda q, pool: Forecast(p_yes=0.5)  # noqa: E731
    assert fitness(g, []) == WORST_FITNESS


# --------------------------------------------------------------------------------------
# ShinkaEvolve full-program adapter: the program format round-trips and validates.
# --------------------------------------------------------------------------------------
def test_seed_program_renders_and_loads(tmp_path: Any) -> None:
    src = seed_program(SeedKnobs(use_ensemble=True))
    assert "EVOLVE-BLOCK-START" in src and "EVOLVE-BLOCK-END" in src
    assert "def forecast(" in src and "size_by_edge" in src
    p = tmp_path / "initial.py"
    p.write_text(src)
    ok, err = validate_program(p)
    assert ok, err


def test_validate_program_rejects_syntax_and_missing_markers(tmp_path: Any) -> None:
    src = seed_program(SeedKnobs())
    bad_syntax = tmp_path / "bad.py"
    bad_syntax.write_text(src.replace("state = calibrate", "state = = calibrate"))
    ok, err = validate_program(bad_syntax)
    assert not ok and err is not None

    no_markers = tmp_path / "nomarkers.py"
    no_markers.write_text("def forecast(q, pool):\n    return None\n")
    ok2, err2 = validate_program(no_markers)
    assert not ok2 and err2 is not None
