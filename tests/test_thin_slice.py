"""END-TO-END thin-slice smoke test: reason + bench + evolve wired together on a MOCK.

This is the integration gate for the four parallel-built modules
(``polyevolve.reason.{dsl,nodes,seed}``, ``polyevolve.bench.{datasets,scoring,splits}`` and
``polyevolve.bench.evaluate_calibration``, ``polyevolve.evolve.{fitness,optimizer}``).

Nothing here touches the network: every model call routes through ``MockModel``, injected
into the node library via ``polyevolve.reason.nodes.build_model`` (so ``bench`` runs the seed
genome offline) and passed explicitly to ``run_evolution`` for the prompt-mutation operator.

What it asserts:
  1. ``make_seed_genome`` -> ``bench.evaluate_calibration`` yields a Brier in [0, 1] over a
     handful of toy resolved Questions.
  2. ``run_evolution`` for 1-2 tiny generations returns knobs that are improved-or-equal vs
     the seed on the selection (train) split, and never regress.

Run:
    uv run pytest tests/test_thin_slice.py -q
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from polyevolve.bench import evaluate_calibration, temporal_split
from polyevolve.evolve import make_calibration_fitness, run_evolution
from polyevolve.reason import nodes
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Question
from polyevolve.reason.seed import SeedKnobs, make_seed_genome

# --------------------------------------------------------------------------------------
# MOCK model - answers every tool the seed pipeline forces, with a deterministic payload.
# --------------------------------------------------------------------------------------


class MockModel:
    """Returns a canned tool-call payload keyed by tool name; records calls; no network."""

    name = "mock"

    def __init__(self, p_yes: float = 0.7, confidence: float = 0.8) -> None:
        self._p_yes = p_yes
        self._confidence = confidence
        self.calls: list[str] = []

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = tool["name"]
        self.calls.append(name)
        payloads: dict[str, dict[str, Any]] = {
            "submit_prediction": {
                "probability_yes": self._p_yes,
                "confidence": self._confidence,
                "reasoning": "mock reasoning",
            },
            "submit_subquestions": {"sub_questions": ["a?", "b?"]},
            "submit_revision": {
                "critique": "mock critique",
                "revised_probability_yes": self._p_yes,
                "confidence": self._confidence,
            },
            "submit_selection": {"indices": [1, 2]},
            "submit_margin": {
                "margin_mean": 1.0,
                "margin_std": 2.0,
                "reasoning": "mock margin",
            },
            "submit_prompt": {
                "system_prompt": "Improved mock forecaster prompt. Call submit_prediction once."
            },
        }
        if name not in payloads:
            raise AssertionError(f"unexpected tool call: {name}")
        return {"input": payloads[name], "usage": {}}


def _mock_build_model(*, model_id: str, anthropic_api_key: str | None = None) -> MockModel:
    return MockModel()


# --------------------------------------------------------------------------------------
# toy corpus: 5 resolved, point-in-time binary questions with a one-item evidence pool.
# --------------------------------------------------------------------------------------


def _toy_questions() -> list[Question]:
    out: list[Question] = []
    for i in range(5):
        out.append(
            Question(
                id=f"q{i}",
                text=f"Will toy event {i} happen?",
                as_of=datetime(2026, 1, 1 + i, tzinfo=UTC),
                resolution_criteria="YES if the toy event occurs.",
                category="toy",
                outcome=(i % 2 == 0),  # mixed outcomes so Brier is non-degenerate
                market_price=0.5,
            )
        )
    return out


def _toy_pools(n: int) -> list[EvidencePool]:
    return [
        EvidencePool(
            items=[
                EvidenceItem(
                    text=f"Dated evidence for toy event {i}.",
                    source="toy",
                    date=datetime(2026, 1, 1 + i, tzinfo=UTC),
                )
            ]
        )
        for i in range(n)
    ]


@pytest.fixture
def _patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "build_model", lambda **_: MockModel())


# --------------------------------------------------------------------------------------
# 1. seed genome -> bench.evaluate_calibration : a valid Brier in [0, 1].
# --------------------------------------------------------------------------------------


def test_seed_genome_evaluates_to_valid_brier(_patched: None) -> None:
    questions = _toy_questions()
    pools = _toy_pools(len(questions))
    genome = make_seed_genome(SeedKnobs())

    result = evaluate_calibration(genome, questions, pools)

    assert result["n"] == float(len(questions))
    assert 0.0 <= result["brier"] <= 1.0
    assert 0.0 <= result["ece"] <= 1.0


# --------------------------------------------------------------------------------------
# 2. run_evolution for 1-2 tiny generations : improved-or-equal knobs, no regression.
# --------------------------------------------------------------------------------------


def test_run_evolution_improves_or_holds(_patched: None) -> None:
    questions = _toy_questions()
    split = temporal_split(questions, train_frac=0.6, val_frac=0.2)
    # align frozen pools 1:1 with each split's questions.
    all_pools = {q.id: p for q, p in zip(questions, _toy_pools(len(questions)), strict=True)}
    train_pools = [all_pools[q.id] for q in split.train]
    val_pools = [all_pools[q.id] for q in split.val]

    result = run_evolution(
        SeedKnobs(),
        split.train,
        split.val,
        generations=2,
        pop=4,
        train_pools=train_pools,
        val_pools=val_pools,
        propose_prompt=True,
        build_model=_mock_build_model,  # prompt-mutation operator stays offline
        seed=0,
    )

    # elitism on the seed guarantees the champion never regresses on the selection split.
    assert result.best_train_fitness >= result.seed_train_fitness
    assert result.improved
    # fitness is -brier, so it lives in [-1, 0].
    assert -1.0 <= result.best_train_fitness <= 0.0
    assert isinstance(result.best_knobs, SeedKnobs)
    assert len(result.history) >= 1


def test_calibration_fitness_matches_negative_brier(_patched: None) -> None:
    questions = _toy_questions()
    pools = _toy_pools(len(questions))
    genome = make_seed_genome(SeedKnobs())

    fit = make_calibration_fitness(pools)(genome, questions)
    brier = evaluate_calibration(genome, questions, pools)["brier"]

    assert fit == pytest.approx(-brier)
