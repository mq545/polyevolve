"""End-to-end tests for the SEED GENOME - all model calls go through a MOCK (no network).

We monkeypatch `polyevolve.reason.nodes.build_model` (the single place node factories build
their client) so the whole seed pipeline runs offline.

Run:
    uv run pytest tests/test_seed.py -q
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from polyevolve.reason import nodes, seed
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Forecast, Question

AS_OF = datetime(2026, 1, 10, tzinfo=UTC)


class MockModel:
    """Returns a canned tool-call payload keyed by tool name (mirrors test_nodes)."""

    name = "mock"

    def __init__(self, responses: dict[str, dict[str, Any]]):
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"tool": tool["name"], "user": user_content, "meta": metadata})
        if tool["name"] not in self._responses:
            raise AssertionError(f"unexpected tool call: {tool['name']}")
        return {"input": self._responses[tool["name"]], "usage": {}}


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: MockModel) -> None:
    monkeypatch.setattr(nodes, "build_model", lambda **_: model)


def _ev(text: str, day: int) -> EvidenceItem:
    return EvidenceItem(text=text, source="src", date=datetime(2026, 1, day, tzinfo=UTC))


def _question(market_price: float | None = 0.5) -> Question:
    return Question(
        id="q1",
        text="Will the Tisza party win the most list votes in the 2026 Hungarian election?",
        as_of=AS_OF,
        resolution_criteria="YES if Tisza receives the plurality of national list votes.",
        category="foreign_politics",
        market_price=market_price,
    )


def _pool() -> EvidencePool:
    return EvidencePool(
        items=[
            _ev("Tisza party leads list votes in latest reliable poll", 5),
            _ev("Unrelated weather report for the weekend", 6),
            _ev("Hungarian election Tisza plurality forecast holds", 7),
        ]
    )


def _assert_valid_forecast(fc: Forecast) -> None:
    assert isinstance(fc, Forecast)
    assert 0.0 <= fc.p_yes <= 1.0
    assert -1.0 <= fc.size <= 1.0
    assert 0.0 <= fc.confidence <= 1.0


def test_seed_genome_runs_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.75, "confidence": 0.8, "reasoning": "r"}}
    )
    _patch_model(monkeypatch, mm)

    genome = seed.make_seed_genome(seed.SeedKnobs(model_id="mock/x"))
    fc = seed.run_genome(genome, _question(market_price=0.5), _pool())

    _assert_valid_forecast(fc)
    # calibrate(coeff=1.3) softens 0.75 toward 0.5 but stays above it.
    assert 0.5 < fc.p_yes < 0.75
    # confident (0.8) + divergent from 0.5 -> positive YES stake.
    assert fc.size > 0
    assert mm.calls[0]["tool"] == "submit_prediction"


def test_seed_forecast_callable_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.6, "confidence": 0.7, "reasoning": "r"}}
    )
    _patch_model(monkeypatch, mm)
    fc = seed.forecast(_question(), _pool(), knobs=seed.SeedKnobs(model_id="mock/x"))
    _assert_valid_forecast(fc)


def test_seed_abstains_when_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.9, "confidence": 0.1, "reasoning": "thin"}}
    )
    _patch_model(monkeypatch, mm)
    knobs = seed.SeedKnobs(model_id="mock/x", abstain_min_conf=0.45)
    fc = seed.forecast(_question(market_price=0.5), _pool(), knobs=knobs)
    assert fc.size == 0.0  # abstained


def test_seed_abstains_when_no_market_price(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.8, "confidence": 0.9, "reasoning": "r"}}
    )
    _patch_model(monkeypatch, mm)
    knobs = seed.SeedKnobs(model_id="mock/x")
    fc = seed.forecast(_question(market_price=None), _pool(), knobs=knobs)
    # high confidence keeps it past abstain, but size_by_edge has no price -> 0.
    assert fc.size == 0.0


def test_seed_ensemble_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class CyclingModel(MockModel):
        def __init__(self, probs: list[float]):
            super().__init__({})
            self._probs = probs
            self._i = 0

        def complete_with_tool(self, **kw: Any) -> dict[str, Any]:  # type: ignore[override]
            p = self._probs[self._i % len(self._probs)]
            self._i += 1
            return {
                "input": {"probability_yes": p, "confidence": 0.8, "reasoning": "d"},
                "usage": {},
            }

    mm = CyclingModel([0.55, 0.6, 0.65])
    _patch_model(monkeypatch, mm)
    knobs = seed.SeedKnobs(model_id="mock/x", use_ensemble=True, ensemble_k=3)
    fc = seed.forecast(_question(market_price=0.4), _pool(), knobs=knobs)
    _assert_valid_forecast(fc)
    assert mm._i == 3  # ensemble ran k=3 model calls


def test_seed_decompose_path(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {
            "submit_subquestions": {"sub_questions": ["a?", "b?"]},
            "submit_prediction": {"probability_yes": 0.65, "confidence": 0.8, "reasoning": "r"},
        }
    )
    _patch_model(monkeypatch, mm)
    knobs = seed.SeedKnobs(model_id="mock/x", use_decompose=True)
    fc = seed.forecast(_question(market_price=0.4), _pool(), knobs=knobs)
    _assert_valid_forecast(fc)
    tools_called = {c["tool"] for c in mm.calls}
    assert tools_called == {"submit_subquestions", "submit_prediction"}


def test_make_seed_genome_default_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.5, "confidence": 0.6, "reasoning": "r"}}
    )
    _patch_model(monkeypatch, mm)
    genome = seed.make_seed_genome()  # no knobs -> defaults
    fc = genome(_question(market_price=0.5), _pool())
    _assert_valid_forecast(fc)


def test_seed_knobs_validation() -> None:
    with pytest.raises(ValueError):
        seed.SeedKnobs(calibrate_coeff=0.0)
    with pytest.raises(ValueError):
        seed.SeedKnobs(ensemble_k=0)
