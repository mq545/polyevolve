"""Tests for joint event inference (joint.py) - MockModel, no network."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from polyevolve.reason import joint as J
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Forecast, Question

AS_OF = datetime(2026, 3, 1, tzinfo=UTC)


class DistModel:
    name = "mock"

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.calls = 0

    def complete_with_tool(self, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {"input": self._payload, "usage": {}}


def _patch(monkeypatch, model):  # noqa: ANN001
    monkeypatch.setattr(J, "build_model", lambda **_: model)


def _q(qid: str, ev: str = "RACE") -> Question:
    return Question(id=qid, text=f"Will {qid} win?", as_of=AS_OF, event_id=ev)


def _pool(txt: str) -> EvidencePool:
    return EvidencePool(items=[EvidenceItem(text=txt, source="polls", date=AS_OF)])


def test_distribution_maps_to_markets(monkeypatch) -> None:  # noqa: ANN001
    model = DistModel(
        {
            "probabilities": [{"index": 1, "probability": 0.6}, {"index": 2, "probability": 0.25}],
            "other_probability": 0.15,
            "confidence": 0.7,
            "reasoning": "A leads",
        }
    )
    _patch(monkeypatch, model)
    fcs = J.forecast_event([_q("A"), _q("B")], [_pool("polls A>B"), _pool("polls A>B")])
    assert len(fcs) == 2
    assert abs(fcs[0].p_yes - 0.6) < 1e-9 and abs(fcs[1].p_yes - 0.25) < 1e-9
    assert model.calls == 1  # ONE call for the whole event
    assert sum(f.p_yes for f in fcs) + 0.15 <= 1.0 + 1e-9  # coherent (+ other)


def test_missing_entry_gets_residual(monkeypatch) -> None:  # noqa: ANN001
    # model only answers outcome 1; outcome 2 must receive the residual mass.
    model = DistModel(
        {
            "probabilities": [{"index": 1, "probability": 0.7}],
            "other_probability": 0.0,
            "confidence": 0.6,
            "reasoning": "x",
        }
    )
    _patch(monkeypatch, model)
    fcs = J.forecast_event([_q("A"), _q("B")], [_pool("p"), _pool("p")])
    assert abs(fcs[0].p_yes - 0.7) < 1e-9
    assert abs(fcs[1].p_yes - 0.3) < 1e-9  # residual 1-0.7


def test_overround_rescaled(monkeypatch) -> None:  # noqa: ANN001
    model = DistModel(
        {
            "probabilities": [{"index": 1, "probability": 0.8}, {"index": 2, "probability": 0.9}],
            "other_probability": 0.0,
            "confidence": 0.6,
            "reasoning": "x",
        }
    )
    _patch(monkeypatch, model)
    fcs = J.forecast_event([_q("A"), _q("B")], [_pool("p"), _pool("p")])
    assert abs(sum(f.p_yes for f in fcs) - 1.0) < 1e-9  # 0.8+0.9 -> rescaled to sum 1


def test_exhaustive_pins_other_zero(monkeypatch) -> None:  # noqa: ANN001
    model = DistModel(
        {
            "probabilities": [{"index": 1, "probability": 0.4}, {"index": 2, "probability": 0.4}],
            "other_probability": 0.5,
            "confidence": 0.6,
            "reasoning": "x",
        }
    )
    _patch(monkeypatch, model)
    fcs = J.forecast_event([_q("A"), _q("B")], [_pool("p"), _pool("p")], exhaustive=True)
    # other forced to 0, listed mass (0.8) kept (<=1, no rescale)
    assert abs(fcs[0].p_yes - 0.4) < 1e-9 and abs(fcs[1].p_yes - 0.4) < 1e-9


def test_model_error_fail_soft(monkeypatch) -> None:  # noqa: ANN001
    class Boom:
        name = "boom"

        def complete_with_tool(self, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("no json")

    _patch(monkeypatch, Boom())
    fcs = J.forecast_event([_q("A"), _q("B")], [_pool("p"), _pool("p")])
    assert all(f.size == 0.0 for f in fcs) and all(0.0 < f.p_yes < 1.0 for f in fcs)


def test_joint_genome_over_groups_and_falls_back(monkeypatch) -> None:  # noqa: ANN001
    model = DistModel(
        {
            "probabilities": [{"index": 1, "probability": 0.6}, {"index": 2, "probability": 0.4}],
            "other_probability": 0.0,
            "confidence": 0.7,
            "reasoning": "x",
        }
    )
    _patch(monkeypatch, model)
    # two siblings in RACE + one lone market in SOLO -> fallback used for the lone one.
    qs = [_q("A", "RACE"), _q("B", "RACE"), _q("Z", "SOLO")]
    pools = [_pool("p"), _pool("p"), _pool("p")]

    def fallback(q, pool):  # noqa: ANN001
        return Forecast(p_yes=0.11, size=0.0, confidence=0.5, rationale="fallback")

    fcs = J.joint_genome_over(qs, pools, fallback=fallback)
    assert len(fcs) == 3
    assert abs(fcs[0].p_yes - 0.6) < 1e-9 and abs(fcs[1].p_yes - 0.4) < 1e-9  # joint
    assert abs(fcs[2].p_yes - 0.11) < 1e-9  # singleton -> fallback
    assert model.calls == 1  # one joint call for the 2-market event
