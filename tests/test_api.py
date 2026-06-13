"""Tests for the public 6-verb facade (polyevolve.api). Offline - no model/network."""

from __future__ import annotations

import json
from datetime import datetime

import polyevolve.api as pe
from polyevolve.reason.dsl import EvidencePool, Forecast, Question

AS_OF = datetime(2024, 6, 1)


def _const_genome(p: float = 0.7, size: float = 0.0):
    def g(q: Question, pool: EvidencePool) -> Forecast:
        return Forecast(p_yes=p, size=size, confidence=0.9, rationale="const")

    return g


def _q(qid: str, *, outcome: bool, price: float) -> Question:
    return Question(
        id=qid,
        text="Will X?",
        as_of=AS_OF,
        category="C",
        outcome=outcome,
        market_price=price,
        crowd_prob=price,
        event_id=qid,
    )


def test_markets_manifold_loads(tmp_path) -> None:  # noqa: ANN001
    row = {
        "id": "m1",
        "question": "Will it rain?",
        "text_desc": "context",
        "T": int(AS_OF.timestamp() * 1000),
        "crowd_at_T": 0.6,
        "resolution": "YES",
    }
    p = tmp_path / "fbench.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    qs = pe.markets(source="manifold", path=p)
    assert len(qs) == 1 and qs[0].id == "m1" and qs[0].outcome is True


def test_markets_unknown_source_raises() -> None:
    try:
        pe.markets(source="nope")
    except ValueError as e:
        assert "unknown markets source" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_score_calibration_combined_score() -> None:
    qs = [_q("a", outcome=True, price=0.5), _q("b", outcome=False, price=0.5)]
    out = pe.score(_const_genome(0.7), qs, objective="calibration")
    assert "combined_score" in out
    assert out["combined_score"] == -out["brier"]  # combined = -brier
    assert out["n"] == 2.0


def test_score_return_combined_score() -> None:
    qs = [_q(f"m{i}", outcome=(i % 2 == 0), price=0.5) for i in range(6)]
    out = pe.score(_const_genome(0.95, size=1.0), qs, objective="return")
    assert "combined_score" in out
    assert out["combined_score"] == out["mean_event_roi"]


def test_forecast_runs_constant_genome() -> None:
    fc = pe.forecast(_const_genome(0.42), _q("a", outcome=True, price=0.5))
    assert isinstance(fc, Forecast) and abs(fc.p_yes - 0.42) < 1e-9


def test_seed_returns_callable_genome() -> None:
    g = pe.seed(calibrate_coeff=1.5, use_ensemble=True, ensemble_k=4)
    assert callable(g)  # not invoked (would need the model)


def test_evolve_delegates_objective_and_split_pools(monkeypatch) -> None:  # noqa: ANN001
    """The facade must build a pools-aware factory and pass per-split pools through."""
    captured: dict[str, object] = {}

    class FakeResult:
        best_knobs = pe.SeedKnobs() if hasattr(pe, "SeedKnobs") else None
        best_train_fitness = 0.1
        best_val_fitness = 0.2
        seed_train_fitness = 0.0
        seed_val_fitness = 0.05
        improved = True

    from polyevolve.reason.seed import SeedKnobs

    fake = FakeResult()
    fake.best_knobs = SeedKnobs()

    def fake_run_evolution(seed_knobs, train_qs, val_qs, **kw):  # noqa: ANN001
        captured.update(kw)
        captured["n_train"] = len(train_qs)
        captured["n_val"] = len(val_qs)
        return fake

    monkeypatch.setattr(pe, "run_evolution", fake_run_evolution)

    train = [_q("a", outcome=True, price=0.5)]
    val = [_q("b", outcome=False, price=0.5), _q("c", outcome=True, price=0.4)]
    tpools = [EvidencePool(items=[])]
    vpools = [EvidencePool(items=[]), EvidencePool(items=[])]

    res = pe.evolve(
        train, tpools, objective="return", val_questions=val, val_pools=vpools, generations=1
    )

    # a pools-aware factory was supplied (not a baked fitness_fn)
    factory = captured["fitness_factory"]
    assert callable(factory)
    # the factory yields a FitnessFn for whatever pools the optimizer hands it
    assert callable(factory(tpools))
    # per-split pools threaded correctly
    assert captured["train_pools"] == tpools
    assert captured["val_pools"] == vpools
    assert captured["n_train"] == 1 and captured["n_val"] == 2
    assert res.genome is not None and res.val_fitness == 0.2 and res.improved
