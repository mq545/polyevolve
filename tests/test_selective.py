"""Selective prediction (reject option): abstention flows to return + coverage metrics."""

from __future__ import annotations

from datetime import datetime

from polyevolve.bench import evaluate_calibration_selective
from polyevolve.bench.returns import evaluate_return
from polyevolve.reason.dsl import EvidencePool, Forecast, Question

AS_OF = datetime(2024, 1, 1)


def _q(qid: str, *, outcome: bool, price: float) -> Question:
    return Question(
        id=qid,
        text="Will X?",
        as_of=AS_OF,
        outcome=outcome,
        market_price=price,
        crowd_prob=price,
        event_id=qid,
    )


def _genome(table: dict[str, Forecast]):
    def g(q: Question, pool: EvidencePool) -> Forecast:
        return table[q.id]

    return g


def test_return_honors_abstention_and_reports_coverage() -> None:
    qs = [
        _q("a", outcome=True, price=0.5),
        _q("b", outcome=False, price=0.5),
        _q("c", outcome=True, price=0.5),
    ]
    # size-aware genome: bets a & c, ABSTAINS on b (size 0)
    g = _genome(
        {
            "a": Forecast(p_yes=0.9, size=0.5, confidence=0.9),
            "b": Forecast(p_yes=0.5, size=0.0, confidence=0.3),
            "c": Forecast(p_yes=0.9, size=0.5, confidence=0.9),
        }
    )
    out = evaluate_return(g, qs)
    # eligible=3, traded=2 (b abstained) -> coverage 2/3
    assert out["n_eligible"] == 3.0
    assert abs(out["coverage"] - 2 / 3) < 1e-9


def test_return_non_sizing_genome_trades_all() -> None:
    qs = [_q("a", outcome=True, price=0.5), _q("b", outcome=False, price=0.5)]
    # genome never sets size (all 0) -> treated as non-selective, coverage 1.0
    g = _genome(
        {
            "a": Forecast(p_yes=0.9, size=0.0, confidence=0.9),
            "b": Forecast(p_yes=0.1, size=0.0, confidence=0.9),
        }
    )
    out = evaluate_return(g, qs)
    assert out["coverage"] == 1.0 and out["n_eligible"] == 2.0


def test_return_respect_abstention_false_trades_all() -> None:
    qs = [_q("a", outcome=True, price=0.5), _q("b", outcome=False, price=0.5)]
    g = _genome(
        {
            "a": Forecast(p_yes=0.9, size=0.5, confidence=0.9),
            "b": Forecast(p_yes=0.5, size=0.0, confidence=0.3),
        }
    )
    out = evaluate_return(g, qs, respect_abstention=False)
    assert out["coverage"] == 1.0  # abstention ignored -> both traded


def test_selective_calibration_sharpens_covered_subset() -> None:
    # a,b: confident AND correct; c,d: unconfident AND wrong. Covering only confident
    # ones should yield a much better Brier than scoring all.
    qs = [
        _q("a", outcome=True, price=0.5),
        _q("b", outcome=False, price=0.5),
        _q("c", outcome=True, price=0.5),
        _q("d", outcome=False, price=0.5),
    ]
    g = _genome(
        {
            "a": Forecast(p_yes=0.95, size=0.5, confidence=0.9),
            "b": Forecast(p_yes=0.05, size=0.5, confidence=0.9),
            "c": Forecast(p_yes=0.10, size=0.0, confidence=0.2),  # wrong + unconfident
            "d": Forecast(p_yes=0.90, size=0.0, confidence=0.2),  # wrong + unconfident
        }
    )
    out = evaluate_calibration_selective(g, qs, min_confidence=0.5)
    assert out["coverage"] == 0.5 and out["n_covered"] == 2.0
    assert out["brier"] < 0.02  # covered subset is sharp
    assert out["brier_all"] > out["brier"]  # all-markets Brier is worse
