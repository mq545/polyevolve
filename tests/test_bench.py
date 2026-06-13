"""Bench (fitness-oracle) tests. No network: the model-backed genome uses a MockModel."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from polyevolve.bench import (
    brier,
    calibration_curve,
    ece,
    evaluate_calibration,
    load_manifold,
    load_manifold_with_pools,
    temporal_split,
)
from polyevolve.bench.splits import event_cluster
from polyevolve.reason.dsl import EvidencePool, Forecast, Question


def _utc(ts_s: float) -> datetime:
    """Naive UTC datetime from a second epoch (matches the loader's mapping)."""
    return datetime.fromtimestamp(ts_s, tz=UTC).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def test_brier_constant_half() -> None:
    # constant 0.5 against any mix of outcomes -> 0.25 exactly
    pairs = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
    assert brier(pairs) == pytest.approx(0.25)


def test_brier_perfect_and_worst() -> None:
    assert brier([(1.0, True), (0.0, False)]) == pytest.approx(0.0)
    assert brier([(0.0, True), (1.0, False)]) == pytest.approx(1.0)


def test_brier_empty_is_nan() -> None:
    assert math.isnan(brier([]))


def test_ece_perfectly_calibrated() -> None:
    # in each bin mean_pred == observed -> ECE 0
    pairs = [(0.0, False), (0.0, False), (1.0, True), (1.0, True)]
    assert ece(pairs) == pytest.approx(0.0)


def test_ece_max_miscalibration() -> None:
    # predict 1.0 but all NO -> single bin, |1.0 - 0.0| weighted 1.0
    assert ece([(1.0, False), (1.0, False)]) == pytest.approx(1.0)


def test_calibration_curve_bins() -> None:
    pairs = [(0.05, False), (0.95, True), (0.95, False)]
    curve = calibration_curve(pairs, bins=10)
    # two non-empty bins: the [0.0,0.1) bin and the [0.9,1.0] bin
    assert len(curve) == 2
    low, high = curve[0], curve[1]
    assert low.count == 1 and low.observed == pytest.approx(0.0)
    assert high.count == 2 and high.observed == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# datasets                                                                     #
# --------------------------------------------------------------------------- #
def _write_corpus(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "a",
            "question": "Will X happen?",
            "text_desc": "Some point-in-time context about X.",
            "created": 1_700_000_000_000,
            "close": 1_710_000_000_000,
            "T": 1_705_000_000_000,
            "crowd_at_T": 0.7,
            "resolution": "YES",
        },
        {
            "id": "b",
            "question": "Will Y happen?",
            "text_desc": "",
            "created": 1_600_000_000_000,
            "close": 1_610_000_000_000,
            "T": 1_605_000_000_000,
            "crowd_at_T": 0.3,
            "resolution": "NO",
        },
    ]
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_manifold_maps_fields(tmp_path: Path) -> None:
    path = _write_corpus(tmp_path)
    qs = load_manifold(path)
    assert len(qs) == 2
    a = next(q for q in qs if q.id == "a")
    assert a.text == "Will X happen?"
    assert a.outcome is True
    assert a.crowd_prob == pytest.approx(0.7)
    assert a.as_of == _utc(1_705_000_000_000 / 1000)
    b = next(q for q in qs if q.id == "b")
    assert b.outcome is False


def test_pool_present_and_empty(tmp_path: Path) -> None:
    path = _write_corpus(tmp_path)
    pairs = load_manifold_with_pools(path)
    by_id = {q.id: pool for q, pool in pairs}
    assert len(by_id["a"].items) == 1
    assert by_id["a"].items[0].text == "Some point-in-time context about X."
    assert by_id["b"].items == []  # empty text_desc -> empty pool


# --------------------------------------------------------------------------- #
# splits                                                                       #
# --------------------------------------------------------------------------- #
def _q(qid: str, ts: int, outcome: bool) -> Question:
    return Question(id=qid, text=qid, as_of=_utc(ts), outcome=outcome)


def test_temporal_split_is_time_ordered() -> None:
    qs = [_q(str(i), 1_000 + i, i % 2 == 0) for i in range(10)]
    split = temporal_split(qs, train_frac=0.6, val_frac=0.2)
    assert len(split.train) == 6 and len(split.val) == 2 and len(split.test) == 2
    # every test as_of is strictly after every train as_of (no leakage)
    assert max(q.as_of for q in split.train) < min(q.as_of for q in split.test)


def test_event_cluster_groups() -> None:
    qs = [_q("e1-a", 1, True), _q("e1-b", 2, False), _q("e2-a", 3, True)]
    clusters = event_cluster(qs, keyfn=lambda q: q.id.split("-")[0])
    assert set(clusters) == {"e1", "e2"}
    assert len(clusters["e1"]) == 2


# --------------------------------------------------------------------------- #
# evaluate_calibration                                                         #
# --------------------------------------------------------------------------- #
def test_evaluate_calibration_constant_genome() -> None:
    """Trivial constant genome (always p_yes=0.5) -> brier ~= 0.25, n correct."""

    def constant_genome(q: Question, pool: EvidencePool) -> Forecast:
        return Forecast(p_yes=0.5)

    qs = [_q("a", 1, True), _q("b", 2, False), _q("c", 3, True), _q("d", 4, False)]
    res = evaluate_calibration(constant_genome, qs)
    assert res["n"] == 4.0
    assert res["brier"] == pytest.approx(0.25)
    # constant 0.5: single bin centered at 0.5, observed 0.5 -> ECE 0
    assert res["ece"] == pytest.approx(0.0)


def test_evaluate_calibration_skips_unresolved_and_crashes() -> None:
    def crashy_genome(q: Question, pool: EvidencePool) -> Forecast:
        if q.id == "boom":
            raise RuntimeError("untrusted genome blew up")
        return Forecast(p_yes=0.9)

    qs = [
        _q("ok", 1, True),
        Question(id="unresolved", text="?", as_of=_utc(2), outcome=None),
        _q("boom", 3, True),
    ]
    res = evaluate_calibration(crashy_genome, qs)
    # unresolved skipped (no outcome), boom skipped (raised) -> only 1 scored
    assert res["n"] == 1.0


# --------------------------------------------------------------------------- #
# model-backed genome with a MOCK model (no network)                          #
# --------------------------------------------------------------------------- #
class MockModel:
    """Implements the polyevolve.contracts.Model surface; returns a canned tool input."""

    def __init__(self, p_yes: float) -> None:
        self._p_yes = p_yes
        self.calls = 0

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        return {"input": {"p_yes": self._p_yes}, "usage": {}}


def test_model_backed_genome_no_network() -> None:
    model = MockModel(p_yes=1.0)
    tool = {"name": "submit_forecast", "input_schema": {"type": "object"}}

    def llm_genome(q: Question, pool: EvidencePool) -> Forecast:
        res = model.complete_with_tool(
            cached_system_blocks=["forecast YES probability"],
            user_content=q.text,
            tool=tool,
        )
        return Forecast(p_yes=float(res["input"]["p_yes"]))

    qs = [_q("a", 1, True), _q("b", 2, True)]
    res = evaluate_calibration(llm_genome, qs)
    assert model.calls == 2
    assert res["brier"] == pytest.approx(0.0)  # predicted 1.0, both YES


def test_genome_cannot_read_outcome_to_cheat() -> None:
    """A genome that tries to read the resolved outcome must NOT score perfectly.

    The bench hands the genome `question.blinded()`, so future/answer fields are stripped.
    A cheater that returns p_yes from q.outcome therefore sees None and cannot reward-hack
    the backtest - the key guard for untrusted, LLM-authored full-program genomes.
    """

    def cheater(q: Question, pool: EvidencePool) -> Forecast:
        return Forecast(p_yes=1.0 if q.outcome else 0.0)

    qs = [_q(str(i), 1_000 + i, i % 2 == 0) for i in range(6)]
    res = evaluate_calibration(cheater, qs)
    # if outcome leaked, brier would be 0.0; blinded -> outcome is None -> always 0.0 pred.
    assert res["brier"] > 0.0


def test_blinded_strips_future_fields_keeps_decision_time_data() -> None:
    q = Question(id="x", text="?", as_of=_utc(1), outcome=True, crowd_prob=0.7, market_price=0.6)
    b = q.blinded()
    assert b.outcome is None and b.crowd_prob is None
    assert b.market_price == 0.6  # known at decision time; needed for sizing
    assert q.outcome is True  # original is untouched (bench scores against it)


def test_forecaster_never_sees_post_asof_evidence() -> None:
    """Evidence dated after the as_of cutoff must not reach the forecaster's context.

    select_evidence builds its candidate set from pool.on_or_before(as_of), so a post-cutoff
    item is dropped before ranking - and therefore never rendered into the prompt. This is
    the point-in-time guard that keeps a backtest from reading the future.
    """
    from polyevolve.reason.dsl import EvidenceItem, ReasoningState
    from polyevolve.reason.nodes import _render_evidence, select_evidence

    as_of = datetime(2025, 6, 1, tzinfo=UTC)
    pool = EvidencePool(
        items=[
            EvidenceItem(text="PRE poll", source="polls", date=datetime(2025, 5, 20, tzinfo=UTC)),
            EvidenceItem(text="POST leak", source="news", date=datetime(2025, 7, 15, tzinfo=UTC)),
        ]
    )
    q = Question(id="x", text="who wins?", as_of=as_of, outcome=True)
    state = select_evidence(k=10, mode="heuristic")(ReasoningState(question=q, pool=pool))
    texts = [e.text for e in state.selected]
    assert "PRE poll" in texts and "POST leak" not in texts
    assert "POST leak" not in _render_evidence(state.selected)  # never enters the prompt
