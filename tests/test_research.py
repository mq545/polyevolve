"""Tests for the agentic research node - fake registry + MockModel, NO network.

Crucially asserts the LEAKAGE boundary: every tool dispatch receives question.as_of.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from polyevolve.reason import research as R
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Question, ReasoningState

AS_OF = datetime(2026, 1, 10, tzinfo=UTC)


class FakeSource:
    """A DataSource stand-in that records the as_of it was called with."""

    def __init__(self, name: str, text: str):
        self.name = name
        self._text = text
        self.seen_as_of: list[Any] = []

    def fetch(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self.seen_as_of.append(ctx.get("as_of"))
        return {"text": self._text, "entities": ctx.get("tags")}

    def render(self, payload: dict[str, Any]) -> str:
        return self._text


class PlanModel:
    """Returns a scripted research plan, then 'enough' on later rounds."""

    name = "mock"

    def __init__(self, plans: list[dict[str, Any]]):
        self._plans = plans
        self.calls = 0

    def complete_with_tool(self, **kw: Any) -> dict[str, Any]:
        i = min(self.calls, len(self._plans) - 1)
        self.calls += 1
        return {"input": self._plans[i], "usage": {}}


def _state() -> ReasoningState:
    q = Question(
        id="q1", text="Will party X win the 2026 election?", as_of=AS_OF, category="Country"
    )
    return ReasoningState(question=q, pool=EvidencePool(items=[]))


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: PlanModel) -> None:
    monkeypatch.setattr(R, "build_model", lambda **_: model)


def test_research_gathers_and_enforces_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    polls = FakeSource("polls", "X 51% Y 49% (Jan 2026)")
    reg = R.ToolRegistry([R.RetrievalTool("polls", "poll tables", polls)])
    model = PlanModel(
        [{"requests": [{"tool": "polls", "entities": ["Country", "X"]}], "enough": True}]
    )
    _patch_model(monkeypatch, model)
    st = _state()
    out = R.research(registry=reg)(st)
    # gathered the poll item, stamped with the cutoff
    assert len(out.selected) == 1 and out.selected[0].source == "polls"
    assert out.selected[0].date == AS_OF
    # LEAKAGE: the tool was called with the question cutoff
    assert polls.seen_as_of == [AS_OF]
    assert "polls" in out.beliefs["research_log"]


def test_research_skips_unknown_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = R.ToolRegistry([R.RetrievalTool("polls", "p", FakeSource("polls", "data"))])
    model = PlanModel([{"requests": [{"tool": "web_search", "entities": ["x"]}], "enough": True}])
    _patch_model(monkeypatch, model)
    out = R.research(registry=reg)(_state())
    assert out.selected == []
    assert "unknown tool" in out.beliefs["research_log"]


def test_research_refines_over_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    news = FakeSource("news", "headline")
    polls = FakeSource("polls", "X 51%")
    reg = R.ToolRegistry([R.RetrievalTool("news", "n", news), R.RetrievalTool("polls", "p", polls)])
    model = PlanModel(
        [
            {"requests": [{"tool": "news", "entities": ["Country"]}], "enough": False},
            {"requests": [{"tool": "polls", "entities": ["Country"]}], "enough": True},
        ]
    )
    _patch_model(monkeypatch, model)
    out = R.research(registry=reg, max_rounds=3)(_state())
    assert {it.source for it in out.selected} == {"news", "polls"}
    assert model.calls == 2  # stopped after 'enough' on round 2


def test_research_merges_into_pool_replacing_same_source(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = R.ToolRegistry([R.RetrievalTool("polls", "p", FakeSource("polls", "fresh polls"))])
    model = PlanModel([{"requests": [{"tool": "polls", "entities": ["X"]}], "enough": True}])
    _patch_model(monkeypatch, model)
    st = _state()
    st.pool.items = [
        EvidenceItem(text="stale polls", source="polls", date=AS_OF),
        EvidenceItem(text="some news", source="news", date=AS_OF),
    ]
    out = R.research(registry=reg)(st)
    pool_by_src = {it.source: it.text for it in out.pool.items}
    assert pool_by_src["polls"] == "fresh polls"  # replaced
    assert pool_by_src["news"] == "some news"  # kept
