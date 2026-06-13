"""Tests for the LLM keyword resolver - dual output shape + fail-soft, no network."""

from __future__ import annotations

from typing import Any

from polyevolve.data_sources import keyword_resolver as K


class FakeModel:
    name = "mock"

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.calls = 0

    def complete_with_tool(self, **_: Any) -> dict[str, Any]:
        self.calls += 1
        return {"input": self._payload, "usage": {}}


def _patch(monkeypatch, model, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr(K, "build_model", lambda **_: model)
    monkeypatch.setattr(K, "_CACHE", tmp_path / "kw.jsonl")
    monkeypatch.setattr(K, "_MEM", None)  # reset module cache between tests


def test_dict_shape(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    model = FakeModel({"terms": [{"index": 1, "query": "Tisza"}, {"index": 2, "query": "DK"}]})
    _patch(monkeypatch, model, tmp_path)
    out = resolve = K.resolve_keywords(["Will TISZA win?", "Will DK win?"], geo="HU")
    assert out == {0: "Tisza", 1: "DK"}
    assert resolve is out and model.calls == 1


def test_positional_string_shape(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    # smaller local models return a flat positional list of strings - must still map.
    model = FakeModel({"terms": ["Tisza", "DK", "LMP"]})
    _patch(monkeypatch, model, tmp_path)
    out = K.resolve_keywords(["q1", "q2", "q3"], geo="HU")
    assert out == {0: "Tisza", 1: "DK", 2: "LMP"}


def test_blank_and_overflow_entries_dropped(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    model = FakeModel({"terms": ["", "Keiko", "spill-over-index"]})
    _patch(monkeypatch, model, tmp_path)
    out = K.resolve_keywords(["q1", "q2"], geo="PE")  # only 2 markets
    assert out == {1: "Keiko"}  # blank dropped, 3rd term beyond n is ignored


def test_cache_hit_skips_model(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    model = FakeModel({"terms": ["A", "B"]})
    _patch(monkeypatch, model, tmp_path)
    K.resolve_keywords(["q1", "q2"], event="E", geo="X")
    K.resolve_keywords(["q1", "q2"], event="E", geo="X")  # identical -> cached
    assert model.calls == 1


def test_model_error_fail_soft(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    class Boom:
        name = "boom"

        def complete_with_tool(self, **_: Any) -> dict[str, Any]:
            raise RuntimeError("down")

    _patch(monkeypatch, Boom(), tmp_path)
    assert K.resolve_keywords(["q1", "q2"], geo="HU") == {}
