"""Unit tests for the node library - all model calls go through a MOCK (no network).

We monkeypatch `polyevolve.reason.nodes.build_model` so every model-using factory gets a
scripted client. Each test asserts the relevant node updates ReasoningState sanely.

Run:
    uv run pytest tests/test_nodes.py -q
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from polyevolve.reason import nodes
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Question, ReasoningState

AS_OF = datetime(2026, 1, 10, tzinfo=UTC)


class MockModel:
    """Returns a canned tool-call payload keyed by the tool name. No network."""

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


def _state(
    *,
    market_price: float | None = None,
    items: list[EvidenceItem] | None = None,
    beliefs: dict[str, Any] | None = None,
) -> ReasoningState:
    q = Question(
        id="q1",
        text="Will the Tisza party win the most list votes in the 2026 Hungarian election?",
        as_of=AS_OF,
        resolution_criteria="YES if Tisza receives the plurality of national list votes.",
        category="foreign_politics",
        market_price=market_price,
    )
    pool = EvidencePool(items=items or [])
    return ReasoningState(question=q, pool=pool, beliefs=beliefs or {})


# --------------------------------------------------------------------------------------
# 1. call_model
# --------------------------------------------------------------------------------------


def test_call_model_writes_beliefs(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.62, "confidence": 0.7, "reasoning": "polls"}}
    )
    _patch_model(monkeypatch, mm)
    st = nodes.call_model(model_id="mock/x")(_state())
    assert st.beliefs["p_yes"] == pytest.approx(0.62)
    assert st.beliefs["confidence"] == pytest.approx(0.7)
    assert st.beliefs["rationale"] == "polls"
    assert mm.calls[0]["tool"] == "submit_prediction"


def test_call_model_clips_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 1.4, "confidence": 9, "reasoning": ""}}
    )
    _patch_model(monkeypatch, mm)
    st = nodes.call_model(model_id="mock/x")(_state())
    assert st.beliefs["p_yes"] == 1.0
    assert st.beliefs["confidence"] == 1.0


# --------------------------------------------------------------------------------------
# 2. decompose
# --------------------------------------------------------------------------------------


def test_decompose_stores_subqs(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel({"submit_subquestions": {"sub_questions": ["a?", "b?", " ", "c?", "d?", "e?"]}})
    _patch_model(monkeypatch, mm)
    st = nodes.decompose(model_id="mock/x", max_subqs=4)(_state())
    assert st.beliefs["subqs"] == ["a?", "b?", "c?", "d?"]


# --------------------------------------------------------------------------------------
# 3. ensemble
# --------------------------------------------------------------------------------------


class CyclingModel(MockModel):
    """Yields a different probability_yes per call to exercise aggregation."""

    def __init__(self, probs: list[float]):
        super().__init__({})
        self._probs = probs
        self._i = 0

    def complete_with_tool(self, **kw: Any) -> dict[str, Any]:  # type: ignore[override]
        p = self._probs[self._i % len(self._probs)]
        self._i += 1
        return {
            "input": {"probability_yes": p, "confidence": 0.6, "reasoning": f"draw{self._i}"},
            "usage": {},
        }


def test_ensemble_aggregates_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = CyclingModel([0.4, 0.5, 0.6])
    _patch_model(monkeypatch, mm)
    st = nodes.ensemble(k=3, model_id="mock/x", aggregate="mean")(_state())
    assert st.beliefs["p_yes"] == pytest.approx(0.5)
    assert st.beliefs["samples"] == [0.4, 0.5, 0.6]
    assert 0.0 <= st.beliefs["confidence"] <= 1.0


def test_ensemble_trimmed_mean_drops_outlier(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = CyclingModel([0.5, 0.5, 0.5, 0.95, 0.05])
    _patch_model(monkeypatch, mm)
    st = nodes.ensemble(k=5, model_id="mock/x", aggregate="trimmed_mean")(_state())
    assert st.beliefs["p_yes"] == pytest.approx(0.5, abs=0.05)


def test_ensemble_multi_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = CyclingModel([0.3, 0.7])
    _patch_model(monkeypatch, mm)
    st = nodes.ensemble(model_id=["m/a", "m/b"], aggregate="mean")(_state())
    assert st.beliefs["p_yes"] == pytest.approx(0.5)
    assert len(st.beliefs["samples"]) == 2


# --------------------------------------------------------------------------------------
# 4. debate_critique
# --------------------------------------------------------------------------------------


def test_debate_critique_revises(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {
            "submit_revision": {
                "critique": "base-rate neglect",
                "revised_probability_yes": 0.55,
                "confidence": 0.65,
            }
        }
    )
    _patch_model(monkeypatch, mm)
    st = _state(beliefs={"p_yes": 0.8, "rationale": "init"})
    out = nodes.debate_critique(model_id="mock/x")(st)
    assert out.beliefs["p_yes_proposed"] == pytest.approx(0.8)
    assert out.beliefs["p_yes"] == pytest.approx(0.55)
    assert "base-rate neglect" in out.beliefs["rationale"]


def test_debate_critique_bootstraps_when_no_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {
            "submit_prediction": {"probability_yes": 0.7, "confidence": 0.6, "reasoning": "r"},
            "submit_revision": {"critique": "c", "revised_probability_yes": 0.6},
        }
    )
    _patch_model(monkeypatch, mm)
    out = nodes.debate_critique(model_id="mock/x")(_state())
    assert out.beliefs["p_yes_proposed"] == pytest.approx(0.7)
    assert out.beliefs["p_yes"] == pytest.approx(0.6)


# --------------------------------------------------------------------------------------
# 5. select_evidence
# --------------------------------------------------------------------------------------


def _ev(text: str, day: int, source: str = "src") -> EvidenceItem:
    return EvidenceItem(text=text, source=source, date=datetime(2026, 1, day, tzinfo=UTC))


def test_select_evidence_heuristic_ranks_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        _ev("Tisza party leads list votes in latest poll", 5),
        _ev("Unrelated weather report for the weekend", 6),
        _ev("Hungarian election Tisza plurality forecast", 7),
        _ev("Sports scores from yesterday", 8),
    ]
    st = _state(items=items)
    out = nodes.select_evidence(k=2, mode="heuristic")(st)
    assert len(out.selected) == 2
    texts = " ".join(e.text for e in out.selected).lower()
    assert "tisza" in texts


def test_select_evidence_respects_leakage(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        _ev("on-time tisza item", 5),
        EvidenceItem(text="future leaked tisza item", date=datetime(2026, 2, 1, tzinfo=UTC)),
    ]
    st = _state(items=items)
    out = nodes.select_evidence(k=5, mode="heuristic")(st)
    assert all(e.date is None or e.date <= AS_OF for e in out.selected)
    assert len(out.selected) == 1


def test_select_evidence_embed_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the lazy embed import to "not installed".
    monkeypatch.setattr(nodes, "_embedding_rank", lambda *a, **k: None)
    items = [_ev("tisza plurality poll", 5), _ev("noise item", 6)]
    out = nodes.select_evidence(k=1, mode="embed")(_state(items=items))
    assert len(out.selected) == 1
    assert "heuristic(embed-unavailable)" in out.trace[-1]


def test_select_evidence_llm_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel({"submit_selection": {"indices": [2, 1]}})
    _patch_model(monkeypatch, mm)
    items = [_ev("first", 5), _ev("second", 6)]
    out = nodes.select_evidence(k=2, mode="llm", model_id="mock/x")(_state(items=items))
    assert [e.text for e in out.selected] == ["second", "first"]


# --------------------------------------------------------------------------------------
# 6. latent_to_prob
# --------------------------------------------------------------------------------------


def test_latent_to_prob_cdf(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel({"submit_margin": {"margin_mean": 0.0, "margin_std": 5.0, "reasoning": "tie"}})
    _patch_model(monkeypatch, mm)
    st = nodes.latent_to_prob(model_id="mock/x", threshold=0.0, direction="above")(_state())
    assert st.beliefs["margin_mu"] == 0.0
    assert st.beliefs["sigma"] == 5.0
    assert st.beliefs["p_yes"] == pytest.approx(0.5, abs=1e-6)


def test_latent_to_prob_positive_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel({"submit_margin": {"margin_mean": 10.0, "margin_std": 5.0, "reasoning": "lead"}})
    _patch_model(monkeypatch, mm)
    st = nodes.latent_to_prob(model_id="mock/x", threshold=0.0, direction="above")(_state())
    assert st.beliefs["p_yes"] > 0.95


def test_latent_to_prob_zero_sigma_is_step(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_margin": {"margin_mean": 3.0, "margin_std": 0.0, "reasoning": "certain"}}
    )
    _patch_model(monkeypatch, mm)
    st = nodes.latent_to_prob(model_id="mock/x", threshold=0.0, direction="above")(_state())
    assert st.beliefs["p_yes"] == 1.0


# --------------------------------------------------------------------------------------
# 7. calibrate
# --------------------------------------------------------------------------------------


def test_calibrate_temperature_softens() -> None:
    st = _state(beliefs={"p_yes": 0.95})
    out = nodes.calibrate(coeff=2.0, method="temperature")(st)
    assert out.beliefs["p_yes_raw"] == pytest.approx(0.95)
    assert 0.5 < out.beliefs["p_yes"] < 0.95  # pulled toward 0.5


def test_calibrate_temperature_sharpens() -> None:
    out = nodes.calibrate(coeff=0.5, method="temperature")(_state(beliefs={"p_yes": 0.7}))
    assert out.beliefs["p_yes"] > 0.7


def test_calibrate_identity_at_t1() -> None:
    out = nodes.calibrate(coeff=1.0, method="temperature")(_state(beliefs={"p_yes": 0.73}))
    assert out.beliefs["p_yes"] == pytest.approx(0.73, abs=1e-4)


def test_calibrate_platt() -> None:
    out = nodes.calibrate(coeff=1.0, bias=0.0, method="platt")(_state(beliefs={"p_yes": 0.6}))
    assert out.beliefs["p_yes"] == pytest.approx(0.6, abs=1e-4)


# --------------------------------------------------------------------------------------
# 8. abstain
# --------------------------------------------------------------------------------------


def test_abstain_low_confidence() -> None:
    st = _state(market_price=0.5, beliefs={"p_yes": 0.9, "confidence": 0.2})
    out = nodes.abstain(min_conf=0.4, min_div=0.05)(st)
    assert out.beliefs["size"] == 0.0


def test_abstain_too_close_to_market() -> None:
    st = _state(market_price=0.62, beliefs={"p_yes": 0.63, "confidence": 0.9})
    out = nodes.abstain(min_conf=0.4, min_div=0.05)(st)
    assert out.beliefs["size"] == 0.0


def test_abstain_keeps_confident_divergent() -> None:
    st = _state(market_price=0.4, beliefs={"p_yes": 0.7, "confidence": 0.9})
    out = nodes.abstain(min_conf=0.4, min_div=0.05)(st)
    assert "size" not in out.beliefs  # untouched -> downstream sizing decides


def test_abstain_no_market_keeps_on_confidence() -> None:
    st = _state(market_price=None, beliefs={"p_yes": 0.8, "confidence": 0.9})
    out = nodes.abstain(min_conf=0.4, min_div=0.05)(st)
    assert "size" not in out.beliefs


# --------------------------------------------------------------------------------------
# 9. size_by_edge
# --------------------------------------------------------------------------------------


def test_size_by_edge_yes_side() -> None:
    st = _state(market_price=0.4, beliefs={"p_yes": 0.7})
    out = nodes.size_by_edge(kelly_frac=0.5)(st)
    # f* = (0.7-0.4)/(1-0.4) = 0.5; size = 0.5*0.5 = 0.25
    assert out.beliefs["size"] == pytest.approx(0.25)


def test_size_by_edge_no_side_is_negative() -> None:
    st = _state(market_price=0.6, beliefs={"p_yes": 0.3})
    out = nodes.size_by_edge(kelly_frac=0.5)(st)
    assert out.beliefs["size"] < 0


def test_size_by_edge_no_price_is_zero() -> None:
    st = _state(market_price=None, beliefs={"p_yes": 0.9})
    out = nodes.size_by_edge()(st)
    assert out.beliefs["size"] == 0.0


def test_size_by_edge_respects_prior_abstain() -> None:
    st = _state(market_price=0.4, beliefs={"p_yes": 0.9, "size": 0.0})
    out = nodes.size_by_edge(kelly_frac=0.5)(st)
    assert out.beliefs["size"] == 0.0  # stays abstained


# --------------------------------------------------------------------------------------
# pipeline smoke: compose several nodes like a genome would
# --------------------------------------------------------------------------------------


def test_pipeline_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    mm = MockModel(
        {"submit_prediction": {"probability_yes": 0.75, "confidence": 0.8, "reasoning": "r"}}
    )
    _patch_model(monkeypatch, mm)
    items = [_ev("tisza plurality poll lead", 5), _ev("noise", 6)]
    st = _state(market_price=0.5, items=items)
    for node in (
        nodes.select_evidence(k=1, mode="heuristic"),
        nodes.call_model(model_id="mock/x"),
        nodes.calibrate(coeff=1.5, method="temperature"),
        nodes.abstain(min_conf=0.4, min_div=0.05),
        nodes.size_by_edge(kelly_frac=0.25),
    ):
        st = node(st)
    fc = st.to_forecast()
    assert 0.5 < fc.p_yes < 0.75  # calibrated down from 0.75
    assert fc.size > 0  # confident + divergent -> positive YES stake
    assert len(st.trace) == 5


# --------------------------------------------------------------------------------------
# validate_evidence + extract_features (deep-reasoning nodes)
# --------------------------------------------------------------------------------------


def test_validate_evidence_filters_and_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [
        _ev("Fidesz 39, TISZA 51 (Medián, Jan 2026)", day=5, source="polls"),
        _ev("The following graph presents the average of all polls.", day=6, source="polls"),
        _ev("unrelated weather note", day=7, source="news"),
    ]
    model = MockModel(
        {
            "submit_validation": {
                "usable_indices": [1],
                "data_quality": 0.8,
                "key_signal_present": True,
                "notes": "one numeric poll usable",
            }
        }
    )
    _patch_model(monkeypatch, model)
    st = _state(items=items)
    st.selected = list(items)
    out = nodes.validate_evidence()(st)
    assert len(out.selected) == 1 and out.selected[0].source == "polls"
    assert out.beliefs["data_quality"] == 0.8
    assert out.beliefs["key_signal_present"] is True


def test_validate_evidence_no_evidence_sets_zero_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_model(monkeypatch, MockModel({}))  # model must NOT be called
    st = _state(items=[])
    out = nodes.validate_evidence()(st)
    assert out.beliefs["data_quality"] == 0.0
    assert out.beliefs["key_signal_present"] is False


def test_extract_features_builds_feature_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MockModel(
        {
            "submit_features": {
                "features": [
                    {"name": "poll_lead", "value": "TISZA +12 (latest 3-poll avg)"},
                    {"name": "days_to_resolution", "value": "21"},
                    {"name": "bad", "value": ""},  # dropped (empty value)
                ],
                "base_rate": 0.4,
                "summary": "tisza ahead",
            }
        }
    )
    _patch_model(monkeypatch, model)
    st = _state(items=[_ev("poll data", day=5, source="polls")])
    out = nodes.extract_features()(st)
    ft = out.beliefs["features_text"]
    assert "poll_lead: TISZA +12" in ft and "days_to_resolution: 21" in ft
    assert "bad" not in ft
    assert out.beliefs["base_rate"] == 0.4


def test_call_model_consumes_features_and_data_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MockModel(
        {"submit_prediction": {"probability_yes": 0.7, "confidence": 0.6, "reasoning": "ok"}}
    )
    _patch_model(monkeypatch, model)
    st = _state(items=[_ev("x", day=5, source="polls")])
    st.selected = list(st.pool.items)
    st.beliefs["features_text"] = "- poll_lead: +12"
    st.beliefs["data_quality"] = 0.2
    st.beliefs["base_rate"] = 0.45
    nodes.call_model()(st)
    # the deep context must reach the model prompt
    user = model.calls[-1]["user"]
    assert "DERIVED FEATURES" in user and "poll_lead: +12" in user
    assert "DATA-QUALITY: 0.20" in user and "0.45" in user


def test_latent_threshold_band_two_sided(monkeypatch: pytest.MonkeyPatch) -> None:
    # quantity ~ N(95, 10); P(70<=seats<=79) should be small (left tail), NOT a one-sided guess
    model = MockModel(
        {
            "submit_quantity": {
                "quantity": "Tisza seats",
                "mean": 95.0,
                "std": 10.0,
                "condition": "between",
                "low": 70.0,
                "high": 79.0,
                "reasoning": "x",
            }
        }
    )
    _patch_model(monkeypatch, model)
    st = _state(items=[_ev("polls", day=5, source="polls")])
    st.selected = list(st.pool.items)
    out = nodes.latent_threshold()(st)
    p = out.beliefs["p_yes"]
    assert 0.0 < p < 0.10  # 70-79 is well below the mean -> small
    assert out.beliefs["margin_mu"] == 95.0


def test_latent_threshold_at_least(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MockModel(
        {
            "submit_quantity": {
                "quantity": "Tisza seats",
                "mean": 95.0,
                "std": 10.0,
                "condition": "at_least",
                "low": 70.0,
                "high": 0.0,
                "reasoning": "x",
            }
        }
    )
    _patch_model(monkeypatch, model)
    st = _state(items=[_ev("polls", day=5, source="polls")])
    st.selected = list(st.pool.items)
    out = nodes.latent_threshold()(st)
    assert out.beliefs["p_yes"] > 0.95  # P(seats >= 70) when mean 95 -> high


def test_reweight_polls_drops_gov_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    polls = (
        "Pollster | Date | Fidesz | TISZA\n"
        "Nézőpont | Jan 2026 | 49 | 41\n"  # gov-aligned (captured) -> dropped
        "Medián | Jan 2026 | 39 | 51\n"  # independent -> kept
        "Republikon | Jan 2026 | 36 | 48\n"  # independent -> kept
    )
    st = _state(items=[_ev(polls, day=5, source="polls")])
    st.question.text = "Will TISZA win the 2026 Hungarian parliamentary election?"
    st.selected = list(st.pool.items)
    out = nodes.reweight_polls()(st)
    txt = out.selected[0].text
    assert "Nézőpont" not in txt or txt.startswith("[reweighted")  # the data row is gone
    assert "Nézőpont | Jan 2026" not in txt  # the captured row removed
    assert "Medián" in txt and "Republikon" in txt
    assert "[reweighted:" in txt


def test_reweight_polls_noop_without_polls(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _state(items=[_ev("some news", day=5, source="news")])
    st.selected = list(st.pool.items)
    out = nodes.reweight_polls()(st)
    assert out.selected[0].text == "some news"  # untouched
