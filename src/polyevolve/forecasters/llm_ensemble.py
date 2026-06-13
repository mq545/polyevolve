"""LLM self-ensemble forecaster - the kill-test pipeline, packaged as a plugin.

This is the forecasting CORE of `scripts/kill_test.py` lifted to the Forecaster
contract: a K-lens self-ensemble over a frozen, PRICE-FREE prompt, aggregated by a
trimmed mean. The system framing is the "shift-detector" reframe (model FORMING
shifts, not levels) that paired-beats the old level-estimator prompt by +2-3 SE on
the beatable band (see project memory: project_polymarket_killtest_findings,
project_polymarket_upset_modeling).

Ensemble diversity comes NOT from sampling temperature (the local `Model` contract
has none) but from K distinct reasoning-lens instructions appended to the user
content. Each lens is one full forecast; we drop the single min and max and mean
the rest - robust to one over- and one under-confident member.

ABSTAIN-IF-NO-DATA: when the assembled research context is empty / a no-data
marker, the model is never called. We return the base rate (0.5) at low confidence
rather than letting the shift-detector prompt COERCE the model into hallucinating
poll numbers to fill its momentum/dispersion template (the confabulation failure
documented in the retraction in project_polymarket_killtest_findings). This is the
$0, GPU-free, deterministic branch and is what the tests exercise.

The live model path uses `polyevolve.models.build_model` (Ollama qwen3). It is
guarded: the model is built lazily on first real call, so importing this module -
and predicting on empty context - never touches a GPU. CI has no GPU; tests MUST
only hit the abstain branch or inject a fake model.

Self-registers as @register_forecaster("llm_ensemble"). Core never imports this.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from polyevolve.contracts import Model
from polyevolve.core.registry import register_forecaster
from polyevolve.core.types import Confidence, Market, Prediction

logger = logging.getLogger(__name__)

# Local, $0 model. LiteLLM route ("/" in the id) -> Ollama (see models.build_model).
DEFAULT_MODEL = os.environ.get(
    "POLYEVOLVE_FORECAST_MODEL", "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M"
)

# No-data markers the assembler emits when no connector returned usable text. On
# any of these we ABSTAIN (return base rate) instead of calling the model - the
# shift-detector prompt confabulates numbers when handed an empty context.
_NO_DATA_MARKERS = frozenset({"", "[no context available]", "[no data]"})

# Upset-detection system framing. The crowd loses by anchoring a stale base rate
# while a free, public, point-in-time signal already shows a shift FORMING. The
# forecaster's job is to DETECT a forming shift the price hasn't absorbed - not to
# estimate a level. (Verbatim from kill_test.py v2upset; do NOT show the price.)
UPSET_SYSTEM_PROMPT = (
    "You are a calibrated political forecaster whose specialty is catching UPSETS - "
    "cases where the consensus is anchored on a stale base rate while a forming shift "
    "is already visible in the evidence. Most markets resolve near the base rate, so "
    "stay calibrated; but your edge comes from the MINORITY of cases where signals of "
    "a forming shift are present and under-weighted.\n\n"
    "Across many real upsets, the tells were never the level of any single number - "
    "they were SHIFTS, DISPERSION, and CONTRADICTIONS:\n"
    "1. MOMENTUM over levels: weigh the TREND and its acceleration (is a poll/attention "
    "line moving and speeding up?), not just the latest value. Weight the most RECENT "
    "window heavily; a dated event (debate, scandal, defection, escalation) should "
    "dominate older framing.\n"
    "2. DISPERSION is signal, not noise: when sources disagree (a lone dissenting poll "
    "vs the consensus, a bimodal spread), two crowds exist and one is about to be wrong "
    "- do NOT average it away; ask which side the momentum favors.\n"
    "3. DIVERGENCE inverts the naive prior: when two streams CONTRADICT (attention "
    "surging for the trailing candidate; 'talks' optimism while forces stage; record "
    "turnout while the electorate was structurally reshaped), trust the harder signal.\n"
    "4. The DENOMINATOR can change: who is IN the electorate (voter-roll purges, "
    "registration/boundary changes, diaspora, turnout-universe shifts) can decide a "
    "race while every poll surveys a phantom electorate. Watch for it.\n"
    "5. UPSET-RISK co-occurrence: when a front-running favorite + a large undecided/"
    "abstention pool + at least one dissenting signal all co-occur, WIDEN your "
    "probability away from the base rate toward the upset.\n"
    "Stay well-calibrated overall (when you say 70%, it should happen ~70%); spend your "
    "deviations on the cases where these forming-shift tells are genuinely present.\n\n"
    "GROUNDING: only reason from numbers actually present in the context. If the "
    "context has no concrete poll/attention figures, say so and fall back to the base "
    "rate at LOW confidence - do NOT invent percentages or seat trajectories.\n\n"
    "Call submit_prediction exactly once. No text outside the tool call."
)

# K=5 diverse reasoning lenses (verbatim from kill_test.py v2upset). The Model
# contract takes no temperature, so ensemble diversity comes from steering each
# member down a distinct reasoning path.
REASONING_LENSES: tuple[str, ...] = (
    # 0: base-rate anchor (the disciplined prior)
    "REASONING LENS - BASE-RATE ANCHOR: Fix the reference-class base rate for this "
    "kind of event BEFORE the case specifics. State it explicitly. This is your prior; "
    "most markets resolve near it. Deviate only for concrete forming-shift evidence.",
    # 1: momentum & recency (derivatives, not levels)
    "REASONING LENS - MOMENTUM & RECENCY: Ignore static levels; track the TREND and "
    "its acceleration in polls, attention, and events over the final window. Is a line "
    "moving and speeding up? Let the freshest, post-event data dominate older framing. "
    "A fast-forming move the market hasn't absorbed is the edge.",
    # 2: dispersion & dissent (the outlier is the signal)
    "REASONING LENS - DISPERSION & DISSENT: Look for DISAGREEMENT across sources - a "
    "lone dissenting poll, a bimodal spread, a local source contradicting the national "
    "frame. Do NOT average it away. Treat a credible outlier as possibly the truest "
    "snapshot, especially if momentum and undecideds back it.",
    # 3: divergence & contradiction (trust the harder signal)
    "REASONING LENS - DIVERGENCE & CONTRADICTION: Find where two streams CONTRADICT "
    "(attention vs polls; reassuring headlines vs on-the-ground staging; turnout vs a "
    "reshaped electorate). When they diverge, distrust the soft/narrative stream and "
    "weight the harder, structural one - even if it inverts your first instinct.",
    # 4: upset-risk & denominator
    "REASONING LENS - UPSET-RISK & DENOMINATOR: Score upset risk: does a complacent "
    "front-runner + a large undecided/abstention pool + at least one dissenting signal "
    "co-occur? If so, WIDEN away from the base rate toward the upset. Also ask whether "
    "the DENOMINATOR shifted (voter-roll/registration/turnout-universe changes the "
    "polls can't see).",
)
K = len(REASONING_LENSES)

# Forced-tool schema for each ensemble member's forecast.
PREDICTION_TOOL: dict[str, Any] = {
    "name": "submit_prediction",
    "description": (
        "Submit a calibrated probability estimate that the market resolves YES, "
        "along with reasoning. This is the only output channel - do not produce text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "probability_yes": {
                "type": "number",
                "description": (
                    "P(YES) in [0, 1]. Must be CALIBRATED: if you say 0.70, the event "
                    "must resolve YES roughly 70% of the time."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Your confidence in the point estimate, in [0, 1]. "
                    "Lower if data is sparse, noisy, or the question is genuinely uncertain."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Concise step-by-step reasoning for the estimate.",
            },
        },
        "required": ["probability_yes", "confidence", "reasoning"],
    },
}


def trimmed_mean(values: list[float]) -> float | None:
    """Drop the single min and single max, mean the rest (the middle 3 of 5).

    Robust to one outlier-high and one outlier-low ensemble member. With fewer
    than 3 values, trimming would discard everything, so fall back to the plain
    mean (or None if empty). Verbatim behavior from kill_test.py.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) < 3:
        return sum(vals) / len(vals)
    ordered = sorted(vals)
    middle = ordered[1:-1]
    return sum(middle) / len(middle)


def _is_no_data(context: str) -> bool:
    """True when the assembled context carries no usable signal -> abstain."""
    return context.strip() in _NO_DATA_MARKERS


def _build_user_content(question: str, context: str, lens: str) -> str:
    """Forecasting prompt: question + frozen context + one reasoning lens.

    Deliberately PRICE-FREE - the market price is never shown (showing it collapses
    edge-vs-market to ~0 by construction, the leakage footgun from the agent).
    """
    return "\n".join(
        [
            f"MARKET QUESTION: {question}",
            "\n" + (context.strip() or "[no context available]"),
            "\n" + lens,
            "\nProduce a calibrated probability. Call submit_prediction with your estimate.",
        ]
    )


def _confidence_band(mean_conf: float, members: int) -> Confidence:
    """Map the ensemble's mean self-reported confidence to the coarse band.

    Thin ensembles (one usable member) are capped at "low" - a single forecast is
    not corroboration. Otherwise: <0.5 low, <0.75 medium, else high.
    """
    if members < 2 or mean_conf < 0.5:
        return "low"
    if mean_conf < 0.75:
        return "medium"
    return "high"


@register_forecaster("llm_ensemble")
class LLMEnsembleForecaster:
    """K-lens self-ensemble + trimmed mean, shift-detector framing. Key: 'llm_ensemble'.

    The live model is built lazily and only when there is real context to reason
    over; on empty/no-data context we ABSTAIN to the base rate without any model
    call. Inject `model=` (a `Model`) in tests to exercise the live path GPU-free.
    """

    key = "llm_ensemble"

    def __init__(self, model: Model | None = None, model_id: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._model_id = model_id

    def _get_model(self) -> Model:
        """Lazily build the local model. Guarded: never called in the abstain path."""
        if self._model is None:
            from polyevolve.models import build_model

            self._model = build_model(model_id=self._model_id, anthropic_api_key=None)
        return self._model

    def predict(self, market: Market, context: str) -> Prediction:
        # ABSTAIN: no signal -> base rate, no model call (no GPU, no confabulation).
        if _is_no_data(context):
            return Prediction(
                prob_yes=0.5,
                confidence="low",
                reasoning="abstain: no research context available - returning base rate 0.5",
            )

        model = self._get_model()
        probs: list[float] = []
        confs: list[float] = []
        reasonings: list[str] = []
        failed = 0
        for i, lens in enumerate(REASONING_LENSES):
            user_content = _build_user_content(market.question, context, lens)
            # Fail-soft per member: one lens that errors/won't parse must not sink
            # the whole forecast (mirrors the evaluator / kill_test behavior).
            try:
                result = model.complete_with_tool(
                    cached_system_blocks=[UPSET_SYSTEM_PROMPT],
                    user_content=user_content,
                    tool=PREDICTION_TOOL,
                    metadata={"lens": i, "market_external_id": market.external_id},
                )
                pred = result["input"]
                prob = min(1.0, max(0.0, float(pred["probability_yes"])))
            except Exception:
                failed += 1
                logger.warning(
                    "llm_ensemble: lens %d failed for market %s - skipping",
                    i,
                    market.external_id,
                )
                continue
            probs.append(prob)
            confs.append(float(pred.get("confidence", 0.0)))
            reasonings.append(str(pred.get("reasoning", "")))

        agg = trimmed_mean(probs)
        if agg is None:
            # Every member failed: abstain rather than emit a fake number.
            return Prediction(
                prob_yes=0.5,
                confidence="low",
                reasoning=f"abstain: all {K} ensemble members failed - returning base rate 0.5",
            )

        mean_conf = sum(confs) / len(confs) if confs else 0.0
        confidence = _confidence_band(mean_conf, len(probs))
        reasoning = (
            f"K={K} lens self-ensemble (used {len(probs)}, failed {failed}), "
            f"trimmed-mean P(YES)={agg:.3f}, mean member confidence={mean_conf:.2f}.\n"
            + "\n".join(f"  [lens {i}] {r[:300]}" for i, r in enumerate(reasonings))
        )
        return Prediction(prob_yes=agg, confidence=confidence, reasoning=reasoning)
