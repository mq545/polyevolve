"""The NODE LIBRARY - the stable, unit-tested primitives the genome composes.

Each public name here is a **factory**: call it with config and it returns a `Node`
(a callable `(ReasoningState) -> ReasoningState`). Evolution tunes the *config* and the
*composition* (which nodes, in what order); the node bodies stay frozen and tested.

Contract (see polyevolve.reason.dsl, which we MUST NOT modify):
- Everything flows through `ReasoningState`. Nodes mutate-and-return the same state object
  (pydantic models here are mutable; the genome treats the pipeline as a fold).
- Belief keys this library reads/writes:
    p_yes       float in [0,1]   - current probability estimate (the headline output)
    rationale   str              - human-readable why
    confidence  float in [0,1]   - self-reported confidence (drives abstain/size)
    size        float in [-1,1]  - signed stake fraction; 0.0 == ABSTAIN
    subqs       list[str]        - decomposition sub-questions
    margin_mu   float            - latent margin mean (latent_to_prob)
    sigma       float            - latent margin 1-sigma
    p_yes_raw   float            - pre-calibration p_yes (audit)
    samples     list[float]      - per-draw p_yes from ensemble (audit)
- Leakage rule: evidence-touching nodes use `pool.on_or_before(question.as_of)` only.
- Network: ONLY via an injected model built by `build_model`. Every model-using factory
  takes `model_id` (and optional `anthropic_api_key`) and builds the client lazily at call
  time, so tests can monkeypatch `polyevolve.reason.nodes.build_model` and hit no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import norm

from polyevolve.contracts import Model
from polyevolve.data_sources.pollster_bias import pollster_lean
from polyevolve.models import build_model
from polyevolve.reason.dsl import EvidenceItem, Node, Question, ReasoningState

__all__ = [
    "abstain",
    "calibrate",
    "call_model",
    "debate_critique",
    "decompose",
    "ensemble",
    "extract_features",
    "latent_threshold",
    "latent_to_prob",
    "reweight_polls",
    "select_evidence",
    "size_by_edge",
    "validate_evidence",
]

# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def _clip01(x: float) -> float:
    """Clamp into the open-ish [0,1] interval and coerce non-finite to 0.5."""
    if not np.isfinite(x):
        return 0.5
    return float(min(1.0, max(0.0, x)))


def _make_model(model_id: str, anthropic_api_key: str | None) -> Model:
    """Build a model client. Isolated so tests monkeypatch `build_model` here."""
    return build_model(model_id=model_id, anthropic_api_key=anthropic_api_key)


def _render_evidence(items: Sequence[EvidenceItem], limit: int = 40) -> str:
    """Render selected evidence into a compact, dated block for the prompt."""
    if not items:
        return "(no evidence selected)"
    lines: list[str] = []
    for i, e in enumerate(items[:limit]):
        when = e.date.date().isoformat() if e.date is not None else "undated"
        src = f" [{e.source}]" if e.source else ""
        lines.append(f"{i + 1}. ({when}){src} {e.text}")
    return "\n".join(lines)


def _question_block(q: Question) -> str:
    parts = [f"QUESTION: {q.text}", f"AS OF (decision date): {q.as_of.date().isoformat()}"]
    if q.resolution_criteria:
        parts.append(f"RESOLUTION CRITERIA (exact YES condition): {q.resolution_criteria}")
    return "\n".join(parts)


# Reused prediction tool: a single calibrated P(YES) channel (mirrors
# foreign_politics_agent.PREDICTION_TOOL, kept local so the node library is self-contained).
_PREDICTION_TOOL: dict[str, Any] = {
    "name": "submit_prediction",
    "description": (
        "Submit a calibrated probability that the question resolves YES, with reasoning. "
        "This is the only output channel - do not produce free text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "probability_yes": {
                "type": "number",
                "description": "P(YES) in [0,1]. Must be calibrated across all your predictions.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence in the estimate in [0,1]; lower when data is thin.",
            },
            "reasoning": {"type": "string", "description": "Concise step-by-step reasoning."},
        },
        "required": ["probability_yes", "confidence", "reasoning"],
    },
}


# --------------------------------------------------------------------------------------
# 1. call_model
# --------------------------------------------------------------------------------------


def call_model(
    system_prompt: str | None = None,
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
) -> Node:
    """Forecast P(YES) over the question + `state.selected` evidence.

    Writes beliefs['p_yes'], beliefs['confidence'], beliefs['rationale'].
    """
    sys_prompt = system_prompt or (
        "You are an elite calibrated forecaster. Read the question, its exact resolution "
        "criteria, and the dated evidence. Estimate a single calibrated P(YES) using ONLY "
        "the evidence and general knowledge as of the decision date. Do not invent numbers "
        "absent from the evidence. Call submit_prediction exactly once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        model = _make_model(model_id, anthropic_api_key)
        # Feature-aware + grounded (additive): if upstream nodes built derived features or
        # assessed data quality, reason over those; otherwise behave exactly as before.
        feats = str(state.beliefs.get("features_text", "")).strip()
        feat_block = f"DERIVED FEATURES (reason over these first):\n{feats}\n\n" if feats else ""
        dq = state.beliefs.get("data_quality")
        ground = ""
        if dq is not None:
            br = float(state.beliefs.get("base_rate", 0.5))
            ground = (
                f"\nEVIDENCE DATA-QUALITY: {float(dq):.2f} (0-1). If it is low, ANCHOR to the "
                f"reference base rate ({br:.2f}) and lower confidence; do not invent specifics.\n"
            )
        user = (
            f"{_question_block(state.question)}\n\n"
            f"{feat_block}"
            f"EVIDENCE:\n{_render_evidence(state.selected)}\n"
            f"{ground}\n"
            "Give a calibrated probability via submit_prediction."
        )
        res = model.complete_with_tool(
            cached_system_blocks=[sys_prompt],
            user_content=user,
            tool=_PREDICTION_TOOL,
            metadata={"question_id": state.question.id, "node": "call_model"},
        )
        out = res["input"]
        p = _clip01(float(out["probability_yes"]))
        state.beliefs["p_yes"] = p
        state.beliefs["confidence"] = _clip01(float(out.get("confidence", 0.5)))
        state.beliefs["rationale"] = str(out.get("reasoning", ""))
        return state.log(f"call_model[{model_id}] p_yes={p:.3f}")

    return _node


# --------------------------------------------------------------------------------------
# 2. decompose
# --------------------------------------------------------------------------------------

_DECOMPOSE_TOOL: dict[str, Any] = {
    "name": "submit_subquestions",
    "description": "Break the question into 2-4 decisive sub-questions. Only output channel.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "2 to 4 concrete, separately-answerable sub-questions whose answers "
                    "jointly determine the main question's resolution."
                ),
            }
        },
        "required": ["sub_questions"],
    },
}


def decompose(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    max_subqs: int = 4,
) -> Node:
    """LLM splits the question into 2-4 sub-questions; store in beliefs['subqs']."""
    sys_prompt = (
        "You are a forecasting analyst. Decompose the question into the 2-4 sub-questions "
        "that, if answered, would most reduce uncertainty about whether it resolves YES. "
        "Call submit_subquestions exactly once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        model = _make_model(model_id, anthropic_api_key)
        res = model.complete_with_tool(
            cached_system_blocks=[sys_prompt],
            user_content=_question_block(state.question),
            tool=_DECOMPOSE_TOOL,
            metadata={"question_id": state.question.id, "node": "decompose"},
        )
        subqs = [str(s).strip() for s in res["input"].get("sub_questions", []) if str(s).strip()]
        subqs = subqs[:max_subqs]
        state.beliefs["subqs"] = subqs
        return state.log(f"decompose -> {len(subqs)} subqs")

    return _node


# --------------------------------------------------------------------------------------
# 3. ensemble
# --------------------------------------------------------------------------------------


def ensemble(
    k: int = 3,
    model_id: str | Sequence[str] = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    aggregate: str = "trimmed_mean",
    system_prompt: str | None = None,
) -> Node:
    """Run the forecast k times (or across k models) and aggregate p_yes.

    `model_id` may be a single id (sampled k times) or a list of ids (one run each;
    `k` ignored). `aggregate` in {"mean", "median", "trimmed_mean"}.
    """
    models: list[str]
    if isinstance(model_id, str):
        models = [model_id] * max(1, k)
    else:
        models = list(model_id)
        if not models:
            raise ValueError("ensemble: model_id list must be non-empty")

    def _aggregate(samples: list[float]) -> float:
        arr = np.asarray(samples, dtype=float)
        if aggregate == "mean":
            return float(arr.mean())
        if aggregate == "median":
            return float(np.median(arr))
        if aggregate == "trimmed_mean":
            if arr.size >= 3:
                lo, hi = np.percentile(arr, [25, 75])
                kept = arr[(arr >= lo) & (arr <= hi)]
                if kept.size:
                    return float(kept.mean())
            return float(arr.mean())
        raise ValueError(f"ensemble: unknown aggregate {aggregate!r}")

    def _node(state: ReasoningState) -> ReasoningState:
        samples: list[float] = []
        rationales: list[str] = []
        for mid in models:
            sub = call_model(
                system_prompt=system_prompt, model_id=mid, anthropic_api_key=anthropic_api_key
            )(state)
            samples.append(_clip01(float(sub.beliefs.get("p_yes", 0.5))))
            if sub.beliefs.get("rationale"):
                rationales.append(str(sub.beliefs["rationale"]))
        agg = _clip01(_aggregate(samples))
        state.beliefs["samples"] = samples
        state.beliefs["p_yes"] = agg
        # spread -> confidence: tight agreement => higher confidence.
        spread = float(np.std(samples)) if len(samples) > 1 else 0.0
        state.beliefs["confidence"] = _clip01(1.0 - 2.0 * spread)
        if rationales:
            state.beliefs["rationale"] = rationales[0]
        return state.log(
            f"ensemble[n={len(samples)},{aggregate}] p_yes={agg:.3f} spread={spread:.3f}"
        )

    return _node


# --------------------------------------------------------------------------------------
# 4. debate_critique
# --------------------------------------------------------------------------------------

_CRITIQUE_TOOL: dict[str, Any] = {
    "name": "submit_revision",
    "description": (
        "Critique the proposed probability, then submit a revised, better-calibrated P(YES)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "critique": {
                "type": "string",
                "description": "What is wrong, over/under-weighted, or uncalibrated here.",
            },
            "revised_probability_yes": {
                "type": "number",
                "description": "Your revised, better-calibrated P(YES) in [0,1].",
            },
            "confidence": {"type": "number", "description": "Confidence in the revision, [0,1]."},
        },
        "required": ["critique", "revised_probability_yes"],
    },
}


def debate_critique(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    system_prompt: str | None = None,
) -> Node:
    """Propose -> critique/refute -> revise the p_yes once.

    If no prior p_yes exists in beliefs, runs `call_model` first to get the proposal.
    """

    def _node(state: ReasoningState) -> ReasoningState:
        if "p_yes" not in state.beliefs:
            state = call_model(
                system_prompt=system_prompt, model_id=model_id, anthropic_api_key=anthropic_api_key
            )(state)
        proposed = _clip01(float(state.beliefs.get("p_yes", 0.5)))
        prior_rationale = str(state.beliefs.get("rationale", ""))

        model = _make_model(model_id, anthropic_api_key)
        sys_prompt = (
            "You are a skeptical red-team forecaster. A colleague proposed a probability. "
            "Find the strongest reasons it is mis-calibrated (anchoring, base-rate neglect, "
            "over-reaction to salient evidence), then submit a revised P(YES). Call "
            "submit_revision exactly once."
        )
        user = (
            f"{_question_block(state.question)}\n\n"
            f"EVIDENCE:\n{_render_evidence(state.selected)}\n\n"
            f"COLLEAGUE PROPOSAL: P(YES) = {proposed:.3f}\n"
            f"COLLEAGUE REASONING: {prior_rationale or '(none provided)'}\n\n"
            "Critique it, then submit your revised probability."
        )
        res = model.complete_with_tool(
            cached_system_blocks=[sys_prompt],
            user_content=user,
            tool=_CRITIQUE_TOOL,
            metadata={"question_id": state.question.id, "node": "debate_critique"},
        )
        out = res["input"]
        revised = _clip01(float(out["revised_probability_yes"]))
        state.beliefs["p_yes_proposed"] = proposed
        state.beliefs["p_yes"] = revised
        if "confidence" in out:
            state.beliefs["confidence"] = _clip01(float(out["confidence"]))
        state.beliefs["rationale"] = (
            f"{prior_rationale}\n[critique] {out.get('critique', '')}".strip()
        )
        return state.log(f"debate_critique {proposed:.3f} -> {revised:.3f}")

    return _node


# --------------------------------------------------------------------------------------
# 5. select_evidence
# --------------------------------------------------------------------------------------

_SELECT_TOOL: dict[str, Any] = {
    "name": "submit_selection",
    "description": "Choose the most decisive, on-topic evidence items by index.",
    "input_schema": {
        "type": "object",
        "properties": {
            "indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "1-based indices of the most relevant items, best first.",
            }
        },
        "required": ["indices"],
    },
}


def _embedding_rank(query: str, items: list[EvidenceItem], k: int) -> list[EvidenceItem] | None:
    """Optional semantic rank via sentence-transformers (extra 'embed').

    LAZY import; returns None if the package is not installed so the caller falls back.
    """
    try:
        from sentence_transformers import SentenceTransformer, util  # noqa: PLC0415
    except Exception:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [e.text for e in items]
    emb = model.encode([query, *texts], convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(emb[0], emb[1:])[0]
    order = sorted(range(len(items)), key=lambda i: float(scores[i]), reverse=True)
    return [items[i] for i in order[:k]]


def _heuristic_rank(query: str, items: list[EvidenceItem], k: int) -> list[EvidenceItem]:
    """LLM-free fallback: token-overlap relevance, recency tiebreak."""
    q_tokens = {t for t in query.lower().split() if len(t) > 3}

    def score(e: EvidenceItem) -> tuple[float, float]:
        e_tokens = {t for t in e.text.lower().split() if len(t) > 3}
        overlap = len(q_tokens & e_tokens) / (len(q_tokens) + 1)
        recency = e.date.timestamp() if e.date is not None else 0.0
        return (overlap, recency)

    return sorted(items, key=score, reverse=True)[:k]


def select_evidence(
    k: int = 8,
    *,
    mode: str = "heuristic",
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    anthropic_api_key: str | None = None,
) -> Node:
    """Choose/rank up to k leakage-safe items into state.selected.

    mode:
      "heuristic"  - token-overlap + recency (no network, default).
      "embed"      - semantic cosine via sentence-transformers; falls back to heuristic
                     if the optional extra is not installed.
      "llm"        - ask the model to pick indices; falls back to heuristic on failure.
    """

    def _node(state: ReasoningState) -> ReasoningState:
        candidates = state.pool.on_or_before(state.question.as_of)
        query = f"{state.question.text} {state.question.resolution_criteria}".strip()
        chosen: list[EvidenceItem]
        used = mode

        if not candidates:
            chosen = []
        elif mode == "embed":
            ranked = _embedding_rank(query, candidates, k)
            if ranked is None:
                used = "heuristic(embed-unavailable)"
                chosen = _heuristic_rank(query, candidates, k)
            else:
                chosen = ranked
        elif mode == "llm":
            try:
                model = _make_model(model_id, anthropic_api_key)
                listing = _render_evidence(candidates, limit=60)
                res = model.complete_with_tool(
                    cached_system_blocks=[
                        "Select the most decisive, on-topic evidence for the question. "
                        "Call submit_selection exactly once."
                    ],
                    user_content=f"{_question_block(state.question)}\n\nITEMS:\n{listing}",
                    tool=_SELECT_TOOL,
                    metadata={"question_id": state.question.id, "node": "select_evidence"},
                )
                idxs = [int(i) - 1 for i in res["input"].get("indices", [])]
                chosen = [candidates[i] for i in idxs if 0 <= i < len(candidates)][:k]
                if not chosen:
                    used = "heuristic(llm-empty)"
                    chosen = _heuristic_rank(query, candidates, k)
            except Exception:
                used = "heuristic(llm-error)"
                chosen = _heuristic_rank(query, candidates, k)
        else:  # heuristic / default
            used = "heuristic"
            chosen = _heuristic_rank(query, candidates, k)

        state.selected = chosen
        return state.log(f"select_evidence[{used},k={k}] -> {len(chosen)}/{len(candidates)}")

    return _node


# --------------------------------------------------------------------------------------
# 6. latent_to_prob
# --------------------------------------------------------------------------------------

_MARGIN_TOOL: dict[str, Any] = {
    "name": "submit_margin",
    "description": (
        "Submit the underlying latent margin/level as a normal distribution (mean, 1-sigma). "
        "Only output channel - do not produce free text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "margin_mean": {
                "type": "number",
                "description": (
                    "Your single best point estimate of the latent quantity that decides the "
                    "question (e.g. vote margin in points, or level minus threshold)."
                ),
            },
            "margin_std": {
                "type": "number",
                "description": "1-sigma uncertainty on it (larger when sources conflict).",
            },
            "reasoning": {"type": "string", "description": "Concise step-by-step reasoning."},
        },
        "required": ["margin_mean", "margin_std", "reasoning"],
    },
}


_QUANTITY_TOOL: dict[str, Any] = {
    "name": "submit_quantity",
    "description": (
        "Estimate the latent QUANTITY this question is really about as a normal distribution, "
        "and classify the YES condition over it. The only output channel - no free text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "quantity": {
                "type": "string",
                "description": (
                    "What you are estimating, in natural units - e.g. 'Tisza seat count', "
                    "'PVV vote share %', or for a plurality question the winning margin in points."
                ),
            },
            "mean": {"type": "number", "description": "Best point estimate of the quantity."},
            "std": {
                "type": "number",
                "description": "1-sigma uncertainty (>0; larger if sources conflict).",
            },
            "condition": {
                "type": "string",
                "enum": ["at_least", "at_most", "between"],
                "description": (
                    "How YES maps to the quantity: 'at_least' (>= low), 'at_most' (<= high), "
                    "'between' (low..high inclusive band). A plurality/'win the most' question is "
                    "'at_least' with low=0 on the winning-margin quantity."
                ),
            },
            "low": {"type": "number", "description": "Lower threshold/bound (at_least & between)."},
            "high": {"type": "number", "description": "Upper threshold/bound (at_most & between)."},
            "reasoning": {"type": "string"},
        },
        "required": ["quantity", "mean", "std", "condition", "low", "high"],
    },
}


def latent_threshold(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
) -> Node:
    """Coherence node for THRESHOLD/BAND questions (the proper seat-market fix).

    The model estimates the underlying quantity (seat count, vote-share %) as one N(mu, sigma)
    AND classifies the YES condition (>=, <=, or a band); we read p_yes off the matching CDF
    region. Unlike `latent_to_prob` (one-sided), this handles exact bands ("win 70-79 seats")
    correctly - so sibling seat markets are coherent by construction rather than each guessed
    independently. Plurality maps to margin >= 0. Writes margin_mu, sigma, p_yes.
    """
    sys_prompt = (
        "You are an elite calibrated forecaster. Identify the single latent QUANTITY the "
        "question hinges on (seats, vote-share %, or winning margin), estimate it as ONE normal "
        "distribution, and classify how YES maps to it. Reliability-weight the evidence and "
        "DOWN-WEIGHT government-aligned/partisan pollsters. Call submit_quantity exactly once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        model = _make_model(model_id, anthropic_api_key)
        feats = str(state.beliefs.get("features_text", "")).strip()
        feat_block = f"DERIVED FEATURES:\n{feats}\n\n" if feats else ""
        user = (
            f"{_question_block(state.question)}\n\n"
            f"{feat_block}EVIDENCE:\n{_render_evidence(state.selected)}\n\n"
            "Estimate the latent quantity + condition via submit_quantity."
        )
        try:
            res = model.complete_with_tool(
                cached_system_blocks=[sys_prompt],
                user_content=user,
                tool=_QUANTITY_TOOL,
                metadata={"question_id": state.question.id, "node": "latent_threshold"},
            )
            out = res["input"]
            mu = float(out["mean"])
            sigma = abs(float(out["std"]))
            cond = str(out.get("condition", "at_least"))
            lo = float(out.get("low", 0.0))
            hi = float(out.get("high", 0.0))
        except Exception:  # noqa: BLE001 - degrade: leave p_yes as-is / neutral
            state.beliefs.setdefault("p_yes", 0.5)
            return state.log("latent_threshold: model error -> kept prior p_yes")

        if sigma <= 0:
            sigma = 1e-6
        if cond == "between":
            lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
            p = float(norm.cdf(hi2, loc=mu, scale=sigma) - norm.cdf(lo2, loc=mu, scale=sigma))
        elif cond == "at_most":
            p = float(norm.cdf(hi, loc=mu, scale=sigma))
        else:  # at_least (default; plurality = margin >= 0)
            p = float(norm.sf(lo, loc=mu, scale=sigma))
        state.beliefs["margin_mu"] = mu
        state.beliefs["sigma"] = sigma
        state.beliefs["p_yes"] = _clip01(p)
        state.beliefs["rationale"] = str(out.get("reasoning", ""))
        return state.log(f"latent_threshold[{cond}] mu={mu:.1f} sig={sigma:.1f} -> p={p:.3f}")

    return _node


def latent_to_prob(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    threshold: float = 0.0,
    direction: str = "above",
    latent_description: str = (
        "the underlying margin in percentage points that decides this question (positive "
        "favors YES); estimate it as a single normal distribution"
    ),
) -> Node:
    """Elicit a latent quantity ~N(mu, sigma), derive p_yes via the normal CDF.

    Coherent-by-construction (generalizes scripts/predict_margin.py): one distribution,
    every threshold probability read off it. p_yes = P(latent >= threshold) for
    direction="above", else P(latent <= threshold). Writes margin_mu, sigma, p_yes.
    """
    if direction not in ("above", "below"):
        raise ValueError("latent_to_prob: direction must be 'above' or 'below'")

    sys_prompt = (
        "You are an elite calibrated forecaster. Reliability-weight the evidence and estimate "
        f"{latent_description}. Then call submit_margin exactly once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        model = _make_model(model_id, anthropic_api_key)
        user = (
            f"{_question_block(state.question)}\n\n"
            f"EVIDENCE:\n{_render_evidence(state.selected)}\n\n"
            "Submit the latent quantity as mean and 1-sigma via submit_margin."
        )
        res = model.complete_with_tool(
            cached_system_blocks=[sys_prompt],
            user_content=user,
            tool=_MARGIN_TOOL,
            metadata={"question_id": state.question.id, "node": "latent_to_prob"},
        )
        out = res["input"]
        mu = float(out["margin_mean"])
        sigma = abs(float(out["margin_std"]))
        if sigma <= 0:
            p = 1.0 if (mu >= threshold) == (direction == "above") else 0.0
        elif direction == "above":
            p = float(norm.sf(threshold, loc=mu, scale=sigma))
        else:
            p = float(norm.cdf(threshold, loc=mu, scale=sigma))
        state.beliefs["margin_mu"] = mu
        state.beliefs["sigma"] = sigma
        state.beliefs["p_yes"] = _clip01(p)
        state.beliefs["rationale"] = str(out.get("reasoning", ""))
        return state.log(
            f"latent_to_prob N({mu:.2f},{sigma:.2f}) P(x{'>=' if direction == 'above' else '<='}"
            f"{threshold:g})={p:.3f}"
        )

    return _node


# --------------------------------------------------------------------------------------
# 7. calibrate
# --------------------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def calibrate(
    coeff: float = 1.0,
    *,
    bias: float = 0.0,
    method: str = "temperature",
) -> Node:
    """Data-fit-style transform on beliefs['p_yes']; `coeff` is the tunable parameter.

    Preserves p_yes_raw for audit.
      method="temperature": invert-to-logit, divide by `coeff` (T), re-sigmoid. T>1 softens
        (pulls toward 0.5 - fixes overconfidence, our qwen3 finding); T<1 sharpens.
      method="platt": Platt scaling, sigmoid(coeff * logit(p) + bias).
    """
    if coeff == 0:
        raise ValueError("calibrate: coeff (temperature/scale) must be non-zero")

    def _node(state: ReasoningState) -> ReasoningState:
        raw = _clip01(float(state.beliefs.get("p_yes", 0.5)))
        state.beliefs["p_yes_raw"] = raw
        z = _logit(raw)
        if method == "temperature":
            cal = _sigmoid(z / coeff)
        elif method == "platt":
            cal = _sigmoid(coeff * z + bias)
        else:
            raise ValueError(f"calibrate: unknown method {method!r}")
        state.beliefs["p_yes"] = _clip01(cal)
        return state.log(f"calibrate[{method},c={coeff:g}] {raw:.3f} -> {cal:.3f}")

    return _node


# --------------------------------------------------------------------------------------
# 8. abstain
# --------------------------------------------------------------------------------------


def abstain(min_conf: float = 0.4, min_div: float = 0.05) -> Node:
    """Set beliefs['size']=0 unless confidence>=min_conf AND we diverge enough from market.

    Divergence test: if there is no market_price, the confidence gate alone decides (we
    keep the trade). If there is a price, require |p_yes - market_price| >= min_div.
    """

    def _node(state: ReasoningState) -> ReasoningState:
        conf = _clip01(float(state.beliefs.get("confidence", 0.5)))
        p = _clip01(float(state.beliefs.get("p_yes", 0.5)))
        price = state.question.market_price

        confident = conf >= min_conf
        divergent = price is None or abs(p - float(price)) >= min_div
        if confident and divergent:
            return state.log(f"abstain: KEEP (conf={conf:.2f}, div ok)")
        state.beliefs["size"] = 0.0
        reason = "low-conf" if not confident else "too-close-to-market"
        return state.log(f"abstain: ABSTAIN ({reason}, conf={conf:.2f})")

    return _node


# --------------------------------------------------------------------------------------
# 9. size_by_edge
# --------------------------------------------------------------------------------------


def size_by_edge(kelly_frac: float = 0.25, *, cap: float = 1.0) -> Node:
    """Fractional-Kelly stake vs question.market_price. Signed: + = buy YES, - = buy NO.

    Kelly for a binary at price m with belief p: f* = (p - m) / (1 - m) on the YES side,
    and the symmetric NO-side formula when p < m. We scale by kelly_frac and clip to ±cap.
    size=0 if no price, or if a prior node already abstained (size explicitly 0.0).
    """

    def _node(state: ReasoningState) -> ReasoningState:
        if state.beliefs.get("size") == 0.0 and "size" in state.beliefs:
            return state.log("size_by_edge: skipped (already abstained)")
        price = state.question.market_price
        if price is None:
            state.beliefs["size"] = 0.0
            return state.log("size_by_edge: no market price -> size 0")
        m = _clip01(float(price))
        p = _clip01(float(state.beliefs.get("p_yes", 0.5)))
        if p >= m:
            f = (p - m) / (1 - m) if m < 1 else 0.0  # YES side
            sign = 1.0
        else:
            f = (m - p) / m if m > 0 else 0.0  # NO side
            sign = -1.0
        size = sign * kelly_frac * f
        size = float(np.clip(size, -cap, cap))
        state.beliefs["size"] = size
        return state.log(f"size_by_edge: p={p:.3f} m={m:.3f} -> size={size:+.3f}")

    return _node


# --------------------------------------------------------------------------------------
# 10. validate_evidence  (DATA VALIDATION BEFORE INFERENCE - guards against garbage-in)
# --------------------------------------------------------------------------------------

_VALIDATION_TOOL: dict[str, Any] = {
    "name": "submit_validation",
    "description": (
        "Audit the evidence for THIS question before any forecast: which items are usable and "
        "how good is the data overall. The only output channel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "usable_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "1-based indices of evidence items that are ON-TOPIC and SUBSTANTIVE "
                    "(concrete decision-relevant facts/numbers), in priority order. EXCLUDE "
                    "section headers, captions, chart descriptions, empty 'no data' notes, "
                    "fetch errors, and off-topic text."
                ),
            },
            "data_quality": {
                "type": "number",
                "description": (
                    "0-1: how much trustworthy, decision-relevant evidence is present overall "
                    "(0 = nothing usable, 1 = rich direct evidence such as recent numeric polls)."
                ),
            },
            "key_signal_present": {
                "type": "boolean",
                "description": (
                    "True iff the single most decisive evidence for THIS question is present "
                    "(e.g. recent numeric opinion polls for an election)."
                ),
            },
            "notes": {"type": "string", "description": "One line: what is usable vs missing."},
        },
        "required": ["usable_indices", "data_quality", "key_signal_present"],
    },
}


def validate_evidence(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
    min_keep: int = 0,
) -> Node:
    """Validate evidence BEFORE inference - the guard against garbage-in.

    The genome was previously willing to reason confidently over non-numeric poll captions and
    off-topic text. This node has the model audit `state.selected` (falling back to the leakage-
    safe pool), KEEP only on-topic substantive items, and score overall ``data_quality`` +
    ``key_signal_present``. Downstream `call_model` reads ``data_quality`` to anchor on the base
    rate when evidence is thin, so low-quality input yields humility, not a confident hallucination.
    Writes beliefs['data_quality','key_signal_present','validation_notes']; rewrites state.selected.
    """
    sys_prompt = (
        "You are a meticulous research auditor. BEFORE any forecast, judge the EVIDENCE for this "
        "question: which items are on-topic and substantive (real facts/numbers), and how good is "
        "the data overall. Be strict - section headers, chart/graph captions, empty 'no data' "
        "notes, fetch errors, and off-topic text are NOT usable. Call submit_validation once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        items = list(state.selected) or state.pool.on_or_before(state.question.as_of)
        if not items:
            state.beliefs["data_quality"] = 0.0
            state.beliefs["key_signal_present"] = False
            state.beliefs["validation_notes"] = "no evidence"
            return state.log("validate_evidence: no evidence -> dq=0.0")
        model = _make_model(model_id, anthropic_api_key)
        user = (
            f"{_question_block(state.question)}\n\n"
            f"EVIDENCE:\n{_render_evidence(items)}\n\n"
            "Audit the evidence via submit_validation."
        )
        try:
            res = model.complete_with_tool(
                cached_system_blocks=[sys_prompt],
                user_content=user,
                tool=_VALIDATION_TOOL,
                metadata={"question_id": state.question.id, "node": "validate_evidence"},
            )
            out = res["input"]
        except Exception:  # noqa: BLE001 - degrade gracefully: keep all evidence, neutral quality
            state.selected = items
            state.beliefs["data_quality"] = 0.5
            state.beliefs["key_signal_present"] = False
            state.beliefs["validation_notes"] = "validation model error -> kept all"
            return state.log("validate_evidence: model error -> keep all, dq=0.5")
        idx = [int(i) for i in out.get("usable_indices", []) if isinstance(i, int | float)]
        kept = [items[i - 1] for i in idx if 1 <= i <= len(items)]
        if not kept and min_keep > 0:
            kept = items[:min_keep]
        state.selected = kept
        state.beliefs["data_quality"] = _clip01(float(out.get("data_quality", 0.5)))
        state.beliefs["key_signal_present"] = bool(out.get("key_signal_present", False))
        state.beliefs["validation_notes"] = str(out.get("notes", ""))
        return state.log(
            f"validate_evidence: kept {len(kept)}/{len(items)} "
            f"dq={state.beliefs['data_quality']:.2f}"
        )

    return _node


# --------------------------------------------------------------------------------------
# 11. extract_features  (LLM-DRIVEN OPEN-ENDED FEATURE CONSTRUCTION)
# --------------------------------------------------------------------------------------

_FEATURES_TOOL: dict[str, Any] = {
    "name": "submit_features",
    "description": (
        "Identify and COMPUTE the decision-relevant features from the evidence, plus a "
        "reference base rate. Do not forecast yet. The only output channel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "string"},
                        "why_relevant": {"type": "string"},
                    },
                    "required": ["name", "value"],
                },
                "description": (
                    "The decisive quantities you DERIVE from the evidence - e.g. latest poll "
                    "lead, trend across the most recent polls, a pollster-weighted average that "
                    "DOWN-WEIGHTS government-aligned/partisan pollsters, days until resolution, "
                    "swing vs the last comparable election, incumbency. COMPUTE values from the "
                    "evidence; do not merely restate raw text."
                ),
            },
            "base_rate": {
                "type": "number",
                "description": "0-1 reference-class base rate / prior, before specific evidence.",
            },
            "summary": {"type": "string"},
        },
        "required": ["features", "base_rate"],
    },
}


def extract_features(
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    *,
    anthropic_api_key: str | None = None,
) -> Node:
    """LLM-driven, open-ended FEATURE CONSTRUCTION - the deeper-reasoning lever.

    Instead of asking the model to read a raw text blob and guess in one pass, this node has it
    first DERIVE the decisive quantities (poll lead/trend, pollster-reweighted average, days-out,
    swing, ...) plus a base rate. `call_model` then reasons over these computed features. The
    feature set is open-ended (the model decides what matters), making this the neurosymbolic
    bridge: a symbolic feature layer the optimizer can later evolve. Writes
    beliefs['features_text','base_rate'].
    """
    sys_prompt = (
        "You are a quantitative forecasting analyst. From the evidence, IDENTIFY and COMPUTE "
        "the features that most determine this question's outcome. Prefer hard numbers: poll lead, "
        "trend across the latest polls, an average that DOWN-WEIGHTS government-aligned or "
        "partisan pollsters, days until resolution, swing vs the last comparable election. Give "
        "the reference-class base rate. Do NOT forecast yet - just build the features. Call "
        "submit_features once."
    )

    def _node(state: ReasoningState) -> ReasoningState:
        items = list(state.selected) or state.pool.on_or_before(state.question.as_of)
        model = _make_model(model_id, anthropic_api_key)
        user = (
            f"{_question_block(state.question)}\n\n"
            f"EVIDENCE:\n{_render_evidence(items)}\n\n"
            "Build the decisive features via submit_features."
        )
        try:
            res = model.complete_with_tool(
                cached_system_blocks=[sys_prompt],
                user_content=user,
                tool=_FEATURES_TOOL,
                metadata={"question_id": state.question.id, "node": "extract_features"},
            )
            out = res["input"]
        except Exception:  # noqa: BLE001 - degrade gracefully: skip features, reason over raw
            state.beliefs["features_text"] = ""
            state.beliefs.setdefault("base_rate", 0.5)
            return state.log("extract_features: model error -> skipped")
        lines: list[str] = []
        for f in out.get("features", []) or []:
            if not isinstance(f, dict):
                continue
            nm = str(f.get("name", "")).strip()
            val = str(f.get("value", "")).strip()
            if nm and val:
                lines.append(f"- {nm}: {val}")
        state.beliefs["features_text"] = "\n".join(lines) if lines else "(no features extracted)"
        state.beliefs["base_rate"] = _clip01(float(out.get("base_rate", 0.5)))
        return state.log(
            f"extract_features -> {len(lines)} features, base_rate={state.beliefs['base_rate']:.2f}"
        )

    return _node


# --------------------------------------------------------------------------------------
# 12. reweight_polls  (CAPTURED-POLLING fix - symbolic pollster filtering)
# --------------------------------------------------------------------------------------


def reweight_polls(*, drop_leans: tuple[str, ...] = ("gov",)) -> Node:
    """Drop captured (government-aligned) pollsters from the polls evidence before reasoning.

    The crowd over-trusts a captured poll average; this symbolic node removes the flagged
    pollsters' rows (see ``data_sources.pollster_bias``) so the forecaster reads only the
    independent polls, and prepends an auditable note of what was dropped. No-op when no
    polls item is present or nothing matches. This is the neurosymbolic half of the
    captured-polling edge thesis: deterministic filtering feeding the LLM estimate.
    """

    def _node(state: ReasoningState) -> ReasoningState:
        ctx = state.question.text
        dropped: set[str] = set()
        new_selected: list[EvidenceItem] = []
        for it in state.selected:
            if it.source != "polls" or not it.text:
                new_selected.append(it)
                continue
            kept_lines: list[str] = []
            for line in it.text.split("\n"):
                cells = [c.strip() for c in line.split("|")]
                hit = next(
                    (c for c in cells if c and pollster_lean(c, context_hint=ctx) in drop_leans),
                    None,
                )
                if hit:
                    dropped.add(hit)
                    continue
                kept_lines.append(line)
            text = "\n".join(kept_lines)
            if dropped:
                text = (
                    f"[reweighted: dropped government-aligned pollsters "
                    f"({', '.join(sorted(dropped))}) as likely captured]\n{text}"
                )
            new_selected.append(EvidenceItem(text=text, source="polls", date=it.date))
        state.selected = new_selected
        return state.log(f"reweight_polls: dropped {len(dropped)} captured pollster row-type(s)")

    return _node
