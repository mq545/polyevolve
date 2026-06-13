"""JOINT event inference - forecast a connected set of markets as ONE problem.

The per-market genome answers "will X win?" N times in isolation: it can never reason
"A's gain is B's loss", it re-gathers the same evidence N times, and its sibling
probabilities are incoherent until post-hoc normalization. This module forecasts the
EVENT: all sibling markets in one context, one shared evidence pool, one model call
eliciting a categorical DISTRIBUTION over the outcomes plus an "other" bucket (most
candidate lists are non-exhaustive). Coherent by construction, and - the discrimination
lever - the model reasons COMPARATIVELY across the field instead of scoring each
outcome blind to its rivals.

`forecast_event` is the unit; `joint_genome_over` adapts it to the bench: it groups
questions by `event_id` (Polymarket's own event grouping, carried through ingest),
forecasts each multi-market group jointly (singletons fall back to the per-market
genome), and returns per-question Forecasts aligned with the input.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from polyevolve.models import build_model, coerce_rows
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Forecast, Genome, Question

__all__ = ["forecast_event", "joint_genome_over"]

_DISTRIBUTION_TOOL: dict[str, Any] = {
    "name": "submit_distribution",
    "description": (
        "Submit your probability for EACH listed outcome of this event, as one coherent "
        "distribution. The only output channel - no free text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "probabilities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "Outcome number from the list.",
                        },
                        "probability": {
                            "type": "number",
                            "description": "P(this outcome) in [0,1].",
                        },
                    },
                    "required": ["index", "probability"],
                },
                "description": "One entry per listed outcome, in any order.",
            },
            "other_probability": {
                "type": "number",
                "description": (
                    "P(none of the listed outcomes) if the list is not exhaustive; 0 if it is."
                ),
            },
            "confidence": {"type": "number", "description": "Overall confidence in [0,1]."},
            "reasoning": {"type": "string", "description": "Concise comparative reasoning."},
        },
        "required": ["probabilities", "confidence", "reasoning"],
    },
}

_SYS_PROMPT = (
    "You are an elite calibrated forecaster. The outcomes below belong to ONE event (e.g. the "
    "candidates of one election), so at most one resolves YES - reason about them AGAINST EACH "
    "OTHER: who leads whom in the same polls, momentum, structural advantages. Use ONLY the "
    "dated evidence and general knowledge as of the decision date; do not invent numbers. "
    "Spread probability over the outcomes (plus 'other' if the list is not exhaustive) so it "
    "sums to ~1. Call submit_distribution exactly once."
)


def _merged_pool(pools: Sequence[EvidencePool]) -> list[EvidenceItem]:
    """Union of the group's evidence, deduped by (source, text prefix)."""
    seen: set[tuple[str, str]] = set()
    out: list[EvidenceItem] = []
    for pool in pools:
        for it in pool.items:
            key = (it.source, it.text[:200])
            if key not in seen:
                seen.add(key)
                out.append(it)
    return out


def forecast_event(
    questions: Sequence[Question],
    pools: Sequence[EvidencePool],
    *,
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    anthropic_api_key: str | None = None,
    exhaustive: bool = False,
) -> list[Forecast]:
    """Forecast one event's sibling markets jointly. Returns Forecasts aligned 1:1.

    One model call sees every outcome plus the merged evidence and emits a coherent
    distribution. Missing/garbled entries get the residual mass spread evenly; if the
    listed mass plus 'other' exceeds 1 we rescale (the distribution property is enforced
    in CODE, not trusted from the model). On model failure every market gets a neutral
    abstaining Forecast (fail-soft).
    """
    if len(questions) != len(pools):
        raise ValueError("questions and pools must align 1:1")
    qs = list(questions)
    n = len(qs)
    as_of = max(q.as_of for q in qs)
    evidence = _merged_pool(pools)
    ev_block = (
        "\n".join(
            f"- ({it.date.date().isoformat() if it.date else 'undated'}) "
            f"[{it.source}] {it.text[:1200]}"
            for it in evidence[:12]
        )
        or "(no evidence)"
    )
    outcomes = "\n".join(f"{i + 1}. {q.text}" for i, q in enumerate(qs))
    user = (
        f"EVENT OUTCOMES (at most one resolves YES):\n{outcomes}\n\n"
        f"AS OF (decision date): {as_of.date().isoformat()}\n"
        f"LIST IS {'EXHAUSTIVE' if exhaustive else 'NOT exhaustive (allow other_probability)'}\n\n"
        f"EVIDENCE (shared for the whole event):\n{ev_block}\n\n"
        "Submit the full distribution via submit_distribution."
    )
    model = build_model(model_id=model_id, anthropic_api_key=anthropic_api_key)
    try:
        res = model.complete_with_tool(
            cached_system_blocks=[_SYS_PROMPT],
            user_content=user,
            tool=_DISTRIBUTION_TOOL,
            metadata={"question_id": qs[0].event_id or qs[0].id, "node": "forecast_event"},
        )
        out = res["input"]
    except Exception:  # noqa: BLE001 - fail-soft: neutral abstain on every leg
        return [
            Forecast(p_yes=1.0 / max(2, n), size=0.0, confidence=0.3, rationale="joint model error")
            for _ in qs
        ]

    # coerce_rows accepts both the schema shape [{index, probability}] and a flat
    # positional list of floats [0.6, 0.25] (the anti-false-null seam) - without it a
    # flat-shape reply would silently collapse to a neutral abstain on every leg.
    probs = [0.0] * n
    got = [False] * n
    for entry in coerce_rows(out.get("probabilities"), scalar_field="probability"):
        try:
            i = int(entry["index"]) - 1
            pv = float(entry["probability"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n:
            probs[i] = min(1.0, max(0.0, pv))
            got[i] = True
    other = min(1.0, max(0.0, float(out.get("other_probability", 0.0) or 0.0)))
    if exhaustive:
        other = 0.0
    # spread any residual mass over unanswered outcomes; then de-overround if needed
    missing = [i for i in range(n) if not got[i]]
    if missing:
        residual = max(0.0, 1.0 - other - sum(probs))
        for i in missing:
            probs[i] = residual / len(missing)
    if sum(probs) + other > 1.0 and sum(probs) > 0:
        scale = (1.0 - other) / sum(probs) if other < 1.0 else 0.0
        probs = [p * scale for p in probs]
    conf = min(1.0, max(0.0, float(out.get("confidence", 0.5))))
    why = str(out.get("reasoning", ""))
    return [
        Forecast(p_yes=min(1.0, max(0.0, p)), size=0.0, confidence=conf, rationale=why)
        for p in probs
    ]


def joint_genome_over(
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None,
    *,
    fallback: Genome,
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    anthropic_api_key: str | None = None,
) -> list[Forecast]:
    """Forecast a corpus with JOINT inference per multi-market event.

    Groups by `event_id`; each group of >=2 goes through `forecast_event` (ONE call per
    event); singletons / event-less questions use the per-market `fallback` genome.
    Returns Forecasts aligned 1:1 with `questions`.
    """
    qs = list(questions)
    ps = list(pools) if pools is not None else [EvidencePool(items=[]) for _ in qs]
    if len(qs) != len(ps):
        raise ValueError("questions and pools must align 1:1")

    by_event: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs):
        if q.event_id:
            by_event[q.event_id].append(i)

    out: list[Forecast | None] = [None] * len(qs)
    for idxs in by_event.values():
        if len(idxs) < 2:
            continue
        fcs = forecast_event(
            [qs[i] for i in idxs],
            [ps[i] for i in idxs],
            model_id=model_id,
            anthropic_api_key=anthropic_api_key,
        )
        for i, fc in zip(idxs, fcs, strict=True):
            out[i] = fc
    for i, q in enumerate(qs):
        if out[i] is None:
            try:
                out[i] = fallback(q, ps[i])
            except Exception:  # noqa: BLE001
                out[i] = Forecast(p_yes=0.5, size=0.0, confidence=0.3, rationale="fallback error")
    return [fc for fc in out if fc is not None]
