"""The SEED GENOME - the crowd-parity scaffold ShinkaEvolve mutates.

A *genome* is a function `forecast(Question, EvidencePool) -> Forecast` (see
polyevolve.reason.dsl.Genome). This module provides the seed: a fixed, sane composition of
the frozen node library (polyevolve.reason.nodes) wrapped so an optimizer can mutate it two
ways:

1. KNOBS - the `SeedKnobs` dataclass at the top exposes the scalar/flag/string parameters
   (prompt text, calibration temperature, abstain gates, kelly fraction, decompose/ensemble
   switches). An optimizer mutates a `SeedKnobs` and calls `make_seed_genome(knobs)`.
2. COMPOSITION - the body between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` inside
   `forecast()` is the node pipeline. ShinkaEvolve rewrites *that span* (which nodes, in
   what order) while leaving the contract and node bodies frozen.

Pipeline (crowd-parity scaffold):
    select_evidence
      -> (optional) decompose            # widen the question into sub-questions
      -> call_model | ensemble           # reliability-weighting forecasting system prompt
      -> calibrate                       # temperature-soften (fixes qwen3 overconfidence)
      -> abstain                         # confidence + divergence gate
      -> size_by_edge                    # fractional-Kelly stake vs market price

Everything flows through `ReasoningState`; the final `Forecast` is read off beliefs via
`ReasoningState.to_forecast()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polyevolve.reason.dsl import EvidencePool, Forecast, Genome, Question, ReasoningState
from polyevolve.reason.nodes import (
    abstain,
    calibrate,
    call_model,
    decompose,
    ensemble,
    extract_features,
    latent_threshold,
    reweight_polls,
    select_evidence,
    size_by_edge,
    validate_evidence,
)
from polyevolve.reason.research import research

__all__ = ["SeedKnobs", "forecast", "make_seed_genome", "run_genome"]

# Default model id (kept identical to the node library so tests that monkeypatch
# `polyevolve.reason.nodes.build_model` work without configuring anything).
DEFAULT_MODEL_ID = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M"

# The reliability-weighting forecasting system prompt. This is a KNOB (mutable text), but
# the seed value encodes our crowd-parity doctrine: weight sources by reliability, respect
# the as-of cutoff, prefer humility over invented precision.
SEED_SYSTEM_PROMPT = (
    "You are an elite, calibrated forecaster competing against an efficient prediction "
    "market. Read the question, its exact resolution criteria, and the dated evidence.\n"
    "Method:\n"
    "1. RELIABILITY-WEIGHT the evidence: trust independent, recent, primary, track-record "
    "sources; discount partisan, stale, second-hand, or methodologically suspect ones "
    "(e.g. captured pollsters). Note conflicts explicitly.\n"
    "2. Start from the base rate for this class of event, then update only as far as the "
    "RELIABLE evidence justifies. Do not over-react to a single salient headline.\n"
    "3. Use ONLY evidence dated on or before the decision date and general knowledge as of "
    "that date. Do NOT invent numbers (polls, vote shares) that are absent from the "
    "evidence; if the data is thin, widen your uncertainty and lower your confidence.\n"
    "4. Output a single calibrated P(YES). Across many such questions your stated "
    "probabilities should match observed frequencies. Call submit_prediction exactly once."
)


# --------------------------------------------------------------------------------------
# TUNABLE KNOBS - an optimizer mutates an instance of this and calls make_seed_genome().
# --------------------------------------------------------------------------------------


@dataclass
class SeedKnobs:
    """The mutable parameters of the seed genome.

    Every field is something an optimizer is free to perturb. The EVOLVE-BLOCK body of
    `forecast()` reads these (and nothing else) to configure the node pipeline.
    """

    # forecasting system prompt handed to call_model / ensemble (reliability-weighting).
    system_prompt: str = SEED_SYSTEM_PROMPT
    # calibration temperature for `calibrate(method="temperature")`. >1 softens toward 0.5
    # (counters qwen3 overconfidence); <1 sharpens; ==1 is identity. Must be non-zero.
    calibrate_coeff: float = 1.3
    # abstain gates: keep a trade only if confidence >= min_conf AND we diverge from the
    # market price by >= min_div.
    abstain_min_conf: float = 0.45
    abstain_min_div: float = 0.06
    # fractional-Kelly multiplier for size_by_edge (0 == never stake; 1 == full Kelly).
    kelly_frac: float = 0.25
    # composition switches the EVOLVE-BLOCK reads.
    use_decompose: bool = False
    use_ensemble: bool = False
    ensemble_k: int = 3
    # COHERENCE lever: estimate a latent margin/quantity ~N(mu,sigma) and read p_yes off the
    # normal CDF (coherent by construction) instead of asking for P(YES) directly - fixes the
    # per-binary "YES-machine" incoherence across sibling/threshold markets.
    use_latent: bool = False
    # DEEP-reasoning levers (off by default = the shallow seed): validate evidence before
    # inference (guards garbage-in), and let the model construct its own derived features.
    use_validate: bool = False
    use_features: bool = False
    # AGENTIC lever: gather the genome's own evidence via leakage-safe tools (plan->execute
    # <= as_of -> refine) before reasoning, instead of consuming only the pre-fixed pool.
    use_research: bool = False
    research_rounds: int = 2
    # CAPTURED-POLLING lever: drop government-aligned pollsters from the polls evidence so the
    # forecaster reads only independent polls (the domain Tier-1 edge thesis).
    use_pollster_reweight: bool = False
    # number of evidence items the selector keeps.
    select_k: int = 8
    # model id every model-using node is built with.
    model_id: str = DEFAULT_MODEL_ID
    # optional API key for hosted (e.g. anthropic) models; None for local/ollama.
    anthropic_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.calibrate_coeff == 0:
            raise ValueError("SeedKnobs.calibrate_coeff (temperature) must be non-zero")
        if self.ensemble_k < 1:
            raise ValueError("SeedKnobs.ensemble_k must be >= 1")
        if self.select_k < 0:
            raise ValueError("SeedKnobs.select_k must be >= 0")


# The seed genome runs against these defaults unless an optimizer overrides them.
_DEFAULT_KNOBS = SeedKnobs()


# --------------------------------------------------------------------------------------
# The seed genome: forecast(Question, EvidencePool) -> Forecast
# --------------------------------------------------------------------------------------


def forecast(
    q: Question,
    pool: EvidencePool,
    knobs: SeedKnobs = _DEFAULT_KNOBS,
) -> Forecast:
    """Run the seed crowd-parity scaffold and return a Forecast.

    The composable pipeline lives in the EVOLVE-BLOCK below; everything outside it (state
    construction, knob reading, the final read-off) is the stable harness ShinkaEvolve
    keeps fixed. Only the knobs and the block body are mutated.
    """
    state = ReasoningState(question=q, pool=pool)
    mid = knobs.model_id
    key = knobs.anthropic_api_key

    # EVOLVE-BLOCK-START
    # 0. (optional) AGENTIC GATHER: the genome fetches its own leakage-safe evidence
    # (plan -> execute <= as_of -> refine) instead of consuming only the pre-fixed pool.
    if knobs.use_research:
        state = research(model_id=mid, anthropic_api_key=key, max_rounds=knobs.research_rounds)(
            state
        )

    # 1. retrieve: leakage-safe, ranked evidence into state.selected.
    state = select_evidence(k=knobs.select_k, mode="heuristic")(state)

    # 1a. (optional) CAPTURED-POLLING reweight: drop government-aligned pollsters so the
    # forecaster reads only independent polls (symbolic; runs before validate/features see them).
    if knobs.use_pollster_reweight:
        state = reweight_polls()(state)

    # 1b. (optional) VALIDATE evidence before inference: drop off-topic/contentless items and
    # score data_quality, so the forecaster anchors to the base rate instead of hallucinating
    # from garbage (e.g. non-numeric poll captions).
    if knobs.use_validate:
        state = validate_evidence(model_id=mid, anthropic_api_key=key)(state)

    # 1c. (optional) CONSTRUCT FEATURES: have the model derive the decisive quantities
    # (poll lead/trend, pollster-reweighted average, days-out, swing) before estimating.
    if knobs.use_features:
        state = extract_features(model_id=mid, anthropic_api_key=key)(state)

    # 2. (optional) decompose the question to widen reasoning before estimating.
    if knobs.use_decompose:
        state = decompose(model_id=mid, anthropic_api_key=key)(state)

    # 3. estimate P(YES): a latent margin -> CDF (coherent), an ensemble, or a direct P(YES).
    if knobs.use_latent:
        state = latent_threshold(model_id=mid, anthropic_api_key=key)(state)
    elif knobs.use_ensemble:
        state = ensemble(
            k=knobs.ensemble_k,
            model_id=mid,
            anthropic_api_key=key,
            aggregate="trimmed_mean",
            system_prompt=knobs.system_prompt,
        )(state)
    else:
        state = call_model(system_prompt=knobs.system_prompt, model_id=mid, anthropic_api_key=key)(
            state
        )

    # 4. calibrate: temperature-soften the (often overconfident) raw probability.
    state = calibrate(coeff=knobs.calibrate_coeff, method="temperature")(state)

    # 5. abstain unless confident AND meaningfully divergent from the market.
    state = abstain(min_conf=knobs.abstain_min_conf, min_div=knobs.abstain_min_div)(state)

    # 6. size: fractional-Kelly stake vs market price (0 if abstained / no price).
    state = size_by_edge(kelly_frac=knobs.kelly_frac)(state)
    # EVOLVE-BLOCK-END

    return state.to_forecast()


# --------------------------------------------------------------------------------------
# Optimizer-facing helpers
# --------------------------------------------------------------------------------------


def make_seed_genome(knobs: SeedKnobs | None = None) -> Genome:
    """Factory: bind a `SeedKnobs` into a `Genome` (Question, EvidencePool) -> Forecast.

    This is what an optimizer calls after mutating a knob set: it returns a plain genome
    matching the dsl.Genome signature, with the knobs closed over.
    """
    resolved = knobs if knobs is not None else SeedKnobs()

    def _genome(q: Question, pool: EvidencePool) -> Forecast:
        return forecast(q, pool, knobs=resolved)

    return _genome


def run_genome(genome: Genome, question: Question, pool: EvidencePool) -> Forecast:
    """Run any genome on one (question, pool). Thin, uniform entry point for bench/evolve."""
    return genome(question, pool)
