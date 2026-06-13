"""The evolvable program: a full ``forecast()`` body ShinkaEvolve rewrites.

This is the keystone of *full-program* evolution. ShinkaEvolve mutates ONLY the code
between the EVOLVE-BLOCK markers - the entire reasoning pipeline. The fixed scaffold
around it (state construction, the final read-off) never changes, so every champion is a
drop-in `Genome` ((Question, EvidencePool) -> Forecast).

`seed_program()` renders a working seed from a `SeedKnobs` (the LLM starts from a real
pipeline, not a blank page). `load_program_genome()` execs a (possibly mutated) program
file back into a callable genome. `validate_program()` is the smoke test the eval bridge
uses to decide ``correct`` - broken LLM code drops out of selection instead of crashing.
"""
# ruff: noqa: E501 - the toolbox doc + scaffold are formatted text shown to the mutation LLM.

from __future__ import annotations

import importlib.util
from pathlib import Path

from polyevolve.reason.dsl import Genome
from polyevolve.reason.seed import SeedKnobs

# The node vocabulary the mutator may compose. Shown verbatim to the mutation LLM so it
# knows the toolbox - keep names/signatures in sync with polyevolve.reason.nodes.
TOOLBOX_DOC = """Available nodes (each is a factory returning a callable state -> state;
apply as `state = node(...)(state)`). Compose them freely inside the EVOLVE-BLOCK:

  select_evidence(k=8, mode="heuristic"|"embedding")   rank leakage-safe evidence -> state.selected
  research(model_id=mid, max_rounds=2)                 agentic gather (<= as_of)
  reweight_polls(drop_leans=("gov",))                  drop captured pollsters from evidence
  validate_evidence(model_id=mid)                      drop off-topic items, score quality
  extract_features(model_id=mid)                       derive decisive quantities
  decompose(model_id=mid)                              split into sub-questions
  call_model(system_prompt=..., model_id=mid)          direct P(YES)
  ensemble(k=3, model_id=mid, aggregate="trimmed_mean", system_prompt=...)
  debate_critique(model_id=mid)                        propose -> refute -> revise
  latent_threshold(model_id=mid)                       coherent margin~N(mu,sig) -> CDF
  latent_to_prob(model_id=mid)                         latent quantity -> p_yes via CDF
  calibrate(coeff=1.3, method="temperature")           soften overconfident p_yes (coeff>1 -> 0.5)
  abstain(min_conf=0.45, min_div=0.06)                 size=0 unless confident AND diverges from price
  size_by_edge(kelly_frac=0.25)                        signed fractional-Kelly stake vs market price

Rules: an estimate node (call_model/ensemble/latent_*/debate_critique) MUST run before
calibrate/abstain/size, and end with size_by_edge so the forecast carries a stake. You are
rewriting the top-level `predict(q, pool, mid) -> Forecast` function: build a
ReasoningState, pipe it through nodes, and return `state.to_forecast()`. You MAY add or refactor
helper functions in the block and call them from predict. `mid` is predict's arg and
`SYSTEM_PROMPT` is a module global - both already in scope. Do not add imports or I/O."""


_SCAFFOLD = '''"""A PolyEvolve forecasting genome - the predict() function is evolved by ShinkaEvolve.

{toolbox}
"""
# ruff: noqa

from __future__ import annotations

from polyevolve.reason.dsl import EvidencePool, Forecast, Question, ReasoningState
from polyevolve.reason.nodes import (
    abstain,
    calibrate,
    call_model,
    debate_critique,
    decompose,
    ensemble,
    extract_features,
    latent_threshold,
    latent_to_prob,
    reweight_polls,
    select_evidence,
    size_by_edge,
    validate_evidence,
)
from polyevolve.reason.research import research

MODEL_ID = {model_id!r}
SYSTEM_PROMPT = {system_prompt!r}


# EVOLVE-BLOCK-START
def predict(q: Question, pool: EvidencePool, mid: str) -> Forecast:
    """The evolvable genome: compose nodes into the state, then read off the Forecast.

    Free to restructure entirely and to add helper functions in this block. Keep the
    signature and the final `return state.to_forecast()`.
    """
    state = ReasoningState(question=q, pool=pool)
{block}
    return state.to_forecast()
# EVOLVE-BLOCK-END


def forecast(q: Question, pool: EvidencePool) -> Forecast:
    """Fixed entrypoint (the Genome the platform loads). Delegates to the evolved predict()."""
    return predict(q, pool, MODEL_ID)
'''


def _seed_block(k: SeedKnobs) -> str:
    """Render the seed EVOLVE-BLOCK body from knobs - a real, working pipeline."""
    lines = [f'state = select_evidence(k={k.select_k}, mode="heuristic")(state)']
    if k.use_pollster_reweight:
        lines.append("state = reweight_polls()(state)")
    if k.use_validate:
        lines.append("state = validate_evidence(model_id=mid)(state)")
    if k.use_features:
        lines.append("state = extract_features(model_id=mid)(state)")
    if k.use_decompose:
        lines.append("state = decompose(model_id=mid)(state)")
    if k.use_latent:
        lines.append("state = latent_threshold(model_id=mid)(state)")
    elif k.use_ensemble:
        lines.append(
            f"state = ensemble(k={k.ensemble_k}, model_id=mid, "
            'aggregate="trimmed_mean", system_prompt=SYSTEM_PROMPT)(state)'
        )
    else:
        lines.append("state = call_model(system_prompt=SYSTEM_PROMPT, model_id=mid)(state)")
    lines.append(f'state = calibrate(coeff={k.calibrate_coeff}, method="temperature")(state)')
    lines.append(
        f"state = abstain(min_conf={k.abstain_min_conf}, min_div={k.abstain_min_div})(state)"
    )
    lines.append(f"state = size_by_edge(kelly_frac={k.kelly_frac})(state)")
    return "\n".join(" " * 4 + ln for ln in lines)


def seed_program(knobs: SeedKnobs) -> str:
    """Render a complete, runnable seed program (initial.py) from a SeedKnobs."""
    return _SCAFFOLD.format(
        toolbox=TOOLBOX_DOC,
        model_id=knobs.model_id,
        system_prompt=knobs.system_prompt,
        block=_seed_block(knobs),
    )


def load_program_genome(path: str | Path) -> Genome:
    """Exec a program file and return its `forecast` as a Genome callable."""
    spec = importlib.util.spec_from_file_location("evolved_genome", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load program at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "forecast", None)
    if not callable(fn):
        raise AttributeError(f"program at {path} has no callable forecast()")
    return fn  # type: ignore[no-any-return]


def validate_program(path: str | Path) -> tuple[bool, str | None]:
    """Static smoke-test a (possibly mutated) program before spending an eval on it.

    Returns (ok, error). Checks the EVOLVE-BLOCK markers survived and the module imports
    and exposes a callable `forecast` (this catches syntax errors, bad imports the LLM may
    have added, and module-scope NameErrors). Runtime errors *inside* forecast() are caught
    per-market by the eval bridge - a genome that errors on every market scores correct=False.
    """
    text = Path(path).read_text()
    if "EVOLVE-BLOCK-START" not in text or "EVOLVE-BLOCK-END" not in text:
        return False, "EVOLVE-BLOCK markers missing"
    try:
        genome = load_program_genome(path)
    except Exception as exc:  # noqa: BLE001 - any load failure is an invalid program
        return False, f"load failed: {exc!r}"
    if not callable(genome):
        return False, "forecast is not callable"
    return True, None
