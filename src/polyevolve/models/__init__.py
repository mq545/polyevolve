"""Model implementations and factory.

Every model goes through LiteLLM, which resolves the provider from the model id and reads
that provider's config from the environment itself - so the model id IS the config:

  ollama/<model>        -> local Ollama (defaults to http://localhost:11434; OLLAMA_API_BASE
                           to override)
  hosted_vllm/<model>   -> local vLLM (VLLM_API_BASE to override)
  anthropic/<model>     -> Anthropic (reads ANTHROPIC_API_KEY)
  openai/<model>        -> OpenAI (reads OPENAI_API_KEY)

No per-provider routing or threaded keys here: pick the model id and set the standard env
var if the provider needs one. When a recorder is supplied, the model is wrapped in
TracedModel so every call is logged to the observability layer.
"""

from __future__ import annotations

from polyevolve.contracts import Model
from polyevolve.observability import LLMCallRecorder

from .litellm_model import LiteLLMModel
from .structured import coerce_rows
from .traced import TracedModel

__all__ = ["LiteLLMModel", "TracedModel", "build_model", "coerce_rows"]


def build_model(*, model_id: str, recorder: LLMCallRecorder | None = None) -> Model:
    base: Model = LiteLLMModel(model_id=model_id)
    if recorder is not None:
        return TracedModel(base, recorder)
    return base
