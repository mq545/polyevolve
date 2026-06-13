"""Model implementations and factory.

Routing rule: a model ID containing "/" routes through LiteLLM (local Ollama,
vLLM, OpenAI, etc.). A bare Claude ID (e.g. "claude-sonnet-4-6") goes to the
direct Anthropic SDK path, which preserves prompt caching and adaptive thinking.

Local serving providers and where their base URL comes from:
  ollama/<model>        -> OLLAMA_API_BASE  (default http://localhost:11434)
  hosted_vllm/<model>   -> VLLM_API_BASE    (default http://localhost:8000/v1)

When a recorder is supplied, the resulting model is wrapped in TracedModel so
every call is logged to the observability layer.
"""

from __future__ import annotations

import os

from polyevolve.contracts import Model
from polyevolve.observability import LLMCallRecorder

from .anthropic_model import AnthropicModel
from .litellm_model import LiteLLMModel
from .structured import coerce_rows
from .traced import TracedModel

__all__ = ["AnthropicModel", "LiteLLMModel", "TracedModel", "build_model", "coerce_rows"]


def build_model(
    *,
    model_id: str,
    anthropic_api_key: str | None = None,
    recorder: LLMCallRecorder | None = None,
) -> Model:
    is_litellm_route = "/" in model_id

    base: Model
    if is_litellm_route:
        provider = model_id.split("/", 1)[0]
        api_base: str | None = None
        if provider == "ollama":
            api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        elif provider == "hosted_vllm":
            api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        base = LiteLLMModel(model_id=model_id, api_base=api_base)
    elif not anthropic_api_key:
        raise RuntimeError(
            f"ANTHROPIC_API_KEY is required for direct Anthropic model {model_id!r}. "
            f"For local inference, set DEFAULT_MODEL to a LiteLLM-routed string like "
            f"'ollama/qwen2.5:14b'."
        )
    else:
        base = AnthropicModel(api_key=anthropic_api_key, model_id=model_id)

    if recorder is not None:
        return TracedModel(base, recorder)
    return base
