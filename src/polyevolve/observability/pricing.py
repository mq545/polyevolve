"""Per-call cost estimation.

Local (LiteLLM-routed, e.g. ollama/*) calls are treated as free. Anthropic
direct calls are priced from the published per-1M-token rates, accounting for
cache read (~0.1x input) and cache write (~1.25x input).
"""

from __future__ import annotations

from typing import Any

# (input_per_1M, output_per_1M) in USD.
_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost_usd(model_name: str, usage: dict[str, Any]) -> float | None:
    """Return estimated USD cost for one call, or None if rate is unknown.

    A "/" in the model name means it routed through LiteLLM (local Ollama /
    other provider) - treated as free for our purposes.
    """
    if "/" in model_name:
        return 0.0

    rates = _RATES.get(model_name)
    if rates is None:
        return None

    in_rate, out_rate = rates
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0

    cost = (
        inp * in_rate + cache_write * in_rate * 1.25 + cache_read * in_rate * 0.1 + out * out_rate
    ) / 1_000_000
    return round(cost, 6)
