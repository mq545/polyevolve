"""LiteLLM-based model wrapper for multi-provider support (Ollama, vLLM, OpenAI).

Use this for local/dev inference. For Anthropic in production, prefer
AnthropicModel directly - it preserves prompt caching and adaptive thinking
which are silently dropped on this path.

Structured output strategy - JSON mode, not forced tool_use. Local models served
through Ollama vary wildly in their support for forced `tool_choice`: notably
Qwen3 *thinking* models get stuck in a `<think>` loop and never emit the forced
call (finish_reason=stop, no tool_calls). Asking for a JSON object instead works
across thinking and non-thinking models alike. So we derive a JSON instruction
from the tool's input_schema, let the model reason freely, then strip any
`<think>…</think>` block and parse the JSON object out of the content. This keeps
the same Model contract (returns {"input": dict, "usage": dict}).
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm

from polyevolve.contracts import Model

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# also tolerate an unterminated trailing <think> (model ran out of tokens mid-thought)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _json_instruction(tool: dict[str, Any]) -> str:
    """Build a 'output exactly this JSON' instruction from a tool's input_schema."""
    schema = tool.get("input_schema", {})
    props = schema.get("properties", {})
    required = set(schema.get("required", list(props.keys())))
    lines = ["Output ONLY a single JSON object with exactly these fields:"]
    for name, spec in props.items():
        typ = spec.get("type", "string")
        desc = spec.get("description", "")
        req = "required" if name in required else "optional"
        lines.append(f'  "{name}" ({typ}, {req}): {desc}')
    lines.append(
        "Think first if you wish, then output the JSON object as the LAST thing in "
        "your response. No prose or code fences after the JSON."
    )
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Strip thinking, then extract the last balanced top-level JSON object."""
    cleaned = _THINK_RE.sub("", text)
    cleaned = _OPEN_THINK_RE.sub("", cleaned)  # drop any unclosed think tail
    cleaned = cleaned.replace("```json", "").replace("```", "")

    # Scan for balanced { ... } objects; keep the last complete one (the answer
    # usually comes after the reasoning).
    candidates: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(cleaned):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(cleaned[start : i + 1])
                start = -1

    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("no parseable JSON object found in model output")


class LiteLLMModel:
    name: str

    def __init__(
        self,
        *,
        model_id: str,
        max_tokens: int = 4096,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._api_base = api_base
        self._api_key = api_key
        self._temperature = temperature
        self.name = model_id

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Caching is Anthropic-specific; on this path the cached blocks are just
        # concatenated into one system message.
        system_text = "\n\n".join([*cached_system_blocks, _json_instruction(tool)])

        # Local instruct models (qwen via Ollama) intermittently emit prose / fenced / partial
        # output that fails JSON extraction. Retry ONCE with a stricter, deterministic repair
        # prompt before giving up - this recovered the ~5-7% per-call failures that were
        # silently dropping markets from backtests and crashing the deep/agentic genome.
        repair = (
            "\n\nCRITICAL: Respond with ONLY a single valid JSON object matching the schema - "
            "no prose, no markdown code fences, no commentary before or after."
        )
        choice = None
        for attempt in range(2):
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": system_text + (repair if attempt else "")},
                    {"role": "user", "content": user_content},
                ],
            }
            if attempt:
                kwargs["temperature"] = 0.0  # deterministic repair attempt
            elif self._temperature is not None:
                kwargs["temperature"] = self._temperature
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = litellm.completion(**kwargs)
            choice = response.choices[0]
            content = getattr(choice.message, "content", None) or ""
            try:
                args = _extract_json_object(content)
                break
            except RuntimeError as e:
                if attempt == 1:
                    raise RuntimeError(
                        f"{self._model_id}: {e}. finish_reason="
                        f"{getattr(choice, 'finish_reason', 'unknown')}"
                    ) from e

        usage = getattr(response, "usage", None)
        return {
            "input": args,
            "usage": {
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }


_: type[Model] = LiteLLMModel  # protocol satisfaction check
