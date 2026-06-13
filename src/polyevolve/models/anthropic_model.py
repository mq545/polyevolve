"""Anthropic Claude wrapper with prompt caching and forced tool_use structured output.

Designed for the research-agent loop where many per-market calls share a stable
system + domain-context prefix. The first call writes the cache; subsequent
calls in the same run read from it at ~0.1x cost.

Cache layout: tools (static) -> system blocks (cached) -> user content (volatile).
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from polyevolve.contracts import Model


class AnthropicModel:
    name: str

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
        thinking: bool = True,
    ) -> None:
        self._client = Anthropic(api_key=api_key)
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._thinking = thinking
        self.name = model_id

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(cached_system_blocks) > 4:
            raise ValueError("Anthropic supports at most 4 cache_control breakpoints per request")

        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": block,
                "cache_control": {"type": "ephemeral"},
            }
            for block in cached_system_blocks
        ]

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": self._max_tokens,
            "system": system_blocks,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "messages": [{"role": "user", "content": user_content}],
        }
        if self._thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        response = self._client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                return {
                    "input": dict(block.input) if isinstance(block.input, dict) else {},
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cache_creation_input_tokens": getattr(
                            response.usage, "cache_creation_input_tokens", 0
                        ),
                        "cache_read_input_tokens": getattr(
                            response.usage, "cache_read_input_tokens", 0
                        ),
                    },
                }

        raise RuntimeError(
            f"Model did not call tool {tool['name']!r}. stop_reason={response.stop_reason}"
        )


_: type[Model] = AnthropicModel  # protocol satisfaction check
