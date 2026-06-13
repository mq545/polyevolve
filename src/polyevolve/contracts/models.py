from typing import Any, Protocol


class Model(Protocol):
    name: str

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the model and force a single tool_use response.

        Each entry in `cached_system_blocks` becomes a cached system text block
        (rendered in order, max 4). `user_content` is per-call dynamic content.
        `tool` is a JSON-schema tool definition; the model is forced to call it.
        `metadata` is opaque to the model call itself; tracing wrappers use it
        (e.g. to link a call to a market). Plain wrappers ignore it.

        Returns: {"input": dict, "usage": dict} where input is the tool call
        arguments and usage carries token + cache stats.
        """
