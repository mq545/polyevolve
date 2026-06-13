"""TracedModel - wraps any Model to record each call to the observability layer."""

from __future__ import annotations

import time
from typing import Any

from polyevolve.contracts import Model
from polyevolve.observability import LLMCallRecorder


class TracedModel:
    name: str

    def __init__(self, inner: Model, recorder: LLMCallRecorder) -> None:
        self._inner = inner
        self._recorder = recorder
        self.name = inner.name

    def complete_with_tool(
        self,
        *,
        cached_system_blocks: list[str],
        user_content: str,
        tool: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.monotonic()
        error: str | None = None
        result: dict[str, Any] | None = None
        try:
            result = self._inner.complete_with_tool(
                cached_system_blocks=cached_system_blocks,
                user_content=user_content,
                tool=tool,
            )
            return result
        except Exception as e:
            error = repr(e)
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            self._recorder.record(
                model_name=self._inner.name,
                cached_system_blocks=cached_system_blocks,
                user_content=user_content,
                result=result,
                latency_ms=latency_ms,
                error=error,
                metadata=metadata or {},
            )


_: type[Model] = TracedModel  # protocol satisfaction check
