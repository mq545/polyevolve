"""Records every LLM call to Postgres (always) and Langfuse (if configured)."""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from psycopg.types.json import Json

from .pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


class LLMCallRecorder:
    def __init__(self, db_url: str, langfuse: Any | None = None) -> None:
        self._db_url = db_url
        self._langfuse = langfuse

    def record(
        self,
        *,
        model_name: str,
        cached_system_blocks: list[str],
        user_content: str,
        result: dict[str, Any] | None,
        latency_ms: int,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        usage = (result or {}).get("usage", {})
        cost = estimate_cost_usd(model_name, usage)
        system_chars = sum(len(b) for b in cached_system_blocks)

        try:
            with psycopg.connect(self._db_url) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_calls (
                        agent_name, model_name, market_external_id, latency_ms,
                        input_tokens, output_tokens, cache_read_tokens,
                        cache_creation_tokens, estimated_cost_usd,
                        system_prompt_chars, user_prompt, response, error
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        metadata.get("agent_name"),
                        model_name,
                        metadata.get("market_external_id"),
                        latency_ms,
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                        usage.get("cache_read_input_tokens"),
                        usage.get("cache_creation_input_tokens"),
                        cost,
                        system_chars,
                        user_content,
                        Json((result or {}).get("input")),
                        error,
                    ),
                )
                conn.commit()
        except Exception:
            logger.exception("failed to record llm_call to postgres")

        if self._langfuse is not None:
            self._record_langfuse(
                model_name=model_name,
                user_content=user_content,
                result=result,
                usage=usage,
                cost=cost,
                error=error,
                metadata=metadata,
            )

    def _record_langfuse(
        self,
        *,
        model_name: str,
        user_content: str,
        result: dict[str, Any] | None,
        usage: dict[str, Any],
        cost: float | None,
        error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        # Defensive: never let tracing break the pipeline, and tolerate SDK
        # version differences by catching everything.
        langfuse = self._langfuse
        if langfuse is None:
            return
        try:
            gen = langfuse.start_generation(
                name="predict",
                model=model_name,
                input=user_content,
                metadata=metadata,
            )
            gen.update(
                output=(result or {}).get("input"),
                usage_details={
                    "input": usage.get("input_tokens"),
                    "output": usage.get("output_tokens"),
                },
                cost_details={"total": cost} if cost is not None else None,
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
            gen.end()
        except Exception:
            logger.debug("langfuse record failed (non-fatal)", exc_info=True)
