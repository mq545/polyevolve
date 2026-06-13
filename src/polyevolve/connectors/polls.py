"""Wikipedia opinion-poll ResearchConnector plugin.

Wraps the proven `polyevolve.data_sources.polls.WikipediaPollsSource`
(local-language Wikipedia opinion-poll text, point-in-time via revision history,
$0) to the new `core.interfaces.ResearchConnector` contract. The legacy source
already has the right `fetch(dict)->dict` / `render(dict)->str` shape and enforces
the strictly-before-as_of revision leakage guard internally; this adapter maps the
core `ResearchContext` into the dict the legacy `fetch` expects and declares the
plugin `key` + `categories` for discovery.

Polls are an elections signal, so categories is `("politics",)`.

Self-registers as @register_connector("polls"). Core never imports this file.
"""

from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext
from polyevolve.data_sources.polls import WikipediaPollsSource


@register_connector("polls")
class PollsConnector:
    """Point-in-time Wikipedia opinion-poll text. Plugin key: 'polls'."""

    key = "polls"
    categories: tuple[str, ...] = ("politics",)

    def __init__(self, inner: WikipediaPollsSource | None = None) -> None:
        self._inner = inner or WikipediaPollsSource()

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # The legacy source reads question / as_of / tags from a plain dict and
        # enforces the strictly-before-as_of revision leakage guard internally.
        return self._inner.fetch(
            {
                "question": ctx.question,
                "as_of": ctx.as_of,
                "tags": list(ctx.tags),
            }
        )

    def render(self, payload: dict[str, Any]) -> str:
        return self._inner.render(payload)
