"""GDELT DOC 2.0 news ResearchConnector plugin.

Wraps the proven `polyevolve.data_sources.gdelt_doc.GdeltDocSource` (full-text
relevance search + article-body scraping, point-in-time, $0) to the new
`core.interfaces.ResearchConnector` contract. The legacy source already has the
right `fetch(dict)->dict` / `render(dict)->str` shape and enforces the
`seendate < as_of` leakage guard internally; this adapter just maps the core
`ResearchContext` into the dict the legacy `fetch` expects and declares the
plugin `key` + `categories` for discovery.

News applies to every category (`("*",)`): a forecaster on any market benefits
from on-topic, leakage-guarded coverage.

Self-registers as @register_connector("news"). Core never imports this file.
"""

from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext
from polyevolve.data_sources.gdelt_doc import GdeltDocSource


@register_connector("news")
class NewsConnector:
    """GDELT DOC 2.0 relevance search + body text. Plugin key: 'news'."""

    key = "news"
    categories: tuple[str, ...] = ("*",)

    def __init__(self, inner: GdeltDocSource | None = None) -> None:
        self._inner = inner or GdeltDocSource()

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # The legacy source reads question / as_of from a plain dict and enforces
        # the client-side seendate < as_of leakage guard internally.
        return self._inner.fetch(
            {
                "question": ctx.question,
                "as_of": ctx.as_of,
            }
        )

    def render(self, payload: dict[str, Any]) -> str:
        return self._inner.render(payload)
