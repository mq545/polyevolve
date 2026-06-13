"""Wikipedia pageviews ResearchConnector plugin.

Wraps the proven `polyevolve.data_sources.pageviews.WikipediaPageviewsSource`
(local-language attention momentum, point-in-time, $0) to the new
`core.interfaces.ResearchConnector` contract. The legacy source already has the
right `fetch(dict)->dict` / `render(dict)->str` shape; this adapter just maps the
core `ResearchContext` into the dict the legacy `fetch` expects and declares the
plugin `key` + `categories` for discovery.

Self-registers as @register_connector("pageviews"). Core never imports this file.
"""

from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext
from polyevolve.data_sources.pageviews import WikipediaPageviewsSource


@register_connector("pageviews")
class PageviewsConnector:
    """Wikipedia attention momentum. Plugin key: 'pageviews'."""

    key = "pageviews"
    categories: tuple[str, ...] = ("politics",)

    def __init__(self, inner: WikipediaPageviewsSource | None = None) -> None:
        self._inner = inner or WikipediaPageviewsSource()

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # The legacy source reads question / as_of / tags from a plain dict and
        # enforces the strictly-before-as_of leakage guard internally.
        return self._inner.fetch(
            {
                "question": ctx.question,
                "as_of": ctx.as_of,
                "tags": list(ctx.tags),
            }
        )

    def render(self, payload: dict[str, Any]) -> str:
        return self._inner.render(payload)
