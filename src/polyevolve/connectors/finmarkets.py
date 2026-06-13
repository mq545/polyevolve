"""Financial-market movement ResearchConnector plugin.

Wraps the proven `polyevolve.data_sources.finmarkets.FinancialMarketsSource`
(pre-as_of price movement of mapped instruments as a leading indicator, $0 Yahoo
chart JSON) to the new `core.interfaces.ResearchConnector` contract. The legacy
source already has the right `fetch(dict)->dict` / `render(dict)->str` shape and
enforces the strictly-before-as_of bar leakage guard internally; this adapter maps
the core `ResearchContext` into the dict the legacy `fetch` expects and declares
the plugin `key` + `categories` for discovery.

The instrument map covers election (equity index + FX) and conflict (defense/oil/
gold) questions, so categories is `("politics", "geopolitics")`.

Self-registers as @register_connector("markets"). Core never imports this file.
"""

from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext
from polyevolve.data_sources.finmarkets import FinancialMarketsSource


@register_connector("markets")
class FinMarketsConnector:
    """Pre-as_of financial-market movement as a leading indicator. Key: 'markets'."""

    key = "markets"
    categories: tuple[str, ...] = ("politics", "geopolitics")

    def __init__(self, inner: FinancialMarketsSource | None = None) -> None:
        self._inner = inner or FinancialMarketsSource()

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # The legacy source reads question / as_of / tags from a plain dict and
        # enforces the strictly-before-as_of bar leakage guard internally.
        return self._inner.fetch(
            {
                "question": ctx.question,
                "as_of": ctx.as_of,
                "tags": list(ctx.tags),
            }
        )

    def render(self, payload: dict[str, Any]) -> str:
        return self._inner.render(payload)
