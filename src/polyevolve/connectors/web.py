"""Web-research ResearchConnector plugin - placeholder for an agent-backed source.

A real web-research connector is AGENT-BACKED: given the market question and a
point-in-time `as_of` cutoff, an LLM agent would issue targeted web searches,
fetch and read candidate pages, and distill a leakage-guarded, price-free brief.
That requires an LLM/tool loop and careful point-in-time discipline (every fetched
page must be filtered to content published strictly before `as_of`, which is hard
on the open web), so it is intentionally NOT implemented here.

This stub keeps the plugin slot wired into discovery with the correct interface
shape, so experiments can name `web` and a real implementation can drop in later
without touching anything else. Until then it is a clean no-data source:
`fetch` returns `{}` and `render` returns "" (the explicit no-data signal the
forecaster sees as a gap rather than fabricated content).

Intended interface for the real implementation:
  fetch(ctx) -> {
      "query": str,                 # the search query/queries the agent ran
      "as_of": str | None,          # ISO cutoff echoed for audit
      "findings": [                 # distilled, point-in-time, price-free items
          {"url": str, "published": str, "summary": str}, ...
      ],
  }
  render(payload) -> a compact, LLM-readable brief, or "" when no findings,
      mirroring the error / no-data / found three-state convention of the other
      connectors (fail loud on hard errors, never silent-empty).

Categories is `("*",)`: web research applies to any market.

Self-registers as @register_connector("web"). Core never imports this file.
"""

from __future__ import annotations

from typing import Any

from polyevolve.core.registry import register_connector
from polyevolve.core.types import ResearchContext


@register_connector("web")
class WebConnector:
    """Agent-backed web research (placeholder). Plugin key: 'web'."""

    key = "web"
    categories: tuple[str, ...] = ("*",)

    def fetch(self, ctx: ResearchContext) -> dict[str, Any]:
        # TODO(web): replace with an agent-backed search/read/distill loop that
        # enforces a strictly-before-`ctx.as_of` publication cutoff per page and
        # returns the structured `findings` payload documented in the module
        # docstring. Until then this is a clean no-data source.
        return {}

    def render(self, payload: dict[str, Any]) -> str:
        # No-data: an empty string is the explicit "no signal" marker the
        # forecaster reads as a gap rather than fabricated content.
        return ""
