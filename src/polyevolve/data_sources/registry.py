"""Data source registry - runs a pluggable list of sources, assembles context.

Holds a list of DataSource impls (the Protocol in contracts.py) and gathers each
one's rendered context for a market. New signal sources = append a DataSource to
the list; nothing else changes. Each source's raw payload is cached to
raw_fetches (audit + replay).

FAIL LOUD: a source that errors or returns an error payload contributes an
EXPLICIT error marker to the context - never a silent omission. The 0/150
empty-context disaster happened precisely because failures were swallowed and
rendered as innocuous "(no news found)". A forecaster must be able to tell
"nothing happened" from "we failed to look".
"""

from __future__ import annotations

import logging
from datetime import datetime

import psycopg

from polyevolve.contracts import DataSource, Market
from polyevolve.storage import db

from .gdelt import GdeltSource  # noqa: F401 - kept as a secondary/legacy source
from .gdelt_doc import GdeltDocSource
from .pageviews import WikipediaPageviewsSource
from .polls import WikipediaPollsSource

logger = logging.getLogger(__name__)

# Sentinel prefix for context values that represent a FETCH FAILURE (not genuine
# absence of data). Downstream code / humans can grep for it; the model sees it
# and knows the signal is missing rather than empty.
ERROR_PREFIX = "[SOURCE ERROR]"


def default_sources(
    *,
    enable_gdelt: bool = True,
    enable_pageviews: bool = True,
    enable_polls: bool = True,
    enable_markets: bool = False,
) -> list[DataSource]:
    """The standard production set of data sources.

    One place to define what runs, so call sites don't each hardcode it. As we
    add connectors (BigQuery GDELT for historical, a keyed news API, on-chain),
    they get appended here.
    """
    sources: list[DataSource] = []
    if enable_gdelt:
        # Primary signal source: DOC 2.0 relevance search + scraped body text
        # (key "news"). The legacy GdeltSource (titles only, ~90d) is kept as a
        # secondary; the gdelt_bq BigQuery path is retired (noise, no content).
        sources.append(GdeltDocSource())
    if enable_pageviews:
        # Local-language Wikipedia pageview momentum (key "pageviews"). Free, no
        # key, immutable daily history => genuinely point-in-time, low leakage.
        sources.append(WikipediaPageviewsSource())
    if enable_polls:
        # Local-language Wikipedia opinion-poll text (key "polls"). Free, no key.
        # Leakage-safe ONLY because it reads the article's wikitext as of the
        # latest revision STRICTLY BEFORE as_of (the live poll page is post-edited).
        sources.append(WikipediaPollsSource())
    if enable_markets:
        # UNPRICED leading indicator (key "markets"): equity index + FX + conflict
        # basket (defense/oil/gold) movement STRICTLY before as_of, from Yahoo. Free,
        # no key, deep history => point-in-time for any cutoff. Off by default; the
        # edge-hunt (Track B) turns it on to test whether unpriced signal beats news.
        from .finmarkets import FinancialMarketsSource

        sources.append(FinancialMarketsSource())
    return sources


class DataRegistry:
    """Holds enabled data sources and gathers context per market."""

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        # Default to the production set; tests/callers can inject an explicit list
        # (including fakes) for full control.
        self._sources = sources if sources is not None else default_sources()

    @property
    def sources(self) -> list[DataSource]:
        return self._sources

    def gather(
        self,
        market: Market,
        conn: psycopg.Connection | None = None,
        as_of: datetime | None = None,
    ) -> dict[str, str]:
        """Return {source_name: rendered_context} for the agent.

        If conn is provided, raw payloads (including error payloads) are cached to
        raw_fetches. If as_of is provided, sources only return data from before
        that instant (point-in-time view for backtesting). None => latest data.

        A source raising an exception does NOT abort the others - but it DOES
        leave a loud error marker in its slot, so a failed fetch can never be
        mistaken for "no data found".
        """
        context: dict[str, str] = {}
        # Tags are curated entity labels (e.g. country names) some sources map to
        # their own keys (GDELT FIPS codes) - pass them through alongside the
        # question. Sources that don't use tags simply ignore the key.
        tags = market.metadata.get("tags", [])
        for source in self._sources:
            name = source.name
            try:
                payload = source.fetch({"question": market.question, "as_of": as_of, "tags": tags})
                if conn is not None:
                    db.insert_raw_fetch(
                        conn,
                        source=name,
                        endpoint=f"market:{market.external_id}",
                        payload=payload,
                    )
                # A payload carrying an `error` is a failure, not empty data -
                # render it loudly so it can't masquerade as "(no news found)".
                if isinstance(payload, dict) and payload.get("error"):
                    context[name] = f"{ERROR_PREFIX} {name}: {payload['error']}"
                else:
                    context[name] = source.render(payload)
            except Exception as exc:  # noqa: BLE001 - one source must not kill the rest
                logger.exception("data source %s failed for market %s", name, market.external_id)
                if conn is not None:
                    db.insert_raw_fetch(
                        conn,
                        source=name,
                        endpoint=f"market:{market.external_id}",
                        payload={"error": "exception", "detail": repr(exc)},
                    )
                context[name] = f"{ERROR_PREFIX} {name}: exception {exc!r}"
        return context
