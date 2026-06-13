"""Per-event search-interest features - the validated genome upgrade.

Backtest (scripts/backtest_trends_llm.py, n=85 winner-markets) showed that blending the
genome's P(YES) 50/50 with each candidate's normalized search SHARE across its event field
lifts resolution 0.033 -> 0.054 (near the crowd's 0.063) and cuts Brier 27%. Search interest
is PRICED by the crowd (it adds nothing on top of the crowd) but it is exactly the
attention/momentum signal our genome underweights, so it is incremental to US.

This module packages that lever for live use: given a set of sibling markets (one event),
resolve each to the term locals search (LLM keyword resolver), pull point-in-time interest
(leakage-safe via trends.py), and return each market's share of the field. `geo_for` maps an
event title to an ISO-2 locale (worldwide if unknown). All $0, all leakage-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from polyevolve.data_sources.keyword_resolver import resolve_keywords
from polyevolve.data_sources.trends import interest_shares

__all__ = ["GEO", "event_search_shares", "geo_for"]

# event-title keyword -> ISO-2 search locale. Covers the live foreign-election universe;
# anything unmatched falls back to worldwide ("") which still ranks a name field reasonably.
GEO = {
    "hungar": "HU",
    "tisza": "HU",
    "peru": "PE",
    "cyprus": "CY",
    "japan": "JP",
    "west bengal": "IN",
    "tamil nadu": "IN",
    "india": "IN",
    "dublin": "IE",
    "galway": "IE",
    "ireland": "IE",
    "british columbia": "CA",
    "b.c.": "CA",
    "canada": "CA",
    "netherlands": "NL",
    "dutch": "NL",
    "brazil": "BR",
    "colombia": "CO",
    "los angeles": "US",
    "makerfield": "GB",
    "farrer": "AU",
    "venice": "IT",
    "italy": "IT",
    "thailand": "TH",
    "germany": "DE",
    "france": "FR",
    "poland": "PL",
    "argentina": "AR",
}


def geo_for(title: str) -> str:
    """Best-effort ISO-2 locale for an event title; '' (worldwide) if unknown."""
    t = (title or "").lower()
    for k, v in GEO.items():
        if k in t:
            return v
    return ""


def event_search_shares(
    questions: Sequence[str],
    as_of: datetime,
    *,
    event: str = "",
    geo: str | None = None,
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    anthropic_api_key: str | None = None,
) -> dict[int, float]:
    """Each market's normalized search share across its event field, by input index.

    Returns ``{index: share}`` for the markets we could resolve+fetch (shares sum to ~1
    over the returned indices); markets that fail keyword resolution or have no interest
    are simply absent. Empty dict if the whole field is unresolvable (caller leaves the
    genome estimate untouched). Leakage-safe: interest is requested up to ``as_of`` only.
    """
    qs = list(questions)
    g = geo_for(event) if geo is None else geo
    kw = resolve_keywords(
        qs, event=event, geo=g, model_id=model_id, anthropic_api_key=anthropic_api_key
    )  # {idx: term}
    if len(kw) < 2:
        return {}
    terms = list({t for t in kw.values()})
    shares = interest_shares(terms, as_of, geo=g)
    recent = {i: shares.get(term, {}).get("recent", 0.0) for i, term in kw.items()}
    total = sum(recent.values())
    if total <= 0:
        return {}
    return {i: v / total for i, v in recent.items()}
