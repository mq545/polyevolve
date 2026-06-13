"""Foreign-politics market filtering.

Polymarket tags live on the *event* (not the market). The /events endpoint
supports server-side tag_slug filtering, so discovery pre-filters to political
tags. This module is defense-in-depth + the exclusion logic that tag_slug
alone can't express (drop US-headline and non-political markets that share a
political tag).

Scorecard decision (see MARKET_SCORECARD.md): foreign politics, non-headline.
US/UK headline politics is explicitly excluded - dominated by sharp anglophone
flow, no remaining edge.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from polyevolve.contracts import Market

# Tags we actively pull from the /events endpoint (server-side tag_slug filter).
INCLUDE_TAGS: tuple[str, ...] = (
    "politics",
    "elections",
    "geopolitics",
    "world",
)

# Country/region slugs that mark a market as foreign (non-US/UK) politics.
# Used to *promote* a market that has a political tag into our scope.
FOREIGN_COUNTRY_TAGS: frozenset[str] = frozenset(
    {
        "france",
        "germany",
        "brazil",
        "mexico",
        "argentina",
        "venezuela",
        "india",
        "indonesia",
        "poland",
        "turkey",
        "south-korea",
        "korea",
        "japan",
        "italy",
        "spain",
        "netherlands",
        "canada",
        "australia",
        "israel",
        "iran",
        "china",
        "russia",
        "ukraine",
        "thailand",
        "philippines",
        "nigeria",
        "south-africa",
        "colombia",
        "chile",
        "peru",
        "ireland",
        "greece",
        "portugal",
        "sweden",
        "norway",
        "finland",
        "denmark",
        "austria",
        "belgium",
        "romania",
        "hungary",
        "czech",
        "taiwan",
        "pakistan",
        "bangladesh",
        "vietnam",
        "egypt",
        "lebanon",
        "syria",
    }
)

# Hard exclusions - US/UK headline politics (no edge per scorecard).
EXCLUDE_HEADLINE_TAGS: frozenset[str] = frozenset(
    {
        "us-election",
        "us-elections",
        "2024-election",
        "2026-midterms",
        "midterms",
        "trump",
        "biden",
        "harris",
        "us-politics",
        "congress",
        "supreme-court",
        "uk",  # UK politics is anglophone-saturated
        "starmer",
    }
)

# Hard exclusions - not politics at all (defense against mis-tagged markets).
EXCLUDE_CATEGORY_TAGS: frozenset[str] = frozenset(
    {
        "sports",
        "soccer",
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "epl",
        "ucl",
        "crypto",
        "bitcoin",
        "ethereum",
        "pop-culture",
        "celebrities",
        "music",
        "movies",
        "awards",
        "sports-betting",
        "tennis",
        "golf",
    }
)


def market_tags(market: Market) -> set[str]:
    raw = market.metadata.get("tags") or []
    return {str(t).lower() for t in raw}


# Neg-risk multi-candidate events seed placeholder slots ("Person H", "Other",
# "Person A") that never get a real order book. They have no price history and
# aren't real candidates, so they pollute backtest samples. Match on question.
_PLACEHOLDER_RE = re.compile(
    r"\b(person\s+[a-z]|candidate\s+[a-z]|other|another\s+candidate|someone\s+else)\b",
    re.IGNORECASE,
)


def is_placeholder_market(market: Market) -> bool:
    """True for neg-risk placeholder candidate slots that never traded."""
    q = market.question or ""
    # "Will Person H win ..." / "Will Other win ..." -> placeholder.
    return bool(_PLACEHOLDER_RE.search(q))


def is_foreign_politics(market: Market) -> bool:
    """Keep a market iff it is political, foreign, and not a hard exclusion."""
    tags = market_tags(market)

    if tags & EXCLUDE_CATEGORY_TAGS:
        return False
    if tags & EXCLUDE_HEADLINE_TAGS:
        return False

    is_political = bool(tags & set(INCLUDE_TAGS))
    if not is_political:
        return False

    # Require a foreign-country marker OR geopolitics tag to exclude generic /
    # US-defaulted political markets that slipped through.
    is_foreign = bool(tags & FOREIGN_COUNTRY_TAGS) or "geopolitics" in tags
    return is_foreign


# --------------------------------------------------------------------------------------
# Domain selection - parameterizes ingestion so the platform can target any category, not
# just foreign politics. A domain = (discovery tag_slugs to pull, a keep-predicate). Pass a
# domain name to the snapshot builder / market listing to cast the net at a new market type.
# --------------------------------------------------------------------------------------
SPORTS_TAGS: tuple[str, ...] = (
    "sports",
    "soccer",
    "epl",
    "ucl",
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "tennis",
    "golf",
    "cfb",
    "cbb",
    "f1",
    "fifa-world-cup",
)
CRYPTO_TAGS: tuple[str, ...] = ("crypto", "bitcoin", "ethereum", "solana")
CULTURE_TAGS: tuple[str, ...] = ("pop-culture", "celebrities", "music", "movies", "awards")


def _tag_membership(include: frozenset[str]) -> Callable[[Market], bool]:
    """A keep-predicate: market is in-domain iff its tags intersect `include`."""

    def keep(market: Market) -> bool:
        return bool(market_tags(market) & include)

    return keep


@dataclass(frozen=True)
class Domain:
    """One ingestion target: which event tags to discover, and which markets to keep."""

    name: str
    tags: tuple[str, ...]  # event tag_slugs to pull from /events
    keep: Callable[[Market], bool]  # in-domain test applied after discovery


DOMAINS: dict[str, Domain] = {
    "foreign_politics": Domain("foreign_politics", INCLUDE_TAGS, is_foreign_politics),
    "sports": Domain("sports", SPORTS_TAGS, _tag_membership(frozenset(SPORTS_TAGS))),
    "crypto": Domain("crypto", CRYPTO_TAGS, _tag_membership(frozenset(CRYPTO_TAGS))),
    "culture": Domain("culture", CULTURE_TAGS, _tag_membership(frozenset(CULTURE_TAGS))),
    # everything we know how to discover; keep any non-placeholder market.
    "all": Domain(
        "all",
        INCLUDE_TAGS + SPORTS_TAGS + CRYPTO_TAGS + CULTURE_TAGS,
        lambda _m: True,
    ),
}


def domain(name: str) -> Domain:
    """Look up an ingestion domain by name (e.g. 'foreign_politics', 'sports', 'all')."""
    try:
        return DOMAINS[name]
    except KeyError:
        raise ValueError(f"unknown domain {name!r}; choose from {sorted(DOMAINS)}") from None
