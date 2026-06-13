"""Scout: WHERE on Polymarket is it worth hunting for an edge - a live, read-only
category x thinness map.

This is the experiment-funnel's first stage: before forecasting anything, point
the user at the corners of the venue where the crowd is THIN, because thin markets
(low liquidity, low volume, wide spreads) are where mispricing can survive long
enough to trade. We do NOT forecast, score, or place anything here - we just sample
the currently-OPEN markets across a set of category tags and summarize how thin each
category is, so the user can decide where to dig.

Unlike `scripts/efficiency_map.py` (an offline BACKTEST that scores resolved-market
prices against outcomes from the DB), this is a $0, network-only LIVE scan with no
DB and no leakage concerns - every number is a current snapshot of open markets.

The scan reuses the proven Gamma `/events` tag-filtered discovery (tags live on
events, not markets) already used by `market_sources.polymarket`: per category tag
we pull open events, flatten their nested markets, and read each market's live
`liquidityNum` / `volumeNum` / `spread` (bestAsk - bestBid). We then aggregate per
category into a THINNESS score (our inefficiency proxy) and return a ranked table.

THINNESS (0 = deep/efficient, 1 = thin/likely-inefficient) blends three signals,
each mapped to [0, 1] so no single unit dominates:
  - liquidity: thin = LOW on-book liquidity        -> 1 - squash(liquidity)
  - volume:    thin = LOW traded volume            -> 1 - squash(volume)
  - spread:    thin = WIDE bid/ask spread          -> squash_spread(spread)
We use a per-market thinness = mean of the available signals, then the category
thinness = mean over its markets. A higher category thinness ranks higher: that is
where to hunt. We report the components too, so the user sees WHY a category ranks.

Pure helpers (squashing, scoring, aggregation) are stdlib-only and unit-testable
without a network; only `efficiency_map()` and `_scan_tag()` touch Gamma.

    from polyevolve.scout.efficiency import efficiency_map
    rows = efficiency_map(["politics", "geopolitics", "crypto"])
    for r in rows:   # ranked thinnest-first
        print(r.category, r.thinness, r.n_markets)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
_HTTP_TIMEOUT_S = 30.0
# Default category tags to sample when the caller names none. These are the
# event tag_slugs Polymarket uses; the same political/world set the rest of the
# platform's experiments live in, plus a few liquid contrast categories so the
# ranking has something to rank against.
DEFAULT_CATEGORIES: tuple[str, ...] = (
    "politics",
    "geopolitics",
    "world",
    "elections",
    "crypto",
    "economy",
)

# Squash scales (USDC). A market at the scale value squashes to ~0.5; the map is
# log-based so it spans the orders-of-magnitude range real volume/liquidity show.
# Stated, not fit - they only set where the 0/1 thinness knee sits, and every
# category is judged on the same scale, so the RANKING is robust to their exact
# values.
_LIQUIDITY_SCALE = 20_000.0
_VOLUME_SCALE = 100_000.0
# A spread at/above this (YES-share price units) reads as maximally thin.
_SPREAD_CAP = 0.10


# ---------------------------------------------------------------------------
# Pure scoring helpers (stdlib only; unit-tested without a network).
# ---------------------------------------------------------------------------


def _squash_log(value: float, scale: float) -> float:
    """Map a non-negative magnitude to [0, 1] on a log scale (0 -> 0, scale -> 0.5).

    Uses x / (x + scale) on log1p so that an order-of-magnitude range spreads out
    instead of saturating. Negative/NaN inputs clamp to 0.0 (treated as "none")."""
    if not math.isfinite(value) or value <= 0.0:
        return 0.0
    lv = math.log1p(value)
    ls = math.log1p(scale)
    if ls <= 0.0:
        return 0.0
    return lv / (lv + ls)


def liquidity_thinness(liquidity: float | None) -> float | None:
    """Thinness from on-book liquidity: LOW liquidity == thin (-> near 1.0).

    None (missing) -> None so it drops out of the per-market mean rather than
    biasing it."""
    if liquidity is None or not math.isfinite(liquidity):
        return None
    return 1.0 - _squash_log(liquidity, _LIQUIDITY_SCALE)


def volume_thinness(volume: float | None) -> float | None:
    """Thinness from traded volume: LOW volume == thin (-> near 1.0). None drops."""
    if volume is None or not math.isfinite(volume):
        return None
    return 1.0 - _squash_log(volume, _VOLUME_SCALE)


def spread_thinness(spread: float | None) -> float | None:
    """Thinness from the bid/ask spread: WIDE spread == thin (-> near 1.0).

    Linearly scaled by _SPREAD_CAP and clamped to [0, 1]. None drops."""
    if spread is None or not math.isfinite(spread) or spread < 0.0:
        return None
    return min(1.0, spread / _SPREAD_CAP)


def _mean(xs: Sequence[float]) -> float | None:
    """Mean, or None for an empty sequence."""
    return sum(xs) / len(xs) if xs else None


@dataclass(frozen=True)
class MarketThinness:
    """Per-market thinness sample: the raw signals + their blended score.

    `thinness` is the mean of whichever component signals were available (None
    components are skipped); None only if NO signal was present at all."""

    external_id: str
    question: str
    liquidity: float | None
    volume: float | None
    spread: float | None
    liquidity_thinness: float | None
    volume_thinness: float | None
    spread_thinness: float | None
    thinness: float | None

    @classmethod
    def of(
        cls,
        external_id: str,
        question: str,
        liquidity: float | None,
        volume: float | None,
        spread: float | None,
    ) -> MarketThinness:
        lt = liquidity_thinness(liquidity)
        vt = volume_thinness(volume)
        st = spread_thinness(spread)
        present = [c for c in (lt, vt, st) if c is not None]
        return cls(
            external_id=external_id,
            question=question,
            liquidity=liquidity,
            volume=volume,
            spread=spread,
            liquidity_thinness=lt,
            volume_thinness=vt,
            spread_thinness=st,
            thinness=_mean(present),
        )


@dataclass(frozen=True)
class CategoryRow:
    """A category's aggregated thinness over the open markets sampled for it.

    `thinness` (the inefficiency proxy) is the mean per-market thinness; the
    component means explain WHY it ranks where it does. `thinnest` lists the few
    thinnest individual markets so the user has concrete starting points."""

    category: str
    n_markets: int
    thinness: float | None
    median_liquidity: float | None
    median_volume: float | None
    mean_spread: float | None
    liquidity_thinness: float | None
    volume_thinness: float | None
    spread_thinness: float | None
    thinnest: tuple[MarketThinness, ...]


def _median(xs: Sequence[float]) -> float | None:
    """Median of present values, or None if empty."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def aggregate_category(
    category: str, markets: Iterable[MarketThinness], top_n: int = 5
) -> CategoryRow:
    """Roll per-market thinness samples up into one ranked CategoryRow.

    Only markets that produced at least one signal (thinness is not None) count
    toward the category score and `n_markets`. `top_n` thinnest markets are kept
    as concrete leads, thinnest first."""
    scored = [m for m in markets if m.thinness is not None]
    scored.sort(key=lambda m: m.thinness or 0.0, reverse=True)

    def col(attr: str) -> list[float]:
        return [v for m in scored if (v := getattr(m, attr)) is not None]

    return CategoryRow(
        category=category,
        n_markets=len(scored),
        thinness=_mean([m.thinness for m in scored if m.thinness is not None]),
        median_liquidity=_median(col("liquidity")),
        median_volume=_median(col("volume")),
        mean_spread=_mean(col("spread")),
        liquidity_thinness=_mean(col("liquidity_thinness")),
        volume_thinness=_mean(col("volume_thinness")),
        spread_thinness=_mean(col("spread_thinness")),
        thinnest=tuple(scored[:top_n]),
    )


def rank_categories(rows: Iterable[CategoryRow]) -> list[CategoryRow]:
    """Sort category rows thinnest-first (highest thinness == hunt here first).

    Categories with no scored markets (thinness None) sort last."""
    return sorted(rows, key=lambda r: (r.thinness is not None, r.thinness or 0.0), reverse=True)


# ---------------------------------------------------------------------------
# Live Gamma scan (the only network-touching code).
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Coerce a Gamma numeric field to float, or None if absent/unparseable."""
    if isinstance(value, bool):  # guard: bool is an int subclass
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _market_spread(raw: dict[str, Any]) -> float | None:
    """Live round-trip spread: bestAsk - bestBid when both present, else the
    reported `spread` field. None if neither is usable."""
    bid, ask = _num(raw.get("bestBid")), _num(raw.get("bestAsk"))
    if bid is not None and ask is not None and ask >= bid:
        return ask - bid
    return _num(raw.get("spread"))


def market_thinness_from_raw(raw: dict[str, Any]) -> MarketThinness | None:
    """Build a MarketThinness from a raw Gamma market dict.

    Returns None for closed/idless markets (a thinness map of open hunting ground
    must not be polluted by settled markets, whose book is frozen)."""
    ext_id = str(raw.get("id", ""))
    if not ext_id or raw.get("closed"):
        return None
    liquidity = _num(raw.get("liquidityNum"))
    if liquidity is None:
        liquidity = _num(raw.get("liquidity"))
    volume = _num(raw.get("volumeNum"))
    if volume is None:
        volume = _num(raw.get("volume"))
    return MarketThinness.of(
        external_id=ext_id,
        question=str(raw.get("question") or ""),
        liquidity=liquidity,
        volume=volume,
        spread=_market_spread(raw),
    )


def _scan_tag(tag: str, http: httpx.Client, limit: int, seen: set[str]) -> list[MarketThinness]:
    """Fetch open events for one category tag and extract per-market thinness.

    Reuses the proven /events tag_slug discovery. `seen` dedupes markets that
    appear under more than one tag, so each market scores its FIRST category only
    (categories are scanned in caller order). Fail-soft: a tag that errors
    contributes nothing rather than aborting the whole map."""
    try:
        resp = http.get(
            f"{GAMMA_BASE}/events",
            params={"limit": limit, "closed": "false", "active": "true", "tag_slug": tag},
            timeout=_HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        events = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    out: list[MarketThinness] = []
    for event in events:
        for raw_market in event.get("markets") or []:
            mt = market_thinness_from_raw(raw_market)
            if mt is None or mt.external_id in seen:
                continue
            seen.add(mt.external_id)
            out.append(mt)
    return out


def efficiency_map(
    categories: Sequence[str] | None = None,
    *,
    limit_per_tag: int = 100,
    top_n: int = 5,
    http: httpx.Client | None = None,
) -> list[CategoryRow]:
    """Live category x thinness map of CURRENTLY-OPEN Polymarket markets.

    For each category tag, sample open markets and aggregate their liquidity /
    volume / spread into a thinness (inefficiency-proxy) score, then return the
    categories ranked THINNEST-FIRST - i.e. where to hunt for an edge. Pure
    read-only, $0 (Gamma needs no auth); writes nothing.

    Args:
      categories:    event tag_slugs to scan (default: DEFAULT_CATEGORIES).
      limit_per_tag: events fetched per tag.
      top_n:         thinnest individual markets kept per category as leads.
      http:          optional shared client (e.g. for tests); one is made if None.

    Returns the ranked CategoryRow list. A category with no open markets still
    appears (n_markets=0, thinness=None) and sorts last, so the caller sees it
    was scanned and came back empty.
    """
    cats = tuple(categories) if categories else DEFAULT_CATEGORIES
    owns_http = http is None
    client = http or httpx.Client(timeout=_HTTP_TIMEOUT_S)
    seen: set[str] = set()
    try:
        rows = [
            aggregate_category(cat, _scan_tag(cat, client, limit_per_tag, seen), top_n=top_n)
            for cat in cats
        ]
    finally:
        if owns_http:
            client.close()
    return rank_categories(rows)


# ---------------------------------------------------------------------------
# CLI-friendly rendering (a plain table the cli.py `scout` command can print).
# ---------------------------------------------------------------------------


def _f(x: float | None, places: int = 3) -> str:
    return f"{x:.{places}f}" if x is not None else "  n/a"


def _money(x: float | None) -> str:
    return f"{x:>12,.0f}" if x is not None else "         n/a"


def format_table(rows: Sequence[CategoryRow]) -> str:
    """Render ranked category rows as a fixed-width text table for the CLI.

    One line per category (thinnest first) with the thinness score, market count,
    median liquidity/volume, mean spread, and the component sub-scores; then a
    short 'leads' block of the thinnest individual markets per category."""
    lines: list[str] = []
    lines.append("=" * 96)
    lines.append("POLYMARKET SCOUT | category x THINNESS (inefficiency proxy) | live, read-only")
    lines.append("=" * 96)
    lines.append(
        f"{'category':<14} {'thin':>6} {'n':>4} {'med_liq':>12} {'med_vol':>12} "
        f"{'spread':>7} {'liqT':>5} {'volT':>5} {'sprT':>5}"
    )
    lines.append("-" * 96)
    for r in rows:
        lines.append(
            f"{r.category:<14} {_f(r.thinness):>6} {r.n_markets:>4} "
            f"{_money(r.median_liquidity)} {_money(r.median_volume)} "
            f"{_f(r.mean_spread):>7} {_f(r.liquidity_thinness, 2):>5} "
            f"{_f(r.volume_thinness, 2):>5} {_f(r.spread_thinness, 2):>5}"
        )
    lines.append("-" * 96)
    lines.append("thin = inefficiency proxy in [0,1] (1 = thinnest); hunt the top rows first.")
    lines.append("\nTHINNEST MARKETS PER CATEGORY (concrete leads):")
    for r in rows:
        if not r.thinnest:
            continue
        lines.append(f"\n  [{r.category}]")
        for m in r.thinnest:
            lines.append(
                f"    thin={_f(m.thinness)} liq={_money(m.liquidity)} "
                f"vol={_money(m.volume)} spr={_f(m.spread)}  {m.question[:54]}"
            )
    return "\n".join(lines)
