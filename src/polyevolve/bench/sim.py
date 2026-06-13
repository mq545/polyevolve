"""The adversarial trading simulator - PolyEvolve's net-of-spread execution model.

The return fitness in ``bench.returns`` runs forecasts through this. It is the pure
simulation core: sizing math + cost models + the event-clustered P&L sim. It NEVER fills at
the mid - every fill crosses the spread via a swappable :data:`CostModel` (a liquidity-tier
half-spread, or a real order-book walk), stake is capped by BOTH a per-EVENT fractional-Kelly
budget AND the book depth, and all significance is computed on EVENTS (correlated legs of one
event are one observation) so a handful of markets can't masquerade as a powered sample.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CostModel",
    "Fill",
    "OrderBook",
    "SimResult",
    "Trade",
    "TradeRecord",
    "kelly_fraction",
    "orderbook_walk_cost_model",
    "passes_edge_gate",
    "position_pnl",
    "run_adversarial_sim",
    "tier_round_trip",
    "tiered_cost_model",
]

# Honest-trading defaults.
DEFAULT_EDGE_THRESHOLD = 0.08
DEFAULT_KELLY_FRACTION = 0.25  # quarter-Kelly
DEFAULT_MAX_POSITION = 0.05  # cap any single EVENT at 5% of bankroll
DEFAULT_DEPTH_PARTICIPATION = 0.10  # consume at most 10% of a market's liquidity


# ----------------------------------------------------------------------------
# Sizing math (pure; previously scripts/trading_sim.py - inlined to keep the
# platform sim self-contained, with no DB/kill_test dependency).
# ----------------------------------------------------------------------------
def kelly_fraction(p_win: float, entry: float) -> float:
    """Full-Kelly fraction of bankroll for a binary share bought at ``entry``.

    Net decimal odds b = (1-entry)/entry; f* = p_win - (1-p_win)/b. Floored at 0 (a
    non-positive-edge bet is "don't bet"); the upside is left unclamped because callers
    apply a fractional multiplier and a hard cap.
    """
    if entry <= 0.0 or entry >= 1.0:
        return 0.0
    b = (1.0 - entry) / entry
    f = (p_win * b - (1.0 - p_win)) / b
    return f if f > 0.0 else 0.0


def position_pnl(direction: str, entry: float, y: int, stake: float) -> float:
    """Realised P&L (bankroll units) of a resolved position: ``stake * (payoff-entry)/entry``."""
    if entry <= 0.0:
        raise ValueError("entry price must be > 0")
    won = (y == 1) if direction == "YES" else (y == 0)
    payoff = 1.0 if won else 0.0
    return stake * (payoff - entry) / entry


def passes_edge_gate(p_calibrated: float, p_mkt: float, edge_threshold: float) -> bool:
    """True iff ``|p_calibrated - p_mkt| >= edge_threshold`` (take a position)."""
    return abs(p_calibrated - p_mkt) >= edge_threshold


# ----------------------------------------------------------------------------
# Records + order book.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderBook:
    """A minimal one-sided-per-direction CLOB snapshot for depth-aware fills.

    ``yes_asks`` / ``no_asks`` are (price, size_usd) levels best-first. Walking them gives a
    true volume-weighted fill price AND a hard depth limit - the only honest resting fill.
    """

    yes_asks: tuple[tuple[float, float], ...] = ()
    no_asks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class TradeRecord:
    """One tradeable market: our signal vs the crowd mid, the outcome, and regime keys.

    Correlated markets (legs of one election) MUST share ``event_id`` so they collapse to one
    independent observation. ``book``, if present, is walked for depth-aware fills.
    """

    market_id: str
    signal_prob: float
    crowd_price: float
    outcome: str  # "YES" | "NO"
    liquidity: float | None = None
    event_id: str | None = None
    lead: int | None = None
    category: str | None = None
    book: OrderBook | None = None

    @property
    def y(self) -> int:
        return 1 if self.outcome == "YES" else 0


# ----------------------------------------------------------------------------
# Cost models - explicit, swappable. NEVER mid-fill.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Fill:
    """Crossing the spread for one side: effective ``entry`` price + a depth ``max_notional``."""

    entry: float
    max_notional: float | None


# (direction, mid_price, liquidity, book, desired_notional) -> Fill
CostModel = Callable[[str, float, float | None, "OrderBook | None", float], Fill]

LIQ_THICK = 10_000.0  # >= $10k liquidity: ~1% round-trip
LIQ_MID = 2_000.0  # >= $2k: ~3% round-trip
THICK_RT = 0.01
MID_RT = 0.03
THIN_RT = 0.05


def tier_round_trip(liquidity: float | None) -> float:
    """Round-trip cost (price units) for a market's liquidity tier (mid when unknown)."""
    if liquidity is None:
        return MID_RT
    if liquidity >= LIQ_THICK:
        return THICK_RT
    if liquidity >= LIQ_MID:
        return MID_RT
    return THIN_RT


def tiered_cost_model(
    direction: str,
    mid: float,
    liquidity: float | None,
    book: OrderBook | None,
    desired_notional: float,
) -> Fill:
    """Default cost model: cross half the tier's round-trip spread; cap depth by liquidity."""
    half_spread = tier_round_trip(liquidity) / 2.0
    base = mid if direction == "YES" else (1.0 - mid)
    entry = base + half_spread
    max_notional = liquidity * DEFAULT_DEPTH_PARTICIPATION if liquidity is not None else None
    return Fill(entry=entry, max_notional=max_notional)


def orderbook_walk_cost_model(
    direction: str,
    mid: float,
    liquidity: float | None,
    book: OrderBook | None,
    desired_notional: float,
) -> Fill:
    """Walk a real CLOB book for a volume-weighted fill + true depth cap; tiered fallback."""
    if book is None:
        return tiered_cost_model(direction, mid, liquidity, book, desired_notional)
    levels = book.yes_asks if direction == "YES" else book.no_asks
    if not levels:
        return tiered_cost_model(direction, mid, liquidity, book, desired_notional)
    spent = 0.0
    shares = 0.0
    book_depth = 0.0
    remaining = desired_notional
    for price, size_usd in levels:
        book_depth += size_usd
        if remaining <= 0.0:
            continue
        take = min(size_usd, remaining)
        shares += take / price
        spent += take
        remaining -= take
    if shares <= 0.0:
        best_price = levels[0][0]
        return Fill(entry=best_price, max_notional=book_depth)
    return Fill(entry=spent / shares, max_notional=book_depth)


# ----------------------------------------------------------------------------
# Core simulation over records.
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    market_id: str
    event_id: str
    direction: str
    entry: float
    won: bool
    roi: float  # per unit staked, AFTER costs
    stake: float  # $ actually deployed (depth- and Kelly-capped)
    pnl: float  # stake * roi
    liquidity: float | None
    lead: int | None
    category: str | None


@dataclass
class SimResult:
    trades: list[Trade]
    n_markets: int
    n_events: int
    event_pnls: dict[str, float]
    event_rois: dict[str, float]  # P&L / staked, per event
    net_roi: float
    mean_event_roi: float
    tstat_events: float  # significance on EVENTS, not markets
    win_rate: float
    total_pnl: float
    total_staked: float
    breakdowns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    worst_window: dict[str, Any] = field(default_factory=dict)


def _liquidity_decile_label(liq: float | None) -> str:
    if liq is None:
        return "unknown"
    if liq >= LIQ_THICK:
        return f">={int(LIQ_THICK)}"
    if liq >= LIQ_MID:
        return f"{int(LIQ_MID)}-{int(LIQ_THICK)}"
    return f"<{int(LIQ_MID)}"


def _lead_bucket_label(lead: int | None) -> str:
    if lead is None:
        return "unknown"
    if lead <= 7:
        return "<=7d"
    if lead <= 30:
        return "8-30d"
    if lead <= 60:
        return "31-60d"
    return ">60d"


def _tstat(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 0.0:
        return float("nan")
    return mean / (sd / math.sqrt(n))


def run_adversarial_sim(
    records: Sequence[TradeRecord],
    *,
    cost_model: CostModel = tiered_cost_model,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    kelly_fraction_mult: float = DEFAULT_KELLY_FRACTION,
    max_position: float = DEFAULT_MAX_POSITION,
    bankroll: float = 1.0,
) -> SimResult:
    """Run the honest trading rule over ``records`` and return all breakdowns.

    Fills cross the spread via ``cost_model`` (never the mid); stake is capped by both the
    per-EVENT fractional-Kelly budget and the book depth; aggregates/significance are on
    EVENTS, not markets (correlated legs of one event share one ``max_position`` allowance).
    """
    event_budget_used: dict[str, float] = defaultdict(float)
    event_budget = max_position * bankroll

    trades: list[Trade] = []
    for rec in records:
        event_id = rec.event_id or rec.market_id
        pc = rec.signal_prob
        p_mkt = rec.crowd_price
        if not passes_edge_gate(pc, p_mkt, edge_threshold):
            continue
        direction = "YES" if pc > p_mkt else "NO"
        p_win = pc if direction == "YES" else 1.0 - pc

        remaining_budget = event_budget - event_budget_used[event_id]
        if remaining_budget <= 0.0:
            continue

        kelly_f = kelly_fraction(p_win, p_mkt if direction == "YES" else 1.0 - p_mkt)
        desired = min(kelly_fraction_mult * kelly_f, 1.0) * remaining_budget
        if desired <= 0.0:
            continue

        fill = cost_model(direction, p_mkt, rec.liquidity, rec.book, desired)
        entry = fill.entry
        if entry <= 0.0 or entry >= 1.0:
            continue

        stake = desired
        if fill.max_notional is not None:
            stake = min(stake, fill.max_notional)
        stake = min(stake, remaining_budget)
        if stake <= 0.0:
            continue

        roi = position_pnl(direction, entry, rec.y, 1.0)
        won = (rec.y == 1) if direction == "YES" else (rec.y == 0)
        pnl = stake * roi
        event_budget_used[event_id] += stake
        trades.append(
            Trade(
                market_id=rec.market_id,
                event_id=event_id,
                direction=direction,
                entry=entry,
                won=won,
                roi=roi,
                stake=stake,
                pnl=pnl,
                liquidity=rec.liquidity,
                lead=rec.lead,
                category=rec.category,
            )
        )

    n_markets = len({r.market_id for r in records})
    n_events = len({(r.event_id or r.market_id) for r in records})

    event_pnls: dict[str, float] = defaultdict(float)
    event_staked: dict[str, float] = defaultdict(float)
    for t in trades:
        event_pnls[t.event_id] += t.pnl
        event_staked[t.event_id] += t.stake
    event_rois = {e: (event_pnls[e] / event_staked[e]) for e in event_pnls if event_staked[e] > 0}

    total_pnl = sum(t.pnl for t in trades)
    total_staked = sum(t.stake for t in trades)
    net_roi = total_pnl / total_staked if total_staked > 0 else 0.0
    mean_event_roi = sum(event_rois.values()) / len(event_rois) if event_rois else 0.0
    win_rate = (sum(1 for t in trades if t.won) / len(trades)) if trades else 0.0

    breakdowns = _build_breakdowns(trades)
    return SimResult(
        trades=trades,
        n_markets=n_markets,
        n_events=n_events,
        event_pnls=dict(event_pnls),
        event_rois=event_rois,
        net_roi=net_roi,
        mean_event_roi=mean_event_roi,
        tstat_events=_tstat(list(event_rois.values())),
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_staked=total_staked,
        breakdowns=breakdowns,
        worst_window=_worst_subwindow(breakdowns),
    )


# ----------------------------------------------------------------------------
# Per-regime breakdowns + worst sub-window.
# ----------------------------------------------------------------------------
def _regime_cells(trades: list[Trade], key: Callable[[Trade], str]) -> list[dict[str, Any]]:
    """Aggregate trades into regime cells, with EVENT-level n and significance."""
    by_cell: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_cell[key(t)].append(t)
    cells: list[dict[str, Any]] = []
    for label, ts in sorted(by_cell.items()):
        pnl = sum(t.pnl for t in ts)
        staked = sum(t.stake for t in ts)
        ev_pnl: dict[str, float] = defaultdict(float)
        ev_staked: dict[str, float] = defaultdict(float)
        for t in ts:
            ev_pnl[t.event_id] += t.pnl
            ev_staked[t.event_id] += t.stake
        ev_rois = [ev_pnl[e] / ev_staked[e] for e in ev_pnl if ev_staked[e] > 0]
        cells.append(
            {
                "cell": label,
                "n_markets": len(ts),
                "n_events": len(ev_pnl),
                "roi": pnl / staked if staked > 0 else 0.0,
                "pnl": pnl,
                "win_rate": sum(1 for t in ts if t.won) / len(ts) if ts else 0.0,
                "tstat_events": _tstat(ev_rois),
            }
        )
    return cells


def _build_breakdowns(trades: list[Trade]) -> dict[str, list[dict[str, Any]]]:
    return {
        "category": _regime_cells(trades, lambda t: t.category or "unknown"),
        "liquidity_decile": _regime_cells(trades, lambda t: _liquidity_decile_label(t.liquidity)),
        "lead_bucket": _regime_cells(trades, lambda t: _lead_bucket_label(t.lead)),
    }


def _worst_subwindow(breakdowns: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The single worst regime cell (>=1 event) - so a bad cell can't hide in the mean."""
    worst: dict[str, Any] | None = None
    for dim, cells in breakdowns.items():
        for cell in cells:
            if cell["n_events"] < 1:
                continue
            if worst is None or cell["roi"] < worst["roi"]:
                worst = {**cell, "dimension": dim}
    return worst or {}
