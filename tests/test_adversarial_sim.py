"""Self-test for the adversarial simulator: PROVE it doesn't lie.

The whole point of the hardened sim is to be trustworthy. So we feed it two
synthetic strategies with KNOWN ground-truth edge and assert the sim reports the
right answer AFTER costs:

  * KNOWN-POSITIVE: we secretly peek at each outcome and emit a signal that is
    right with prob (1 - noise). A genuine edge must survive the spread and show
    up as a clearly positive net ROI with a significant EVENT t-stat.
  * KNOWN-ZERO: a random signal independent of the outcome. After crossing the
    spread this must report ~0 (in fact slightly negative - costs with no edge),
    and must NOT be flagged significant. If a zero-edge strategy looked positive,
    the sim would be lying.

We also test the honesty machinery directly:
  - fills cross the spread (never the mid);
  - the order-book walk caps size at real depth and worsens the VWAP;
  - correlated legs collapse to events (n_events < n_markets) and the per-event
    Kelly cap is shared across legs;
  - the worst-sub-window is surfaced.

The sim is a first-class platform module (`polyevolve.bench.sim`).
"""

from __future__ import annotations

import math
import random

import pytest

from polyevolve.bench import sim  # the sim now lives in the platform

TradeRecord = sim.TradeRecord
OrderBook = sim.OrderBook


# ----------------------------------------------------------------------------
# Synthetic data generators with KNOWN edge.
# ----------------------------------------------------------------------------
def _make_records(
    n: int,
    *,
    edge_kind: str,
    noise: float,
    seed: int,
    liquidity: float = 50_000.0,
    events: int | None = None,
) -> list:
    """Build n synthetic markets.

    CRITICAL for an honest zero-edge baseline: the crowd price must equal the
    TRUE probability of YES, and the outcome is drawn from that probability. (If
    instead the crowd were 0.4 while the truth were 0.5, betting YES would have a
    REAL +EV edge - a mispriced market, not zero edge. That subtlety is exactly
    what a forgiving sim hides, so we get it right here.)

    edge_kind="positive": the signal is nudged toward the REALIZED outcome and is
        RIGHT with prob (1 - noise) on gated trades - a genuine information edge.
    edge_kind="zero": the signal is a random ±0.15 shove independent of the
        outcome (pure noise) on an EFFICIENT crowd price -> no edge. After the
        spread this must net ~0 / slightly negative.

    `events`: if set, markets are assigned round-robin to that many event_ids so
        we can also exercise event de-dup; default = each market its own event.
    """
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        # Efficient crowd: the price IS the true probability; outcome ~ Bernoulli(price).
        crowd = rng.uniform(0.35, 0.65)
        y = 1 if rng.random() < crowd else 0
        outcome = "YES" if y == 1 else "NO"
        if edge_kind == "positive":
            # Signal points the right way with prob (1-noise); push it far enough
            # past the crowd to clear the 0.08 edge gate on the correct side.
            correct = rng.random() > noise
            target_high = y == 1  # if YES, we want signal > crowd
            point_high = target_high if correct else (not target_high)
            signal = crowd + 0.15 if point_high else crowd - 0.15
        elif edge_kind == "zero":
            # Independent of y: random direction, fixed magnitude to clear the gate.
            point_high = rng.random() < 0.5
            signal = crowd + 0.15 if point_high else crowd - 0.15
        else:  # pragma: no cover
            raise ValueError(edge_kind)
        signal = min(0.99, max(0.01, signal))
        event_id = None if events is None else f"ev{i % events}"
        recs.append(
            TradeRecord(
                market_id=f"m{i}",
                signal_prob=signal,
                crowd_price=crowd,
                outcome=outcome,
                liquidity=liquidity,
                event_id=event_id,
                lead=30,
                category="synthetic",
            )
        )
    return recs


# ----------------------------------------------------------------------------
# THE PROOF: known-positive edge -> positive net AFTER costs.
# ----------------------------------------------------------------------------
def test_known_positive_edge_is_reported_positive_after_costs() -> None:
    # 10% noise => signal right ~90% of the time on gated trades: a large, real
    # edge that must survive a 1% round-trip spread (thick-liquidity tier).
    recs = _make_records(600, edge_kind="positive", noise=0.10, seed=1)
    res = sim.run_adversarial_sim(recs, cost_model=sim.tiered_cost_model)
    assert res.net_roi > 0.05, f"known +edge should net clearly positive, got {res.net_roi}"
    assert res.win_rate > 0.8, f"~90%-accurate signal should win most trades, got {res.win_rate}"
    # A real edge over hundreds of independent events must be significant.
    assert res.tstat_events > 2.0, f"event t-stat should be significant, got {res.tstat_events}"


def test_positive_edge_magnitude_is_in_the_right_ballpark() -> None:
    # With ~90% accuracy at an avg entry near 0.5+half_spread, expected per-trade
    # ROI is large and positive. We don't pin an exact number (entries vary), but
    # it must be solidly positive and not absurd (< full +100%/trade on average).
    recs = _make_records(600, edge_kind="positive", noise=0.10, seed=7)
    res = sim.run_adversarial_sim(recs, cost_model=sim.tiered_cost_model)
    assert 0.05 < res.net_roi < 1.5


# ----------------------------------------------------------------------------
# THE PROOF: known-zero edge -> ~0 / slightly negative, NOT significant.
# ----------------------------------------------------------------------------
def test_known_zero_edge_is_reported_near_zero_after_costs() -> None:
    recs = _make_records(600, edge_kind="zero", noise=0.0, seed=2)
    res = sim.run_adversarial_sim(recs, cost_model=sim.tiered_cost_model)
    # No information: net ROI must sit near zero. Costs make it <= a hair positive.
    assert res.net_roi < 0.03, f"zero-edge must not look profitable, got {res.net_roi}"
    # And it must NOT be flagged as a significant positive edge.
    significant_positive = (
        not math.isnan(res.tstat_events) and res.tstat_events > 2.0 and res.net_roi > 0
    )
    assert not significant_positive, "zero-edge strategy was falsely flagged significant+positive"


def test_zero_edge_is_worse_than_positive_edge() -> None:
    pos = sim.run_adversarial_sim(_make_records(600, edge_kind="positive", noise=0.10, seed=3))
    zero = sim.run_adversarial_sim(_make_records(600, edge_kind="zero", noise=0.0, seed=3))
    assert pos.net_roi > zero.net_roi + 0.1


# ----------------------------------------------------------------------------
# NEVER mid-fill: the entry must be worse than the mid by the half-spread.
# ----------------------------------------------------------------------------
def test_fill_crosses_the_spread_never_the_mid() -> None:
    # Single YES bet: signal 0.7, crowd 0.5, thick liquidity (1% round-trip).
    rec = TradeRecord("m", 0.7, 0.5, "YES", liquidity=50_000.0, event_id="e", lead=30, category="c")
    res = sim.run_adversarial_sim([rec], edge_threshold=0.08)
    assert len(res.trades) == 1
    # half_spread = 0.01/2 = 0.005 -> entry = 0.505, strictly worse than mid 0.5.
    assert res.trades[0].entry == pytest.approx(0.505)
    assert res.trades[0].entry > 0.5


def test_thin_liquidity_costs_more_than_thick() -> None:
    thick = TradeRecord("m", 0.7, 0.5, "YES", liquidity=50_000.0, event_id="e")
    thin = TradeRecord("m", 0.7, 0.5, "YES", liquidity=100.0, event_id="e")
    r_thick = sim.run_adversarial_sim([thick], edge_threshold=0.08)
    r_thin = sim.run_adversarial_sim([thin], edge_threshold=0.08)
    assert r_thin.trades[0].entry > r_thick.trades[0].entry


def test_unknown_liquidity_uses_conservative_mid_tier() -> None:
    # liquidity=None must NOT get the optimistic thick tier.
    assert sim.tier_round_trip(None) == sim.MID_RT
    assert sim.tier_round_trip(None) > sim.THICK_RT


# ----------------------------------------------------------------------------
# Order-book walk: real depth caps size and worsens the VWAP.
# ----------------------------------------------------------------------------
def test_orderbook_walk_caps_size_at_depth() -> None:
    # Only $50 resting on the YES ask; we must not fill more than that.
    book = OrderBook(yes_asks=((0.52, 50.0),))
    rec = TradeRecord("m", 0.9, 0.5, "YES", liquidity=1_000_000.0, event_id="e", book=book)
    res = sim.run_adversarial_sim([rec], cost_model=sim.orderbook_walk_cost_model)
    assert len(res.trades) == 1
    assert res.trades[0].stake <= 50.0 + 1e-9


def test_orderbook_walk_vwap_is_worse_for_bigger_orders() -> None:
    book = sim.OrderBook(yes_asks=((0.50, 10.0), (0.60, 1000.0)))
    small = sim.orderbook_walk_cost_model("YES", 0.50, None, book, 5.0)
    big = sim.orderbook_walk_cost_model("YES", 0.50, None, book, 500.0)
    assert big.entry > small.entry  # eating into the 0.60 level raises the VWAP


# ----------------------------------------------------------------------------
# Event-level correlation: de-dup + shared per-event Kelly cap.
# ----------------------------------------------------------------------------
def test_events_collapse_correlated_markets() -> None:
    # 30 markets, 3 events -> n_events must be 3, not 30 (defeats the n~=10 trap).
    recs = _make_records(30, edge_kind="positive", noise=0.1, seed=5, events=3)
    res = sim.run_adversarial_sim(recs)
    assert res.n_markets == 30
    assert res.n_events == 3


def test_per_event_kelly_cap_is_shared_across_legs() -> None:
    # Many correlated YES legs of ONE event; total stake across legs must not
    # exceed the per-event budget (max_position * bankroll), not be charged 10x.
    recs = [
        TradeRecord(f"m{i}", 0.9, 0.5, "YES", liquidity=1e9, event_id="same_event")
        for i in range(10)
    ]
    res = sim.run_adversarial_sim(recs, max_position=0.05, bankroll=1.0)
    total_stake = sum(t.stake for t in res.trades)
    assert total_stake <= 0.05 + 1e-9, f"per-event cap breached: {total_stake}"


def test_significance_uses_events_not_markets() -> None:
    # 100 correlated copies of ONE coin-flip event collapse to a single event ->
    # the event t-stat must be nan (n_events == 1), NOT a fake large market t-stat.
    recs = [
        TradeRecord(f"m{i}", 0.9, 0.5, "YES", liquidity=1e9, event_id="one") for i in range(100)
    ]
    res = sim.run_adversarial_sim(recs)
    assert res.n_events == 1
    assert math.isnan(res.tstat_events)


# ----------------------------------------------------------------------------
# Per-regime reporting + worst sub-window.
# ----------------------------------------------------------------------------
def test_breakdowns_cover_all_dimensions() -> None:
    recs = _make_records(120, edge_kind="positive", noise=0.1, seed=9)
    res = sim.run_adversarial_sim(recs)
    assert set(res.breakdowns) == {"category", "liquidity_decile", "lead_bucket"}
    for cells in res.breakdowns.values():
        assert cells  # every dimension produced at least one cell


def test_worst_subwindow_surfaces_the_losing_cell() -> None:
    # Mix a strongly winning category with a strongly losing one; the worst-window
    # must point at the loser even though the overall average may be positive.
    rng = random.Random(11)
    recs = []
    for i in range(60):  # winners: signal always correct
        y = 1 if rng.random() < 0.5 else 0
        crowd = 0.5
        sig = 0.7 if y == 1 else 0.3
        recs.append(
            TradeRecord(
                f"w{i}",
                sig,
                crowd,
                "YES" if y else "NO",
                liquidity=50_000.0,
                event_id=f"w{i}",
                category="winner",
            )
        )
    for i in range(60):  # losers: signal always WRONG
        y = 1 if rng.random() < 0.5 else 0
        crowd = 0.5
        sig = 0.3 if y == 1 else 0.7  # bets the wrong way
        recs.append(
            TradeRecord(
                f"l{i}",
                sig,
                crowd,
                "YES" if y else "NO",
                liquidity=50_000.0,
                event_id=f"l{i}",
                category="loser",
            )
        )
    res = sim.run_adversarial_sim(recs)
    assert res.worst_window
    assert res.worst_window["cell"] == "loser"
    assert res.worst_window["roi"] < 0
