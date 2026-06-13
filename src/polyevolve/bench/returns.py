"""Net-of-spread RETURN scorer - PolyEvolve's betting-market fitness.

This is the measurement that makes PolyEvolve a platform for *betting markets*, not just
calibration: a genome's forecasts are run through the adversarial trading sim (never
mid-fill, cross the spread, walk the book, per-EVENT Kelly budget, event-clustered
significance) to get honest net-of-spread P&L. Edge - if any - shows up here. The sim is a
first-class platform module (:mod:`polyevolve.bench.sim`), so the return fitness is fully
self-contained.
"""

from __future__ import annotations

from collections.abc import Sequence

from polyevolve.bench import sim
from polyevolve.reason.dsl import EvidencePool, Genome, Question


def evaluate_return(
    genome: Genome,
    questions: Sequence[Question],
    pools: Sequence[EvidencePool] | None = None,
    *,
    edge_threshold: float = 0.05,
    respect_abstention: bool = True,
) -> dict[str, float]:
    """Run `genome` over resolved betting markets and score honest net-of-spread return.

    The genome supplies the SIGNAL (p_yes); the adversarial sim owns execution
    (edge gate, fractional-Kelly sizing, spread-crossing fills, depth caps) so the
    result cannot be inflated by a forgiving fill.

    SELECTIVE betting: when ``respect_abstention`` and the genome is size-aware (it set a
    non-zero stake on at least one market), markets it abstained on (``Forecast.size == 0``)
    are NOT traded - the genome trades coverage for precision. ``coverage`` (traded / eligible)
    is reported so a thin, selective policy is never confused with a broad one. A genome that
    never sets a size (no abstain/size node) is treated as non-selective and trades all markets.

    Returns the EVENT-aggregated aggregates (the honest n): {'net_roi','mean_event_roi',
    'tstat_events','n_events','n_markets','total_pnl','win_rate','coverage','n_eligible'}.
    """
    if pools is not None and len(pools) != len(questions):
        raise ValueError("pools length must match questions")

    forecasts = []
    for i, q in enumerate(questions):
        if q.outcome is None or q.market_price is None:
            continue
        pool = pools[i] if pools is not None else EvidencePool(items=[])
        try:
            fc = genome(q, pool)
        except Exception:
            continue
        forecasts.append((q, fc))

    n_eligible = len(forecasts)
    # Only honor abstention if the genome actually sizes (else size==0 is just the default
    # and would zero out every market). This keeps non-sizing genomes unchanged.
    size_aware = respect_abstention and any(float(fc.size) != 0.0 for _, fc in forecasts)

    records = []
    for q, fc in forecasts:
        if size_aware and float(fc.size) == 0.0:
            continue  # genome ABSTAINED on this market
        price = q.market_price
        if price is None:  # already filtered above; re-narrow for the type checker
            continue
        records.append(
            sim.TradeRecord(
                market_id=q.id,
                signal_prob=float(fc.p_yes),
                crowd_price=float(price),
                outcome="YES" if q.outcome else "NO",
                liquidity=q.liquidity,
                event_id=q.event_id,
                lead=q.lead_days,
                category=q.category or None,
            )
        )

    coverage = (len(records) / n_eligible) if n_eligible else 0.0
    if not records:
        return {
            "net_roi": 0.0,
            "mean_event_roi": 0.0,
            "tstat_events": 0.0,
            "n_events": 0.0,
            "n_markets": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "coverage": coverage,
            "n_eligible": float(n_eligible),
        }

    res = sim.run_adversarial_sim(records, edge_threshold=edge_threshold)
    ts = res.tstat_events
    return {
        "net_roi": float(res.net_roi),
        "mean_event_roi": float(res.mean_event_roi),
        "tstat_events": float(ts) if ts is not None and ts == ts else 0.0,
        "n_events": float(res.n_events),
        "n_markets": float(res.n_markets),
        "total_pnl": float(res.total_pnl),
        "win_rate": float(res.win_rate),
        "coverage": coverage,
        "n_eligible": float(n_eligible),
    }
