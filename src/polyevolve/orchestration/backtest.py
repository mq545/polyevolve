"""Backtest harness - point-in-time replay against resolved markets.

This is the FAST DEVELOPMENT LOOP, not a validation gate. It can falsify cheaply
(fail the backtest -> kill) but cannot validate (passing earns the right to
forward-test, not to trade). See RESEARCH_CONTRACT.md.

Discipline baked in:
- Every market is replayed with as_of = resolution time, so GDELT only returns
  news that existed before the market resolved (no retrieval contamination).
- Each row is tagged is_clean = (market resolved after model cutoff + margin).
  CONTAMINATED rows are dev-only and must never be a fitness/validation signal.
- A deterministic train/holdout split is assigned to clean rows so an evolution
  loop has an untouched holdout.

Usage:
    uv run python -m polyevolve.orchestration.backtest [--limit N] [--run-id ID]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta

from polyevolve.agents.foreign_politics_agent import ForeignPoliticsAgent
from polyevolve.config import Config
from polyevolve.data_sources.registry import DataRegistry
from polyevolve.market_sources.filters import is_foreign_politics, is_placeholder_market
from polyevolve.market_sources.polymarket import PolymarketSource, ResolvedMarket
from polyevolve.models import build_model
from polyevolve.models.cutoffs import get_cutoff, is_clean_for_backtest
from polyevolve.orchestration.scoring import assign_splits, brier
from polyevolve.storage import db

logger = logging.getLogger(__name__)


def _run_id(now: datetime, model_name: str) -> str:
    stamp = now.strftime("%Y%m%dT%H%M%S")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return f"bt_{stamp}_{safe_model}"


def compute_as_of(
    resolved_at: datetime,
    created_at: datetime | None,
    lead_days: int,
    min_lead_days: int = 3,
) -> datetime | None:
    """Pick the prediction instant: lead_days before actual resolution.

    Clamped to >= created_at (can't predict before the market existed). Returns
    None if the market's life is shorter than min_lead_days - too close to
    resolution to count as a forecast, so it's dropped rather than leak a
    near-resolution prediction into the calibration set.
    """
    target = resolved_at - timedelta(days=lead_days)
    as_of = max(target, created_at) if created_at is not None else target
    if as_of >= resolved_at:
        return None
    if (resolved_at - as_of) < timedelta(days=min_lead_days):
        return None
    return as_of


def run_backtest(
    *,
    cfg: Config,
    now: datetime,
    limit: int,
    lead_days: int = 7,
    max_per_event: int = 3,
    run_id: str | None = None,
) -> str:
    polymarket = PolymarketSource()
    registry = DataRegistry()
    model = build_model(model_id=cfg.default_model, anthropic_api_key=cfg.anthropic_api_key)
    agent = ForeignPoliticsAgent(model)

    rid = run_id or _run_id(now, model.name)
    cutoff = get_cutoff(model.name)
    logger.info(
        "Backtest run_id=%s model=%s cutoff=%s lead_days=%d",
        rid,
        model.name,
        cutoff.date.date() if cutoff else "UNKNOWN(all contaminated)",
        lead_days,
    )

    # 1. Discover genuinely past-resolved foreign-politics markets, excluding
    #    neg-risk placeholder candidate slots ("Person H", "Other") that never
    #    traded and have no price history.
    discovered: list[ResolvedMarket] = [
        rm
        for rm in polymarket.list_resolved_markets({"now": now, "pages_per_tag": 6})
        if is_foreign_politics(rm.market) and not is_placeholder_market(rm.market)
    ]

    # 2. Compute each market's as_of (lead_days before ACTUAL resolution) and
    #    cap markets-per-event so one election can't dominate the sample (which
    #    would shrink the effective n far below the row count). Drop markets too
    #    short-lived to give a valid forecast horizon. No silent truncation.
    plan: list[tuple[ResolvedMarket, datetime]] = []
    per_event: dict[str, int] = {}
    dropped_short = 0
    dropped_event_cap = 0
    for rm in discovered:
        as_of = compute_as_of(rm.resolved_at, rm.created_at, lead_days)
        if as_of is None:
            dropped_short += 1
            continue
        event = str(rm.market.metadata.get("event_title") or rm.market.external_id)
        if per_event.get(event, 0) >= max_per_event:
            dropped_event_cap += 1
            continue
        per_event[event] = per_event.get(event, 0) + 1
        plan.append((rm, as_of))
        if len(plan) >= limit:
            break
    logger.info(
        "Discovered %d FP markets; %d usable across %d events "
        "(dropped %d short-lived, %d over per-event cap of %d)",
        len(discovered),
        len(plan),
        len(per_event),
        dropped_short,
        dropped_event_cap,
        max_per_event,
    )

    # 3. Deterministic train/holdout split over the CLEAN subset only.
    clean_ids = [
        rm.market.external_id for rm, _ in plan if is_clean_for_backtest(model.name, rm.resolved_at)
    ]
    splits = assign_splits(clean_ids)
    logger.info(
        "Clean (post-cutoff): %d/%d  [train=%d holdout=%d]",
        len(clean_ids),
        len(plan),
        sum(1 for s in splits.values() if s == "train"),
        sum(1 for s in splits.values() if s == "holdout"),
    )

    # 4. Replay each market point-in-time at its as_of.
    with db.connection(cfg.db_url) as conn:
        for i, (rm, as_of) in enumerate(plan, 1):
            mkt = rm.market
            is_clean = is_clean_for_backtest(model.name, rm.resolved_at)
            try:
                data = registry.gather(mkt, conn=conn, as_of=as_of)
                pred = agent.predict(mkt, data=data)

                mkt_price = polymarket.price_at(rm.yes_token_id, as_of) if rm.yes_token_id else None

                b_agent = brier(pred.probability_yes, rm.outcome)
                b_market = brier(mkt_price, rm.outcome) if mkt_price is not None else None

                db.insert_backtest(
                    conn,
                    {
                        "run_id": rid,
                        "agent_name": agent.name,
                        "model_name": model.name,
                        "market_external_id": mkt.external_id,
                        "question": mkt.question,
                        "as_of": as_of,
                        "resolved_at": rm.resolved_at,
                        "outcome": rm.outcome,
                        "probability_yes": pred.probability_yes,
                        "confidence": pred.confidence,
                        "market_price_at_as_of": mkt_price,
                        "is_clean": is_clean,
                        "split": splits.get(mkt.external_id),
                        "brier_agent": b_agent,
                        "brier_market": b_market,
                        "reasoning": pred.reasoning,
                        "data_sources_used": list(data.keys()),
                    },
                )
                conn.commit()
                logger.info(
                    "[%d/%d] %s as_of=%s p=%.3f mkt=%s out=%s brier=%.3f %s | %s",
                    i,
                    len(plan),
                    "CLEAN " if is_clean else "contam",
                    as_of.date(),
                    pred.probability_yes,
                    f"{mkt_price:.3f}" if mkt_price is not None else "n/a",
                    rm.outcome,
                    b_agent,
                    splits.get(mkt.external_id, "-"),
                    mkt.question[:38],
                )
            except Exception:
                logger.exception("backtest failed for market %s", mkt.external_id)
                conn.rollback()

    return rid


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="polyevolve-backtest")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--lead-days", type=int, default=7, help="predict N days before resolution")
    parser.add_argument("--max-per-event", type=int, default=3, help="cap markets per event")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_env()
    now = datetime.now(UTC)
    rid = run_backtest(
        cfg=cfg,
        now=now,
        limit=args.limit,
        lead_days=args.lead_days,
        max_per_event=args.max_per_event,
        run_id=args.run_id,
    )

    logger.info("Backtest complete: run_id=%s", rid)
    logger.info("Inspect: uv run python -m polyevolve.cli backtest --run %s", rid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
