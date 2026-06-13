"""Build a frozen, model-agnostic eval snapshot.

Run ONCE per named snapshot set. Discovers resolved foreign-politics markets,
computes each market's as_of (lead before actual resolution), and freezes the
GDELT research context + historical market price + outcome at that instant into
eval_snapshots. After this, the evaluator never touches the network - every
candidate sees identical, reproducible inputs.

Model-agnostic on purpose: clean/contaminated and train/holdout are decided per
model at eval time, so one snapshot serves qwen now and Claude later.

Usage:
    uv run python -m polyevolve.orchestration.snapshot --set fp_v1 --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

import psycopg
from psycopg.types.json import Json

from polyevolve.config import Config
from polyevolve.data_sources.registry import DataRegistry
from polyevolve.market_sources.filters import domain as get_domain
from polyevolve.market_sources.filters import is_placeholder_market
from polyevolve.market_sources.polymarket import PolymarketSource
from polyevolve.orchestration.backtest import compute_as_of
from polyevolve.storage import db

logger = logging.getLogger(__name__)


def _insert(
    conn: psycopg.Connection,
    snapshot_set: str,
    rm: object,
    as_of: datetime,
    lead_days: int,
    price: float | None,
    context: dict[str, str],
) -> bool:
    m = rm.market  # type: ignore[attr-defined]
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_snapshots (
                snapshot_set, market_external_id, venue, question, event_title,
                tags, as_of, resolved_at, created_market_at, lead_days, outcome,
                market_price_at_as_of, research_context
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (snapshot_set, market_external_id) DO NOTHING
            RETURNING id
            """,
            (
                snapshot_set,
                m.external_id,
                m.venue,
                m.question,
                m.metadata.get("event_title"),
                Json(m.metadata.get("tags", [])),
                as_of,
                rm.resolved_at,  # type: ignore[attr-defined]
                rm.created_at,  # type: ignore[attr-defined]
                lead_days,
                rm.outcome,  # type: ignore[attr-defined]
                price,
                Json(context),
            ),
        )
        return cur.fetchone() is not None


def build_snapshot(
    *,
    cfg: Config,
    now: datetime,
    snapshot_set: str,
    limit: int,
    lead_days: int = 7,
    max_per_event: int = 3,
    domain: str = "foreign_politics",
    min_volume: float = 0.0,
    gather_research: bool = True,
    pages_per_tag: int = 10,
) -> int:
    polymarket = PolymarketSource()
    registry = DataRegistry()

    dom = get_domain(domain)

    def _volume(market: object) -> float:
        try:
            return float(market.metadata.get("volume") or 0.0)  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            return 0.0

    # Liquidity gate at DISCOVERY: only keep markets with real volume, so we never spend a
    # point-in-time price fetch on the dust/auto-generated long tail (e.g. 187k sports props).
    discovered = [
        rm
        for rm in polymarket.list_resolved_markets(
            {"now": now, "pages_per_tag": pages_per_tag, "tags": list(dom.tags)}
        )
        if dom.keep(rm.market)
        and not is_placeholder_market(rm.market)
        and _volume(rm.market) >= min_volume
    ]

    plan = []
    per_event: dict[str, int] = {}
    dropped_short = dropped_cap = 0
    for rm in discovered:
        as_of = compute_as_of(rm.resolved_at, rm.created_at, lead_days)
        if as_of is None:
            dropped_short += 1
            continue
        event = str(rm.market.metadata.get("event_title") or rm.market.external_id)
        if per_event.get(event, 0) >= max_per_event:
            dropped_cap += 1
            continue
        per_event[event] = per_event.get(event, 0) + 1
        plan.append((rm, as_of))
        if len(plan) >= limit:
            break

    logger.info(
        "Discovered %d FP markets; planning %d across %d events "
        "(dropped %d short-lived, %d over per-event cap)",
        len(discovered),
        len(plan),
        len(per_event),
        dropped_short,
        dropped_cap,
    )

    inserted = 0
    with db.connection(cfg.db_url) as conn:
        for i, (rm, as_of) in enumerate(plan, 1):
            try:
                # gather_research=False -> a fast price+outcome dataset (no per-market
                # context fetch); the calibration/return studies don't need the prose.
                context = (
                    registry.gather(rm.market, conn=conn, as_of=as_of) if gather_research else {}
                )
                price = polymarket.price_at(rm.yes_token_id, as_of) if rm.yes_token_id else None
                if _insert(conn, snapshot_set, rm, as_of, lead_days, price, context):
                    inserted += 1
                conn.commit()
                logger.info(
                    "[%d/%d] frozen as_of=%s price=%s out=%s | %s",
                    i,
                    len(plan),
                    as_of.date(),
                    f"{price:.3f}" if price is not None else "n/a",
                    rm.outcome,
                    rm.market.question[:44],
                )
            except Exception:
                logger.exception("snapshot insert failed for %s", rm.market.external_id)
                conn.rollback()

    logger.info("Snapshot '%s': %d markets frozen", snapshot_set, inserted)
    return inserted


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="polyevolve-snapshot")
    parser.add_argument("--set", dest="snapshot_set", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--lead-days", type=int, default=7)
    parser.add_argument("--max-per-event", type=int, default=3)
    parser.add_argument(
        "--domain",
        default="foreign_politics",
        help="ingestion domain: foreign_politics | sports | crypto | culture | all",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="discovery liquidity gate: skip markets with volume below this ($)",
    )
    parser.add_argument(
        "--no-research",
        action="store_true",
        help="skip per-market research gathering (fast price+outcome dataset)",
    )
    parser.add_argument(
        "--pages-per-tag",
        type=int,
        default=10,
        help="event pages to pull per tag (raise to scale n)",
    )
    args = parser.parse_args()

    cfg = Config.from_env()
    build_snapshot(
        cfg=cfg,
        now=datetime.now(UTC),
        snapshot_set=args.snapshot_set,
        limit=args.limit,
        lead_days=args.lead_days,
        max_per_event=args.max_per_event,
        domain=args.domain,
        min_volume=args.min_volume,
        gather_research=not args.no_research,
        pages_per_tag=args.pages_per_tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
