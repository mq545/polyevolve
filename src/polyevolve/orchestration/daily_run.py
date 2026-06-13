"""Daily run: fetch foreign-politics markets, gather research data, predict, store."""

from __future__ import annotations

import logging
import sys

from polyevolve.agents.foreign_politics_agent import ForeignPoliticsAgent
from polyevolve.config import Config
from polyevolve.data_sources.registry import DataRegistry
from polyevolve.market_sources.filters import is_foreign_politics
from polyevolve.market_sources.polymarket import PolymarketSource
from polyevolve.models import build_model
from polyevolve.observability import LLMCallRecorder
from polyevolve.observability.langfuse_client import get_langfuse
from polyevolve.storage import db

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = Config.from_env()

    polymarket = PolymarketSource()
    registry = DataRegistry()
    recorder = LLMCallRecorder(cfg.db_url, langfuse=get_langfuse())
    model = build_model(
        model_id=cfg.default_model,
        recorder=recorder,
    )
    agent = ForeignPoliticsAgent(model)

    logger.info("Using model: %s", model.name)

    with db.connection(cfg.db_url) as conn:
        all_markets = list(polymarket.list_markets({}))
        markets = [m for m in all_markets if is_foreign_politics(m)]
        logger.info(
            "Fetched %d markets; %d matched foreign-politics filter",
            len(all_markets),
            len(markets),
        )

        for market in markets:
            try:
                market_db_id = db.upsert_market(conn, market, raw_fetch_id=None)
                data = registry.gather(market, conn)
                prediction = agent.predict(market, data=data)
                pred_id = db.insert_prediction(conn, market_db_id, prediction)
                conn.commit()
                logger.info(
                    "market=%s p_yes=%.3f conf=%.3f sources=%s pred_id=%d | %s",
                    market.external_id,
                    prediction.probability_yes,
                    prediction.confidence,
                    list(data.keys()),
                    pred_id,
                    market.question[:50],
                )
            except Exception:
                logger.exception("Failed processing market %s", market.external_id)
                conn.rollback()

    return 0


if __name__ == "__main__":
    sys.exit(main())
