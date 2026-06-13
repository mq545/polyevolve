"""Resolution sync: find past-close markets without a resolution, fetch + persist.

Run periodically (e.g. daily). Calibration can only be computed once resolutions
exist, so this is the other half of the measurement loop.
"""

from __future__ import annotations

import logging
import sys

from polyevolve.config import Config
from polyevolve.market_sources.polymarket import PolymarketSource
from polyevolve.storage import db

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = Config.from_env()
    polymarket = PolymarketSource()

    resolved = 0
    skipped = 0

    with db.connection(cfg.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.id, m.external_id
                FROM markets m
                LEFT JOIN resolutions r ON r.market_id = m.id
                WHERE m.venue = 'polymarket'
                  AND r.market_id IS NULL
                  AND m.close_time IS NOT NULL
                  AND m.close_time < NOW()
                """
            )
            pending = cur.fetchall()

        logger.info("%d markets past close without resolution", len(pending))

        for market_id, external_id in pending:
            try:
                resolution = polymarket.get_resolution(external_id)
                if resolution is None:
                    skipped += 1
                    continue
                db.upsert_resolution(conn, market_id, resolution)
                conn.commit()
                resolved += 1
                logger.info("resolved market=%s outcome=%s", external_id, resolution.outcome)
            except Exception:
                logger.exception("Failed resolving market %s", external_id)
                conn.rollback()

    logger.info("Done. resolved=%d skipped(ambiguous/unsettled)=%d", resolved, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
