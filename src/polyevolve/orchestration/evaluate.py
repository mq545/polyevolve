"""Evaluate calibration of stored predictions against resolved markets."""

from __future__ import annotations

import logging
import sys

from polyevolve.config import Config
from polyevolve.storage import db

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = Config.from_env()

    with db.connection(cfg.db_url) as conn:
        rows = db.fetch_calibration(conn)
        if not rows:
            logger.info("No resolved predictions yet. Nothing to evaluate.")
            return 0

        header = (
            f"{'agent':<25} {'model':<25} {'bucket':<8} {'n':<6} "
            f"{'predicted':<10} {'actual':<10} {'brier':<10}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['agent_name']:<25} {r['model_name']:<25} "
                f"{float(r['probability_bucket']):<8.2f} {r['n']:<6} "
                f"{float(r['avg_predicted']):<10.3f} {float(r['actual_yes_rate']):<10.3f} "
                f"{float(r['brier_component']):<10.4f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
