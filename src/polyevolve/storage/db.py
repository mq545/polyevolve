"""Postgres helpers for raw_fetches, markets, predictions, resolutions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.types.json import Json

from polyevolve.contracts import Market, Prediction, Resolution


@contextmanager
def connection(db_url: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(db_url) as conn:
        yield conn


def checksum(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def insert_raw_fetch(
    conn: psycopg.Connection,
    source: str,
    endpoint: str,
    payload: Any,
) -> int | None:
    """Append-only insert. Returns the new id, or None if (source, endpoint, checksum) collides."""
    cs = checksum(payload)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw_fetches (source, endpoint, payload, checksum)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source, endpoint, checksum) DO NOTHING
            RETURNING id
            """,
            (source, endpoint, Json(payload), cs),
        )
        row = cur.fetchone()
        return row[0] if row else None


def upsert_market(
    conn: psycopg.Connection,
    market: Market,
    raw_fetch_id: int | None,
    category: str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO markets (
                venue, external_id, cross_venue_id, question, category,
                close_time, status, raw_fetch_id, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (venue, external_id) DO UPDATE SET
                status = EXCLUDED.status,
                close_time = EXCLUDED.close_time,
                last_seen_at = NOW(),
                metadata = EXCLUDED.metadata
            RETURNING id
            """,
            (
                market.venue,
                market.external_id,
                market.cross_venue_id,
                market.question,
                category,
                market.close_time,
                market.status,
                raw_fetch_id,
                Json(market.metadata),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def insert_prediction(conn: psycopg.Connection, market_id: int, p: Prediction) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO predictions (
                market_id, agent_name, model_name,
                probability_yes, confidence, market_price_at_prediction,
                reasoning, key_factors, uncertainty_drivers, data_sources_used
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                market_id,
                p.agent_name,
                p.model_name,
                p.probability_yes,
                p.confidence,
                p.market_price_at_prediction,
                p.reasoning,
                Json(p.key_factors),
                Json(p.uncertainty_drivers),
                Json(p.data_sources_used),
            ),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def upsert_resolution(
    conn: psycopg.Connection,
    market_id: int,
    r: Resolution,
    raw_fetch_id: int | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO resolutions (market_id, outcome, resolved_at, raw_fetch_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (market_id) DO UPDATE SET
                outcome = EXCLUDED.outcome,
                resolved_at = EXCLUDED.resolved_at
            """,
            (market_id, r.outcome, r.resolved_at, raw_fetch_id),
        )


def fetch_calibration(
    conn: psycopg.Connection, agent_name: str | None = None
) -> list[dict[str, Any]]:
    where = ""
    params: tuple[Any, ...] = ()
    if agent_name:
        where = "WHERE agent_name = %s"
        params = (agent_name,)
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM calibration {where}", params)
        cols = [d.name for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def insert_backtest(conn: psycopg.Connection, row: dict[str, Any]) -> None:
    """Insert one backtest result. Idempotent on (run_id, market_external_id)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO backtests (
                run_id, agent_name, model_name, market_external_id, question,
                as_of, resolved_at, outcome, probability_yes, confidence,
                market_price_at_as_of, is_clean, split,
                brier_agent, brier_market, reasoning, data_sources_used
            ) VALUES (
                %(run_id)s, %(agent_name)s, %(model_name)s, %(market_external_id)s,
                %(question)s, %(as_of)s, %(resolved_at)s, %(outcome)s,
                %(probability_yes)s, %(confidence)s, %(market_price_at_as_of)s,
                %(is_clean)s, %(split)s, %(brier_agent)s, %(brier_market)s,
                %(reasoning)s, %(data_sources_used)s
            )
            ON CONFLICT (run_id, market_external_id) DO NOTHING
            """,
            {**row, "data_sources_used": Json(row.get("data_sources_used"))},
        )


def fetch_backtest_calibration(
    conn: psycopg.Connection, run_id: str | None = None
) -> list[dict[str, Any]]:
    where = ""
    params: tuple[Any, ...] = ()
    if run_id:
        where = "WHERE run_id = %s"
        params = (run_id,)
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM v_backtest_calibration {where}", params)
        cols = [d.name for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def latest_backtest_run(conn: psycopg.Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT run_id FROM backtests ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        return str(row[0]) if row else None
