"""Dataset loaders for the fitness bench.

The bench scores genomes against RESOLVED, point-in-time forecasting questions.
`load_manifold` reads the dev corpus produced by `scripts/fbench_select.py`
(re-fetch / selection logic lives there; here we only parse the frozen jsonl).

Each line of the jsonl has the shape::

    {"id", "question", "text_desc", "created", "close", "T" (ms epoch),
     "crowd_at_T", "resolution" ('YES'|'NO'), ...}

and is mapped onto the frozen DSL types (`Question`, `EvidencePool`,
`EvidenceItem`). Ground truth (`outcome`) and the crowd baseline (`crowd_prob`)
live on `Question` but are NEVER exposed to a genome - only the bench reads them.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Question

__all__ = [
    "load_manifold",
    "load_manifold_with_pools",
    "pool_for",
    "parse_question",
    "load_polymarket_resolved",
    "load_polymarket_resolved_with_pools",
]


def _utc_from_ms(ms: float) -> datetime:
    """Naive UTC datetime from a millisecond epoch (matches `datetime.utcfromtimestamp`)."""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).replace(tzinfo=None)


def parse_question(row: dict[str, object]) -> Question:
    """Map one fbench jsonl record onto a frozen-DSL `Question`."""
    resolution = str(row.get("resolution", "")).upper()
    crowd = row.get("crowd_at_T")
    price = row.get("market_price")
    return Question(
        id=str(row["id"]),
        text=str(row["question"]),
        as_of=_utc_from_ms(float(row["T"])),  # type: ignore[arg-type]
        category=str(row.get("category", "")),
        outcome=(resolution == "YES"),
        crowd_prob=None if crowd is None else float(crowd),  # type: ignore[arg-type]
        market_price=None if price is None else float(price),  # type: ignore[arg-type]
    )


def pool_for(row: dict[str, object]) -> EvidencePool:
    """Build the (frozen) evidence pool for one record.

    The dev corpus carries at most a single description blob (`text_desc`); when
    present it becomes one undated `EvidenceItem`, otherwise the pool is empty.
    Richer point-in-time acquisition is an upstream concern (scout/connectors).
    """
    text_desc = str(row.get("text_desc") or "").strip()
    if not text_desc:
        return EvidencePool(items=[])
    return EvidencePool(items=[EvidenceItem(text=text_desc, source="manifold:text_desc")])


def load_manifold(path: str | Path) -> list[Question]:
    """Load resolved point-in-time questions from an fbench jsonl file.

    Pools are reconstructed on demand via `pool_for`; if a caller needs the
    paired pool, use `load_manifold_with_pools`.
    """
    return [q for q, _ in load_manifold_with_pools(path)]


def load_manifold_with_pools(path: str | Path) -> list[tuple[Question, EvidencePool]]:
    """Load (Question, EvidencePool) pairs from an fbench jsonl file."""
    p = Path(path)
    out: list[tuple[Question, EvidencePool]] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append((parse_question(row), pool_for(row)))
    return out


def _row_to_question(
    mid: Any,
    text: Any,
    as_of: Any,
    price: Any,
    outcome: Any,
    event_title: Any,
    lead_days: Any,
    tags: Any,
) -> Question:
    cat = ""
    if tags:
        cat = tags[0] if isinstance(tags, list) and tags else str(tags)
    return Question(
        id=str(mid),
        text=str(text or ""),
        as_of=as_of if isinstance(as_of, datetime) else datetime.now(UTC).replace(tzinfo=None),
        category=cat,
        outcome=(str(outcome).upper() in ("YES", "TRUE", "1")),
        market_price=float(price),
        crowd_prob=float(price),
        event_id=str(event_title) if event_title else None,
        lead_days=int(lead_days) if lead_days is not None else None,
    )


def _resolved_query(
    snapshot_set: str | None,
    price_band: tuple[float, float] | None,
    limit: int | None,
    *,
    with_context: bool,
) -> tuple[str, list[object]]:
    cols = (
        "market_external_id, question, as_of, market_price_at_as_of, "
        "outcome, event_title, lead_days, tags"
    )
    if with_context:
        cols += ", research_context"
    q = (
        f"select {cols} from eval_snapshots "
        "where outcome is not null and market_price_at_as_of is not null"
    )
    params: list[object] = []
    if snapshot_set:
        q += " and snapshot_set = %s"
        params.append(snapshot_set)
    if price_band:
        q += " and market_price_at_as_of between %s and %s"
        params.extend([price_band[0], price_band[1]])
    if limit:
        q += " limit %s"
        params.append(limit)
    return q, params


def load_polymarket_resolved(
    db_url: str | None = None,
    limit: int | None = None,
    snapshot_set: str | None = None,
    price_band: tuple[float, float] | None = None,
) -> list[Question]:
    """Load resolved POLYMARKET markets (with price + outcome) for the RETURN bench.

    Reads `eval_snapshots` - the real betting-market dataset behind PolyEvolve's
    net-of-spread return fitness. Each row carries `market_price_at_as_of` (the
    crowd mid the sim crosses the spread against), the resolved `outcome`, plus the
    execution metadata the adversarial sim needs (event_id from `event_title` so
    correlated legs collapse to one observation, `lead_days`, `category` from tags).
    Liquidity is not stored -> None (sim treats as the conservative mid tier).
    ``price_band`` (lo, hi) optionally restricts to a crowd-price window (e.g.
    (0.3, 0.7) - the mid-band "beatable" regime).
    """
    import psycopg

    from polyevolve.config import Config

    url = db_url or Config.from_env().db_url
    q, params = _resolved_query(snapshot_set, price_band, limit, with_context=False)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(q, params)
        return [_row_to_question(*row) for row in cur.fetchall()]


def load_polymarket_resolved_with_pools(
    db_url: str | None = None,
    limit: int | None = None,
    snapshot_set: str | None = None,
    price_band: tuple[float, float] | None = None,
) -> list[tuple[Question, EvidencePool]]:
    """Like `load_polymarket_resolved`, but also returns each market's FROZEN pool.

    The pool is reconstructed from the stored `research_context` jsonb (the same
    ``{source: rendered_text}`` shape `gather` produces), already point-in-time and
    leakage-audited from the original eval run - so the RETURN loop can run over real
    evidence with zero re-fetch. Markets without context get an empty pool.
    """
    import psycopg

    from polyevolve.bench.pools import context_to_pool
    from polyevolve.config import Config

    url = db_url or Config.from_env().db_url
    q, params = _resolved_query(snapshot_set, price_band, limit, with_context=True)
    out: list[tuple[Question, EvidencePool]] = []
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(q, params)
        for row in cur.fetchall():
            question = _row_to_question(*row[:8])
            rc = row[8]
            ctx = rc if isinstance(rc, dict) else {}
            pool = context_to_pool({str(k): str(v) for k, v in ctx.items()}, question.as_of)
            out.append((question, pool))
    return out
