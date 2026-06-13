"""``gather`` - the connector-based, leakage-safe EvidencePool builder.

This is the data lever of PolyEvolve: it turns a `Question` into the FROZEN,
point-in-time evidence corpus the genome reasons over. It wraps the existing
`DataRegistry` (GDELT DOC news + local-language Wikipedia pageviews/polls + any
future connector), gathers each source's context constrained to ``<= as_of``, and
records the result as an `EvidencePool` of one `EvidenceItem` per source.

Two properties make this honest and reusable:

- **Leakage-safe.** Every source is called with ``as_of = question.as_of`` so it
  only returns data published before the cutoff (point-in-time view). Each item is
  stamped with that cutoff date, so the DSL's ``on_or_before`` guard keeps it.
- **Frozen + cached.** Pools are written to a jsonl cache keyed by question id;
  re-gathering reuses the cache. The evolutionary loop therefore sees a *fixed*
  corpus - features-as-fetched are frozen, only retrieval/selection + reasoning
  evolve - and runs deterministically, offline, with zero re-fetch cost.

FAIL LOUD (inherited from `DataRegistry`): a source that errors contributes an
explicit ``[SOURCE ERROR]`` marker item, never a silent omission - so a genome (or
a human) can tell "nothing happened" from "we failed to look".
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from polyevolve.contracts.markets import Market
from polyevolve.reason.dsl import EvidenceItem, EvidencePool, Question

if TYPE_CHECKING:  # avoid importing the (network-touching) registry at module load
    from polyevolve.data_sources.registry import DataRegistry

__all__ = ["gather_pool", "gather_pools", "context_to_pool"]


def _question_to_market(q: Question) -> Market:
    """Adapt a DSL `Question` to the `Market` shape `DataRegistry` consumes.

    Only the fields the sources read are meaningful: ``question`` (the query text)
    and ``metadata['tags']`` (curated entity labels - e.g. country names - that the
    local-language sources map to Wikipedia languages / GDELT codes). We seed tags
    from the question category so single-label questions still route correctly.
    """
    tags = [q.category] if q.category else []
    return Market(
        venue="polyevolve",
        external_id=q.id,
        cross_venue_id=None,
        question=q.text,
        close_time=None,
        status="open",
        metadata={"tags": tags},
    )


def context_to_pool(context: dict[str, str], as_of: datetime | None) -> EvidencePool:
    """Convert a `DataRegistry.gather` result ({source: rendered_text}) to a pool.

    One `EvidenceItem` per non-empty source, stamped with the ``as_of`` cutoff so
    the leakage guard keeps it. Error markers are preserved as items (fail loud).
    """
    items: list[EvidenceItem] = []
    for source, rendered in context.items():
        text = (rendered or "").strip()
        if not text:
            continue
        items.append(EvidenceItem(text=text, source=source, date=as_of))
    return EvidencePool(items=items)


def gather_pool(
    question: Question,
    registry: DataRegistry | None = None,
    *,
    conn: object | None = None,
) -> EvidencePool:
    """Gather the frozen, leakage-safe `EvidencePool` for a single question.

    ``registry`` defaults to the production `DataRegistry` (touches the network);
    inject a fake for tests. ``conn`` (optional psycopg connection) enables raw-
    payload auditing to ``raw_fetches``.
    """
    if registry is None:
        from polyevolve.data_sources.registry import DataRegistry

        registry = DataRegistry()
    market = _question_to_market(question)
    context = registry.gather(market, conn=conn, as_of=question.as_of)  # type: ignore[arg-type]
    return context_to_pool(context, question.as_of)


def gather_pools(
    questions: Sequence[Question],
    *,
    cache_path: str | Path | None = None,
    registry: DataRegistry | None = None,
    conn: object | None = None,
    refresh: bool = False,
) -> list[EvidencePool]:
    """Gather pools for many questions, returning a list ALIGNED 1:1 with `questions`.

    Pools are cached to ``cache_path`` (jsonl, one ``{"id", "pool"}`` record per
    line). Cached questions are reused; only missing ones are fetched (set
    ``refresh=True`` to re-fetch everything). The aligned output is exactly what
    `bench.evaluate_calibration` / `bench.returns.evaluate_return` expect as
    ``pools`` - so ``gather_pools`` is the bridge from raw markets to fitness.
    """
    cache: dict[str, EvidencePool] = {}
    path = Path(cache_path) if cache_path else None
    if path is not None and path.exists() and not refresh:
        cache = _load_cache(path)

    out: list[EvidencePool] = []
    new_records: list[tuple[str, EvidencePool]] = []
    for q in questions:
        pool = cache.get(q.id)
        if pool is None:
            pool = gather_pool(q, registry, conn=conn)
            cache[q.id] = pool
            new_records.append((q.id, pool))
        out.append(pool)

    if path is not None and (new_records or refresh):
        _write_cache(path, cache if refresh else None, new_records)
    return out


def _load_cache(path: Path) -> dict[str, EvidencePool]:
    cache: dict[str, EvidencePool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[str(rec["id"])] = EvidencePool.model_validate(rec["pool"])
    return cache


def _write_cache(
    path: Path,
    full: dict[str, EvidencePool] | None,
    new_records: Sequence[tuple[str, EvidencePool]],
) -> None:
    """Append new records, or rewrite the whole cache when ``full`` is given (refresh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if full is not None:
        with path.open("w", encoding="utf-8") as fh:
            for qid, pool in full.items():
                fh.write(json.dumps({"id": qid, "pool": pool.model_dump(mode="json")}) + "\n")
        return
    with path.open("a", encoding="utf-8") as fh:
        for qid, pool in new_records:
            fh.write(json.dumps({"id": qid, "pool": pool.model_dump(mode="json")}) + "\n")
