"""Tests for the `gather` verb (bench.pools) - fake registry, no network."""

from __future__ import annotations

from datetime import datetime

from polyevolve.bench.pools import context_to_pool, gather_pool, gather_pools
from polyevolve.reason.dsl import Question

AS_OF = datetime(2024, 6, 1)


class FakeRegistry:
    """Records the as_of it was called with; returns a fixed context dict."""

    def __init__(self, context: dict[str, str]) -> None:
        self._context = context
        self.calls: list[tuple[str, object]] = []

    def gather(self, market, conn=None, as_of=None):  # noqa: ANN001
        self.calls.append((market.external_id, as_of))
        return dict(self._context)


def _q(qid: str) -> Question:
    return Question(id=qid, text="Will X win the 2024 election?", as_of=AS_OF, category="Country")


def test_context_to_pool_one_item_per_source_stamped_with_cutoff() -> None:
    pool = context_to_pool({"news": "headline body", "polls": "A 51% B 49%"}, AS_OF)
    assert len(pool.items) == 2
    assert {i.source for i in pool.items} == {"news", "polls"}
    assert all(i.date == AS_OF for i in pool.items)
    # leakage guard keeps cutoff-stamped items
    assert len(pool.on_or_before(AS_OF)) == 2


def test_context_to_pool_skips_empty_keeps_error_marker() -> None:
    pool = context_to_pool({"news": "", "polls": "[SOURCE ERROR] polls: boom"}, AS_OF)
    assert len(pool.items) == 1
    assert pool.items[0].source == "polls"
    assert "[SOURCE ERROR]" in pool.items[0].text


def test_gather_pool_passes_as_of_for_leakage_control() -> None:
    reg = FakeRegistry({"news": "x"})
    pool = gather_pool(_q("m1"), reg)
    assert len(pool.items) == 1
    # the source MUST be called with the question cutoff (point-in-time)
    assert reg.calls == [("m1", AS_OF)]


def test_gather_pools_aligned_and_cached(tmp_path) -> None:  # noqa: ANN001
    reg = FakeRegistry({"news": "x"})
    cache = tmp_path / "pools.jsonl"
    qs = [_q("a"), _q("b")]

    pools = gather_pools(qs, cache_path=cache, registry=reg)
    assert len(pools) == 2
    assert cache.exists()
    assert len(reg.calls) == 2

    # second call hits the cache for both -> no new fetches
    reg2 = FakeRegistry({"news": "DIFFERENT"})
    pools2 = gather_pools(qs, cache_path=cache, registry=reg2)
    assert len(reg2.calls) == 0
    assert pools2[0].items[0].text == "x"  # served from cache, not the new registry


def test_gather_pools_fetches_only_missing(tmp_path) -> None:  # noqa: ANN001
    reg = FakeRegistry({"news": "x"})
    cache = tmp_path / "pools.jsonl"
    gather_pools([_q("a")], cache_path=cache, registry=reg)
    assert len(reg.calls) == 1

    # add a new question -> only the new one is fetched
    reg.calls.clear()
    pools = gather_pools([_q("a"), _q("b")], cache_path=cache, registry=reg)
    assert len(pools) == 2
    assert reg.calls == [("b", AS_OF)]


def test_gather_pools_refresh_refetches_all(tmp_path) -> None:  # noqa: ANN001
    reg = FakeRegistry({"news": "x"})
    cache = tmp_path / "pools.jsonl"
    gather_pools([_q("a")], cache_path=cache, registry=reg)

    reg2 = FakeRegistry({"news": "NEW"})
    pools = gather_pools([_q("a")], cache_path=cache, registry=reg2, refresh=True)
    assert pools[0].items[0].text == "NEW"
    assert len(reg2.calls) == 1
