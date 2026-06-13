"""Tests for Genome hashing + serialization (the cache-key contract)."""

from polyevolve.evolution.genome import Genome


def _g(**kw: object) -> Genome:
    base = {"system_prompt": "sys", "domain_context": "dom"}
    base.update(kw)
    return Genome(**base)  # type: ignore[arg-type]


def test_hash_is_deterministic() -> None:
    assert _g().hash() == _g().hash()


def test_hash_changes_with_prompt() -> None:
    assert _g(system_prompt="a").hash() != _g(system_prompt="b").hash()


def test_hash_changes_with_effort() -> None:
    assert _g(effort="low").hash() != _g(effort="high").hash()


def test_hash_changes_with_data_weights() -> None:
    assert (
        _g(data_weights={"gdelt_news": 1.0}).hash() != _g(data_weights={"gdelt_news": 0.5}).hash()
    )


def test_hash_stable_across_weight_key_order() -> None:
    # dict insertion order must not change the hash (json sort_keys)
    a = _g(data_weights={"a": 1.0, "b": 0.5})
    b = _g(data_weights={"b": 0.5, "a": 1.0})
    assert a.hash() == b.hash()


def test_roundtrip_dict() -> None:
    g = _g(effort="high", max_context_chars=5000)
    assert Genome.from_dict(g.to_dict()) == g
