"""Tests for coerce_rows - the anti-false-null seam for LLM structured output."""

from __future__ import annotations

from polyevolve.models import coerce_rows


def test_schema_shape_kept() -> None:
    rows = coerce_rows(
        [{"index": 1, "role": "INCUMBENT"}, {"index": 2, "role": "CHALLENGER"}], scalar_field="role"
    )
    assert rows == [{"index": 1, "role": "INCUMBENT"}, {"index": 2, "role": "CHALLENGER"}]


def test_flat_positional_strings() -> None:
    # the exact shape that silently false-nulled keyword resolution + econ tagging
    rows = coerce_rows(["INCUMBENT", "CHALLENGER", "CHALLENGER"], scalar_field="role")
    assert rows == [
        {"index": 1, "role": "INCUMBENT"},
        {"index": 2, "role": "CHALLENGER"},
        {"index": 3, "role": "CHALLENGER"},
    ]


def test_flat_positional_floats() -> None:
    # joint inference: probabilities as a flat list of floats
    rows = coerce_rows([0.6, 0.25, 0.15], scalar_field="probability")
    assert rows == [
        {"index": 1, "probability": 0.6},
        {"index": 2, "probability": 0.25},
        {"index": 3, "probability": 0.15},
    ]


def test_index_keyed_dict() -> None:
    rows = coerce_rows({"1": "A", "2": "B"}, scalar_field="role")
    assert {r["index"]: r["role"] for r in rows} == {1: "A", 2: "B"}


def test_dict_items_missing_index_backfilled() -> None:
    rows = coerce_rows([{"query": "Tisza"}, {"query": "DK"}], scalar_field="query")
    assert rows == [{"index": 1, "query": "Tisza"}, {"index": 2, "query": "DK"}]


def test_dict_items_extra_keys_preserved() -> None:
    rows = coerce_rows([{"index": 2, "probability": 0.4, "note": "x"}], scalar_field="probability")
    assert rows == [{"index": 2, "probability": 0.4, "note": "x"}]


def test_custom_index_base() -> None:
    rows = coerce_rows(["A", "B"], scalar_field="role", index_base=0)
    assert [r["index"] for r in rows] == [0, 1]


def test_non_list_yields_empty() -> None:
    assert coerce_rows(None, scalar_field="role") == []
    assert coerce_rows("INCUMBENT", scalar_field="role") == []
    assert coerce_rows(42, scalar_field="role") == []


def test_single_record_dict_is_not_a_row_list() -> None:
    # a dict whose values are NOT all scalars is one record, not an index->scalar map
    assert coerce_rows({"index": 1, "items": [1, 2]}, scalar_field="role") == []


def test_nested_and_none_items_skipped() -> None:
    rows = coerce_rows(["A", None, ["x"], "B"], scalar_field="role")
    assert [r["role"] for r in rows] == ["A", "B"]
    assert [r["index"] for r in rows] == [1, 4]  # positional index preserved
