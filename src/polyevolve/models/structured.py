"""Coercion for LLM structured-output shape drift - the anti-false-null seam.

A recurring, expensive failure mode: a tool/JSON schema asks for an array of records
(``[{"index": 1, "role": "INCUMBENT"}, ...]``) but a local model returns a flat
positional list of scalars (``["INCUMBENT", "CHALLENGER", ...]``) - or a dict keyed by
index. A hand-rolled parser expecting only the schema shape silently drops every item,
returns empty, and the caller reads "no signal" when the signal was fine - a *false NULL*
indistinguishable from a real negative result.

``coerce_rows`` normalizes all three observed shapes into one list of dicts, each carrying
an ``index`` and the named scalar field, so callers never hand-roll this again. It is the
single chokepoint every consumer of an LLM "list of records" field should pass through.
"""

from __future__ import annotations

from typing import Any

__all__ = ["coerce_rows"]

_SCALAR = (str, int, float, bool)


def coerce_rows(
    value: Any,
    *,
    scalar_field: str,
    index_field: str = "index",
    index_base: int = 1,
) -> list[dict[str, Any]]:
    """Normalize an LLM 'array of records' field into a uniform list of dicts.

    Accepts the three shapes local models actually emit and returns the same thing for
    each: a list of dicts that always have ``index_field`` (positional if absent) and,
    for scalar items, ``scalar_field``:

      - schema shape    ``[{"index": 1, "<field>": X}, ...]``  -> kept, index filled if missing
      - flat positional ``[X, Y, ...]``                        -> ``[{index:1, field:X}, ...]``
      - index-keyed dict ``{"1": X, "2": Y}``                  -> ``[{index:1, field:X}, ...]``

    Dict items are preserved as-is (all their keys kept) with only a missing ``index``
    back-filled positionally, so callers reading other named keys (e.g. ``probability``)
    still work. Non-dict, non-scalar items are skipped. A non-list/non-dict ``value``
    (None, etc.) yields ``[]``. Coercion never raises - it cannot turn a recoverable
    shape into a false null.
    """
    items: list[Any]
    if isinstance(value, dict):
        # index-keyed mapping {"1": X, ...} only if the values are scalars; otherwise a
        # single record dict, which is not a row list -> nothing to coerce.
        if value and all(isinstance(v, _SCALAR) for v in value.values()):
            rows: list[dict[str, Any]] = []
            for k, v in value.items():
                try:
                    idx = int(k)
                except (TypeError, ValueError):
                    idx = len(rows) + index_base
                rows.append({index_field: idx, scalar_field: v})
            return rows
        return []
    if isinstance(value, list):
        items = value
    else:
        return []

    out: list[dict[str, Any]] = []
    for pos, item in enumerate(items):
        if isinstance(item, dict):
            row = dict(item)
            if index_field not in row:
                row[index_field] = pos + index_base
            out.append(row)
        elif isinstance(item, _SCALAR):
            out.append({index_field: pos + index_base, scalar_field: item})
        # lists / nested / None items carry no usable record -> skip
    return out
