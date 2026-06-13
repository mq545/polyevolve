"""LLM keyword resolver - turn a market question into the term people actually SEARCH.

The Trends signal is real (hand-keyed "Tisza" tracked the winning surge) but dies on
naive extraction: a regex over "Will X win?" yields generic party labels ("the Indian
National Congress", "DIKO") that nobody Googles, so the series is flat noise. This node
asks the model, once per event, to map each sibling market to the single best Google
Trends query for that candidate - the common personal name or short party tag in the
language locals search, stripped of English boilerplate ("party", "coalition", honorifics).

It is a LOOKUP, not a forecast: it sees only the question text + locale, never the
outcome, so it is leakage-safe. One call per event keeps the field consistent (all terms
resolved against each other) and cheap. Disk-cached by (event, sorted questions, geo).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from polyevolve.models import build_model, coerce_rows

__all__ = ["resolve_keywords"]

_CACHE = Path("scripts/.cache/keyword_cache.jsonl")
_MEM: dict[str, dict[str, str]] | None = None

_TOOL: dict[str, Any] = {
    "name": "submit_terms",
    "description": (
        "Return the single best Google Trends search query for EACH listed market - the "
        "term real people in that locale type to follow this candidate. The only output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Market number from the list."},
                        "query": {
                            "type": "string",
                            "description": (
                                "Bare search query: the candidate's common personal name or "
                                "short party tag as locals search it (native script if that is "
                                "what they use). No 'party'/'coalition'/honorifics/English gloss."
                            ),
                        },
                    },
                    "required": ["index", "query"],
                },
            },
        },
        "required": ["terms"],
    },
}

_SYS = (
    "You map election-market questions to the exact phrase people SEARCH on Google for that "
    "candidate or party - not the formal English name. Rules: use the common personal name "
    "(surname locals use) for a person; the short colloquial tag for a party (e.g. 'Tisza' "
    "not 'Tisza Party', 'BJP' or the Hindi name not 'Bharatiya Janata Party'); native script "
    "if that locale searches in it; drop honorifics, 'party', 'coalition', 'alliance', and any "
    "parenthetical gloss. Return ONLY the bare name itself - do NOT append words like "
    "'election', 'campaign', 'kampany', 'vote', or the office. One query per market, in the "
    "SAME order as the markets are listed. This is a lookup of how the name is searched, "
    "NOT a prediction. Call submit_terms exactly once."
)


def _load() -> dict[str, dict[str, str]]:
    global _MEM
    if _MEM is None:
        _MEM = {}
        if _CACHE.exists():
            for line in _CACHE.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _MEM[rec["key"]] = rec["val"]
    return _MEM


def _save(key: str, val: dict[str, str]) -> None:
    cache = _load()
    cache[key] = val
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "val": val}) + "\n")


def resolve_keywords(
    questions: Sequence[str],
    *,
    event: str = "",
    geo: str = "",
    model_id: str = "ollama/qwen3:30b-a3b-instruct-2507-q4_K_M",
    anthropic_api_key: str | None = None,
) -> dict[int, str]:
    """Map each question (by its index) to the search term locals use. Disk-cached.

    Returns ``{index: query}`` for every index the model resolved; indices it omits are
    simply absent (caller falls back). Fail-soft to ``{}`` on model error.
    """
    qs = list(questions)
    key = f"{geo}|{event}|" + "||".join(qs)
    cache = _load()
    if key in cache:
        return {int(k): v for k, v in cache[key].items()}

    listing = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(qs))
    user = (
        f"LOCALE/GEO: {geo or 'unknown (use the candidates home-country search habits)'}\n"
        f"EVENT: {event or '(unspecified election)'}\n\n"
        f"MARKETS:\n{listing}\n\n"
        "Return one search query per market via submit_terms."
    )
    model = build_model(model_id=model_id, anthropic_api_key=anthropic_api_key)
    try:
        res = model.complete_with_tool(
            cached_system_blocks=[_SYS],
            user_content=user,
            tool=_TOOL,
            metadata={"question_id": event or (qs[0] if qs else ""), "node": "resolve_keywords"},
        )
        out = res["input"]
    except Exception:  # noqa: BLE001 - fail-soft; caller falls back to naive extraction
        return {}

    # coerce_rows normalizes the schema shape AND the flat-positional-string shape local
    # models emit (the anti-false-null seam), so we never hand-roll that parsing again.
    mapping: dict[int, str] = {}
    for row in coerce_rows(out.get("terms"), scalar_field="query"):
        try:
            i = int(row["index"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        q = str(row.get("query", "")).strip()
        if 0 <= i < len(qs) and q:
            mapping[i] = q
    _save(key, {str(k): v for k, v in mapping.items()})
    return mapping
