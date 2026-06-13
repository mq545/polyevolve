"""Google Trends search-interest connector - a point-in-time, crowd-underweighted signal.

Search interest is a leading indicator of *who is surging* - exactly the discrimination
signal (rank the winner) our forecaster lacks, and one the English-speaking Polymarket
crowd plausibly underweights on obscure foreign races. Leakage-safe by construction: we
request a timeframe ENDING at ``as_of`` so Trends only returns historical (dated) values
on/before the cutoff.

Two hard lessons baked in:
- KEYWORD RESOLUTION is everything: query bare proper nouns (a candidate's common name),
  NOT "<X> party" in English - "Tisza party" returns 0 while "Tisza" tracks the surge.
- RATE LIMITS: Trends 429s aggressively, so every query is disk-cached and batched
  (<=5 terms, scored on one shared 0-100 scale) with backoff.

`interest_shares(terms, as_of, geo)` returns each term's recent mean interest + momentum,
normalized to a share across the batch (the comparative signal a race needs).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_CACHE = Path("scripts/.cache/trends_cache.jsonl")
_MEM: dict[str, dict[str, Any]] | None = None


def _load_cache() -> dict[str, dict[str, Any]]:
    global _MEM
    if _MEM is None:
        _MEM = {}
        if _CACHE.exists():
            for line in _CACHE.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _MEM[rec["key"]] = rec["val"]
    return _MEM


def _save_cache(key: str, val: dict[str, Any]) -> None:
    cache = _load_cache()
    cache[key] = val
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "val": val}) + "\n")


def _raw_interest(
    terms: list[str], start: str, end: str, geo: str, *, retries: int = 3
) -> dict[str, list[float]] | None:
    """Per-term daily interest over [start, end] on a shared 0-100 scale. None on failure."""
    from pytrends.request import TrendReq  # noqa: PLC0415 - heavy, lazy

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
            pt.build_payload(terms[:5], timeframe=f"{start} {end}", geo=geo)
            df = pt.interest_over_time()
            if df is None or df.empty:
                return {t: [] for t in terms}
            cols = [c for c in df.columns if c != "isPartial"]
            return {c: [float(v) for v in df[c].tolist()] for c in cols}
        except Exception as exc:  # noqa: BLE001 - flaky API; backoff + retry
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    print(f"[trends] gave up on {terms} ({geo}): {last_err!r}")
    return None


def interest_shares(
    terms: list[str],
    as_of: datetime,
    *,
    geo: str = "",
    window_days: int = 60,
    momentum_days: int = 14,
) -> dict[str, dict[str, float]]:
    """Comparative search interest for `terms` as of `as_of` (data <= as_of only).

    Returns ``{term: {"recent", "share", "momentum"}}`` where ``recent`` is the mean
    interest over the last ``momentum_days`` of the window, ``share`` is that normalized
    across the batch (the rank signal), and ``momentum`` is recent vs prior-window change.
    Disk-cached by (terms, geo, as_of-day); fail-soft to empty on API failure.
    """
    end = as_of.date().isoformat()
    start = (as_of - timedelta(days=window_days)).date().isoformat()
    key = f"{geo}|{start}|{end}|{'|'.join(sorted(terms))}"
    cache = _load_cache()
    if key in cache:
        series = cache[key]
    else:
        raw = _raw_interest([t for t in terms if t], start, end, geo)
        series = raw if raw is not None else {}
        _save_cache(key, series)

    out: dict[str, dict[str, float]] = {}
    for t in terms:
        vals = series.get(t, []) or []
        if not vals:
            out[t] = {"recent": 0.0, "share": 0.0, "momentum": 0.0}
            continue
        recent = sum(vals[-momentum_days:]) / max(1, len(vals[-momentum_days:]))
        prior = vals[:-momentum_days] or vals
        prior_mean = sum(prior) / max(1, len(prior))
        out[t] = {"recent": recent, "share": 0.0, "momentum": recent - prior_mean}
    total = sum(v["recent"] for v in out.values())
    if total > 0:
        for v in out.values():
            v["share"] = v["recent"] / total
    return out
