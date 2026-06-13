"""World Bank macro connector - vintage-correct economic fundamentals, leakage-safe.

The incumbent economic-penalty mechanism (high inflation + unemployment, low growth ->
incumbent loses) is the most-documented structural prior in political science, and it is a
candidate UNPRICED signal: the English-speaking Polymarket crowd may underweight local
macro conditions on obscure foreign races. Public API, no key.

Leakage-safe by construction: `macro(iso, as_of)` returns, per indicator, the most recent
ANNUAL value dated STRICTLY BEFORE the election year - the figure that would have been
published by decision time (World Bank annual data lags ~1 year; the election-year value is
not yet out). Caveat: World Bank serves the latest REVISED annual figure, so this is
vintage-approximate at annual granularity (revisions are small), not a true point-in-time
ALFRED-style vintage. Disk-cached.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

__all__ = ["INDICATORS", "macro"]

_BASE = "https://api.worldbank.org/v2"
_CACHE = Path("scripts/.cache/worldbank_cache.jsonl")
_MEM: dict[str, dict[str, float | None]] | None = None

# indicator code -> friendly key
INDICATORS = {
    "FP.CPI.TOTL.ZG": "inflation",  # CPI inflation, annual %
    "NY.GDP.MKTP.KD.ZG": "growth",  # real GDP growth, annual %
    "SL.UEM.TOTL.ZS": "unemployment",  # unemployment, % of labour force (ILO modelled)
}


def _load_cache() -> dict[str, dict[str, float | None]]:
    global _MEM
    if _MEM is None:
        _MEM = {}
        if _CACHE.exists():
            for line in _CACHE.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _MEM[rec["key"]] = rec["val"]
    return _MEM


def _save_cache(key: str, val: dict[str, float | None]) -> None:
    cache = _load_cache()
    cache[key] = val
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "val": val}) + "\n")


def _series(iso: str, indicator: str, lo: int, hi: int) -> dict[int, float]:
    url = f"{_BASE}/country/{iso}/indicator/{indicator}?format=json&date={lo}:{hi}&per_page=100"
    req = urllib.request.Request(url, headers={"User-Agent": "polyevolve-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as fh:  # noqa: S310 - fixed https host
            raw = json.load(fh)
    except (urllib.error.URLError, json.JSONDecodeError):
        return {}  # unknown country code / transient error -> treat as no data
    if not isinstance(raw, list) or len(raw) < 2 or not raw[1]:
        return {}
    out: dict[int, float] = {}
    for d in raw[1]:
        if d.get("value") is not None:
            try:
                out[int(d["date"])] = float(d["value"])
            except (TypeError, ValueError):
                continue
    return out


def macro(iso: str, as_of: datetime) -> dict[str, float | None]:
    """Latest pre-election-year value per indicator, plus a misery index. Leakage-safe.

    Returns ``{"inflation", "growth", "unemployment", "misery"}`` where misery =
    inflation + unemployment - growth (higher = worse economy = larger incumbent penalty).
    Any indicator with no value before the election year is ``None``; misery is ``None``
    unless all three components are present. Disk-cached by (iso, election-year).
    """
    year = as_of.year
    key = f"{iso}|{year}"
    cache = _load_cache()
    if key in cache:
        return cache[key]

    out: dict[str, float | None] = {}
    for code, name in INDICATORS.items():
        series = _series(iso, code, year - 4, year - 1)  # strictly before election year
        out[name] = series[max(series)] if series else None
    if all(out.get(k) is not None for k in ("inflation", "growth", "unemployment")):
        out["misery"] = (
            float(out["inflation"]) + float(out["unemployment"]) - float(out["growth"])  # type: ignore[arg-type]
        )
    else:
        out["misery"] = None
    _save_cache(key, out)
    return out
