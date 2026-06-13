"""Shared article post-processing: dedup + ranking.

This is the seam the GDELT "Agentic AI At Scale" article argues is mandatory:
raw article COUNT is fake signal because news is massively syndicated - 5,000
articles can be one wire story rewritten. A forecaster handed 8 copies of the
same report infers false corroboration, which is *worse* than empty context.

So every article-based DataSource runs its results through here BEFORE rendering:
  1. dedup_articles  - collapse near-identical stories (wire copies, rewrites)
  2. rank_articles   - cap to the top-N, preferring source/domain diversity

Kept deliberately simple (normalized-title + token-Jaccard dedup; diversity-
greedy ranking). The function boundary is the real deliverable: when we add
embedding-based relevance ranking later, only rank_articles changes - sources
and the registry don't.
"""

from __future__ import annotations

import re
from typing import Any

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace - for dup comparison."""
    t = _PUNCT_RE.sub(" ", title.lower())
    return _WS_RE.sub(" ", t).strip()


def _token_set(normalized_title: str) -> frozenset[str]:
    return frozenset(normalized_title.split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_articles(
    articles: list[dict[str, Any]], *, jaccard_threshold: float = 0.8
) -> list[dict[str, Any]]:
    """Collapse near-identical articles (syndication, rewrites, translations).

    Two articles are treated as the same story if their normalized titles are
    identical OR share >= jaccard_threshold of their word tokens. The first
    occurrence is kept and annotated with `dup_count` (how many raw articles it
    represents) - so a source can HONESTLY surface "covered by N outlets" without
    letting raw volume masquerade as independent corroboration.
    """
    kept: list[dict[str, Any]] = []
    kept_tokens: list[frozenset[str]] = []
    for art in articles:
        norm = _normalize_title(art.get("title", ""))
        if not norm:
            # Untitled: can't dedup reliably; keep as-is.
            kept.append({**art, "dup_count": 1})
            kept_tokens.append(frozenset())
            continue
        toks = _token_set(norm)
        match_idx = None
        for i, kt in enumerate(kept_tokens):
            if kt == toks or _jaccard(toks, kt) >= jaccard_threshold:
                match_idx = i
                break
        if match_idx is None:
            kept.append({**art, "dup_count": 1})
            kept_tokens.append(toks)
        else:
            kept[match_idx]["dup_count"] += 1
    return kept


def rank_articles(articles: list[dict[str, Any]], *, max_n: int) -> list[dict[str, Any]]:
    """Return the top max_n articles, greedily preferring domain diversity.

    Diversity matters because the goal is breadth of independent perspective, not
    N takes from one outlet. Within that, higher dup_count (a more widely-carried
    story) and original order (the source's own sort, e.g. recency) break ties.

    SEAM: this is where embedding/relevance ranking plugs in later. For now it's
    a deterministic diversity-greedy pass - no model calls, fully reproducible.
    """
    if max_n <= 0 or not articles:
        return []

    # Stable sort by dup_count desc; original order is preserved for ties because
    # Python's sort is stable, so a more-syndicated (more salient) story floats up
    # without discarding the source's own recency ordering.
    by_salience = sorted(articles, key=lambda a: a.get("dup_count", 1), reverse=True)

    selected: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    # First pass: one article per domain (maximize distinct outlets).
    for art in by_salience:
        if len(selected) >= max_n:
            break
        domain = art.get("domain", "")
        if domain and domain in seen_domains:
            continue
        seen_domains.add(domain)
        selected.append(art)
    # Second pass: backfill remaining slots if diversity left us short.
    if len(selected) < max_n:
        chosen = {id(a) for a in selected}
        for art in by_salience:
            if len(selected) >= max_n:
                break
            if id(art) not in chosen:
                selected.append(art)
    return selected
