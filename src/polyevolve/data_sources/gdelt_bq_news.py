"""BQ-discovery news source: BigQuery finds relevant URLs, we scrape the content.

Why this exists: the GDELT DOC 2.0 API rate-limits per IP and put us in a sticky
multi-day penalty box, so DOC-based snapshot builds stall. BigQuery's GKG table
has NO rate limit (billed on bytes, partition-pruned to ~$0 within the free tier),
so it's the throttle-free way to DISCOVER article URLs at scale - and we get the
actual article TEXT the same way the DOC source does: by scraping (scraping.py).

This fixes the original gdelt_bq failure (country-firehose noise) by matching the
QUESTION'S ENTITIES (people/orgs/places, proper nouns) in V2Persons /
V2Organizations / V2Locations - not just the country - so the URLs are on-topic.
GKG has no headline/body, but we don't need it to: we scrape the URL for content.

Same source key ("news") and rendered shape as GdeltDocSource, so snapshots and
the genome's data_weights are interchangeable across the two discovery backends.

Point-in-time: partition-prune on _PARTITIONTIME < as_of AND filter DATE < as_of
(GKG's 15-min stamp), so no article seen at/after the cutoff can enter.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from .gdelt_bq import tags_to_fips
from .processing import dedup_articles, rank_articles
from .scraping import fetch_article_text

# FIPS country code -> set of ccTLDs that signal a LOCAL (in-country) publisher.
# Used only for a SOFT ranking boost (the thesis prefers local press over English
# wire). Keys mirror the verified _TAG_TO_FIPS map in gdelt_bq.py; values are the
# country's ccTLD plus common local-news second-level domains (.co.kr, .co.il).
# Not exhaustive - a missing entry just means "no local boost", never a filter.
_FIPS_TO_TLD: dict[str, tuple[str, ...]] = {
    "IR": (".ir",),
    "IS": (".il", ".co.il"),
    "UP": (".ua",),
    "HU": (".hu",),
    "NL": (".nl",),
    "KS": (".kr", ".co.kr"),
    "CA": (".ca",),
    "LE": (".lb",),
    "EI": (".ie",),
    "IN": (".in", ".co.in"),
    "JA": (".jp", ".co.jp"),
    "VE": (".ve",),
    "TH": (".th", ".co.th"),
    "CB": (".kh",),
    "IT": (".it",),
    "CY": (".cy", ".com.cy"),
    "AS": (".au", ".com.au"),
    "MX": (".mx", ".com.mx"),
    "CH": (".cn",),
    "QA": (".qa",),
    "TW": (".tw", ".com.tw"),
    "RS": (".ru",),
    "IZ": (".iq",),
}
# Domains/TLDs that are English-language international wire - the thesis wants to
# DOWN-weight these relative to local press. A soft penalty, not a filter.
_WIRE_TLDS: tuple[str, ...] = (".com", ".org", ".net", ".co.uk", ".uk", ".gov")
_WIRE_DOMAINS: frozenset[str] = frozenset(
    {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "cnn.com",
        "nytimes.com",
        "washingtonpost.com",
        "theguardian.com",
        "bloomberg.com",
        "aljazeera.com",
        "ft.com",
        "wsj.com",
        "politico.com",
        "politico.eu",
        "foxnews.com",
        "nbcnews.com",
        "abcnews.go.com",
    }
)

logger = logging.getLogger(__name__)

_GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"
_MAX_BYTES_BILLED = int(os.environ.get("BQ_MAX_BYTES_BILLED", str(30 * 1024**3)))
_LOOKBACK_DAYS = int(os.environ.get("BQ_LOOKBACK_DAYS", "21"))
_GKG_DT_FMT = "%Y%m%d%H%M%S"


class GdeltBqNewsSource:
    """BigQuery entity-discovery + scraping. Source key: 'news'."""

    name = "news"

    def __init__(
        self,
        client: Any | None = None,
        project: str | None = None,
        max_records: int = 10,
        scrape_top_k: int = 8,
        lookback_days: int = _LOOKBACK_DAYS,
        scrape: bool = True,
    ) -> None:
        self._client = client
        self._project = project or os.environ.get("GCP_PROJECT")
        self._max_records = max_records
        self._scrape_top_k = scrape_top_k
        self._lookback_days = lookback_days
        self._scrape = scrape

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import bigquery  # noqa: PLC0415 - lazy by design

            self._client = bigquery.Client(project=self._project)
        return self._client

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"articles": [], "query": "", "error": "empty_question"}

        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")
        cutoff = as_of if as_of is not None else datetime.now(UTC)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        start = cutoff - timedelta(days=self._lookback_days)

        from google.cloud import bigquery  # noqa: PLC0415

        # Relevance is the whole game (GKG has no relevance ranking, only recency).
        # OR-ing loose single tokens + recency = noise (a "Korea" market matched a
        # Supreme Court story). So we extract DISTINCTIVE entities and require
        # CO-OCCURRENCE (AND), case-insensitively, across the entity columns:
        #   - prefer multi-word proper-noun PHRASES ("Bank of Korea", "Andrea
        #     Martella") - a single such phrase is usually precise on its own;
        #   - else AND the two longest single proper nouns;
        #   - else fall back to country FIPS co-occurrence.
        terms = _match_terms(question)
        params: list[Any] = []
        order_expr = "DATE DESC"
        if terms:
            # Recall-oriented OR match across entity columns, but RANK by how many
            # distinct query terms an article hits (a relevance proxy GKG lacks):
            # an article mentioning more of the question's entities sorts first, so
            # a "KRG + Iraq" market surfaces KRG-independence pieces over the Iraq
            # firehose. Recency only breaks ties.
            cols = ["V2Persons", "V2Organizations", "V2Locations"]
            per_term = []
            for i, term in enumerate(terms):
                params.append(bigquery.ScalarQueryParameter(f"e{i}", "STRING", f"%{term.lower()}%"))
                per_term.append("(" + " OR ".join(f"LOWER({c}) LIKE @e{i}" for c in cols) + ")")
            predicate = "(" + " OR ".join(per_term) + ")"
            score = " + ".join(f"CAST({clause} AS INT64)" for clause in per_term)
            order_expr = f"({score}) DESC, DATE DESC"
            query_desc = " | ".join(terms)
        else:
            codes = tags_to_fips(context.get("tags") or [])
            if not codes:
                return {"articles": [], "query": "", "error": "no_entities"}
            predicate = " AND ".join(f"V2Locations LIKE @c{i}" for i in range(len(codes)))
            params = [
                bigquery.ScalarQueryParameter(f"c{i}", "STRING", f"%#{code}#%")
                for i, code in enumerate(codes)
            ]
            query_desc = "country:" + "+".join(codes)

        sql = f"""
            SELECT DocumentIdentifier AS url, DATE AS seendate, V2Tone AS tone
            FROM `{_GKG_TABLE}`
            WHERE _PARTITIONTIME >= TIMESTAMP(@start)
              AND _PARTITIONTIME <  TIMESTAMP(@cutoff)
              AND DATE < @cutoff_int
              AND DocumentIdentifier IS NOT NULL
              AND {predicate}
            ORDER BY {order_expr}
            LIMIT @maxrec
        """
        params += [
            bigquery.ScalarQueryParameter("start", "STRING", start.strftime("%Y-%m-%d")),
            bigquery.ScalarQueryParameter("cutoff", "STRING", cutoff.strftime("%Y-%m-%d %H:%M:%S")),
            bigquery.ScalarQueryParameter("cutoff_int", "INT64", int(cutoff.strftime(_GKG_DT_FMT))),
            bigquery.ScalarQueryParameter("maxrec", "INT64", self._max_records * 5),
        ]
        job_config = bigquery.QueryJobConfig(
            query_parameters=params, maximum_bytes_billed=_MAX_BYTES_BILLED
        )
        try:
            job = self._get_client().query(sql, job_config=job_config)
            rows = list(job.result())
            bytes_billed = int(job.total_bytes_billed or 0)
        except Exception as exc:  # noqa: BLE001 - fail LOUD, never silent-empty
            logger.exception("bq news discovery failed")
            return {"articles": [], "query": query_desc, "error": f"bq_error: {exc!r}"[:200]}

        articles = [
            {
                "url": r["url"],
                "seendate": str(r["seendate"]),
                "domain": _domain(r["url"]),
                "language": "",
                "tone": _first_tone(r["tone"]),
            }
            for r in rows
        ]
        articles = rank_articles(dedup_articles(articles), max_n=self._max_records * 2)

        if self._scrape:
            # PRECISION GATE: discovery (BQ) is recall-oriented and can't certify
            # relevance, so judge the actual scraped TEXT against the question.
            # We score each body with a WEIGHTED lexical relevance (distinctive /
            # rare question terms - proper-noun entities, long topic nouns - count
            # for more than a country mention), and add a SOFT local-source boost
            # so in-country press outranks English wire. This drops off-topic
            # matches the GKG structure can't (the EU-sanctions-for-a-strike fail).
            weights = _term_weights(question)
            tlds = _local_tlds(context.get("tags") or [])
            min_score = _min_relevance(weights)
            relevant: list[tuple[float, dict[str, Any]]] = []
            scraped_rest: list[tuple[float, dict[str, Any]]] = []  # body, below gate
            for a in articles[: self._scrape_top_k * 2]:
                body = fetch_article_text(a.get("url", ""))
                if not body:
                    continue
                a["text"] = body
                rel = _relevance_score(body, weights)
                boost = _locality_boost(a.get("domain", ""), tlds)
                ranked = rel + boost
                if rel >= min_score:
                    relevant.append((ranked, a))
                else:
                    # Keep the locality-adjusted score so backfill still prefers
                    # local press among the (English-gate-rejected) remainder.
                    scraped_rest.append((ranked, a))
            relevant.sort(key=lambda t: t[0], reverse=True)
            kept = [a for _, a in relevant]
            # The English relevance gate can wrongly reject LOCAL-LANGUAGE bodies
            # (Dutch "kabinet", Korean text) - exactly the coverage the thesis
            # wants. GKG discovery is already language-normalized (proper-noun names
            # match across languages), so backfill from the best-ranked scraped
            # remainder rather than emptying a foreign-language market.
            if len(kept) < self._scrape_top_k:
                scraped_rest.sort(key=lambda t: t[0], reverse=True)
                kept += [a for _, a in scraped_rest[: self._scrape_top_k - len(kept)]]
            articles = kept[: self._scrape_top_k]
        else:
            articles = articles[: self._max_records]

        return {
            "articles": articles,
            "query": query_desc,
            "as_of": cutoff.isoformat(),
            "_bytes_billed": bytes_billed,
        }

    def render(self, payload: dict[str, Any]) -> str:
        query = payload.get("query", "")
        as_of = payload.get("as_of")
        suffix = f", as of {as_of}" if as_of else ""

        error = payload.get("error")
        if error:
            return f"[SOURCE ERROR] news fetch failed ({error})"

        articles = payload.get("articles", [])
        if not articles:
            return f"(BigQuery searched but found no coverage for: {query}{suffix})"

        lines = [f"Relevant news (GDELT/BigQuery discovery, entities={query}{suffix}):"]
        for a in articles:
            lines.append(f"\n• ({a['domain']}, {a['seendate']})")
            body = a.get("text")
            if body:
                lines.append(f"  {body}")
            else:
                lines.append(f"  {a['url']}")
        return "\n".join(lines)


_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
# Capitalized question words / generic nouns that are NOT entities - never start or
# join a phrase, never used as a match term.
_STOP = {
    "will",
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "is",
    "are",
    "does",
    "did",
    "do",
    "the",
    "a",
    "an",
    "by",
    "before",
    "after",
    "win",
    "wins",
}
# Lowercase connectors that may sit BETWEEN two capitalized words in one entity.
_CONNECTORS = {"of", "the", "and", "de", "da", "van", "von", "del", "la", "le", "du", "di", "al"}


def _is_entity_cap(tok: str) -> bool:
    if not tok or not tok[0].isupper():
        return False
    return tok.lower() not in _MONTHS and tok.lower() not in _STOP


def _match_terms(question: str, max_terms: int = 4) -> list[str]:
    """Distinctive entity terms for a recall-oriented, relevance-RANKED GKG match.

    Returns BOTH multi-word proper-noun phrases ("Bank of Korea", "Supreme Leader
    of Iran") AND distinctive single proper nouns / acronyms ("Khamenei", "KRG").
    The caller ORs them (recall) but ranks results by how many terms each article
    hits - so the discriminating name pulls the on-topic article to the top even
    when a generic title phrase also matches. Up to max_terms, longest first.
    """
    import re  # noqa: PLC0415

    toks = re.findall(r"[A-Za-z][A-Za-z'’-]*", question)
    phrases: list[str] = []
    i = 0
    while i < len(toks):
        if not _is_entity_cap(toks[i]):
            i += 1
            continue
        parts = [toks[i]]
        k = i + 1
        while k < len(toks):
            if _is_entity_cap(toks[k]) or (
                toks[k].lower() in _CONNECTORS and k + 1 < len(toks) and _is_entity_cap(toks[k + 1])
            ):
                parts.append(toks[k])
                k += 1
            else:
                break
        if sum(1 for p in parts if p[0].isupper()) >= 2:
            phrases.append(" ".join(parts))
        i = k
    # Single proper nouns / acronyms (>=3 chars so KRG/EU survive), excluding any
    # word already inside a phrase (avoid double-weighting the same entity).
    phrase_words = {w.lower() for ph in phrases for w in ph.split()}
    singles = [
        t for t in toks if _is_entity_cap(t) and len(t) >= 3 and t.lower() not in phrase_words
    ]
    terms = list(dict.fromkeys(phrases)) + list(dict.fromkeys(singles))
    return sorted(terms, key=len, reverse=True)[:max_terms]


_TOPIC_STOP = (
    _STOP
    | _MONTHS
    | {
        "win",
        "wins",
        "next",
        "most",
        "make",
        "change",
        "election",
        "elections",
        "vote",
        "votes",
        "out",
        "new",
        "base",
        "rate",
        "before",
        "after",
        "first",
        "this",
        "that",
        "with",
        "from",
        "into",
        "year",
        "seats",
        "than",
    }
)


# Weighting tiers for the scraped-body relevance gate. Distinctive question
# terms (proper-noun entities, the specific event/topic nouns) must be PRESENT in
# the body for it to count as on-topic; a bare country mention must NOT suffice
# (the EU-sanctions-for-an-Israel/Iran-strike failure). So entities outweigh
# generic topic words, which outweigh nothing.
_ENTITY_WEIGHT = 3.0
_TOPIC_WEIGHT = 1.0
# A LOCAL (in-country ccTLD) publisher gets this soft additive boost; an English
# international-wire domain gets this soft penalty. Tuned to re-order ties, not to
# override a real relevance difference (entity weight 3.0 dominates).
_LOCAL_BOOST = 1.0
_WIRE_PENALTY = -0.5


def _term_weights(question: str) -> dict[str, float]:
    """Map distinctive lowercased question terms to relevance weights.

    Proper-noun ENTITIES (people/orgs/places/acronyms, via _match_terms - phrases
    are split into their words so "Bank of Korea" contributes korea+bank) weigh
    _ENTITY_WEIGHT; other content TOPIC words weigh _TOPIC_WEIGHT. Fillers, months
    and country-generic stopwords are dropped. The body's relevance is the sum of
    the weights of the DISTINCT terms it contains - so an article that names the
    question's specific actor/event scores far above one that only shares a country.
    """
    import re  # noqa: PLC0415

    weights: dict[str, float] = {}
    for term in _match_terms(question):
        for w in term.lower().split():
            w = _strip_possessive(w)
            if len(w) >= 3 and w not in _TOPIC_STOP:
                weights[w] = _ENTITY_WEIGHT
    for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", question.lower()):
        w = _strip_possessive(w)
        if len(w) >= 4 and w not in _TOPIC_STOP and w not in weights:
            weights[w] = _TOPIC_WEIGHT
    return weights


def _strip_possessive(word: str) -> str:
    """Drop a trailing English possessive ('s / ') so the stem substring-matches
    the body - "Iran's" must match a body that says only "Iran"."""
    for suffix in ("'s", "’s", "'", "’"):
        if word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _relevance_score(body: str, weights: dict[str, float]) -> float:
    """Sum of weights of DISTINCT weighted terms that appear in the body text.

    Pure and language-agnostic for proper nouns (names match across languages), so
    a local-language article that names the question's entities still scores."""
    low = body.lower()
    return sum(w for term, w in weights.items() if term in low)


def _min_relevance(weights: dict[str, float]) -> float:
    """Gate threshold: require the body to hit at least one distinctive ENTITY,
    or (when the question has no entity) two topic words. This is what excludes the
    only-mentions-the-country off-topic article that the flat word-count let pass."""
    if any(w >= _ENTITY_WEIGHT for w in weights.values()):
        return _ENTITY_WEIGHT
    return 2.0 * _TOPIC_WEIGHT if len(weights) >= 2 else _TOPIC_WEIGHT


def _local_tlds(tags: list[str]) -> tuple[str, ...]:
    """ccTLD suffixes that mark a local (in-country) publisher for these tags."""
    out: list[str] = []
    for code in tags_to_fips(tags):
        out.extend(_FIPS_TO_TLD.get(code, ()))
    return tuple(out)


def _locality_boost(domain: str, local_tlds: tuple[str, ...]) -> float:
    """Soft additive ranking adjustment: reward in-country local press, mildly
    penalize English international wire. Never a hard filter - the thesis wants
    local-language local coverage to OUTRANK the US/UK wire, not to drop the wire.
    """
    d = domain.lower()
    if not d:
        return 0.0
    if any(d == w or d.endswith("." + w) for w in _WIRE_DOMAINS):
        return _WIRE_PENALTY
    if local_tlds and any(d.endswith(t) for t in local_tlds):
        return _LOCAL_BOOST
    # Non-US/non-UK ccTLD (anything that isn't a generic wire TLD) is plausibly
    # more local than .com wire - a small nudge up.
    if not any(d.endswith(t) for t in _WIRE_TLDS):
        return _LOCAL_BOOST * 0.5
    return _WIRE_PENALTY * 0.5


def _domain(url: str) -> str:
    from urllib.parse import urlparse  # noqa: PLC0415

    try:
        return urlparse(url).netloc
    except ValueError:
        return ""


def _first_tone(v2tone: str | None) -> float | None:
    if not v2tone:
        return None
    try:
        return float(v2tone.split(",")[0])
    except (ValueError, IndexError):
        return None
