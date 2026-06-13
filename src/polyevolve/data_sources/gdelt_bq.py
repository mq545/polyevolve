"""GDELT via BigQuery - historical, point-in-time news at batch scale.

The Doc API can't serve our use case: it's capped to ~90 days of history and
rate-limits hard, so historical snapshot builds were impossible (the 0/150
empty-context disaster). The full GDELT 2.0 dataset is a PUBLIC BigQuery dataset
(`gdelt-bq.gdeltv2`), which gives arbitrary historical date cutoffs and no
per-request rate limit - billed only on bytes scanned.

COST DISCIPLINE (the only real risk here):
  - GKG is ~3.6 TB; a `SELECT *` with no date filter scans all of it (~$22).
  - We ALWAYS partition-prune on `_PARTITIONTIME` (only the window before a
    market's cutoff) and select a handful of columns.
  - Every query also sets `maximum_bytes_billed` as a HARD ceiling, so a buggy
    query FAILS instead of silently scanning a terabyte.

This is implemented as a DataSource (same Protocol as GdeltSource): fetch() runs
one pruned query per market; render() turns the aggregate into prompt text.

NOTE: the GKG table is keyed by article URL (`DocumentIdentifier`) and a 15-min
`DATE` timestamp - there is NO clean headline field. Its rich signal is themes
(`V2Themes`), named entities (`V2Persons`/`V2Locations`), and tone (`V2Tone`).
So context here is "volume + dominant themes + tone + sample URLs", which for
forecasting is arguably richer than headlines (it captures what CHANGED).

Schema/columns MUST be verified against the live table before trusting this
query - run scripts/bq_probe.py first. Treat the SQL below as provisional until
that probe confirms column names.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Public GDELT dataset - partitioned variant (prune on _PARTITIONTIME).
_GKG_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# Hard per-query ceiling. A correct pruned query scans well under this; a query
# missing its date filter would try to scan ~3.6 TB and will instead ERROR here.
# 50 GB ~= 5% of the 1 TiB free monthly tier per query.
_MAX_BYTES_BILLED = int(os.environ.get("BQ_MAX_BYTES_BILLED", str(50 * 1024**3)))

# How far back before each market's as_of to gather coverage.
_LOOKBACK_DAYS = int(os.environ.get("BQ_LOOKBACK_DAYS", "14"))


class GdeltBigQuerySource:
    name = "gdelt_bq"

    def __init__(
        self,
        client: Any | None = None,
        project: str | None = None,
        max_records: int = 8,
        lookback_days: int = _LOOKBACK_DAYS,
    ) -> None:
        # Lazy import so the rest of the app doesn't hard-depend on the BQ client
        # (and so tests can inject a fake client without it installed).
        self._client = client
        self._project = project or os.environ.get("GCP_PROJECT")
        self._max_records = max_records
        self._lookback_days = lookback_days

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

        # Entities come from the market's curated TAGS (clean country labels),
        # mapped to GDELT's FIPS 10-4 country codes - NOT parsed from the question
        # (which turned "May" into an entity). Codes verified against GDELT, see
        # _TAG_TO_FIPS. Falls back to question proper-nouns only if no country tag.
        tags = context.get("tags") or []
        codes = tags_to_fips(tags)
        if codes:
            # Require ALL mapped countries to CO-OCCUR in V2Locations (AND), and
            # match the structured #CC# code token, not a loose substring (so
            # "%LE%" can't match "Lebanese" or random text). This is the precision
            # fix: an Israel-Lebanon market matches only Israel+Lebanon articles.
            predicate = " AND ".join(f"V2Locations LIKE @c{i}" for i in range(len(codes)))
            match_params = [
                bigquery.ScalarQueryParameter(f"c{i}", "STRING", f"%#{code}#%")
                for i, code in enumerate(codes)
            ]
            query_desc = " + ".join(codes)
        else:
            # No country tag (e.g. topic-only market) - fall back to proper-noun
            # OR match across entity columns. Looser, but flagged in query_desc.
            entities = _question_to_entities(question)
            if not entities:
                return {"articles": [], "query": "", "error": "no_entities"}
            predicate = "(" + _entity_predicate(len(entities)) + ")"
            match_params = [
                bigquery.ScalarQueryParameter(f"e{i}", "STRING", f"%{ent}%")
                for i, ent in enumerate(entities)
            ]
            query_desc = "fallback:" + " OR ".join(entities)

        sql = f"""
            SELECT
              DocumentIdentifier AS url,
              DATE AS seendate,
              V2Themes AS themes,
              V2Tone AS tone
            FROM `{_GKG_TABLE}`
            WHERE _PARTITIONTIME >= TIMESTAMP(@start)
              AND _PARTITIONTIME <  TIMESTAMP(@cutoff)
              AND DocumentIdentifier IS NOT NULL
              AND {predicate}
            ORDER BY DATE DESC
            LIMIT @maxrec
        """
        params = [
            bigquery.ScalarQueryParameter("start", "STRING", start.strftime("%Y-%m-%d")),
            bigquery.ScalarQueryParameter("cutoff", "STRING", cutoff.strftime("%Y-%m-%d %H:%M:%S")),
            bigquery.ScalarQueryParameter("maxrec", "INT64", self._max_records * 4),
            *match_params,
        ]

        job_config = bigquery.QueryJobConfig(
            query_parameters=params,
            maximum_bytes_billed=_MAX_BYTES_BILLED,
        )
        try:
            client = self._get_client()
            job = client.query(sql, job_config=job_config)
            rows = list(job.result())
            bytes_billed = int(job.total_bytes_billed or 0)
        except Exception as exc:  # noqa: BLE001 - fail LOUD, never silent-empty
            logger.exception("bigquery gdelt fetch failed")
            return {
                "articles": [],
                "query": " OR ".join(entities),
                "error": f"bq_error: {exc!r}"[:200],
            }

        articles = [
            {
                "url": r["url"],
                "seendate": str(r["seendate"]),
                "themes": (r["themes"] or "")[:300],
                "tone": _first_tone(r["tone"]),
            }
            for r in rows
        ]
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
            return f"[SOURCE ERROR] gdelt_bq fetch failed ({error})"

        articles = payload.get("articles", [])
        if not articles:
            return f"(GDELT/BigQuery searched but found no coverage for: {query}{suffix})"

        tones = [a["tone"] for a in articles if a.get("tone") is not None]
        avg_tone = sum(tones) / len(tones) if tones else None
        lines = [
            f"News coverage (GDELT/BigQuery, entities={query}{suffix}): "
            f"{len(articles)} articles"
            + (f", avg tone {avg_tone:+.1f}" if avg_tone is not None else "")
        ]
        for a in articles:
            lines.append(f"- {a['seendate']} {a['url']}")
        return "\n".join(lines)


# Curated map: our market TAGS -> GDELT FIPS 10-4 country code. EVERY code here
# was verified against live GDELT V2Locations data 2026-05-31 (NOT typed from
# memory - FIPS differs from ISO for many: Ukraine UP not UA, S.Korea KS not KR,
# Russia RS not RU, Ireland EI, Japan JA, China CH, Cambodia CB, Australia AS,
# Iraq IZ). Only the countries appearing in fp_v1 tags are mapped (YAGNI - extend
# as new domains arrive). Topic tags (politics, elections, ...) have no entry and
# are correctly ignored.
_TAG_TO_FIPS: dict[str, str] = {
    "iran": "IR",
    "israel": "IS",
    "ukraine": "UP",
    "ukraine-map": "UP",
    "hungary": "HU",
    "hungary-election": "HU",
    "netherlands": "NL",
    "dutch": "NL",
    "dutch-election": "NL",
    "south-korea": "KS",
    "canada": "CA",
    "lebanon": "LE",
    "ireland": "EI",
    "irish": "EI",
    "india": "IN",
    "indian-elections": "IN",
    "japan": "JA",
    "venezuela": "VE",
    "thailand": "TH",
    "thailand-election": "TH",
    "cambodia": "CB",
    "italy": "IT",
    "cyprus": "CY",
    "australia": "AS",
    "mexico": "MX",
    "china": "CH",
    "qatar": "QA",
    "taiwan": "TW",
    "russia": "RS",
    "iraq": "IZ",
}


def tags_to_fips(tags: list[str], max_codes: int = 3) -> list[str]:
    """Map market tags to a de-duplicated list of GDELT FIPS country codes.

    Only tags present in the verified _TAG_TO_FIPS map contribute; topic tags are
    ignored. Capped at max_codes (a market spanning >3 countries is rare and the
    AND co-occurrence would over-narrow). Order preserved for determinism.
    """
    codes: list[str] = []
    for t in tags:
        code = _TAG_TO_FIPS.get(t.lower())
        if code and code not in codes:
            codes.append(code)
            if len(codes) >= max_codes:
                break
    return codes


def _question_to_entities(question: str, max_terms: int = 4) -> list[str]:
    """Extract proper-noun-ish entity terms from a market question.

    Mirrors the Doc-API builder's intent (entities carry the signal) but returns
    a list so the SQL can OR them across GKG's entity columns.
    """
    import re  # noqa: PLC0415

    stop = {"Will", "The", "A", "An"}
    words = re.findall(r"[A-Z][A-Za-z'-]+", question)  # capitalized tokens
    out: list[str] = []
    for w in words:
        if w in stop or w in out:
            continue
        out.append(w)
        if len(out) >= max_terms:
            break
    return out


def _entity_predicate(n: int) -> str:
    """OR of LIKE clauses across the entity-bearing GKG columns for n entities."""
    cols = ["V2Persons", "V2Organizations", "V2Locations", "V2Themes"]
    clauses = []
    for i in range(n):
        for c in cols:
            clauses.append(f"{c} LIKE @e{i}")
    return " OR ".join(clauses)


def _first_tone(v2tone: str | None) -> float | None:
    """V2Tone is a comma-list; the first value is the overall document tone."""
    if not v2tone:
        return None
    try:
        return float(v2tone.split(",")[0])
    except (ValueError, IndexError):
        return None
