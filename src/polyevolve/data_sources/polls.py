"""Wikipedia opinion-poll text - a point-in-time POLLING signal for elections.

Why this exists: polls are the single strongest naive predictor of an election,
and for obscure foreign races they live on the local-language Wikipedia "Opinion
polling for the … election" article - exactly the source a US Polymarket trader
won't read. They are:
  - $0 (free MediaWiki API, no key);
  - LOCAL-LANGUAGE aware: we search the country's OWN-language wiki first
    (Italy->it, Japan->ja, ...), then English, via the same FIPS->lang map the
    pageviews source uses, so coverage stays in sync;
  - leakage-safe ONLY VIA REVISION HISTORY. The *live* poll page is silently
    post-edited (a poll with a pre-D fieldwork date may have been ADDED after D,
    or a value corrected), so scraping the current page LEAKS. The fix, and the
    whole point of this source: fetch the article's wikitext AS IT WAS at the
    latest revision STRICTLY BEFORE as_of, and hard-assert that revision's
    timestamp < as_of. That gives the polls exactly as the page showed them on D.

What it does, per market (question + as_of + tags):
  1. Build search terms from the question's entities (reuse the news source's
     `_match_terms`) + "election"/"opinion polling" + any year in the question,
     and search the local-language wiki first (then en) via the MediaWiki search
     API for the relevant ELECTION article (prefer an "Opinion polling for …"
     article when one matches).
  2. POINT-IN-TIME via the revisions API: fetch the latest revision whose
     timestamp is STRICTLY BEFORE as_of (rvstart=as_of, rvdir=older), and reject
     any revision >= as_of. (as_of None -> current revision.)
  3. From that revision's wikitext, locate the opinion-polling section by its
     heading (multilingual: "Opinion polls", "Polling", "Sondaggi", "Umfragen",
     ...) and lightly clean the wiki markup to readable text, truncated to a size
     the LLM can read (~2200 chars) so the model itself reads the recent polls -
     we deliberately do NOT brittle-parse every country's bespoke table format.
  4. render() a compact, LLM-readable block - distinguishing error / no-data /
     found exactly like the other sources (fail loud, never silent-empty).

Fail-soft per step: a search miss or a missing polling section is no-data, not a
crash. We fail LOUD only on hard API errors (every request erroring at the
transport level), so "no poll article exists" can never masquerade as "we failed
to look".
"""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import USER_AGENT
from .gdelt_bq import tags_to_fips
from .gdelt_bq_news import _match_terms
from .pageviews import _FALLBACK_LANG, _FIPS_TO_WIKI_LANG

logger = logging.getLogger(__name__)

# Wikimedia's User-Agent policy asks for a descriptive UA with contact info (see config).
_USER_AGENT = USER_AGENT

_TIMEOUT_S = 15.0
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0

# Wikipedia revision timestamps are ISO-8601 Zulu, e.g. "2024-05-30T19:37:44Z".
_REV_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# How many candidate articles to consider from one search, and how much polling
# text to hand the LLM. ~2200 chars comfortably covers the most-recent rows of a
# reverse-chronological poll table without blowing up the prompt.
_SEARCH_LIMIT = 6
_MAX_POLL_CHARS = 3000
# Keep at most this many of the NEWEST poll rows (tables are reverse-chronological,
# newest first) so the cleaned block fits the char budget without slicing a row
# mid-number.
_MAX_POLL_ROWS = 14

# Section headings, per language, that mark the opinion-polling block. Matched
# case-insensitively as a substring of a heading. English is always included as a
# fallback because local-language articles often borrow English headings.
_POLL_HEADINGS: dict[str, tuple[str, ...]] = {
    "en": ("opinion poll", "polling", "polls"),
    "it": ("sondaggi",),
    "de": ("umfrage", "umfragen", "meinungsumfrage"),
    "nl": ("peiling",),
    "es": ("encuesta", "sondeo"),
    "fr": ("sondage",),
    "el": ("δημοσκοπ",),  # δημοσκόπηση / δημοσκοπήσεις
    "uk": ("опитуван",),  # опитування
    "ru": ("опрос",),
    "fa": ("نظرسنجی",),
    "he": ("סקר",),
    "ja": ("世論調査", "情勢調査"),
    "ko": ("여론 조사", "여론조사"),
    "ar": ("استطلاع",),
    "th": ("โพล", "สำรวจ"),
    "km": ("ការស្ទង់មតិ",),
    "hi": ("सर्वेक्षण", "मतदान"),
    "hu": ("közvélemény", "felmérés"),
    "zh": ("民意調查", "民調", "民意调查"),
}

# Election-article search hints per language: appended to the entity terms so the
# search lands on the election / polling article rather than a candidate's bio.
_ELECTION_HINTS: dict[str, tuple[str, ...]] = {
    "en": ("election", "opinion polling"),
    "it": ("elezioni", "sondaggi"),
    "de": ("wahl", "umfragen"),
    "nl": ("verkiezingen", "peilingen"),
    "es": ("elecciones", "encuestas"),
    "fr": ("élection", "sondages"),
    "el": ("εκλογές",),
    "uk": ("вибори",),
    "ru": ("выборы",),
    "fa": ("انتخابات",),
    "he": ("בחירות",),
    "ja": ("選挙",),
    "ko": ("선거",),
    "ar": ("انتخابات",),
    "th": ("การเลือกตั้ง",),
    "km": ("ការបោះឆ្នោត",),
    "hi": ("चुनाव",),
    "hu": ("választás",),
    "zh": ("選舉",),
}


class WikipediaPollsSource:
    """Point-in-time Wikipedia opinion-poll text. Source key: 'polls'."""

    name = "polls"

    def __init__(
        self,
        http: httpx.Client | None = None,
        max_entities: int = 4,
        max_poll_chars: int = _MAX_POLL_CHARS,
    ) -> None:
        self._http = http or httpx.Client(timeout=_TIMEOUT_S, headers={"User-Agent": _USER_AGENT})
        self._max_entities = max_entities
        self._max_poll_chars = max_poll_chars

    def fetch(self, context: dict[str, Any]) -> dict[str, Any]:
        question = context.get("question", "")
        if not question:
            return {"article": None, "query": "", "error": "empty_question"}

        as_of = context.get("as_of")
        if as_of is not None and not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime or None")
        cutoff = as_of
        if cutoff is not None and cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)

        terms = _match_terms(question, max_terms=self._max_entities)
        langs = _wiki_langs(context.get("tags") or [])
        year = _extract_year(question)
        # Acceptable election years for the relevance gate: the question's year if
        # stated, else the as_of year; plus the following year (an early-year
        # election is often polled the year before and titled by the election year).
        base_year = year or (str(cutoff.year) if cutoff else None)
        years_ok = {base_year, str(int(base_year) + 1)} if base_year else set()
        query_desc = " | ".join(_search_query(terms, year, langs[0]))

        any_request_ok = False
        for lang in langs:
            query = _search_query(terms, year, lang)
            titles, ok = self._search(lang, query)
            any_request_ok = any_request_ok or ok
            for title in titles:
                if not _title_relevant(title, years_ok, terms, lang):
                    continue
                rev, rok = self._revision_before(lang, title, cutoff)
                any_request_ok = any_request_ok or rok
                if rev is None:
                    continue
                wikitext, rev_ts = rev
                section = _extract_poll_section(wikitext, lang)
                # Prefer the dense NUMERIC poll table; the section text otherwise
                # leads with bias/caption prose and truncates the real numbers off.
                table_text = _extract_poll_tables(wikitext, section)
                text = (table_text or (_clean_wikitext(section) if section else "")).strip()
                text = text[: self._max_poll_chars]
                # Caption/bias-only text (no real poll figures) is worse than nothing
                # - it makes the LLM fabricate numbers. Require >=2 poll figures
                # (percent OR bare numbers), else treat as no-data so the forecaster
                # falls back to its prior.
                if _figure_count(text) < 2:
                    continue
                return {
                    "article": title,
                    "lang": lang,
                    "revision_ts": rev_ts.isoformat(),
                    "as_of": cutoff.isoformat() if cutoff else None,
                    "polls_text": text,
                    "query": query_desc,
                }

        # Fail LOUD only if EVERY API call errored at the transport level (so we
        # can't tell "no article/section" from "API down"). A clean search/empty
        # result is legitimate no-data, not an error.
        if not any_request_ok:
            return {"article": None, "query": query_desc, "error": "api_unreachable"}

        return {
            "article": None,
            "query": query_desc,
            "as_of": cutoff.isoformat() if cutoff else None,
        }

    def _search(self, lang: str, query: list[str]) -> tuple[list[str], bool]:
        """Search one wiki; return (candidate titles, request_ok).

        Titles are ordered to prefer an "opinion polling"-style article (which
        holds the table) over the bare election article, then by search rank.
        request_ok is True if the HTTP call completed (incl. empty results);
        False only on transport/5xx exhaustion.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": " ".join(query),
            "srlimit": _SEARCH_LIMIT,
            "srnamespace": 0,
            "format": "json",
            "formatversion": 2,
        }
        data, ok = self._api_get(lang, params)
        if data is None:
            return [], ok
        hits = data.get("query", {}).get("search", [])
        titles = [h["title"] for h in hits if "title" in h]
        # Prefer a polling-aggregation article (it carries the table) over the
        # plain election article, preserving search rank within each group.
        poll_kw = _POLL_HEADINGS.get(lang, ()) + _POLL_HEADINGS["en"]
        polling = [t for t in titles if any(k in t.lower() for k in poll_kw)]
        rest = [t for t in titles if t not in polling]
        return polling + rest, ok

    def _revision_before(
        self, lang: str, title: str, cutoff: datetime | None
    ) -> tuple[tuple[str, datetime] | None, bool]:
        """Fetch the latest revision STRICTLY BEFORE cutoff (or current if None).

        Returns ((wikitext, revision_timestamp), request_ok). The leakage guard:
        the returned revision's timestamp is verified < cutoff and any revision
        at/after cutoff is rejected (returns None) - a revision we should never
        have been able to see is never used.
        """
        params: dict[str, Any] = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": 1,
            "rvdir": "older",
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": 2,
        }
        if cutoff is not None:
            # rvstart with rvdir=older = "newest revision at or before this time".
            # We pass exactly as_of and then enforce STRICTLY-before below.
            params["rvstart"] = cutoff.strftime(_REV_TS_FMT)
        data, ok = self._api_get(lang, params)
        if data is None:
            return None, ok
        pages = data.get("query", {}).get("pages", [])
        if not pages or "revisions" not in pages[0] or not pages[0]["revisions"]:
            return None, ok
        rev = pages[0]["revisions"][0]
        rev_ts = _parse_rev_ts(rev.get("timestamp", ""))
        if rev_ts is None:
            return None, ok
        # LEAKAGE GUARD: never use a revision at or after as_of.
        if cutoff is not None and rev_ts >= cutoff:
            logger.warning(
                "polls: rejected revision %s of %r (ts %s >= as_of %s)",
                rev.get("revid"),
                title,
                rev_ts.isoformat(),
                cutoff.isoformat(),
            )
            return None, ok
        content = rev.get("slots", {}).get("main", {}).get("content")
        if not isinstance(content, str):
            return None, ok
        return (content, rev_ts), ok

    def _api_get(self, lang: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        """GET the MediaWiki API for one wiki with light retry/backoff.

        Returns (json, request_ok). request_ok is False only on transport/5xx
        exhaustion; a clean response with empty results returns (json, True).
        """
        url = f"https://{lang}.wikipedia.org/w/api.php"
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._http.get(url, params=params)
            except httpx.HTTPError:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            try:
                resp.raise_for_status()
                return resp.json(), True
            except (httpx.HTTPError, ValueError):
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
        return None, False

    def render(self, payload: dict[str, Any]) -> str:
        query = payload.get("query", "")

        error = payload.get("error")
        if error:
            return f"[SOURCE ERROR] polls fetch failed ({error})"

        if not payload.get("article"):
            return f"(No Wikipedia polling article/section found for: {query})"

        as_of = payload.get("as_of")
        guard = f" < {as_of}" if as_of else " (current)"
        return (
            f"Opinion polls (Wikipedia '{payload['article']}' {payload['lang']}, "
            f"revision as of {payload['revision_ts']}{guard}):\n"
            f"{payload['polls_text']}"
        )


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without network).
# ---------------------------------------------------------------------------


def _wiki_langs(tags: list[str]) -> list[str]:
    """Country tags -> ordered list of Wikipedia langs to search.

    Local language(s) first (de-duplicated, in tag order), then English as a
    secondary. Always non-empty: a market with no mapped country gets English.
    Mirrors pageviews._wiki_langs so coverage stays in sync.
    """
    langs: list[str] = []
    for code in tags_to_fips(tags):
        lang = _FIPS_TO_WIKI_LANG.get(code)
        if lang and lang not in langs:
            langs.append(lang)
    if _FALLBACK_LANG not in langs:
        langs.append(_FALLBACK_LANG)
    return langs


def _extract_year(question: str) -> str | None:
    """Pull a 4-digit election year (19xx/20xx) from the question, if present.

    The year sharply disambiguates "2026 Venice municipal election" from prior
    cycles. Returns the LAST such year (questions usually trail with the year).
    """
    years = re.findall(r"\b(19|20)\d{2}\b", question)
    if not years:
        return None
    # findall with a group returns only the group; re-scan for the full match.
    full = re.findall(r"\b(?:19|20)\d{2}\b", question)
    return full[-1] if full else None


def _search_query(terms: list[str], year: str | None, lang: str) -> list[str]:
    """Build the MediaWiki search query tokens for a wiki.

    Entity terms from the question + a year (if any) + language-appropriate
    election/polling hints, so the search ranks the election/polling article
    first. Always non-empty (falls back to the hints alone if no entities).
    """
    hints = _ELECTION_HINTS.get(lang, _ELECTION_HINTS["en"])
    query: list[str] = list(terms)
    if year:
        query.append(year)
    query.extend(hints)
    return query


def _parse_rev_ts(timestamp: str) -> datetime | None:
    """Parse an ISO-8601 Zulu revision timestamp to an aware UTC datetime."""
    try:
        return datetime.strptime(timestamp, _REV_TS_FMT).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return None


def _extract_poll_section(wikitext: str, lang: str) -> str | None:
    """Return the wikitext of the opinion-polling section, or None.

    Finds the first section whose heading (==..== / ===..===) contains a known
    polling keyword for `lang` (or English, as a fallback), and returns that
    section plus its immediate sub-sections, up to the next heading of equal or
    higher level. Returns None when no polling heading is present.
    """
    keywords = _POLL_HEADINGS.get(lang, ()) + _POLL_HEADINGS["en"]
    headings = list(re.finditer(r"^(={2,6})[ \t]*(.+?)[ \t]*=*\s*$", wikitext, re.M))
    for idx, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip().lower()
        if not any(k in title for k in keywords):
            continue
        body_start = m.end()
        # Section runs until the next heading of the SAME or HIGHER level (so we
        # keep nested per-year sub-tables, which are deeper headings).
        body_end = len(wikitext)
        for nxt in headings[idx + 1 :]:
            if len(nxt.group(1)) <= level:
                body_end = nxt.start()
                break
        return wikitext[body_start:body_end].strip()
    return None


# A poll figure: 42%, 42.5%, 42,5% (EU decimal comma), "42 %".
_RE_PCT = re.compile(r"\d{1,3}(?:[.,]\d)?\s*%")
# A bare poll figure WITHOUT a "%": many tables (HU, NL seat tables) store the
# party share/seat-count as a bold integer like '''39''' with no percent sign.
# Constrained to 1-3 digit numbers not glued to a word char, a "." or a "%" (so
# we don't (a) re-count a "%" figure, (b) catch the integer part of "22.0%", or
# (c) catch years/ids embedded in words). 0-3 digits with an optional single
# decimal covers shares like 5, 39, 22.0, 51,2.
_RE_NUM = re.compile(r"(?<![\w.,])\d{1,3}(?:[.,]\d)?(?![\w%.,])")
_RE_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _figure_count(text: str) -> int:
    """Count poll FIGURES in a chunk: percent figures PLUS bare poll numbers.

    Bare-number tables (no "%") are common (Hungary seat/share table, Dutch seat
    table) and the old %-only count rejected them as garbage. We count "%" figures
    and, separately, bare 1-3 digit numbers (years stripped, to avoid fieldwork
    dates inflating the count)."""
    pct = len(_RE_PCT.findall(text))
    no_year = _RE_YEAR.sub(" ", text)
    bare = len(_RE_NUM.findall(no_year))
    return pct + bare


def _title_relevant(title: str, years_ok: set[str], terms: list[str], lang: str) -> bool:
    """Reject obviously-wrong search hits before we trust their tables.

    Two guards that together kill the observed failures (a question pulling the
    *2006/2014* election's poll table, or a Dutch question pulling 'Coronacrisis in
    Nederland'): (1) a title carrying an explicit year NOT in the acceptable set
    (the question's year, or the as_of year ± the next year, since an election can
    fall the year after the snapshot) is the wrong election; (2) a kept title must
    look election/polling-related - an election/polling heading word or an
    overlapping entity term.
    """
    low = title.lower()
    title_years = set(_RE_YEAR.findall(title))
    if years_ok and title_years and not (title_years & years_ok):
        return False
    hints = (
        _POLL_HEADINGS.get(lang, ())
        + _POLL_HEADINGS["en"]
        + _ELECTION_HINTS.get(lang, ())
        + _ELECTION_HINTS["en"]
    )
    if any(h in low for h in hints):
        return True
    return any(t.lower() in low for t in terms if len(t) > 3)


def _find_wikitables(text: str) -> list[str]:
    """Return the raw wikitext of each top-level {| ... |} table (nesting-aware)."""
    tables: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("{|", i):
            depth, j = 0, i
            while j < n:
                if text.startswith("{|", j):
                    depth += 1
                    j += 2
                elif text.startswith("|}", j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            tables.append(text[i:j])
            i = j
        else:
            i += 1
    return tables


# A poll RESULTS row carries one figure per party in its OWN CELL, so a real row
# has many standalone-numeric cells; a bias-list / caption / sample-size column
# has at most a couple (its figures are buried inside prose cells). Require this
# many numeric CELLS in a row before it counts as a poll row (kills the 'alleged
# bias' table whose prose rows are figure-dense but cell-sparse).
_POLL_ROW_FIGURE_FLOOR = 5

# A standalone poll-value cell: a percentage, a bare 1-3 digit share/seat number,
# or a dash placeholder ("-"/"–"/"—"/"N/A"). Used to count the numeric cells in a
# row so prose cells (a bias description, an event note) do not count.
_RE_NUMERIC_CELL = re.compile(
    r"^(?:'''|''|\s)*"  # optional leading bold/italic/space
    r"(?:\d{1,3}(?:[.,]\d{1,2})?\s*%?|[-–—]|n/?a)"  # number / pct / dash / n/a
    r"(?:'''|''|\s)*$",
    re.I,
)


def _split_row_cells(row: str) -> list[str]:
    """Cleaned cell values of one table row (cells may be ||-joined OR one-per-line).

    Mirrors the cell-splitting used by the cleaner: split each physical line on
    `||`/`!!`, strip the leading marker, and keep each cell's readable value."""
    cells: list[str] = []
    for line in row.splitlines():
        s = line.strip()
        if not s or s.startswith("{|") or s.startswith("|}"):
            continue
        if s.startswith("!") or s.startswith("|"):
            s = s.lstrip("!|").strip()
            for chunk in re.split(r"\s*(?:\|\||!!)\s*", s):
                val = _cell_value(chunk)
                if val:
                    cells.append(val)
    return cells


def _numeric_cell_count(row: str) -> int:
    """Number of standalone-numeric cells in a row (poll figures in their own cell)."""
    return sum(1 for c in _split_row_cells(row) if _RE_NUMERIC_CELL.match(c.strip()))


# A cell whose readable text is essentially just an election year - "2006",
# "2006-os" (HU suffix), "2006-os3" (trailing ref-marker), "2006年" - the row key
# of a historical results table, NOT a poll row (keyed on pollster / day-date).
_RE_YEAR_CELL = re.compile(r"^(?:19|20)\d{2}\D{0,5}\d?$")
_RE_LINK_LABEL = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]")


def _first_cell_year_rows(raw_table: str) -> int:
    """Count rows whose FIRST cell is (essentially) a bare election year.

    The election-results-table fingerprint: each row keyed on a year (often a
    wikilink labelled with the year, sometimes with a ref-marker suffix),
    distinguishing it from a poll table keyed on a pollster name or a day-level
    fieldwork date. We reduce each row's first cell to its readable text and test
    it against the year shape.
    """
    count = 0
    for row in raw_table.split("|-"):
        # The first data line of the row (skip table-open / attribute lines).
        first_cell = ""
        for line in row.splitlines():
            s = line.strip()
            if not s or s.startswith("{|") or s.startswith("|}") or s.startswith("!"):
                continue
            if s.startswith("|"):
                first_cell = _cell_value(s.lstrip("|"))
                break
        if not first_cell:
            continue
        # Unwrap a [[link|label]] to its label, drop refs / HTML / bold, then test.
        first_cell = _RE_LINK_LABEL.sub(r"\1", first_cell)
        first_cell = _RE_REF.sub("", first_cell)
        first_cell = _RE_HTML_TAG.sub("", first_cell).replace("'''", "").strip()
        if _RE_YEAR_CELL.match(first_cell):
            count += 1
    return count


def _poll_table_score(raw_table: str) -> int:
    """Heuristic density score: a real poll-RESULTS table is full of figures
    spread across many rows (one per party per poll). A pollster 'alleged bias'
    list or a chart-caption block has almost none, so it scores ~0 and is skipped.

    Figures are counted %-OPTIONALLY (bare-integer tables are real poll tables),
    but a %-bearing table still outscores an equally-dense bare table so that, when
    both a seat table and a percentage table exist, the percentage one is preferred.

    A historical election-RESULTS table (rows keyed on an election year, not a
    pollster/fieldwork date) is rejected: it is numeric-dense but is past results,
    not polls, and otherwise hijacks the routing when the search lands on a party
    page instead of the polling article.
    """
    rows = raw_table.split("|-")
    poll_rows = sum(1 for r in rows if _numeric_cell_count(r) >= _POLL_ROW_FIGURE_FLOOR)
    if poll_rows == 0:
        return 0
    # Rows whose first cell is a bare election year -> results table fingerprint.
    # If most numeric rows are year-keyed, it is a historical results table, not a
    # poll table; reject it (it otherwise hijacks routing on a party-bio page).
    year_keyed = _first_cell_year_rows(raw_table)
    if year_keyed >= 3 and year_keyed * 2 >= poll_rows:
        return 0
    base = _figure_count(raw_table) + 5 * poll_rows
    # PREFER a %-bearing table when one exists alongside a bare (seat) table: if
    # the figures are predominantly percentages, scale the score up so a real
    # percentage poll table outranks a seat-count table of the same shape.
    pct = len(_RE_PCT.findall(raw_table))
    if pct * 2 >= _figure_count(raw_table) and pct >= poll_rows:
        return base * 3
    return base


def _extract_poll_tables(wikitext: str, section: str | None) -> str | None:
    """Cleaned text of the densest NUMERIC poll table (newest rows first).

    Looks inside the matched poll section first, then the whole article (so a
    heading match that landed on a bias/intro subsection still finds the real
    results table elsewhere). Returns None if no table has real numeric density -
    captions/bias prose are worse than nothing (they induce the LLM to fabricate
    figures), so the caller treats that as no-data rather than feeding them.
    """
    for scope in (section, wikitext):
        if not scope:
            continue
        scored = sorted(
            ((_poll_table_score(t), t) for t in _find_wikitables(scope)),
            key=lambda x: -x[0],
        )
        if scored and scored[0][0] >= 10:
            return _trim_poll_rows(_clean_wikitext(scored[0][1]).strip())
    return None


def _trim_poll_rows(cleaned: str) -> str:
    """Keep the header row plus the newest _MAX_POLL_ROWS poll rows.

    Poll tables are reverse-chronological (newest first), so after the header the
    first data rows are the most recent - exactly what the forecaster needs. The
    header is the first line that carries no poll figures (party labels); rows that
    follow with >=_POLL_ROW_FIGURE_FLOOR figures are poll rows."""
    lines = [ln for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return cleaned
    # Always keep the first line (the party-label header) as context; then keep
    # following lines until we have collected _MAX_POLL_ROWS actual poll rows.
    kept: list[str] = [lines[0]]
    data_rows = 0
    for ln in lines[1:]:
        if _figure_count(ln) >= _POLL_ROW_FIGURE_FLOOR:
            data_rows += 1
            if data_rows > _MAX_POLL_ROWS:
                break
        kept.append(ln)
    return "\n".join(kept)


# Pre-compiled markup patterns for the cleaner (applied in order).
_RE_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
_RE_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RE_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_RE_FILE_LINK = re.compile(r"\[\[(?:File|Image|Datei|Immagine|Fichier):[^\]]*\]\]", re.I)

# Wikitext image-option tokens that are NOT a caption/label (sizes, positioning,
# the `link=`/`alt=` keyword args). Anything else trailing a File-link is its
# human caption (e.g. the party abbreviation in a poll-table column header).
_RE_IMG_SIZE = re.compile(r"^\d+\s*px$|^x\d+px$|^\d+x\d+px$", re.I)
_IMG_KEYWORDS = {
    "thumb",
    "thumbnail",
    "frame",
    "frameless",
    "border",
    "center",
    "centre",
    "left",
    "right",
    "none",
    "top",
    "middle",
    "bottom",
    "baseline",
    "text-top",
    "text-bottom",
    "super",
    "sub",
    "upright",
    "class",
}


def _file_link_label(match: re.Match[str]) -> str:
    """Party label for a File-image link in a poll-table header cell, or "".

    Column headers store the party as a logo: `[[File:Logo.svg|35px|link=Tisza
    Party|TISZA]]` (caption "TISZA") or `[[File:Fidesz.svg|35px|link=Fidesz–KDNP]]`
    (no caption -> fall back to the `link=` target). A purely decorative image
    (`[[File:chart.svg|thumb|880px]]`) has neither -> "". We run this BEFORE
    deleting File-links so the party labels survive into the header row."""
    inner = match.group(0)[2:-2]  # strip [[ ]]
    parts = inner.split("|")
    # parts[0] is "File:Name.svg"; the rest are options/caption.
    link_target = ""
    for opt in parts[1:]:
        o = opt.strip()
        low = o.lower()
        if low.startswith("link="):
            link_target = o[5:].strip()
            continue
        if low.startswith("alt=") or "=" in o:
            continue
        if _RE_IMG_SIZE.match(low) or low in _IMG_KEYWORDS:
            continue
        if o:
            return o  # an explicit caption label wins
    return link_target


# HTML/wikitable cell attributes (style=, rowspan=2, bgcolor=…) that precede the
# cell's actual value behind a single pipe, or appear as a bare attribute token.
_RE_ATTR_KV = re.compile(
    r"\b(?:class|style|width|height|colspan|rowspan|align|valign|scope|bgcolor|"
    r"color|data-sort-value|id)\s*=\s*(?:\"[^\"]*\"|'[^']*'|\S*)",
    re.I,
)


def _cell_value(cell: str) -> str:
    """Readable value of one table cell.

    A cell may be `attrs | value` (the single pipe separates HTML attributes from
    content). Keep the part after the LAST attribute pipe, strip any leftover bare
    attribute tokens, and return the trimmed value ("" if nothing readable
    remains, so pure-styling cells drop out)."""
    # If a single-pipe attribute prefix is present, keep the content after it.
    if "|" in cell:
        head, _, tail = cell.partition("|")
        # Only treat the head as attributes if it looks like attributes (has '='
        # or is a short style token); otherwise keep the whole thing.
        if "=" in head or not head.strip():
            cell = tail
    cell = _RE_ATTR_KV.sub("", cell)
    return cell.strip(" |!")


def _strip_templates(text: str) -> str:
    """Remove {{...}} templates, handling nesting, but keep a poll-relevant few.

    Most templates are styling/colour noise ({{party color|...}}, {{n/a}}). A
    handful carry the actual datum we want (a percentage / date), so for a couple
    of common shapes we keep the last pipe-arg rather than dropping the template.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth = 0
            j = i
            while j < n:
                if text.startswith("{{", j):
                    depth += 1
                    j += 2
                elif text.startswith("}}", j):
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            inner = text[i + 2 : j - 2] if j - 2 > i + 2 else ""
            out.append(_template_value(inner))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


_KEEP_TEMPLATES = ("nowrap", "decrease", "increase", "steady", "small", "0")

# Fieldwork-DATE templates: a poll row's date is often wrapped in one of these,
# e.g. {{opdrts|26|29|Sep|2025|year}} -> "26-29 Sep 2025". We join the date args
# and drop the trailing control flags ("year", "df", "abbr", ...) so the date
# survives instead of being dropped with the rest of the template noise.
_DATE_TEMPLATES = ("opdrts", "opdr", "dts", "date", "start date", "end date")
_DATE_TEMPLATE_FLAGS = {"year", "df", "abbr", "abbrev", "br", "nbsp"}


def _date_template_value(parts: list[str]) -> str:
    """Render a date template's args to a readable date.

    {{opdrts|26|29|Sep|2025|year}} -> "26-29 Sep 2025": numeric day args before the
    month collapse to "d1-d2", non-flag tokens are kept in order. Control flags
    ("year", "df", ...) and named args (a=b) are dropped."""
    args = [p for p in parts[1:] if p and p.lower() not in _DATE_TEMPLATE_FLAGS and "=" not in p]
    if not args:
        return ""
    # Leading run of bare day-numbers -> a "d1-d2" range; the rest (month, year)
    # follow space-joined.
    days: list[str] = []
    i = 0
    while i < len(args) and args[i].isdigit() and len(args[i]) <= 2:
        days.append(args[i])
        i += 1
    rest = args[i:]
    day_part = "-".join(days)
    return " ".join(p for p in (day_part, *rest) if p).strip()


def _split_top_pipes(inner: str) -> list[str]:
    """Split template args on top-level '|', NOT pipes inside a [[wikilink|x]].

    {{Nowrap|[[GroenLinks–PvdA|GL/PvdA]]}} must yield the whole wikilink as one
    arg (then the link unwrap reduces it to "GL/PvdA"); a naive split would cut it
    at the link's own pipe and leak a dangling "]]"."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i, n = 0, len(inner)
    while i < n:
        if inner.startswith("[[", i):
            depth += 1
            buf.append("[[")
            i += 2
        elif inner.startswith("]]", i):
            depth = max(0, depth - 1)
            buf.append("]]")
            i += 2
        elif inner[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(inner[i])
            i += 1
    parts.append("".join(buf))
    return parts


def _template_value(inner: str) -> str:
    """Best-effort readable value for a stripped template's inner text.

    For value-carrying wrappers like {{nowrap|12.3}} keep the last arg; for
    fieldwork-date templates render the date; otherwise drop entirely (colour /
    formatting templates contribute nothing readable).
    """
    parts = [p.strip() for p in _split_top_pipes(inner)]
    if not parts:
        return ""
    name = parts[0].lower()
    if name in _DATE_TEMPLATES:
        return _date_template_value(parts)
    if name in _KEEP_TEMPLATES and len(parts) > 1:
        return parts[-1]
    # A bare numeric template arg (e.g. {{#expr:...}}) -> nothing useful.
    return ""


def _clean_wikitext(section: str) -> str:
    """Lightly convert poll-section wikitext into readable text for the LLM.

    Not a full parser: strips refs/comments/HTML/templates, turns File-image
    party logos into their party label, unwraps [[link|label]] to the label, drops
    table styling attributes, and - crucially - reconstructs ROW STRUCTURE.

    Poll tables frequently put each cell on its OWN line behind a single leading
    `|` (not `||`-joined), so a naive line-by-line pass scatters one number per
    output line and destroys pollster/date/party alignment. We instead split each
    table on its `|-` ROW separators first, then within a row split cells on BOTH
    `||`/`!!` AND every newline-leading `|`/`!`, emitting ONE line per poll row
    with cells joined by " | ".
    """
    t = section
    t = _RE_COMMENT.sub("", t)
    t = _RE_REF.sub("", t)
    # Recover the party label from header File-image logos BEFORE deleting them,
    # otherwise the column party labels (Fidesz/TISZA/...) vanish from the header.
    t = _RE_FILE_LINK.sub(_file_link_label, t)
    t = _strip_templates(t)
    # [[target|label]] -> label ; [[target]] -> target.
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]", r"\1", t)
    # External links [http://x label] -> label ; [http://x] -> "".
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = _RE_HTML_TAG.sub("", t)
    t = t.replace("'''", "").replace("''", "")

    out_lines: list[str] = []
    # A "logical row" accumulates cells until the next `|-` separator (or a table
    # open/close), so cells spread over several physical lines join into one row.
    row_cells: list[str] = []

    def flush_row() -> None:
        cells = [c for c in row_cells if c]
        if cells:
            out_lines.append(" | ".join(cells))
        row_cells.clear()

    for raw in t.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Row separator -> end the current logical row.
        if line.startswith("|-"):
            flush_row()
            continue
        # Table open/close markup carries no data; treat as a row boundary too.
        if line.startswith("{|") or line.startswith("|}"):
            flush_row()
            continue
        # A header/data line: one OR many cells. Strip the leading marker, then
        # split on `||`/`!!` (same-line cells). Each physical line is appended to
        # the current logical row, so single-`|`-per-line cells accumulate.
        if line.startswith("!") or line.startswith("|"):
            line = line.lstrip("!|").strip()
            for chunk in re.split(r"\s*(?:\|\||!!)\s*", line):
                val = _cell_value(chunk)
                if val:
                    row_cells.append(val)
            continue
        # Free prose inside the section (not a table) -> flush any row, keep line.
        flush_row()
        out_lines.append(line)
    flush_row()

    # Collapse runs of blank lines (none are emitted now, but keep the guard) and
    # trim. Lines are already one-per-row.
    cleaned: list[str] = []
    for line in out_lines:
        if not line and (not cleaned or not cleaned[-1]):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()
