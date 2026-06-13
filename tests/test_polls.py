"""Tests for the Wikipedia opinion-poll (point-in-time) source.

Mostly pure-function tests (no network): the polling-section extraction, the
wiki-markup cleaner, the search-query construction, the country->wiki-lang
mapping, the year extraction, and render states. Mocked fetch() proves the
leakage guard end-to-end: a revision AT/AFTER as_of is never used. One live test
hits the real MediaWiki API for a known election article (skipped offline)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from polyevolve.data_sources.polls import (
    WikipediaPollsSource,
    _cell_value,
    _clean_wikitext,
    _extract_poll_section,
    _extract_poll_tables,
    _extract_year,
    _figure_count,
    _parse_rev_ts,
    _poll_table_score,
    _search_query,
    _strip_templates,
    _template_value,
    _wiki_langs,
)

CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)


# --- country -> wiki-lang mapping (mirrors pageviews) ----------------------


def test_wiki_langs_local_then_en() -> None:
    assert _wiki_langs(["italy"]) == ["it", "en"]
    assert _wiki_langs(["japan"]) == ["ja", "en"]
    assert _wiki_langs(["cyprus"]) == ["el", "en"]
    assert _wiki_langs(["ireland"]) == ["en"]
    assert _wiki_langs(["politics"]) == ["en"]


# --- year extraction -------------------------------------------------------


def test_extract_year() -> None:
    assert _extract_year("Will Andrea Martella win the 2026 Venice mayoral election?") == "2026"
    assert _extract_year("Will the 2024 and 2026 races align?") == "2026"  # last year wins
    assert _extract_year("Who wins the next general election?") is None


# --- search-query construction --------------------------------------------


def test_search_query_en_adds_year_and_hints() -> None:
    q = _search_query(["Andrea Martella", "Venice"], "2026", "en")
    assert q[:2] == ["Andrea Martella", "Venice"]
    assert "2026" in q
    assert "election" in q and "opinion polling" in q


def test_search_query_local_lang_hints() -> None:
    q = _search_query(["Venezia"], "2026", "it")
    assert "elezioni" in q
    assert "sondaggi" in q
    assert "2026" in q


def test_search_query_no_entities_still_non_empty() -> None:
    q = _search_query([], None, "en")
    assert q == ["election", "opinion polling"]


# --- revision timestamp parse ----------------------------------------------


def test_parse_rev_ts() -> None:
    ts = _parse_rev_ts("2024-05-30T19:37:44Z")
    assert ts == datetime(2024, 5, 30, 19, 37, 44, tzinfo=UTC)
    assert _parse_rev_ts("garbage") is None
    assert _parse_rev_ts("") is None


# --- polling-section extraction --------------------------------------------

_WIKI = """\
Intro text.

== Background ==
Some background prose, no polls here.

== Opinion polls ==
{| class="wikitable"
|-
! Fieldwork date !! Pollster !! Party A !! Party B
|-
| 2024-05-20 || YouGov || 32 || 28
|}

=== 2023 ===
older rows here

== Results ==
This must NOT be included.
"""


def test_extract_poll_section_finds_english_heading() -> None:
    section = _extract_poll_section(_WIKI, "en")
    assert section is not None
    assert "YouGov" in section
    assert "older rows here" in section  # nested sub-section kept
    assert "This must NOT be included" not in section  # stopped at next == heading
    assert "Some background prose" not in section


def test_extract_poll_section_italian_heading() -> None:
    wt = "== Sondaggi ==\n| 2024 || SWG || 30 |\n== Note ==\nx"
    section = _extract_poll_section(wt, "it")
    assert section is not None
    assert "SWG" in section


def test_extract_poll_section_none_when_absent() -> None:
    wt = "== Background ==\nno polling section at all\n== Results ==\nx"
    assert _extract_poll_section(wt, "en") is None


# --- wikitext cleaning -----------------------------------------------------


def test_clean_strips_links_refs_templates() -> None:
    raw = (
        "| 2024-05-20 || [[YouGov]] || [[Brothers of Italy|FdI]] "
        "{{nowrap|32.1}}<ref name=x>cite</ref> || {{party color|PD}}28"
    )
    out = _clean_wikitext(raw)
    assert "YouGov" in out
    assert "FdI" in out  # [[target|label]] -> label
    assert "32.1" in out  # kept template value
    assert "ref" not in out.lower()  # ref stripped
    assert "party color" not in out  # styling template dropped
    assert "[[" not in out and "{{" not in out


def test_clean_drops_table_style_attrs() -> None:
    raw = '| style="background:red" | 32 || class="x" | 28'
    out = _clean_wikitext(raw)
    assert out == "32 | 28"
    assert "background" not in out and "class" not in out


def test_clean_drops_unquoted_rowspan_and_bgcolor() -> None:
    # The live-table case: `rowspan=2 | Label` and bare `bgcolor=...` cells.
    raw = "| rowspan=2 | Data di rilevazione || bgcolor=#fff | 23,4"
    out = _clean_wikitext(raw)
    assert out == "Data di rilevazione | 23,4"
    assert "rowspan" not in out and "bgcolor" not in out


def test_cell_value_keeps_content_after_attr_pipe() -> None:
    assert _cell_value("rowspan=2 | Istituto") == "Istituto"
    assert _cell_value('style="x" | 12.3') == "12.3"
    assert _cell_value("plain value") == "plain value"  # no attr pipe -> unchanged
    assert _cell_value("bgcolor=#abc") == ""  # pure attribute -> nothing readable


def test_clean_renders_cells_with_separator() -> None:
    raw = "! Pollster !! Party A !! Party B"
    out = _clean_wikitext(raw)
    assert out == "Pollster | Party A | Party B"


def test_strip_templates_nested() -> None:
    # Nested template must be fully consumed (no stray braces leak through).
    assert "{" not in _strip_templates("a {{outer|{{inner|x}}}} b")
    assert _strip_templates("a {{nowrap|9.9}} b").strip() == "a 9.9 b"


def test_strip_templates_keeps_wikilink_inside_template() -> None:
    # {{Nowrap|[[GroenLinks–PvdA|GL/PvdA]]}} must NOT be cut at the wikilink's own
    # pipe (that leaked a dangling "]]" into the header). The whole link is kept.
    out = _clean_wikitext("! {{Nowrap|[[GroenLinks–PvdA|GL/PvdA]]}}")
    assert out == "GL/PvdA"
    assert "]]" not in out


# --- bare-number tables (the HU/NL bug) ------------------------------------

# A Hungary-style results table: party shares are BARE BOLD INTEGERS (no '%'),
# each cell on its OWN line behind a single leading '|', party labels live in the
# header as File-image logos ([[File:...|link=Party|LABEL]]). The old %-only gate
# rejected this as garbage; the fix must accept it and keep the structure.
_HU_TABLE = """\
{| class="wikitable"
|-
! rowspan="2" |Fieldwork date
! rowspan="2" |Polling firm
! rowspan="2" |Sample size
! [[File:Fidesz_2015.svg|35px|link=Fidesz–KDNP]]
! [[File:Logo.svg|35px|link=Tisza Party|TISZA]]
! [[File:DK.svg|35px|link=Democratic Coalition|DK]]
! [[File:MKKP.svg|35px|link=Two-Tailed Dog|MKKP]]
! [[File:MH.svg|35px|link=Our Homeland|MH]]
! rowspan="2" |Lead
|-
|7–13 Jan 2026
|[https://hvg.hu/x Medián]
|1,000
|'''39'''
|style="background:#CCF4FC" |'''51'''
|1
|5
|3
|style="background:#fff" |12
|-
|5–8 Jan 2026
|[https://x.hu/y Alapjogokért Központ]
|1,000
|'''49'''
|'''41'''
|2
|6
|2
|8
|}"""


def test_bare_integer_table_accepted_and_scored() -> None:
    # Bare-integer poll figures count toward the figure total and the table scores
    # well above the >=10 acceptance threshold (it used to score ~1 and be dropped).
    assert _figure_count("'''39''' | '''51''' | 1 | 12") >= 4
    assert _poll_table_score(_HU_TABLE) >= 10


def test_bare_integer_table_rows_aligned_with_header() -> None:
    out = _extract_poll_tables(_HU_TABLE, _HU_TABLE)
    assert out is not None
    lines = out.splitlines()
    # Header row carries the party labels recovered from the File-image logos.
    assert "Fidesz–KDNP" in lines[0]
    assert "TISZA" in lines[0] and "DK" in lines[0]
    # A poll row keeps pollster + date + the bare party numbers on ONE line.
    median = next(line for line in lines if "Medián" in line)
    assert "7–13 Jan 2026" in median
    assert "39" in median and "51" in median  # Fidesz / TISZA bare integers


def test_one_cell_per_line_rows_are_joined() -> None:
    # NL-style: each cell on its own line behind a single '|'. The cleaner must
    # join them into one row, not scatter one number per output line.
    nl_row = (
        '{| class="wikitable"\n'
        "|-\n"
        "! Polling firm !! Fieldwork date !! PVV !! VVD\n"
        "|-\n"
        "| Verian\n"
        "| {{opdrts|26|29|Sep|2025|year}}\n"
        "| style=\"background:#CDE2FE;\" | '''22.0%'''\n"
        "| 14.6%\n"
        "|}"
    )
    out = _clean_wikitext(nl_row)
    row = next(line for line in out.splitlines() if "Verian" in line)
    assert row.count("|") == 3  # Verian | date | 22.0% | 14.6%  -> 3 separators
    assert "22.0%" in row and "14.6%" in row
    assert "26-29 Sep 2025" in row


def test_opdrts_date_template_rendered() -> None:
    assert _template_value("opdrts|26|29|Sep|2025|year") == "26-29 Sep 2025"
    assert _template_value("opdrts|6|Sep|2025|year") == "6 Sep 2025"
    # A plain date template keeps its readable parts (space-joined, flags dropped).
    assert _template_value("dts|2025|9|6") == "2025 9 6"


def test_party_label_header_from_file_logos_preserved() -> None:
    header = (
        "! [[File:Fidesz_2015.svg|35px|link=Fidesz–KDNP]] "
        "!! [[File:Logo.svg|35px|link=Tisza Party|TISZA]] "
        "!! [[File:chart.svg|thumb|880px]]"
    )
    out = _clean_wikitext(header)
    assert "Fidesz–KDNP" in out  # no caption -> link target kept
    assert "TISZA" in out  # explicit caption label kept
    assert "File" not in out and "880px" not in out  # decorative image dropped


# --- election-results table guard (party-bio routing trap) ------------------

_RESULTS_TABLE = """\
{| class="wikitable"
! Election !! Votes !! Share !! Seats !! Role
|-
| [[2010 election|2010]] || 2 706 292 || 52,73% || 263/386 || government
|-
| [[2014 election|2014]] || 2 264 780 || 45,04% || 133/199 || government
|-
| [[2018 election|2018]] || 2 824 647 || 49,60% || 133/199 || government
|-
| [[2022 election|2022]] || 3 060 706 || 54,13% || 135/199 || government
|}"""


def test_election_results_table_rejected() -> None:
    # A party-page historical-RESULTS table is numeric-dense but keyed on election
    # YEARS, not pollsters; it must score 0 so it can't hijack routing.
    assert _poll_table_score(_RESULTS_TABLE) == 0
    assert _extract_poll_tables(_RESULTS_TABLE, _RESULTS_TABLE) is None


def test_bias_prose_table_rejected() -> None:
    # An 'alleged bias' table: figures are buried in PROSE cells, so it has few
    # standalone-numeric cells per row and must score 0 (no poll rows).
    bias = (
        '{| class="wikitable"\n'
        "! Pollster !! Alleged bias\n"
        "|-\n"
        "| Medián || close to the opposition since 2018, founded in 1990s\n"
        "|-\n"
        "| Nézőpont || strong links to the government, funded since 2010\n"
        "|}"
    )
    assert _poll_table_score(bias) == 0


# --- mocked fetch: leakage guard end to end --------------------------------


class _StubClient:
    """Minimal httpx.Client stand-in for the MediaWiki API.

    Routes by request params: list=search returns `search_titles`; prop=revisions
    returns the revision the test wants (timestamp + content). A revision
    timestamp can be set AT/AFTER as_of to prove the leakage guard rejects it.
    """

    def __init__(
        self,
        *,
        search_titles: list[str],
        rev_ts: str,
        rev_content: str,
        revid: int = 123,
    ) -> None:
        self.search_titles = search_titles
        self.rev_ts = rev_ts
        self.rev_content = rev_content
        self.revid = revid
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        params = params or {}
        self.requests.append({"url": url, **params})
        req = httpx.Request("GET", url, params=params)
        if params.get("list") == "search":
            hits = [{"title": t} for t in self.search_titles]
            return httpx.Response(200, json={"query": {"search": hits}}, request=req)
        if params.get("prop") == "revisions":
            page = {
                "title": params.get("titles"),
                "revisions": [
                    {
                        "revid": self.revid,
                        "timestamp": self.rev_ts,
                        "slots": {"main": {"content": self.rev_content}},
                    }
                ],
            }
            return httpx.Response(200, json={"query": {"pages": [page]}}, request=req)
        return httpx.Response(200, json={"query": {}}, request=req)


_POLL_WIKITEXT = (
    "== Opinion polls ==\n"
    "| 2024-05-20 || YouGov || FdI 30% || PD 22%\n"
    "| 2024-05-10 || Ipsos || FdI 29% || PD 23%\n"
    "== Results ==\nx"
)


def test_fetch_found_is_point_in_time() -> None:
    # Revision strictly before as_of -> used.
    stub = _StubClient(
        search_titles=["Opinion polling for the next Italian general election"],
        rev_ts="2024-05-30T19:37:44Z",  # < as_of
        rev_content=_POLL_WIKITEXT,
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch(
        {
            "question": "Will Giorgia Meloni's coalition win the 2027 Italian election?",
            "as_of": CUTOFF,
            "tags": ["italy"],
        }
    )
    assert "error" not in payload
    assert payload["article"] == "Opinion polling for the next Italian general election"
    assert payload["lang"] == "it"  # local wiki searched first
    assert payload["revision_ts"] == "2024-05-30T19:37:44+00:00"
    assert "YouGov" in payload["polls_text"]
    # The revisions request passed rvstart = as_of and rvdir=older.
    rev_reqs = [r for r in stub.requests if r.get("prop") == "revisions"]
    assert rev_reqs and rev_reqs[0]["rvstart"] == "2024-06-01T00:00:00Z"
    assert rev_reqs[0]["rvdir"] == "older"


def test_fetch_accepts_bare_integer_table() -> None:
    # End-to-end: a polling article whose results table uses BARE integers (no '%')
    # must be accepted (the old %-only gate rejected it as no-data).
    content = "== Opinion polls ==\n" + _HU_TABLE + "\n== Results ==\nx"
    stub = _StubClient(
        search_titles=["Opinion polling for the 2026 Hungarian parliamentary election"],
        rev_ts="2026-02-01T00:00:00Z",
        rev_content=content,
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch(
        {
            "question": "Will Fidesz win the 2026 Hungarian parliamentary election?",
            "as_of": datetime(2026, 3, 14, tzinfo=UTC),
            "tags": ["Hungary"],
        }
    )
    assert "error" not in payload
    assert payload.get("polls_text")
    text = payload["polls_text"]
    assert "Medián" in text and "TISZA" in text  # pollster + header party label
    assert "39" in text and "51" in text  # bare-integer party shares


def test_fetch_rejects_revision_at_or_after_as_of() -> None:
    # THE leakage test: the only revision returned is stamped AT/AFTER as_of, so
    # it must be rejected and the source must report no-data (never use it).
    stub = _StubClient(
        search_titles=["Opinion polling for the next Italian general election"],
        rev_ts="2024-06-01T00:00:00Z",  # == as_of -> MUST be rejected
        rev_content=_POLL_WIKITEXT,
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch(
        {
            "question": "Will the 2027 Italian election favour the right?",
            "as_of": CUTOFF,
            "tags": ["italy"],
        }
    )
    assert payload.get("article") is None
    assert "polls_text" not in payload
    assert "error" not in payload  # clean no-data, not a hard error


def test_fetch_after_as_of_also_rejected() -> None:
    stub = _StubClient(
        search_titles=["X election"],
        rev_ts="2024-06-02T00:00:00Z",  # strictly AFTER as_of
        rev_content=_POLL_WIKITEXT,
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch({"question": "Will X win the 2027 election?", "as_of": CUTOFF, "tags": []})
    assert payload.get("article") is None


def test_fetch_no_polling_section_is_no_data() -> None:
    stub = _StubClient(
        search_titles=["Some candidate biography"],
        rev_ts="2024-05-01T00:00:00Z",
        rev_content="== Early life ==\nNo polls here at all.\n== Career ==\nx",
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch(
        {"question": "Will Foo Bar win the 2027 election?", "as_of": CUTOFF, "tags": []}
    )
    assert payload.get("article") is None
    assert "error" not in payload


def test_fetch_empty_question_errors() -> None:
    src = WikipediaPollsSource(http=_StubClient(search_titles=[], rev_ts="", rev_content=""))
    assert src.fetch({"question": "", "as_of": CUTOFF, "tags": []})["error"] == "empty_question"


def test_fetch_as_of_none_uses_current_revision() -> None:
    stub = _StubClient(
        search_titles=["2024 election"],
        rev_ts="2024-09-01T00:00:00Z",  # future of CUTOFF, but as_of is None -> ok
        rev_content=_POLL_WIKITEXT,
    )
    src = WikipediaPollsSource(http=stub)
    payload = src.fetch({"question": "Will A win the 2024 election?", "as_of": None, "tags": []})
    assert payload["article"] == "2024 election"
    # No rvstart sent when as_of is None (fetch current).
    rev_reqs = [r for r in stub.requests if r.get("prop") == "revisions"]
    assert rev_reqs and "rvstart" not in rev_reqs[0]


def test_fetch_bad_as_of_type_raises() -> None:
    src = WikipediaPollsSource(http=_StubClient(search_titles=[], rev_ts="", rev_content=""))
    with pytest.raises(TypeError):
        src.fetch({"question": "Will A win?", "as_of": "2024-06-01", "tags": []})


# --- render states ---------------------------------------------------------


def test_render_error_state() -> None:
    src = WikipediaPollsSource(http=_StubClient(search_titles=[], rev_ts="", rev_content=""))
    out = src.render({"error": "api_unreachable", "query": "X"})
    assert out.startswith("[SOURCE ERROR] polls fetch failed")


def test_render_no_data_state() -> None:
    src = WikipediaPollsSource(http=_StubClient(search_titles=[], rev_ts="", rev_content=""))
    out = src.render({"article": None, "query": "Andrea Martella 2026 election"})
    assert out.startswith("(No Wikipedia polling article/section found for:")
    assert "Andrea Martella" in out


def test_render_found_state() -> None:
    src = WikipediaPollsSource(http=_StubClient(search_titles=[], rev_ts="", rev_content=""))
    out = src.render(
        {
            "article": "Opinion polling for the next Italian general election",
            "lang": "it",
            "revision_ts": "2024-05-30T19:37:44+00:00",
            "as_of": "2024-06-01T00:00:00+00:00",
            "polls_text": "2024-05-20 | YouGov | FdI 30 | PD 22",
            "query": "x",
        }
    )
    assert out.startswith("Opinion polls (Wikipedia 'Opinion polling for the next Italian")
    assert "it" in out
    assert "revision as of 2024-05-30T19:37:44+00:00 < 2024-06-01T00:00:00+00:00" in out
    assert "YouGov" in out


# --- live sanity check (one known election article) ------------------------


@pytest.mark.skipif(
    os.environ.get("POLYEVOLVE_LIVE_TESTS") != "1",
    reason="live MediaWiki API test; set POLYEVOLVE_LIVE_TESTS=1 to run",
)
def test_live_single_article() -> None:
    src = WikipediaPollsSource()
    as_of = datetime(2024, 6, 1, tzinfo=UTC)
    payload = src.fetch(
        {
            "question": "Will the centre-right win the next Italian general election in 2027?",
            "as_of": as_of,
            "tags": ["italy"],
        }
    )
    assert "error" not in payload
    assert payload.get("article"), "expected to resolve an Italian polling article"
    assert payload["polls_text"].strip(), "expected non-empty polling text"
    # Leakage guard: the chosen revision is strictly before as_of.
    rev_ts = datetime.fromisoformat(payload["revision_ts"])
    assert rev_ts < as_of
