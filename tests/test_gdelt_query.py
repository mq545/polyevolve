"""Tests for GDELT query construction + render state distinction (no network)."""

from polyevolve.data_sources.gdelt import GdeltSource, _question_to_query


def test_query_prefers_proper_nouns() -> None:
    question = "Will a new Cabinet of the Netherlands be sworn in by December 31, 2025?"
    q = _question_to_query(question)
    terms = q.split()
    # Entity tokens must be present; generic filler should not crowd them out.
    assert "Netherlands" in terms
    assert "Cabinet" in terms
    # Capped small so GDELT's AND doesn't over-constrain to zero results.
    assert len(terms) <= 4


def test_query_drops_stopwords() -> None:
    q = _question_to_query("Will Maduro be out by March 31?")
    terms = q.split()
    assert "Maduro" in terms
    assert "Will" not in terms  # stopword
    assert "by" not in terms


def test_query_empty_question() -> None:
    assert _question_to_query("") == ""


def test_render_distinguishes_error_from_empty() -> None:
    s = GdeltSource()
    err = s.render({"error": "out_of_window", "detail": "too old", "query": "Maduro"})
    empty = s.render({"articles": [], "query": "Maduro"})
    found = s.render(
        {
            "articles": [
                {
                    "title": "Maduro flees",
                    "domain": "x.com",
                    "language": "English",
                    "seendate": "20251201T000000Z",
                    "dup_count": 1,
                }
            ],
            "query": "Maduro",
        }
    )
    assert err.startswith("[SOURCE ERROR]")
    assert "out_of_window" in err
    # Genuine empty must NOT look like an error, and must NOT be silent.
    assert not empty.startswith("[SOURCE ERROR]")
    assert "no coverage" in empty
    assert "Maduro flees" in found


def test_render_shows_syndication_count() -> None:
    s = GdeltSource()
    out = s.render(
        {
            "articles": [
                {
                    "title": "Big story",
                    "domain": "x.com",
                    "language": "English",
                    "seendate": "20251201T000000Z",
                    "dup_count": 5,
                }
            ],
            "query": "test",
        }
    )
    assert "carried by ~5 outlets" in out
