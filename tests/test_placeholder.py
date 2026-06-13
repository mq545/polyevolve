"""Tests for neg-risk placeholder candidate detection."""

from polyevolve.contracts import Market
from polyevolve.market_sources.filters import is_placeholder_market


def _m(question: str) -> Market:
    return Market(
        venue="polymarket",
        external_id="x",
        cross_venue_id=None,
        question=question,
        close_time=None,
        status="resolved",
        metadata={},
    )


def test_person_letter_is_placeholder() -> None:
    assert is_placeholder_market(_m("Will Person H win the 2026 Galway West by-election?"))
    assert is_placeholder_market(_m("Will Person A win the 2026 Galway West by-election?"))


def test_other_is_placeholder() -> None:
    assert is_placeholder_market(_m("Will Other win the 2026 Galway West by-election?"))


def test_candidate_letter_is_placeholder() -> None:
    assert is_placeholder_market(_m("Will Candidate B win?"))


def test_real_named_candidate_is_not_placeholder() -> None:
    assert not is_placeholder_market(_m("Will Mike Cubbard win the 2026 Galway West by-election?"))
    assert not is_placeholder_market(_m("Will Niall Murphy win the 2026 Galway West by-election?"))


def test_normal_question_is_not_placeholder() -> None:
    assert not is_placeholder_market(_m("Russia x Ukraine ceasefire by end of 2026?"))
    assert not is_placeholder_market(_m("Macron out by June 30, 2026?"))


def test_substring_person_not_falsely_flagged() -> None:
    # "personal", "personnel" should not trigger the \bperson\s+[a-z]\b pattern
    assert not is_placeholder_market(_m("Will personal income tax rise in France?"))
