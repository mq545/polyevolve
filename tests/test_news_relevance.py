"""Tests for the BQ-news scraped-body relevance gate + local-source boost.

Pure-function tests only - no BigQuery, no network, no scraping, no GPU. These
cover the precision fix (off-topic-but-same-country articles are excluded) and
the soft local-source preference the thesis depends on.
"""

from polyevolve.data_sources.gdelt_bq_news import (
    _ENTITY_WEIGHT,
    _local_tlds,
    _locality_boost,
    _min_relevance,
    _relevance_score,
    _term_weights,
)

_STRIKE_Q = "Will Israel conduct a military strike on Iran's nuclear facilities by July?"


def test_entities_weighted_above_topic_words() -> None:
    w = _term_weights(_STRIKE_Q)
    # Proper-noun entities outweigh generic content words.
    assert w["israel"] == _ENTITY_WEIGHT
    assert w["iran"] == _ENTITY_WEIGHT
    # A generic topic noun present but lighter (or absent if a stopword).
    assert w.get("nuclear", 0.0) <= _ENTITY_WEIGHT


def test_offtopic_same_country_article_is_below_gate() -> None:
    # The real failure: an EU-sanctions story that only mentions Iran in passing
    # must NOT clear the gate for a strike question.
    w = _term_weights(_STRIKE_Q)
    off_topic = (
        "The European Union agreed new sanctions packages targeting trade with "
        "several states. Officials in Brussels said the measures were broad."
    )
    assert _relevance_score(off_topic, w) < _min_relevance(w)


def test_ontopic_article_clears_gate() -> None:
    w = _term_weights(_STRIKE_Q)
    on_topic = (
        "Israel signalled it could launch a strike against Iran's nuclear "
        "facilities, officials said, raising fears of regional escalation."
    )
    assert _relevance_score(on_topic, w) >= _min_relevance(w)


def test_relevance_scores_distinct_terms_once() -> None:
    w = _term_weights(_STRIKE_Q)
    # Repeating an entity does not inflate the score (distinct terms only).
    once = _relevance_score("Iran responded.", w)
    twice = _relevance_score("Iran responded. Iran again. Iran.", w)
    assert once == twice


def test_local_language_proper_noun_still_matches() -> None:
    # Proper nouns match across languages; a Dutch body naming the entity scores.
    q = "Will Geert Wilders become the next Prime Minister of the Netherlands?"
    w = _term_weights(q)
    dutch = "Geert Wilders zei dat het kabinet snel gevormd moet worden in Nederland."
    assert _relevance_score(dutch, w) >= _min_relevance(w)


def test_local_tld_boost_for_in_country_domain() -> None:
    tlds = _local_tlds(["netherlands"])  # -> .nl
    assert _locality_boost("nos.nl", tlds) > 0.0


def test_english_wire_is_penalized() -> None:
    tlds = _local_tlds(["iran"])
    assert _locality_boost("reuters.com", tlds) < 0.0
    assert _locality_boost("bbc.co.uk", tlds) < 0.0


def test_local_press_outranks_wire_on_same_relevance() -> None:
    tlds = _local_tlds(["south-korea"])  # -> .kr / .co.kr
    assert _locality_boost("chosun.co.kr", tlds) > _locality_boost("nytimes.com", tlds)


def test_unknown_cctld_gets_mild_nudge_not_penalty() -> None:
    # A foreign ccTLD we have no local map for is still likely more local than wire.
    tlds = _local_tlds(["iran"])
    assert _locality_boost("somesite.de", tlds) > _locality_boost("somesite.com", tlds)


def test_locality_boost_empty_domain_is_neutral() -> None:
    assert _locality_boost("", ()) == 0.0
