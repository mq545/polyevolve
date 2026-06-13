"""Tests for LiteLLM JSON-mode extraction (the thinking-model fix)."""

import pytest

from polyevolve.models.litellm_model import _extract_json_object, _json_instruction


def test_plain_json() -> None:
    out = _extract_json_object('{"probability_yes": 0.7, "confidence": 0.5}')
    assert out["probability_yes"] == 0.7


def test_strips_think_block() -> None:
    text = '<think>London is rainy, let me reason...</think>\n{"probability_yes": 0.85}'
    assert _extract_json_object(text)["probability_yes"] == 0.85


def test_strips_unclosed_think() -> None:
    # model ran out of tokens but we still want to fail cleanly, not parse think text
    with pytest.raises(RuntimeError):
        _extract_json_object("<think>reasoning that never closed and no json")


def test_handles_code_fence() -> None:
    text = '```json\n{"probability_yes": 0.3, "reasoning": "x"}\n```'
    assert _extract_json_object(text)["probability_yes"] == 0.3


def test_takes_last_object_when_multiple() -> None:
    # reasoning may include an example object; the real answer comes last
    text = 'Example: {"probability_yes": 0.5} ... final: {"probability_yes": 0.9}'
    assert _extract_json_object(text)["probability_yes"] == 0.9


def test_nested_and_arrays() -> None:
    text = '{"probability_yes": 0.4, "key_factors": ["a", "b"], "meta": {"k": 1}}'
    out = _extract_json_object(text)
    assert out["key_factors"] == ["a", "b"]
    assert out["meta"] == {"k": 1}


def test_braces_inside_strings_dont_break_parsing() -> None:
    text = '{"reasoning": "use {curly} braces", "probability_yes": 0.2}'
    assert _extract_json_object(text)["probability_yes"] == 0.2


def test_no_json_raises() -> None:
    with pytest.raises(RuntimeError):
        _extract_json_object("I cannot answer this question.")


def test_instruction_lists_required_fields() -> None:
    tool = {
        "input_schema": {
            "properties": {
                "probability_yes": {"type": "number", "description": "P(YES)"},
                "reasoning": {"type": "string", "description": "why"},
            },
            "required": ["probability_yes", "reasoning"],
        }
    }
    instr = _json_instruction(tool)
    assert "probability_yes" in instr
    assert "reasoning" in instr
