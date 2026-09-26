import pytest

from laya_studio.generation import generate_specialization
from laya_studio.providers import StaticProvider, _decode_json_content


def test_llm_response_unknown_label_rejected(binary_task):
    provider = StaticProvider(
        [
            {
                "examples": [
                    {"input": "routine", "label": "NORMAL", "tags": [], "pair_id": None},
                    {"input": "bad", "label": "MADE_UP", "tags": [], "pair_id": None},
                ]
            }
        ]
    )
    with pytest.raises(ValueError, match="unknown labels"):
        generate_specialization(provider, binary_task, 2, 1)


def test_llm_response_extra_fields_rejected(binary_task):
    provider = StaticProvider(
        [
            {
                "examples": [
                    {
                        "input": "routine",
                        "label": "NORMAL",
                        "tags": [],
                        "pair_id": None,
                        "code": "print(1)",
                    },
                    {"input": "down", "label": "URGENT", "tags": [], "pair_id": None},
                ]
            }
        ]
    )
    with pytest.raises(Exception, match="extra"):
        generate_specialization(provider, binary_task, 2, 1)


def test_llm_over_generation_is_balanced_and_truncated(binary_task):
    provider = StaticProvider(
        [
            {
                "examples": [
                    {"input": "routine one", "label": "NORMAL"},
                    {"input": "routine two", "label": "NORMAL"},
                    {"input": "urgent one", "label": "URGENT"},
                    {"input": "urgent two", "label": "URGENT"},
                ]
            }
        ]
    )
    rows, _ = generate_specialization(provider, binary_task, 2, 1)
    assert [row.label for row in rows] == ["NORMAL", "URGENT"]


def test_json_markdown_fence_and_preamble_are_accepted():
    assert _decode_json_content('```json\n{"examples": []}\n```') == {"examples": []}
    assert _decode_json_content('Here is the result:\n{"examples": []}') == {"examples": []}


def test_llm_under_generation_is_recovered_with_another_batch(binary_task):
    provider = StaticProvider(
        [
            {"examples": [{"input": "only one", "label": "NORMAL"}]},
            {
                "examples": [
                    {"input": "routine", "label": "NORMAL"},
                    {"input": "outage", "label": "URGENT"},
                ]
            },
        ]
    )
    rows, record = generate_specialization(provider, binary_task, 2, 7)
    assert len(rows) == 2
    assert {row.label for row in rows} == {"NORMAL", "URGENT"}
    assert "recovery attempt 2" in (record.prompt or "")
