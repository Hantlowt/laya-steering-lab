import pytest

from laya_steering.generation import generate_specialization
from laya_steering.providers import StaticProvider


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
