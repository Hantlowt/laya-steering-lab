from pathlib import Path

import pytest

from laya_studio.generation import remove_generated_leakage
from laya_studio.io import load_suite, validate_no_leakage
from laya_studio.schemas import Example


def test_cross_split_normalized_duplicate_is_rejected():
    common = dict(label="A", task_name="task", domain="domain")
    rows = [
        Example(id="a", input="Same   Text", split="specialization", **common),
        Example(id="b", input=" same text ", split="hidden", **common),
    ]
    with pytest.raises(ValueError, match="dataset leakage"):
        validate_no_leakage(rows)


def test_checked_in_smoke_suite_is_valid():
    manifest, tasks, splits = load_suite(Path("benchmarks/smoke"))
    assert len(tasks) == 10
    assert len({task.domain for task in tasks}) == 5
    assert len(splits["specialization"]) == 60
    assert manifest.metadata["scientific_evidence"] is False


def test_generated_cross_split_collision_is_removed():
    common = dict(label="A", task_name="task", domain="domain")
    rows = [
        Example(id="spec", input="Same text", split="specialization", **common),
        Example(id="hidden-duplicate", input=" same   text ", split="hidden", **common),
        Example(id="hidden-unique", input="Different text", split="hidden", **common),
    ]
    cleaned, removed = remove_generated_leakage(rows)
    validate_no_leakage(cleaned)
    assert removed == 1
    assert [row.id for row in cleaned] == ["spec", "hidden-unique"]


def test_generated_collision_removes_whole_paraphrase_pair():
    common = dict(label="A", task_name="task", domain="domain")
    rows = [
        Example(id="spec", input="Same text", split="specialization", **common),
        Example(id="para-1", input="Same text", split="paraphrase", pair_id="p1", **common),
        Example(
            id="para-2", input="Equivalent wording", split="paraphrase", pair_id="p1", **common
        ),
    ]
    cleaned, removed = remove_generated_leakage(rows)
    assert removed == 2
    assert [row.id for row in cleaned] == ["spec"]
