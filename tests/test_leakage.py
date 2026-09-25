from pathlib import Path

import pytest

from laya_steering.io import load_suite, validate_no_leakage
from laya_steering.schemas import Example


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
