from pathlib import Path

import pytest

from laya_studio.io import load_suite


def test_incomplete_suite_has_clear_error(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(Exception):
        load_suite(tmp_path)


def test_malformed_jsonl_reports_line(tmp_path):
    smoke = Path("benchmarks/smoke")
    for name in (
        "manifest.json",
        "tasks.yaml",
        "specialization.jsonl",
        "validation.jsonl",
        "paraphrase.jsonl",
        "hard.jsonl",
    ):
        (tmp_path / name).write_bytes((smoke / name).read_bytes())
    (tmp_path / "hidden.jsonl").write_text("{not json}\n")
    with pytest.raises(ValueError, match="hidden.jsonl:1"):
        load_suite(tmp_path)
