from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

import yaml
from pydantic import BaseModel

from .schemas import Example, SuiteManifest, TaskSpec, canonical_json, normalize_text

T = TypeVar("T", bound=BaseModel)


def write_json(path: Path, value: BaseModel | dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        value.model_dump(mode="json", exclude_none=True) if isinstance(value, BaseModel) else value
    )
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_yaml(path: Path, value: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        value.model_dump(mode="json", exclude_none=True) if isinstance(value, BaseModel) else value
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))


def read_yaml(path: Path, model: type[T]) -> T:
    return model.model_validate(yaml.safe_load(path.read_text()))


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = []
    for row in rows:
        raw = row.model_dump(mode="json", exclude_none=True) if isinstance(row, BaseModel) else row
        values.append(canonical_json(raw))
    path.write_text("\n".join(values) + ("\n" if values else ""))


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    values: list[T] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"malformed {path}:{line_no}: {exc}") from exc
    return values


SPLITS = ("specialization", "validation", "hidden", "paraphrase", "hard")


def save_suite(
    root: Path, manifest: SuiteManifest, tasks: list[TaskSpec], rows: list[Example]
) -> None:
    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("task names must be unique within a suite")
    if set(manifest.tasks) != set(names):
        raise ValueError("manifest task names do not match tasks")
    validate_no_leakage(rows)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", manifest)
    write_yaml(root / "tasks.yaml", {"tasks": [x.model_dump(mode="json") for x in tasks]})
    for split in SPLITS:
        write_jsonl(root / f"{split}.jsonl", (x for x in rows if x.split == split))


def load_suite(root: Path) -> tuple[SuiteManifest, list[TaskSpec], dict[str, list[Example]]]:
    try:
        manifest = SuiteManifest.model_validate_json((root / "manifest.json").read_text())
        task_raw = yaml.safe_load((root / "tasks.yaml").read_text())
        tasks = [TaskSpec.model_validate(x) for x in task_raw["tasks"]]
        splits = {split: read_jsonl(root / f"{split}.jsonl", Example) for split in SPLITS}
    except FileNotFoundError as exc:
        raise ValueError(f"incomplete benchmark suite: missing {exc.filename}") from exc
    all_rows = [row for split in SPLITS for row in splits[split]]
    validate_no_leakage(all_rows)
    known = {task.name for task in tasks}
    if set(manifest.tasks) != known:
        raise ValueError("manifest task names do not match tasks.yaml")
    for row in all_rows:
        if row.task_name not in known:
            raise ValueError(f"example {row.id!r} references unknown task {row.task_name!r}")
    return manifest, tasks, splits


def validate_no_leakage(rows: list[Example]) -> None:
    ids: dict[str, str] = {}
    hashes: dict[str, tuple[str, str]] = {}
    texts: dict[str, tuple[str, str]] = {}
    for row in rows:
        if row.id in ids:
            raise ValueError(f"duplicate example id {row.id!r} in {ids[row.id]} and {row.split}")
        ids[row.id] = row.split
        norm = normalize_text(row.input)
        for key, seen, kind in (
            (row.content_hash or "", hashes, "content hash"),
            (norm, texts, "text"),
        ):
            prior = seen.get(key)
            if prior:
                raise ValueError(
                    f"dataset leakage or duplicate: repeated {kind} in {prior[0]} ({prior[1]}) "
                    f"and {row.split} ({row.id})"
                )
            seen[key] = (row.split, row.id)
