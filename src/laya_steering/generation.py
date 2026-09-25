from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .io import SPLITS, save_suite
from .providers import LLMProvider
from .schemas import Example, GenerationRecord, SuiteManifest, TaskSpec, normalize_text


class _GeneratedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1)
    label: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    pair_id: str | None = None


class _ExamplesPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examples: list[_GeneratedExample]


class _TaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: TaskSpec


class _CatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str
    task: str
    labels: list[str] = Field(min_length=2)


class _CatalogPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[_CatalogItem]


EXAMPLES_SCHEMA = _ExamplesPayload.model_json_schema()
TASK_SCHEMA = _TaskPayload.model_json_schema()
CATALOG_SCHEMA = _CatalogPayload.model_json_schema()


def invent_task_catalog(
    provider: LLMProvider, domains: int, tasks_per_domain: int, seed: int
) -> list[tuple[str, list[str]]]:
    total = domains * tasks_per_domain
    prompt = f"""Invent exactly {domains} unrelated decision domains and exactly {tasks_per_domain}
classification tasks per domain ({total} tasks total). Include a broad mix such as operations,
security, commerce, devices, documents, finance, and original synthetic rule systems, but do not
limit the catalog to named examples. Each task needs 2-6 concise unique labels. Return task requests
and labels only; do not generate examples. Avoid near-duplicate policies.
"""
    response = provider.generate_json(prompt, CATALOG_SCHEMA, seed)
    payload = _CatalogPayload.model_validate(response.content)
    if len(payload.tasks) != total or len({x.domain for x in payload.tasks}) != domains:
        raise ValueError(f"provider must return {total} tasks across exactly {domains} domains")
    return [(f"[{item.domain}] {item.task}", item.labels) for item in payload.tasks]


def infer_task(
    provider: LLMProvider, task: str, labels: list[str] | None, seed: int
) -> tuple[TaskSpec, GenerationRecord]:
    label_instruction = (
        f"Allowed labels, exactly as written and in this order: {json.dumps(labels)}"
        if labels
        else "Infer 2 to 6 concise, mutually exclusive labels and order them deterministically."
    )
    prompt = f"""Infer a rigorous classification policy for the requested task.
Task request: {task}
{label_instruction}
Return a short machine-safe name, domain, detailed description, decision schema, class descriptions,
boundary rules, semantic concepts, positive concepts, and negative concepts. Do not produce examples.
"""
    response = provider.generate_json(prompt, TASK_SCHEMA, seed)
    parsed = _TaskPayload.model_validate(response.content)
    if labels and parsed.task.decision.labels != labels:
        raise ValueError("provider changed or reordered the requested labels")
    return parsed.task, _generation_record(response, "specialization")


def generate_specialization(
    provider: LLMProvider, task: TaskSpec, count: int, seed: int
) -> tuple[list[Example], GenerationRecord]:
    prompt = f"""Create exactly {count} labeled SPECIALIZATION examples for this policy:
{task.model_dump_json(indent=2)}
Cover every label, varied phrasing and contexts, and many near-boundary cases. These examples will
construct a classifier and are not a test set. Use only the allowed labels. Return data, not code.
"""
    response = provider.generate_json(prompt, EXAMPLES_SCHEMA, seed)
    payload = _ExamplesPayload.model_validate(response.content)
    rows = _normalize_generated_count(
        _rows(payload.examples, task, "specialization", "spec", provider.model), task, count
    )
    _validate_generated(rows, task, count)
    return rows, _generation_record(response, "specialization")


def generate_benchmark_splits(
    provider: LLMProvider,
    task: TaskSpec,
    counts: dict[str, int],
    seed: int,
) -> tuple[list[Example], GenerationRecord]:
    """Generate all evaluation splits without ever receiving specialization examples."""
    rows: list[Example] = []
    records: list[GenerationRecord] = []
    for offset, split in enumerate(("validation", "hidden", "paraphrase", "hard")):
        count = counts.get(split, 0)
        if not count:
            continue
        split_guidance = {
            "validation": "near-boundary cases for selecting scalar strengths; not final reporting",
            "hidden": "semantically varied unseen cases, balanced across labels",
            "paraphrase": "paired semantic equivalents with very different vocabulary; set pair_id",
            "hard": "ambiguous, adversarial, negated, distractor-heavy, and edge cases",
        }[split]
        prompt = f"""Create exactly {count} {split.upper()} evaluation examples for this policy:
{task.model_dump_json(indent=2)}
Purpose: {split_guidance}.
This is an independently generated benchmark. Do not imitate training-set templates. Use only the
allowed labels. Inputs must be unique. Return data, not code.
"""
        response = provider.generate_json(prompt, EXAMPLES_SCHEMA, seed + 1009 * (offset + 1))
        payload = _ExamplesPayload.model_validate(response.content)
        part = _normalize_generated_count(
            _rows(payload.examples, task, split, split[:4], provider.model), task, count
        )
        _validate_generated(part, task, count)
        rows.extend(part)
        records.append(_generation_record(response, "benchmark"))
    merged_prompt_hash = hashlib.sha256(
        "".join(x.prompt_sha256 for x in records).encode()
    ).hexdigest()
    record = GenerationRecord(
        role="benchmark",
        provider=provider.name,
        model=provider.model,
        seed=seed,
        prompt="\n\n--- INDEPENDENT SPLIT PROMPT ---\n\n".join(x.prompt or "" for x in records),
        prompt_sha256=merged_prompt_hash,
        cache_key=hashlib.sha256("".join(x.cache_key or "" for x in records).encode()).hexdigest(),
    )
    return rows, record


def build_suite(
    output: Path,
    name: str,
    task_requests: list[tuple[str, list[str]]],
    specialization_provider: LLMProvider,
    benchmark_provider: LLMProvider,
    style_control_provider: LLMProvider | None,
    specialization_examples: int,
    counts: dict[str, int],
    seed: int,
) -> None:
    tasks, rows = [], []
    spec_records, benchmark_records = [], []
    for index, (request, labels) in enumerate(task_requests):
        task_seed = seed + index * 100_003
        task, task_record = infer_task(specialization_provider, request, labels, task_seed)
        spec, spec_record = generate_specialization(
            specialization_provider, task, specialization_examples, task_seed + 1
        )
        tests, bench_record = generate_benchmark_splits(
            benchmark_provider, task, counts, task_seed + 50_000
        )
        if style_control_provider is not None and counts.get("hidden", 0):
            style_rows, style_record = generate_benchmark_splits(
                style_control_provider,
                task,
                {"hidden": max(2, counts["hidden"] // 4)},
                task_seed + 75_000,
            )
            tests.extend(
                row.model_copy(update={"id": row.id + "-style-control"}) for row in style_rows
            )
            benchmark_records.append(style_record)
        tasks.append(task)
        rows.extend(spec + tests)
        spec_records.extend((task_record, spec_record))
        benchmark_records.append(bench_record)
    rows, removed = remove_generated_leakage(rows)
    manifest = SuiteManifest(
        name=name,
        tasks=[x.name for x in tasks],
        seed=seed,
        specialization_generation=_merge_records(spec_records, "specialization"),
        benchmark_generation=_merge_records(benchmark_records, "benchmark"),
        metadata={"removed_leakage_collisions": removed},
    )
    save_suite(output, manifest, tasks, rows)


def remove_generated_leakage(rows: list[Example]) -> tuple[list[Example], int]:
    """Drop later generated collisions while preserving split priority and paraphrase pairs."""
    seen: set[str] = set()
    kept: list[Example] = []
    removed = 0
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split]
        if split == "paraphrase":
            grouped: dict[tuple[str, str], list[Example]] = {}
            for row in split_rows:
                grouped.setdefault((row.task_name, row.pair_id or row.id), []).append(row)
            units = list(grouped.values())
        else:
            units = [[row] for row in split_rows]
        for unit in units:
            keys = {normalize_text(row.input) for row in unit}
            if keys & seen or len(keys) != len(unit):
                removed += len(unit)
                continue
            kept.extend(unit)
            seen.update(keys)
    return kept, removed


def _rows(
    values: list[_GeneratedExample], task: TaskSpec, split: str, prefix: str, source_style: str
) -> list[Example]:
    return [
        Example(
            id=f"{task.name}-{prefix}-{i:05d}",
            input=value.input,
            label=value.label,
            split=split,
            task_name=task.name,
            domain=task.domain,
            pair_id=value.pair_id,
            tags=value.tags,
            source_style=source_style,
        )
        for i, value in enumerate(values)
    ]


def _validate_generated(rows: list[Example], task: TaskSpec, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"provider returned {len(rows)} examples, expected exactly {expected}")
    labels = set(task.decision.labels)
    unknown = {x.label for x in rows} - labels
    if unknown:
        raise ValueError(f"provider used unknown labels: {sorted(unknown)}")
    missing = labels - {x.label for x in rows}
    if missing:
        raise ValueError(f"provider omitted labels: {sorted(missing)}")
    if len({x.content_hash for x in rows}) != len(rows):
        raise ValueError("provider returned duplicate inputs")
    if rows and rows[0].split == "paraphrase":
        pairs: dict[str, list[Example]] = {}
        for row in rows:
            if not row.pair_id:
                raise ValueError("every paraphrase example requires a pair_id")
            pairs.setdefault(row.pair_id, []).append(row)
        if any(len(values) < 2 for values in pairs.values()):
            raise ValueError("every paraphrase pair_id must identify at least two examples")
        if any(len({row.label for row in values}) != 1 for values in pairs.values()):
            raise ValueError("paraphrases in a pair must have the same label")


def _normalize_generated_count(rows: list[Example], task: TaskSpec, expected: int) -> list[Example]:
    """Accept provider over-generation while keeping a balanced deterministic subset."""
    if len(rows) < expected:
        raise ValueError(f"provider returned only {len(rows)} examples, expected {expected}")
    if len(rows) == expected:
        return rows

    labels = set(task.decision.labels)
    unknown = {row.label for row in rows} - labels
    if unknown:
        raise ValueError(f"provider used unknown labels: {sorted(unknown)}")
    if len({row.content_hash for row in rows}) != len(rows):
        raise ValueError("provider returned duplicate inputs")

    if rows and rows[0].split == "paraphrase":
        groups: dict[str, list[Example]] = {}
        for row in rows:
            if not row.pair_id:
                raise ValueError("every paraphrase example requires a pair_id")
            groups.setdefault(row.pair_id, []).append(row)
        units = list(groups.values())
    else:
        units = [[row] for row in rows]

    by_label = {label: [] for label in task.decision.labels}
    for unit in units:
        if len({row.label for row in unit}) != 1:
            raise ValueError("paraphrases in a pair must have the same label")
        by_label[unit[0].label].append(unit)

    selected: list[Example] = []
    while len(selected) < expected:
        progressed = False
        for label in task.decision.labels:
            if not by_label[label]:
                continue
            unit = by_label[label].pop(0)
            if len(selected) + len(unit) <= expected:
                selected.extend(unit)
                progressed = True
            if len(selected) == expected:
                return selected
        if not progressed:
            break
    raise ValueError(
        f"provider returned {len(rows)} examples, but paired groups cannot be reduced to {expected}"
    )


def _generation_record(response: Any, role: str) -> GenerationRecord:
    return GenerationRecord(
        role=role,
        provider=response.provider,
        model=response.model,
        seed=response.seed,
        prompt=response.prompt,
        prompt_sha256=hashlib.sha256(response.prompt.encode()).hexdigest(),
        cache_key=response.cache_key,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _merge_records(records: list[GenerationRecord], role: str) -> GenerationRecord:
    if not records:
        raise ValueError("cannot merge an empty generation record list")
    return GenerationRecord(
        role=role,
        provider=",".join(sorted({x.provider for x in records})),
        model=",".join(sorted({x.model for x in records})),
        seed=records[0].seed,
        prompt="\n\n--- GENERATION PROMPT ---\n\n".join(x.prompt or "" for x in records),
        prompt_sha256=hashlib.sha256(
            "".join(x.prompt_sha256 for x in records).encode()
        ).hexdigest(),
        cache_key=hashlib.sha256("".join(x.cache_key or "" for x in records).encode()).hexdigest(),
    )
