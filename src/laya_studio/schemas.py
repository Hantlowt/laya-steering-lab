from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecisionType(str, Enum):
    choice = "choice"
    score = "score"
    noul = "noul"
    ranking = "ranking"


class DecisionSchema(StrictModel):
    type: DecisionType = DecisionType.choice
    labels: list[str] = Field(min_length=2)
    question: str | None = None
    class_descriptions: dict[str, str] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: list[str]) -> list[str]:
        clean = [x.strip() for x in labels]
        if any(not x for x in clean) or len(set(clean)) != len(clean):
            raise ValueError("labels must be non-empty and unique")
        return clean

    @model_validator(mode="after")
    def descriptions_reference_labels(self) -> "DecisionSchema":
        unknown = set(self.class_descriptions) - set(self.labels)
        if unknown:
            raise ValueError(f"class_descriptions contains unknown labels: {sorted(unknown)}")
        if self.type == DecisionType.noul and len(self.labels) != 2:
            raise ValueError("noul decisions require exactly two labels")
        return self


class TaskSpec(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = Field(min_length=10)
    domain: str = Field(min_length=1)
    decision: DecisionSchema
    semantic_concepts: list[str] = Field(default_factory=list)
    positive_concepts: list[str] = Field(default_factory=list)
    negative_concepts: list[str] = Field(default_factory=list)
    policy: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationRecord(StrictModel):
    role: Literal["specialization", "benchmark", "manual_fixture"]
    provider: str
    model: str
    seed: int | None = None
    prompt: str | None = None
    prompt_sha256: str
    cache_key: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Example(StrictModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    label: str = Field(min_length=1)
    split: Literal["specialization", "validation", "hidden", "paraphrase", "hard"]
    task_name: str
    domain: str
    pair_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_style: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None

    @model_validator(mode="after")
    def set_content_hash(self) -> "Example":
        normalized = normalize_text(self.input)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if self.content_hash is not None and self.content_hash != digest:
            raise ValueError("content_hash does not match normalized input")
        object.__setattr__(self, "content_hash", digest)
        return self


class SuiteManifest(StrictModel):
    format: Literal["laya-benchmark-suite"] = "laya-benchmark-suite"
    format_version: Literal[1] = 1
    name: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    specialization_generation: GenerationRecord | None = None
    benchmark_generation: GenerationRecord | None = None
    tasks: list[str]
    seed: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyArtifact(StrictModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    labels: list[str]
    decision_component: Literal["laya_head", "embedding_classifier", "pairwise_laya_head"]
    arrays: list[str] = Field(default_factory=list)


class ExportManifest(StrictModel):
    format: Literal["laya-specialization"] = "laya-specialization"
    format_version: Literal[1] = 1
    name: str
    base_model: str
    base_revision: str | None = None
    backend_compatibility: list[Literal["pytorch", "mlx", "fake"]]
    strategy: str
    decision_schema: dict[str, Any]
    specialization_config: dict[str, Any]
    created_at: str
    package_version: str
    self_contained: bool = False
    base_model_path: str | None = None
    license: str | None = None
    attribution: str | None = None


class ProviderResponse(StrictModel):
    content: dict[str, Any]
    provider: str
    model: str
    seed: int | None
    prompt: str
    cache_key: str


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
