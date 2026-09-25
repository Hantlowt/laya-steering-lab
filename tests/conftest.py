from __future__ import annotations

import pytest

from laya_steering.backends import FakeBackend
from laya_steering.schemas import DecisionSchema, Example, TaskSpec


@pytest.fixture
def binary_task():
    return TaskSpec(
        name="priority",
        domain="support",
        description="Decide whether a support issue is urgent.",
        decision=DecisionSchema(
            labels=["NORMAL", "URGENT"],
            class_descriptions={"NORMAL": "routine help", "URGENT": "outage harm"},
        ),
    )


@pytest.fixture
def rows(binary_task):
    values = [
        ("normal-1", "How do I edit my profile?", "NORMAL", "specialization"),
        ("normal-2", "Where is the documentation?", "NORMAL", "specialization"),
        ("urgent-1", "Production is down for everyone", "URGENT", "specialization"),
        ("urgent-2", "Customer data is actively leaking", "URGENT", "specialization"),
        ("val-1", "The main service is unavailable", "URGENT", "validation"),
        ("val-2", "Please explain account colors", "NORMAL", "validation"),
    ]
    return [
        Example(
            id=i,
            input=text,
            label=label,
            split=split,
            task_name=binary_task.name,
            domain=binary_task.domain,
        )
        for i, text, label, split in values
    ]


@pytest.fixture
def fake_backend():
    return FakeBackend(dimension=16)
