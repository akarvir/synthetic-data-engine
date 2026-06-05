from __future__ import annotations

from synthetic_data_engine.tasks.loader import load_task_spec
from synthetic_data_engine.tasks.spec import TaskSpec
from synthetic_data_engine.tasks.summary import summarize_task


def test_task_min_score_uses_selection_default():
    task = load_task_spec("tasks/general-instruction.yaml")

    assert task.min_score == 0.8


def test_task_min_score_falls_back_to_default():
    task = TaskSpec.from_mapping(
        {
            "name": "example",
            "domain": "reasoning",
            "description": "Example task.",
            "output_schema": {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}}},
        }
    )

    assert task.min_score == 0.8


def test_task_summary_includes_validation_ready_fields():
    task = load_task_spec("tasks/general-instruction.yaml")

    summary = summarize_task(task)

    assert summary["valid"] is True
    assert summary["required_fields"] == ["prompt", "answer"]
    assert summary["selection_min_score"] == 0.8
