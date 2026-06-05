from __future__ import annotations

from typing import Any

from synthetic_data_engine.tasks.spec import TaskSpec


def summarize_task(task: TaskSpec) -> dict[str, Any]:
    return {
        "name": task.name,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "topics": task.topics,
        "required_fields": task.required_fields,
        "selection_min_score": task.min_score,
        "valid": True,
    }
