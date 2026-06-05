from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from synthetic_data_engine.tasks.spec import TaskSpec


def load_task_spec(path: str | Path) -> TaskSpec:
    task_path = Path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_path}")

    text = task_path.read_text(encoding="utf-8")
    if task_path.suffix.lower() == ".json":
        data: Any = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError("Task spec must be a mapping")
    return TaskSpec.from_mapping(data)
