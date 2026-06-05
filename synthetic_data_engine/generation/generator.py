from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from synthetic_data_engine.models.base import ModelClient
from synthetic_data_engine.tasks.spec import TaskSpec


@dataclass(frozen=True)
class Candidate:
    id: str
    item: dict[str, Any]
    generator_model: str
    prompt_messages: list[dict[str, str]]


class Generator:
    def __init__(self, model: ModelClient) -> None:
        self.model = model

    async def generate_one(self, task: TaskSpec, index: int) -> Candidate:
        payload = {
            "mode": "generate",
            "index": index,
            "task": task.to_dict(),
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You generate high-quality synthetic training examples. "
                    "Return only a JSON object matching the requested task schema."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        item = await self.model.complete_json(messages, schema=task.output_schema)
        return Candidate(id=str(uuid.uuid4()), item=item, generator_model=self.model.name, prompt_messages=messages)
