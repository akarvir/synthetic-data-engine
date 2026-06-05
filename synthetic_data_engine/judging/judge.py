from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from synthetic_data_engine.generation.generator import Candidate
from synthetic_data_engine.models.base import ModelClient
from synthetic_data_engine.tasks.spec import TaskSpec
from synthetic_data_engine.tasks.validation import validate_item


@dataclass(frozen=True)
class Judgment:
    id: str
    candidate_id: str
    result: dict[str, Any]
    judge_model: str

    @property
    def score(self) -> float:
        return float(self.result.get("overall_score", 0.0))

    @property
    def verdict(self) -> str:
        return str(self.result.get("verdict", "reject"))


class Judge:
    def __init__(self, model: ModelClient, min_score: float) -> None:
        self.model = model
        self.min_score = min_score

    async def judge_one(self, task: TaskSpec, candidate: Candidate) -> Judgment:
        schema_result = _validate_schema(task, candidate.item)
        if not schema_result["valid"]:
            return Judgment(
                id=str(uuid.uuid4()),
                candidate_id=candidate.id,
                result={
                    "overall_score": 0.0,
                    "scores": {"schema": 0.0},
                    "verdict": "reject",
                    "rationale": schema_result["reason"],
                },
                judge_model="schema-validator",
            )

        payload = {
            "mode": "judge",
            "min_score": self.min_score,
            "task": task.to_dict(),
            "candidate": candidate.item,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You judge synthetic training examples. Score correctness, clarity, novelty, "
                    "difficulty fit, and usefulness. Return only JSON."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        result = await self.model.complete_json(messages)
        result.setdefault("verdict", "accept" if float(result.get("overall_score", 0.0)) >= self.min_score else "reject")
        return Judgment(
            id=str(uuid.uuid4()),
            candidate_id=candidate.id,
            result=result,
            judge_model=self.model.name,
        )


def _validate_schema(task: TaskSpec, item: dict[str, Any]) -> dict[str, Any]:
    return validate_item(task.output_schema, item)
