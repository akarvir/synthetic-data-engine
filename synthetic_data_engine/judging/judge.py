from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from synthetic_data_engine.generation.generator import Candidate
from synthetic_data_engine.models.base import ModelClient
from synthetic_data_engine.tasks.spec import TaskSpec
from synthetic_data_engine.tasks.validation import validate_item

SCORE_FIELDS = ("correctness", "clarity", "novelty", "difficulty_fit", "usefulness")


@dataclass(frozen=True)
class Judgment:
    id: str
    candidate_id: str
    result: dict[str, Any]
    judge_model: str
    prompt_messages: list[dict[str, str]]

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
                prompt_messages=[],
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
                    "You judge synthetic training examples. Return only one JSON object with exactly these keys: "
                    "overall_score, scores, verdict, rationale. overall_score must be a number from 0 to 1. "
                    "scores must be an object with numeric keys correctness, clarity, novelty, difficulty_fit, "
                    "and usefulness. verdict must be accept when overall_score is greater than or equal to "
                    "min_score, otherwise reject. rationale must be a short string. Do not put score dimensions "
                    "at the top level."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        result = normalize_judgment(await self.model.complete_json(messages), min_score=self.min_score)
        return Judgment(
            id=str(uuid.uuid4()),
            candidate_id=candidate.id,
            result=result,
            judge_model=self.model.name,
            prompt_messages=messages,
        )


def _validate_schema(task: TaskSpec, item: dict[str, Any]) -> dict[str, Any]:
    return validate_item(task.output_schema, item)


def normalize_judgment(result: dict[str, Any], min_score: float) -> dict[str, Any]:
    scores = _extract_scores(result)
    overall_score = _extract_overall_score(result, scores)
    verdict = "accept" if overall_score >= min_score else "reject"
    rationale = str(result.get("rationale", "Normalized judge output."))
    return {
        "overall_score": overall_score,
        "scores": scores,
        "verdict": verdict,
        "rationale": rationale,
    }


def _extract_scores(result: dict[str, Any]) -> dict[str, float]:
    nested_scores = result.get("scores")
    if isinstance(nested_scores, dict):
        source = nested_scores
    else:
        source = result
    return {field: _clamp_score(source.get(field, 0.0)) for field in SCORE_FIELDS}


def _extract_overall_score(result: dict[str, Any], scores: dict[str, float]) -> float:
    if "overall_score" in result:
        return _clamp_score(result["overall_score"])
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 3)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(min(1.0, max(0.0, score)), 3)
