from __future__ import annotations

import asyncio
import json

from synthetic_data_engine.generation.generator import Candidate
from synthetic_data_engine.judging.judge import Judge, normalize_judgment
from synthetic_data_engine.models.base import Message, ModelClient
from synthetic_data_engine.tasks.loader import load_task_spec


def test_normalize_judgment_accepts_top_level_dimension_scores():
    result = normalize_judgment(
        {
            "clarity": 0.9,
            "correctness": 1,
            "difficulty_fit": 1,
            "novelty": 0.7,
            "usefulness": 0.8,
            "verdict": "reject",
        },
        min_score=0.8,
    )

    assert result == {
        "overall_score": 0.88,
        "scores": {
            "correctness": 1.0,
            "clarity": 0.9,
            "novelty": 0.7,
            "difficulty_fit": 1.0,
            "usefulness": 0.8,
        },
        "verdict": "accept",
        "rationale": "Normalized judge output.",
    }


def test_normalize_judgment_preserves_strict_shape_and_recomputes_verdict():
    result = normalize_judgment(
        {
            "overall_score": 0.75,
            "scores": {
                "correctness": 1,
                "clarity": 1,
                "novelty": 0.2,
                "difficulty_fit": 0.8,
                "usefulness": 0.9,
            },
            "verdict": "accept",
            "rationale": "Good but below threshold.",
        },
        min_score=0.8,
    )

    assert result["overall_score"] == 0.75
    assert result["verdict"] == "reject"
    assert result["rationale"] == "Good but below threshold."


def test_judge_prompt_requires_exact_output_shape():
    model = CapturingJudgeModel()
    task = load_task_spec("tasks/general-instruction.yaml")
    candidate = Candidate(
        id="candidate",
        item={"prompt": "Question?", "answer": "Answer."},
        generator_model="test",
        prompt_messages=[],
    )

    judgment = asyncio.run(Judge(model=model, min_score=0.8).judge_one(task, candidate))

    assert judgment.verdict == "accept"
    assert "overall_score, scores, verdict, rationale" in model.messages[0]["content"]
    assert json.loads(model.messages[1]["content"])["mode"] == "judge"


class CapturingJudgeModel(ModelClient):
    name = "capturing"

    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def complete_json(self, messages: list[Message], schema=None):
        self.messages = messages
        return {
            "clarity": 0.9,
            "correctness": 1,
            "difficulty_fit": 1,
            "novelty": 0.7,
            "usefulness": 0.8,
            "verdict": "reject",
        }
