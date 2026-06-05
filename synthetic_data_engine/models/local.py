from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from synthetic_data_engine.models.base import Message, ModelClient


class LocalDeterministicModel(ModelClient):
    """Deterministic model for development, tests, and dry runs."""

    name = "local-deterministic"

    async def complete_json(self, messages: list[Message], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _extract_payload(messages)
        mode = payload.get("mode")
        if mode == "judge":
            return self._judge(payload)
        return self._generate(payload)

    def _generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload["task"]
        index = int(payload.get("index", 0))
        topic = _pick(task.get("topics") or [task["domain"]], index)
        difficulty = _pick(task.get("difficulty") or ["medium"], index)
        seed = hashlib.sha256(f"{task['name']}:{topic}:{difficulty}:{index}".encode()).hexdigest()[:10]

        return {
            "prompt": (
                f"Create a {difficulty} {topic} response for task '{task['name']}'. "
                f"Explain the key reasoning and include a concrete example. Seed: {seed}."
            ),
            "answer": (
                f"A strong answer should address {topic}, match {difficulty} difficulty, "
                f"and satisfy the task requirements: {', '.join(task.get('requirements', [])) or 'none'}."
            ),
            "metadata": {
                "topic": topic,
                "difficulty": difficulty,
                "seed": seed,
            },
        }

    def _judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidate = payload["candidate"]
        task = payload["task"]
        serialized = json.dumps(candidate, sort_keys=True)
        jitter = int(hashlib.sha256(serialized.encode()).hexdigest()[:4], 16) / 65535
        required_fields = task.get("output_schema", {}).get("required", [])
        present_ratio = sum(1 for field in required_fields if candidate.get(field)) / max(1, len(required_fields))
        length_score = min(1.0, len(serialized) / 600)
        overall = round((0.72 * present_ratio) + (0.22 * length_score) + (0.06 * jitter), 3)
        verdict = "accept" if overall >= float(payload.get("min_score", 0.8)) else "reject"
        return {
            "overall_score": overall,
            "scores": {
                "schema": round(present_ratio, 3),
                "substance": round(length_score, 3),
                "novelty": round(jitter, 3),
            },
            "verdict": verdict,
            "rationale": "Scored by deterministic local heuristics.",
        }


def _extract_payload(messages: list[Message]) -> dict[str, Any]:
    content = messages[-1]["content"]
    return json.loads(content)


def _pick(values: list[str], index: int) -> str:
    rng = random.Random(index)
    return values[rng.randrange(len(values))]
