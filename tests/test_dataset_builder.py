from __future__ import annotations

from synthetic_data_engine.dataset.builder import build_dataset_rows


def test_build_dataset_rows_caps_by_score_order():
    rows = build_dataset_rows(
        [
            _record("low", score=0.81, difficulty="easy"),
            _record("high", score=0.95, difficulty="hard"),
            _record("mid", score=0.9, difficulty="medium"),
        ],
        min_score=0.8,
        max_items=2,
    )

    assert [row["prompt"] for row in rows] == ["high", "mid"]


def test_build_dataset_rows_balances_difficulty_when_capped():
    rows = build_dataset_rows(
        [
            _record("easy-best", score=0.99, difficulty="easy"),
            _record("easy-next", score=0.98, difficulty="easy"),
            _record("medium-best", score=0.97, difficulty="medium"),
            _record("hard-best", score=0.96, difficulty="hard"),
        ],
        min_score=0.8,
        max_items=4,
        difficulty_distribution={"easy": 0.25, "medium": 0.5, "hard": 0.25},
    )

    assert [row["metadata"]["difficulty"] for row in rows] == ["easy", "medium", "hard", "easy"]


def _record(prompt: str, score: float, difficulty: str) -> dict:
    return {
        "candidate_id": prompt,
        "run_id": "run",
        "generator_model": "generator",
        "judge_model": "judge",
        "score": score,
        "verdict": "accept",
        "item": {
            "prompt": prompt,
            "answer": f"answer for {prompt}",
            "metadata": {"difficulty": difficulty},
        },
    }
