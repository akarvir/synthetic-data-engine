from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from synthetic_data_engine.dataset.builder import build_dataset_rows


def summarize_records(
    records: list[dict[str, Any]],
    min_score: float,
    candidate_count: int | None = None,
    failure_count: int = 0,
    max_items: int | None = None,
    difficulty_distribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    verdicts = Counter(str(record["verdict"]) for record in records)
    accepted_rows = build_dataset_rows(
        records,
        min_score=min_score,
        max_items=max_items,
        difficulty_distribution=difficulty_distribution,
    )

    return {
        "candidate_count": candidate_count if candidate_count is not None else len(records),
        "judged_count": len(records),
        "accepted_count": len(accepted_rows),
        "failure_count": failure_count,
        "rejected_count": verdicts.get("reject", 0),
        "verdicts": dict(sorted(verdicts.items())),
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_score": round(mean(scores), 4) if scores else None,
        "selection_min_score": min_score,
        "selection_max_items": max_items,
        "selection_difficulty_distribution": difficulty_distribution or {},
    }
