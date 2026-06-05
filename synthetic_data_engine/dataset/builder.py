from __future__ import annotations

from typing import Any

from synthetic_data_engine.dataset.dedupe import item_hash


def build_dataset_rows(records: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda row: row["score"], reverse=True):
        if record["score"] < min_score or record["verdict"] != "accept":
            continue
        digest = item_hash(record["item"])
        if digest in seen:
            continue
        seen.add(digest)
        accepted.append(
            {
                **record["item"],
                "metadata": {
                    **dict(record["item"].get("metadata", {})),
                    "candidate_id": record["candidate_id"],
                    "run_id": record["run_id"],
                    "score": record["score"],
                    "generator_model": record["generator_model"],
                    "judge_model": record["judge_model"],
                },
            }
        )

    return accepted
