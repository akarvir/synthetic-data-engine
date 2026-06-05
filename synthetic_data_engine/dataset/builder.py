from __future__ import annotations

from collections import defaultdict
from typing import Any

from synthetic_data_engine.dataset.dedupe import item_hash


def build_dataset_rows(
    records: list[dict[str, Any]],
    min_score: float,
    max_items: int | None = None,
    difficulty_distribution: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
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

    return select_rows(
        accepted,
        max_items=max_items,
        difficulty_distribution=difficulty_distribution,
    )


def select_rows(
    rows: list[dict[str, Any]],
    max_items: int | None = None,
    difficulty_distribution: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if max_items is None:
        return rows
    if max_items <= 0:
        return []
    capped_rows = rows[:max_items]
    if not difficulty_distribution:
        return capped_rows

    quotas = _allocate_quotas(difficulty_distribution, max_items)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        difficulty = str(row.get("metadata", {}).get("difficulty", ""))
        by_difficulty[difficulty].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for difficulty, quota in quotas.items():
        for row in by_difficulty.get(difficulty, [])[:quota]:
            selected.append(row)
            selected_ids.add(str(row["metadata"]["candidate_id"]))

    for row in rows:
        if len(selected) >= max_items:
            break
        candidate_id = str(row["metadata"]["candidate_id"])
        if candidate_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(candidate_id)

    return selected


def _allocate_quotas(distribution: dict[str, float], max_items: int) -> dict[str, int]:
    positive_weights = {key: float(value) for key, value in distribution.items() if float(value) > 0}
    total = sum(positive_weights.values())
    if total <= 0:
        return {}

    raw = {key: (value / total) * max_items for key, value in positive_weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = max_items - sum(quotas.values())
    remainders = sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)
    for key, _value in remainders[:remaining]:
        quotas[key] += 1
    return quotas
