from __future__ import annotations

import asyncio

from synthetic_data_engine.models.local import LocalDeterministicModel
from synthetic_data_engine.pipeline import run_pipeline
from synthetic_data_engine.storage.sqlite import SqliteStore
from synthetic_data_engine.tasks.loader import load_task_spec


def test_local_pipeline_exports_jsonl(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")
    output = tmp_path / "dataset.jsonl"

    try:
        summary = asyncio.run(
            run_pipeline(
                store=store,
                task=task,
                generator_model=LocalDeterministicModel(),
                judge_model=LocalDeterministicModel(),
                count=3,
                min_score=0.5,
                output_path=output,
            )
        )
    finally:
        store.close()

    assert summary.generated == 3
    assert summary.judged == 3
    assert summary.exported == 3
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
    assert "candidate_id" in rows[0]
