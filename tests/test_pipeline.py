from __future__ import annotations

import asyncio
import json

from synthetic_data_engine.models.local import LocalDeterministicModel
from synthetic_data_engine.pipeline import generate_candidates, report_run, run_pipeline
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


def test_report_summarizes_run(tmp_path):
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
                count=4,
                min_score=0.8,
                output_path=output,
            )
        )
        report = report_run(store=store, run_id=summary.run_id, min_score=0.8, max_items=2)
    finally:
        store.close()

    assert report["run_id"] == summary.run_id
    assert report["task_name"] == "general-instruction"
    assert report["summary"]["candidate_count"] == 4
    assert report["summary"]["accepted_count"] == 2
    assert report["summary"]["selection_max_items"] == 2
    assert report["summary"]["verdicts"] == {"accept": 4}
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["metadata"]["run_id"] == summary.run_id


def test_report_counts_unjudged_candidates(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=LocalDeterministicModel(),
                count=2,
            )
        )
        report = report_run(store=store, run_id=run_id, min_score=0.99)
    finally:
        store.close()

    assert report["summary"]["candidate_count"] == 2
    assert report["summary"]["judged_count"] == 0
    assert report["summary"]["accepted_count"] == 0


def test_store_lists_runs_with_counts(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=LocalDeterministicModel(),
                count=2,
            )
        )
        runs = store.list_runs(limit=10)
    finally:
        store.close()

    assert runs == [
        {
            "run_id": run_id,
            "task_name": "general-instruction",
            "created_at": runs[0]["created_at"],
            "candidate_count": 2,
            "judgment_count": 0,
            "failure_count": 0,
        }
    ]


class FailingModel(LocalDeterministicModel):
    name = "failing"

    async def complete_json(self, messages, schema=None):
        raise RuntimeError("model unavailable")


def test_generation_failures_are_recorded(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=FailingModel(),
                count=2,
                retries=1,
            )
        )
        report = report_run(store=store, run_id=run_id, min_score=0.8)
        counts = store.run_counts(run_id)
        failures = store.list_failures(run_id=run_id, limit=10)
    finally:
        store.close()

    assert report["summary"]["candidate_count"] == 0
    assert report["summary"]["failure_count"] == 2
    assert counts["candidate_count"] == 0
    assert len(failures) == 2
    assert failures[0]["phase"] == "generate"
    assert failures[0]["error_type"] == "RuntimeError"
