from __future__ import annotations

import asyncio
import json
import sqlite3

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


def test_candidate_trace_includes_prompts_and_judgment(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        summary = asyncio.run(
            run_pipeline(
                store=store,
                task=task,
                generator_model=LocalDeterministicModel(),
                judge_model=LocalDeterministicModel(),
                count=1,
                min_score=0.8,
                output_path=None,
            )
        )
        candidate_id = store.dataset_records(summary.run_id)[0]["candidate_id"]
        trace = store.candidate_trace(candidate_id)
    finally:
        store.close()

    assert trace["candidate_id"] == candidate_id
    assert trace["generator_messages"][0]["role"] == "system"
    assert trace["judge_messages"][0]["role"] == "system"
    assert trace["judgment"]["verdict"] == "accept"


def test_store_migrates_prompt_columns(tmp_path):
    db_path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            create table runs (
                id text primary key,
                task_name text not null,
                task_spec text not null,
                created_at text not null
            );
            create table candidates (
                id text primary key,
                run_id text not null references runs(id),
                item_json text not null,
                generator_model text not null,
                created_at text not null
            );
            create table judgments (
                id text primary key,
                candidate_id text not null references candidates(id),
                run_id text not null references runs(id),
                judgment_json text not null,
                judge_model text not null,
                score real not null,
                verdict text not null,
                created_at text not null
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    store = SqliteStore(db_path)
    try:
        candidate_columns = {
            row["name"]
            for row in store.connection.execute("pragma table_info(candidates)").fetchall()
        }
        judgment_columns = {
            row["name"]
            for row in store.connection.execute("pragma table_info(judgments)").fetchall()
        }
    finally:
        store.close()

    assert "prompt_messages_json" in candidate_columns
    assert "prompt_messages_json" in judgment_columns


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


def test_generate_candidates_appends_without_duplicate_indexes(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=LocalDeterministicModel(),
                count=1,
            )
        )
        asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=LocalDeterministicModel(),
                count=1,
                run_id=run_id,
            )
        )
        candidates = store.list_candidates(run_id)
    finally:
        store.close()

    prompts = [candidate.item["prompt"] for candidate in candidates]
    assert len(prompts) == 2
    assert prompts[0] != prompts[1]


def test_generate_candidates_continues_after_failed_attempts(tmp_path):
    task = load_task_spec("tasks/general-instruction.yaml")
    store = SqliteStore(tmp_path / "runs.sqlite")

    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=FailingModel(),
                count=1,
                retries=0,
            )
        )
        asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=LocalDeterministicModel(),
                count=1,
                run_id=run_id,
            )
        )
        candidates = store.list_candidates(run_id)
        attempts = store.generation_attempt_count(run_id)
    finally:
        store.close()

    assert len(candidates) == 1
    assert attempts == 2
    assert '"index": 1' in candidates[0].prompt_messages[1]["content"]


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
