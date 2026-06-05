from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from synthetic_data_engine.dataset.builder import build_dataset_rows
from synthetic_data_engine.dataset.exporters import write_jsonl
from synthetic_data_engine.dataset.report import summarize_records
from synthetic_data_engine.generation.generator import Generator
from synthetic_data_engine.judging.judge import Judge
from synthetic_data_engine.models.base import ModelClient
from synthetic_data_engine.retry import with_retries
from synthetic_data_engine.storage.sqlite import SqliteStore
from synthetic_data_engine.tasks.spec import TaskSpec


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    generated: int
    judged: int
    exported: int
    output_path: Path | None


async def generate_candidates(
    store: SqliteStore,
    task: TaskSpec,
    model: ModelClient,
    count: int,
    run_id: str | None = None,
    concurrency: int = 4,
    retries: int = 2,
) -> str:
    actual_run_id = run_id or str(uuid.uuid4())
    if run_id is None:
        store.create_run(actual_run_id, task)

    generator = Generator(model)
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_index(index: int) -> None:
        async with semaphore:
            try:
                candidate = await with_retries(lambda: generator.generate_one(task, index), retries=retries)
            except Exception as exc:
                store.save_failure(str(uuid.uuid4()), actual_run_id, "generate", str(index), exc)
                return
            store.save_candidate(actual_run_id, candidate)

    await asyncio.gather(*(generate_index(index) for index in range(count)))
    return actual_run_id


async def judge_candidates(
    store: SqliteStore,
    run_id: str,
    model: ModelClient,
    min_score: float,
    concurrency: int = 4,
    retries: int = 2,
) -> int:
    task = store.get_task_for_run(run_id)
    candidates = store.list_candidates(run_id, only_unjudged=True)
    initial_judgment_count = store.run_counts(run_id)["judgment_count"]
    judge = Judge(model=model, min_score=min_score)
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_candidate(candidate_id: int) -> None:
        candidate = candidates[candidate_id]
        async with semaphore:
            try:
                judgment = await with_retries(lambda: judge.judge_one(task, candidate), retries=retries)
            except Exception as exc:
                store.save_failure(str(uuid.uuid4()), run_id, "judge", candidate.id, exc)
                return
            store.save_judgment(run_id, judgment)

    await asyncio.gather(*(judge_candidate(index) for index in range(len(candidates))))
    return store.run_counts(run_id)["judgment_count"] - initial_judgment_count


def export_dataset(store: SqliteStore, run_id: str, output_path: str | Path, min_score: float) -> int:
    rows = build_dataset_rows(store.dataset_records(run_id), min_score=min_score)
    write_jsonl(output_path, rows)
    return len(rows)


def report_run(store: SqliteStore, run_id: str, min_score: float) -> dict[str, object]:
    records = store.dataset_records(run_id)
    counts = store.run_counts(run_id)
    return {
        **store.run_metadata(run_id),
        "summary": summarize_records(
            records,
            min_score=min_score,
            candidate_count=counts["candidate_count"],
            failure_count=counts["failure_count"],
        ),
    }


async def run_pipeline(
    store: SqliteStore,
    task: TaskSpec,
    generator_model: ModelClient,
    judge_model: ModelClient,
    count: int,
    min_score: float,
    output_path: str | Path | None,
    concurrency: int = 4,
    retries: int = 2,
) -> RunSummary:
    run_id = await generate_candidates(
        store=store,
        task=task,
        model=generator_model,
        count=count,
        concurrency=concurrency,
        retries=retries,
    )
    generated = store.run_counts(run_id)["candidate_count"]
    judged = await judge_candidates(
        store=store,
        run_id=run_id,
        model=judge_model,
        min_score=min_score,
        concurrency=concurrency,
        retries=retries,
    )
    exported = 0
    if output_path is not None:
        exported = export_dataset(store=store, run_id=run_id, output_path=output_path, min_score=min_score)
    return RunSummary(
        run_id=run_id,
        generated=generated,
        judged=judged,
        exported=exported,
        output_path=Path(output_path) if output_path is not None else None,
    )
