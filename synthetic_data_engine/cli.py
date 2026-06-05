from __future__ import annotations

import argparse
import asyncio
import json

from synthetic_data_engine.dataset.builder import build_dataset_rows
from synthetic_data_engine.models.factory import create_model
from synthetic_data_engine.pipeline import export_dataset, generate_candidates, judge_candidates, report_run, run_pipeline
from synthetic_data_engine.storage.sqlite import SqliteStore
from synthetic_data_engine.tasks.loader import load_task_spec
from synthetic_data_engine.tasks.summary import summarize_task


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sde", description="Synthetic data engine for LLM datasets.")
    parser.add_argument("--db", default="runs/sde.sqlite", help="SQLite database path.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Generate, judge, and export in one command.")
    _add_task_arg(run)
    _add_model_args(run)
    run.add_argument("--count", type=int, default=10)
    run.add_argument("--min-score", type=float)
    run.add_argument("--out", default="datasets/output.jsonl")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--retries", type=int, default=2)
    run.set_defaults(func=_run)

    generate = subparsers.add_parser("generate", help="Generate candidates for a task.")
    _add_task_arg(generate)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--provider", default="local", choices=["local", "openai-compatible"])
    generate.add_argument("--model")
    generate.add_argument("--concurrency", type=int, default=4)
    generate.add_argument("--retries", type=int, default=2)
    generate.set_defaults(func=_generate)

    validate_task = subparsers.add_parser("validate-task", help="Validate and summarize a task spec.")
    _add_task_arg(validate_task)
    validate_task.set_defaults(func=_validate_task)

    list_runs = subparsers.add_parser("list-runs", help="List recent runs with candidate and judgment counts.")
    list_runs.add_argument("--limit", type=int, default=20)
    list_runs.set_defaults(func=_list_runs)

    list_failures = subparsers.add_parser("list-failures", help="List model or validation failures for a run.")
    list_failures.add_argument("--run-id", default="latest")
    list_failures.add_argument("--limit", type=int, default=20)
    list_failures.set_defaults(func=_list_failures)

    judge = subparsers.add_parser("judge", help="Judge unjudged candidates in a run.")
    judge.add_argument("--run-id", default="latest")
    judge.add_argument("--provider", default="local", choices=["local", "openai-compatible"])
    judge.add_argument("--model")
    judge.add_argument("--min-score", type=float)
    judge.add_argument("--concurrency", type=int, default=4)
    judge.add_argument("--retries", type=int, default=2)
    judge.set_defaults(func=_judge)

    build_dataset = subparsers.add_parser("build-dataset", help="Export accepted candidates from a run.")
    build_dataset.add_argument("--run-id", default="latest")
    build_dataset.add_argument("--min-score", type=float)
    build_dataset.add_argument("--out", default="datasets/output.jsonl")
    build_dataset.set_defaults(func=_build_dataset)

    report = subparsers.add_parser("report", help="Print run quality and selection summary.")
    report.add_argument("--run-id", default="latest")
    report.add_argument("--min-score", type=float)
    report.set_defaults(func=_report)

    inspect = subparsers.add_parser("inspect", help="Print accepted dataset rows as JSON lines.")
    inspect.add_argument("--run-id", default="latest")
    inspect.add_argument("--min-score", type=float)
    inspect.set_defaults(func=_inspect)

    return parser


def _add_task_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="Path to a YAML or JSON task spec.")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generator-provider", default="local", choices=["local", "openai-compatible"])
    parser.add_argument("--generator-model")
    parser.add_argument("--judge-provider", default="local", choices=["local", "openai-compatible"])
    parser.add_argument("--judge-model")


def _run(args: argparse.Namespace) -> None:
    task = load_task_spec(args.task)
    min_score = _task_min_score(task, args.min_score)
    store = SqliteStore(args.db)
    try:
        summary = asyncio.run(
            run_pipeline(
                store=store,
                task=task,
                generator_model=create_model(args.generator_provider, args.generator_model),
                judge_model=create_model(args.judge_provider, args.judge_model),
                count=args.count,
                min_score=min_score,
                output_path=args.out,
                concurrency=args.concurrency,
                retries=args.retries,
            )
        )
    finally:
        store.close()

    print(f"run_id={summary.run_id}")
    print(f"generated={summary.generated}")
    print(f"judged={summary.judged}")
    print(f"exported={summary.exported}")
    if summary.output_path:
        print(f"output={summary.output_path}")


def _generate(args: argparse.Namespace) -> None:
    task = load_task_spec(args.task)
    store = SqliteStore(args.db)
    try:
        run_id = asyncio.run(
            generate_candidates(
                store=store,
                task=task,
                model=create_model(args.provider, args.model),
                count=args.count,
                concurrency=args.concurrency,
                retries=args.retries,
            )
        )
        generated = store.run_counts(run_id)["candidate_count"]
    finally:
        store.close()
    print(f"run_id={run_id}")
    print(f"requested={args.count}")
    print(f"generated={generated}")


def _validate_task(args: argparse.Namespace) -> None:
    task = load_task_spec(args.task)
    print(json.dumps(summarize_task(task), indent=2, sort_keys=True))


def _list_runs(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        runs = store.list_runs(limit=args.limit)
    finally:
        store.close()
    print(json.dumps(runs, indent=2, sort_keys=True))


def _list_failures(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        run_id = _resolve_run_id(store, args.run_id)
        failures = store.list_failures(run_id=run_id, limit=args.limit)
    finally:
        store.close()
    print(json.dumps(failures, indent=2, sort_keys=True))


def _judge(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        run_id = _resolve_run_id(store, args.run_id)
        min_score = _run_min_score(store, run_id, args.min_score)
        judged = asyncio.run(
            judge_candidates(
                store=store,
                run_id=run_id,
                model=create_model(args.provider, args.model),
                min_score=min_score,
                concurrency=args.concurrency,
                retries=args.retries,
            )
        )
    finally:
        store.close()
    print(f"run_id={run_id}")
    print(f"judged={judged}")


def _build_dataset(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        run_id = _resolve_run_id(store, args.run_id)
        min_score = _run_min_score(store, run_id, args.min_score)
        exported = export_dataset(store=store, run_id=run_id, output_path=args.out, min_score=min_score)
    finally:
        store.close()
    print(f"run_id={run_id}")
    print(f"exported={exported}")
    print(f"output={args.out}")


def _inspect(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        run_id = _resolve_run_id(store, args.run_id)
        min_score = _run_min_score(store, run_id, args.min_score)
        rows = build_dataset_rows(store.dataset_records(run_id), min_score=min_score)
    finally:
        store.close()
    for row in rows:
        print(json.dumps(row, sort_keys=True))


def _report(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        run_id = _resolve_run_id(store, args.run_id)
        min_score = _run_min_score(store, run_id, args.min_score)
        report = report_run(store=store, run_id=run_id, min_score=min_score)
    finally:
        store.close()
    print(json.dumps(report, indent=2, sort_keys=True))


def _resolve_run_id(store: SqliteStore, run_id: str) -> str:
    if run_id == "latest":
        return store.latest_run_id()
    return run_id


def _run_min_score(store: SqliteStore, run_id: str, provided: float | None) -> float:
    if provided is not None:
        return provided
    return store.get_task_for_run(run_id).min_score


def _task_min_score(task, provided: float | None) -> float:
    if provided is not None:
        return provided
    return task.min_score
