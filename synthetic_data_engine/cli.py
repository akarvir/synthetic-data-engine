from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from synthetic_data_engine.dataset.builder import build_dataset_rows
from synthetic_data_engine.dataset.exporters import write_jsonl
from synthetic_data_engine.models.factory import create_model
from synthetic_data_engine.pipeline import export_dataset, generate_candidates, judge_candidates, run_pipeline
from synthetic_data_engine.storage.sqlite import SqliteStore
from synthetic_data_engine.tasks.loader import load_task_spec


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
    run.add_argument("--min-score", type=float, default=0.8)
    run.add_argument("--out", default="datasets/output.jsonl")
    run.add_argument("--concurrency", type=int, default=4)
    run.set_defaults(func=_run)

    generate = subparsers.add_parser("generate", help="Generate candidates for a task.")
    _add_task_arg(generate)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--provider", default="local", choices=["local", "openai-compatible"])
    generate.add_argument("--model")
    generate.add_argument("--concurrency", type=int, default=4)
    generate.set_defaults(func=_generate)

    judge = subparsers.add_parser("judge", help="Judge unjudged candidates in a run.")
    judge.add_argument("--run-id", required=True)
    judge.add_argument("--provider", default="local", choices=["local", "openai-compatible"])
    judge.add_argument("--model")
    judge.add_argument("--min-score", type=float, default=0.8)
    judge.add_argument("--concurrency", type=int, default=4)
    judge.set_defaults(func=_judge)

    build_dataset = subparsers.add_parser("build-dataset", help="Export accepted candidates from a run.")
    build_dataset.add_argument("--run-id", required=True)
    build_dataset.add_argument("--min-score", type=float, default=0.8)
    build_dataset.add_argument("--out", default="datasets/output.jsonl")
    build_dataset.set_defaults(func=_build_dataset)

    inspect = subparsers.add_parser("inspect", help="Print accepted dataset rows for a run.")
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--min-score", type=float, default=0.8)
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
    store = SqliteStore(args.db)
    try:
        summary = asyncio.run(
            run_pipeline(
                store=store,
                task=task,
                generator_model=create_model(args.generator_provider, args.generator_model),
                judge_model=create_model(args.judge_provider, args.judge_model),
                count=args.count,
                min_score=args.min_score,
                output_path=args.out,
                concurrency=args.concurrency,
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
            )
        )
    finally:
        store.close()
    print(f"run_id={run_id}")
    print(f"generated={args.count}")


def _judge(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        judged = asyncio.run(
            judge_candidates(
                store=store,
                run_id=args.run_id,
                model=create_model(args.provider, args.model),
                min_score=args.min_score,
                concurrency=args.concurrency,
            )
        )
    finally:
        store.close()
    print(f"run_id={args.run_id}")
    print(f"judged={judged}")


def _build_dataset(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        exported = export_dataset(store=store, run_id=args.run_id, output_path=args.out, min_score=args.min_score)
    finally:
        store.close()
    print(f"run_id={args.run_id}")
    print(f"exported={exported}")
    print(f"output={args.out}")


def _inspect(args: argparse.Namespace) -> None:
    store = SqliteStore(args.db)
    try:
        rows = build_dataset_rows(store.dataset_records(args.run_id), min_score=args.min_score)
    finally:
        store.close()
    for row in rows:
        print(row)
