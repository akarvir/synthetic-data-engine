from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthetic_data_engine.generation.generator import Candidate
from synthetic_data_engine.judging.judge import Judgment
from synthetic_data_engine.tasks.spec import TaskSpec


class SqliteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.migrate()

    def close(self) -> None:
        self.connection.close()

    def migrate(self) -> None:
        self.connection.executescript(
            """
            create table if not exists runs (
                id text primary key,
                task_name text not null,
                task_spec text not null,
                created_at text not null
            );

            create table if not exists candidates (
                id text primary key,
                run_id text not null references runs(id),
                item_json text not null,
                generator_model text not null,
                prompt_messages_json text not null default '[]',
                created_at text not null
            );

            create table if not exists judgments (
                id text primary key,
                candidate_id text not null references candidates(id),
                run_id text not null references runs(id),
                judgment_json text not null,
                judge_model text not null,
                prompt_messages_json text not null default '[]',
                score real not null,
                verdict text not null,
                created_at text not null
            );

            create table if not exists failures (
                id text primary key,
                run_id text not null references runs(id),
                phase text not null,
                item_ref text,
                error_type text not null,
                error_message text not null,
                created_at text not null
            );
            """
        )
        self._ensure_column("candidates", "prompt_messages_json", "text not null default '[]'")
        self._ensure_column("judgments", "prompt_messages_json", "text not null default '[]'")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"pragma table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"alter table {table} add column {column} {definition}")

    def create_run(self, run_id: str, task: TaskSpec) -> None:
        self.connection.execute(
            "insert into runs (id, task_name, task_spec, created_at) values (?, ?, ?, ?)",
            (run_id, task.name, json.dumps(task.to_dict(), sort_keys=True), _now()),
        )
        self.connection.commit()

    def save_failure(
        self,
        failure_id: str,
        run_id: str,
        phase: str,
        item_ref: str | None,
        error: BaseException,
    ) -> None:
        self.connection.execute(
            """
            insert into failures (id, run_id, phase, item_ref, error_type, error_message, created_at)
            values (?, ?, ?, ?, ?, ?, ?)
            """,
            (failure_id, run_id, phase, item_ref, type(error).__name__, str(error), _now()),
        )
        self.connection.commit()

    def get_task_for_run(self, run_id: str) -> TaskSpec:
        row = self.connection.execute("select task_spec from runs where id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        return TaskSpec.from_mapping(json.loads(row["task_spec"]))

    def save_candidate(self, run_id: str, candidate: Candidate) -> None:
        self.connection.execute(
            """
            insert into candidates (id, run_id, item_json, generator_model, prompt_messages_json, created_at)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.id,
                run_id,
                json.dumps(candidate.item, sort_keys=True),
                candidate.generator_model,
                json.dumps(candidate.prompt_messages, sort_keys=True),
                _now(),
            ),
        )
        self.connection.commit()

    def save_judgment(self, run_id: str, judgment: Judgment) -> None:
        self.connection.execute(
            """
            insert into judgments (
                id, candidate_id, run_id, judgment_json, judge_model, prompt_messages_json, score, verdict, created_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                judgment.id,
                judgment.candidate_id,
                run_id,
                json.dumps(judgment.result, sort_keys=True),
                judgment.judge_model,
                json.dumps(judgment.prompt_messages, sort_keys=True),
                judgment.score,
                judgment.verdict,
                _now(),
            ),
        )
        self.connection.commit()

    def list_candidates(self, run_id: str, only_unjudged: bool = False) -> list[Candidate]:
        if only_unjudged:
            rows = self.connection.execute(
                """
                select c.* from candidates c
                left join judgments j on j.candidate_id = c.id
                where c.run_id = ? and j.id is null
                order by c.created_at
                """,
                (run_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "select * from candidates where run_id = ? order by created_at",
                (run_id,),
            ).fetchall()
        return [
            Candidate(
                id=row["id"],
                item=json.loads(row["item_json"]),
                generator_model=row["generator_model"],
                prompt_messages=json.loads(row["prompt_messages_json"]),
            )
            for row in rows
        ]

    def dataset_records(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            select
                c.id as candidate_id,
                c.run_id as run_id,
                c.item_json as item_json,
                c.generator_model as generator_model,
                j.judge_model as judge_model,
                j.score as score,
                j.verdict as verdict
            from candidates c
            join judgments j on j.candidate_id = c.id
            where c.run_id = ?
            order by j.score desc
            """,
            (run_id,),
        ).fetchall()
        return [
            {
                "candidate_id": row["candidate_id"],
                "run_id": row["run_id"],
                "item": json.loads(row["item_json"]),
                "generator_model": row["generator_model"],
                "judge_model": row["judge_model"],
                "score": float(row["score"]),
                "verdict": row["verdict"],
            }
            for row in rows
        ]

    def candidate_trace(self, candidate_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            select
                c.id as candidate_id,
                c.run_id as run_id,
                c.item_json as item_json,
                c.generator_model as generator_model,
                c.prompt_messages_json as generator_messages_json,
                c.created_at as candidate_created_at,
                j.id as judgment_id,
                j.judgment_json as judgment_json,
                j.judge_model as judge_model,
                j.prompt_messages_json as judge_messages_json,
                j.score as score,
                j.verdict as verdict,
                j.created_at as judgment_created_at
            from candidates c
            left join judgments j on j.candidate_id = c.id
            where c.id = ?
            order by j.created_at desc
            limit 1
            """,
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Candidate not found: {candidate_id}")
        return {
            "candidate_id": row["candidate_id"],
            "run_id": row["run_id"],
            "item": json.loads(row["item_json"]),
            "generator_model": row["generator_model"],
            "generator_messages": json.loads(row["generator_messages_json"]),
            "candidate_created_at": row["candidate_created_at"],
            "judgment_id": row["judgment_id"],
            "judgment": json.loads(row["judgment_json"]) if row["judgment_json"] else None,
            "judge_model": row["judge_model"],
            "judge_messages": json.loads(row["judge_messages_json"]) if row["judge_messages_json"] else [],
            "score": float(row["score"]) if row["score"] is not None else None,
            "verdict": row["verdict"],
            "judgment_created_at": row["judgment_created_at"],
        }

    def run_metadata(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "select id, task_name, task_spec, created_at from runs where id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        return {
            "run_id": row["id"],
            "task_name": row["task_name"],
            "created_at": row["created_at"],
            "task": json.loads(row["task_spec"]),
        }

    def latest_run_id(self) -> str:
        row = self.connection.execute(
            "select id from runs order by created_at desc limit 1",
        ).fetchone()
        if row is None:
            raise ValueError("No runs found.")
        return str(row["id"])

    def run_counts(self, run_id: str) -> dict[str, int]:
        candidate_count = self.connection.execute(
            "select count(*) as count from candidates where run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        judgment_count = self.connection.execute(
            "select count(*) as count from judgments where run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        failure_count = self.connection.execute(
            "select count(*) as count from failures where run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        return {
            "candidate_count": int(candidate_count),
            "judgment_count": int(judgment_count),
            "failure_count": int(failure_count),
        }

    def generation_attempt_count(self, run_id: str) -> int:
        candidate_count = self.connection.execute(
            "select count(*) as count from candidates where run_id = ?",
            (run_id,),
        ).fetchone()["count"]
        failure_count = self.connection.execute(
            "select count(*) as count from failures where run_id = ? and phase = 'generate'",
            (run_id,),
        ).fetchone()["count"]
        return int(candidate_count) + int(failure_count)

    def list_runs(self, limit: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            select
                r.id as run_id,
                r.task_name as task_name,
                r.created_at as created_at,
                count(distinct c.id) as candidate_count,
                count(distinct j.id) as judgment_count,
                count(distinct f.id) as failure_count
            from runs r
            left join candidates c on c.run_id = r.id
            left join judgments j on j.run_id = r.id
            left join failures f on f.run_id = r.id
            group by r.id
            order by r.created_at desc
            limit ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "task_name": row["task_name"],
                "created_at": row["created_at"],
                "candidate_count": int(row["candidate_count"]),
                "judgment_count": int(row["judgment_count"]),
                "failure_count": int(row["failure_count"]),
            }
            for row in rows
        ]

    def list_failures(self, run_id: str, limit: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            select id, run_id, phase, item_ref, error_type, error_message, created_at
            from failures
            where run_id = ?
            order by created_at desc
            limit ?
            """,
            (run_id, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "phase": row["phase"],
                "item_ref": row["item_ref"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
