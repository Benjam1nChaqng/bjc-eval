"""SQLite storage layer for swe-judge.

Single-file SQLite database. No ORM — just sqlite3 + Pydantic serialization.
Keeps the dependency surface small. For analytics queries beyond what this
module exposes, point DuckDB at the .sqlite file directly.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from swe_judge.tasks import HumanScore, Run, Score, Task


SCHEMA = """
CREATE TABLE IF NOT EXISTS task (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    difficulty INTEGER NOT NULL,
    source TEXT NOT NULL,
    prompt TEXT NOT NULL,
    reference_solution TEXT NOT NULL,
    test_cases_json TEXT NOT NULL,  -- JSON-encoded list[TestCase]
    tags_json TEXT NOT NULL,         -- JSON-encoded list[str]
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    id TEXT PRIMARY KEY,
    model_under_test TEXT NOT NULL,
    judge_models_json TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS score (
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    dimension TEXT NOT NULL,
    value INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    anchor_matched TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id, judge_model, dimension),
    FOREIGN KEY (run_id) REFERENCES run(id),
    FOREIGN KEY (task_id) REFERENCES task(id)
);

CREATE TABLE IF NOT EXISTS human_score (
    task_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    scorer TEXT NOT NULL,
    value INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (task_id, dimension, scorer),
    FOREIGN KEY (task_id) REFERENCES task(id)
);

CREATE INDEX IF NOT EXISTS idx_score_judge ON score(judge_model);
CREATE INDEX IF NOT EXISTS idx_score_dimension ON score(dimension);
CREATE INDEX IF NOT EXISTS idx_human_score_dim ON human_score(dimension);
"""


class Storage:
    """Thin wrapper over sqlite3 for swe-judge persistence."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cursor() as cur:
            cur.executescript(SCHEMA)

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        finally:
            conn.close()

    # ---- tasks ----

    def insert_task(self, task: Task) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO task
                    (id, category, difficulty, source, prompt, reference_solution,
                     test_cases_json, tags_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.category,
                    task.difficulty,
                    task.source,
                    task.prompt,
                    task.reference_solution,
                    json.dumps([tc.model_dump() for tc in task.test_cases]),
                    json.dumps(task.tags),
                    task.created_at.isoformat(),
                ),
            )

    def list_tasks(self) -> list[Task]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM task").fetchall()
        return [self._row_to_task(r) for r in rows]

    @staticmethod
    def _row_to_task(row: tuple[object, ...]) -> Task:
        from swe_judge.tasks import TestCase

        (id_, category, difficulty, source, prompt, reference_solution,
         test_cases_json, tags_json, created_at) = row
        return Task(
            id=id_,  # type: ignore[arg-type]
            category=category,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            source=source,  # type: ignore[arg-type]
            prompt=prompt,  # type: ignore[arg-type]
            reference_solution=reference_solution,  # type: ignore[arg-type]
            test_cases=[TestCase(**tc) for tc in json.loads(test_cases_json)],  # type: ignore[arg-type]
            tags=json.loads(tags_json),  # type: ignore[arg-type]
            created_at=created_at,  # type: ignore[arg-type]
        )

    # ---- runs ----

    def insert_run(self, run: Run) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO run
                    (id, model_under_test, judge_models_json, dataset_version,
                     started_at, completed_at, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.model_under_test,
                    json.dumps(run.judge_models),
                    run.dataset_version,
                    run.started_at.isoformat(),
                    run.completed_at.isoformat() if run.completed_at else None,
                    json.dumps(run.config),
                ),
            )

    # ---- scores ----

    def insert_scores(self, scores: list[Score]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO score
                    (run_id, task_id, judge_model, dimension, value, rationale, anchor_matched)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (s.run_id, s.task_id, s.judge_model, s.dimension,
                     s.value, s.rationale, s.anchor_matched)
                    for s in scores
                ],
            )

    def scores_for_run(self, run_id: str) -> list[Score]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM score WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [
            Score(
                run_id=r[0], task_id=r[1], judge_model=r[2], dimension=r[3],
                value=r[4], rationale=r[5], anchor_matched=r[6],
            )
            for r in rows
        ]

    # ---- human scores ----

    def insert_human_scores(self, human_scores: list[HumanScore]) -> None:
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO human_score
                    (task_id, dimension, scorer, value, rationale, scored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (h.task_id, h.dimension, h.scorer, h.value,
                     h.rationale, h.scored_at.isoformat())
                    for h in human_scores
                ],
            )

    def all_human_scores(self) -> list[HumanScore]:
        with self._cursor() as cur:
            rows = cur.execute("SELECT * FROM human_score").fetchall()
        return [
            HumanScore(
                task_id=r[0], dimension=r[1], scorer=r[2],  # type: ignore[arg-type]
                value=r[3], rationale=r[4], scored_at=r[5],  # type: ignore[arg-type]
            )
            for r in rows
        ]
