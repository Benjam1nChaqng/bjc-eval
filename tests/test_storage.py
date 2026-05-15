"""Tests for the SQLite storage layer."""

from __future__ import annotations

from datetime import datetime, timezone

from swe_judge.storage import Storage
from swe_judge.tasks import HumanScore, Run, Score, Task


class TestStorage:
    def test_insert_and_list_tasks(self, storage: Storage, sample_task: Task) -> None:
        storage.insert_task(sample_task)
        loaded = storage.list_tasks()
        assert len(loaded) == 1
        assert loaded[0].id == sample_task.id
        assert loaded[0].category == sample_task.category
        assert loaded[0].tags == sample_task.tags

    def test_insert_idempotent(self, storage: Storage, sample_task: Task) -> None:
        storage.insert_task(sample_task)
        storage.insert_task(sample_task)
        loaded = storage.list_tasks()
        assert len(loaded) == 1

    def test_insert_run_and_scores(
        self, storage: Storage, sample_task: Task
    ) -> None:
        storage.insert_task(sample_task)
        run = Run(
            id="R-1",
            model_under_test="claude-opus-4-7",
            judge_models=["claude-opus-4-7", "gpt-5.2"],
            dataset_version="ABC123",
            started_at=datetime.now(timezone.utc),
        )
        storage.insert_run(run)
        scores = [
            Score(
                run_id="R-1", task_id=sample_task.id, judge_model="claude-opus-4-7",
                dimension="correctness", value=4,
                rationale="solid", anchor_matched="4 — Strong",
            ),
            Score(
                run_id="R-1", task_id=sample_task.id, judge_model="gpt-5.2",
                dimension="correctness", value=3,
                rationale="ok", anchor_matched="3 — Functional",
            ),
        ]
        storage.insert_scores(scores)
        loaded = storage.scores_for_run("R-1")
        assert len(loaded) == 2

    def test_insert_human_scores(self, storage: Storage, sample_task: Task) -> None:
        storage.insert_task(sample_task)
        hs = HumanScore(
            task_id=sample_task.id, dimension="correctness",
            value=4, rationale="agree", scorer="bjc",
        )
        storage.insert_human_scores([hs])
        all_humans = storage.all_human_scores()
        assert len(all_humans) == 1
        assert all_humans[0].scorer == "bjc"
        assert all_humans[0].value == 4

    def test_foreign_key_violation_on_orphan_score(
        self, storage: Storage
    ) -> None:
        """Inserting a score without its run/task should raise."""
        import sqlite3

        # No matching task — should fail due to FK constraint.
        score = Score(
            run_id="missing", task_id="missing", judge_model="x",
            dimension="correctness", value=4,
            rationale="x", anchor_matched="x",
        )
        try:
            storage.insert_scores([score])
            assert False, "Expected IntegrityError"
        except sqlite3.IntegrityError:
            pass
