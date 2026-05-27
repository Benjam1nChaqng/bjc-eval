"""Tests for the runner orchestrator."""

from __future__ import annotations

import pytest

from swe_judge.judges.base import JudgeError
from swe_judge.judges.mock import MockJudge
from swe_judge.runner import compute_dataset_version, run_evaluation
from swe_judge.tasks import JudgmentResult, Task


class _FailingJudge:
    @property
    def model_name(self) -> str:
        return "fails-always"

    def judge(self, task: Task, model_output: str) -> JudgmentResult:  # noqa: ARG002
        raise JudgeError("synthetic failure for tests")


class _CrashingJudge:
    """Raises a non-JudgeError exception — exercises the catch-all branch."""

    @property
    def model_name(self) -> str:
        return "crashes-always"

    def judge(self, task: Task, model_output: str) -> JudgmentResult:  # noqa: ARG002
        raise ValueError("auth scope wrong: 401 Unauthorized")


class TestRunner:
    def test_dataset_version_is_deterministic(self, sample_tasks: list[Task]) -> None:
        v1 = compute_dataset_version(sample_tasks)
        v2 = compute_dataset_version(sample_tasks)
        assert v1 == v2
        assert len(v1) == 16

    def test_dataset_version_changes_with_content(self, sample_tasks: list[Task]) -> None:
        v1 = compute_dataset_version(sample_tasks)
        tweaked = list(sample_tasks)
        tweaked[0] = tweaked[0].model_copy(update={"prompt": "changed"})
        v2 = compute_dataset_version(tweaked)
        assert v1 != v2

    def test_run_evaluation_produces_full_score_grid(
        self, sample_tasks: list[Task]
    ) -> None:
        judges = [MockJudge(model_name="A"), MockJudge(model_name="B")]
        outputs = {t.id: f"out for {t.id}" for t in sample_tasks}

        result = run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="model-X",
        )

        # 3 tasks × 2 judges × 3 dimensions = 18 score rows
        assert len(result.scores) == 18
        assert result.failures == []
        assert result.run.completed_at is not None

    def test_missing_output_raises(self, sample_tasks: list[Task]) -> None:
        judges = [MockJudge()]
        outputs = {sample_tasks[0].id: "ok"}  # missing for tasks 1, 2

        with pytest.raises(ValueError, match="missing entries"):
            run_evaluation(
                tasks=sample_tasks,
                judges=judges,
                model_outputs=outputs,
                model_under_test="X",
            )

    def test_failures_collected_not_raised(
        self, sample_tasks: list[Task]
    ) -> None:
        judges = [MockJudge(model_name="ok"), _FailingJudge()]
        outputs = {t.id: "any" for t in sample_tasks}

        result = run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="X",
        )

        # MockJudge produces 3 tasks × 3 dims = 9 successful scores
        # FailingJudge produces 3 failures (one per task)
        assert len(result.scores) == 9
        assert len(result.failures) == 3
        assert all(f.judge_model == "fails-always" for f in result.failures)

    def test_failure_records_exception_type_and_message(
        self, sample_tasks: list[Task]
    ) -> None:
        """A JudgeError must land in the failure record with both class and message."""
        judges = [_FailingJudge()]
        outputs = {t.id: "any" for t in sample_tasks}

        result = run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="X",
        )

        assert len(result.failures) == len(sample_tasks)
        f = result.failures[0]
        assert f.error_type == "JudgeError"
        assert "synthetic failure for tests" in f.error

    def test_failure_preserves_arbitrary_exception_class(
        self, sample_tasks: list[Task]
    ) -> None:
        """A non-JudgeError exception must keep its original class name in the record."""
        judges = [_CrashingJudge()]
        outputs = {t.id: "any" for t in sample_tasks}

        result = run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="X",
        )

        assert len(result.failures) == len(sample_tasks)
        f = result.failures[0]
        assert f.error_type == "ValueError"
        assert "auth scope wrong" in f.error
        # No "Unexpected error:" wrapping prefix — the original message
        # is the source of truth.
        assert not f.error.startswith("Unexpected error:")

    def test_runner_writes_failure_summary_to_stderr(
        self, sample_tasks: list[Task], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When the run finishes with failures, the runner emits one stderr line
        per (judge_model, error_type, truncated message) — so a CLI user sees
        why their judges failed without having to read the SQLite DB."""
        judges = [MockJudge(model_name="ok"), _CrashingJudge()]
        outputs = {t.id: "any" for t in sample_tasks}

        run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="X",
        )

        captured = capsys.readouterr()
        assert "crashes-always" in captured.err
        assert "ValueError" in captured.err
        assert "auth scope wrong" in captured.err
        # Successful judge should not be mentioned in the failure summary.
        assert "MockJudge" not in captured.err or "ok" not in captured.err.split("\n")[0]

    def test_progress_callback_invoked(
        self, sample_tasks: list[Task]
    ) -> None:
        judges = [MockJudge(model_name="A")]
        outputs = {t.id: "ok" for t in sample_tasks}
        calls: list[tuple[str, str]] = []

        def cb(task_id: str, judge_model: str, _outcome: object) -> None:
            calls.append((task_id, judge_model))

        run_evaluation(
            tasks=sample_tasks,
            judges=judges,
            model_outputs=outputs,
            model_under_test="X",
            on_progress=cb,
        )

        assert len(calls) == len(sample_tasks)
        assert all(j == "A" for _, j in calls)
