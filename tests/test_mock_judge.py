"""Tests for the MockJudge."""

from __future__ import annotations

from swe_judge.judges.base import Judge
from swe_judge.judges.mock import MockJudge
from swe_judge.tasks import Task


class TestMockJudge:
    def test_implements_judge_protocol(self) -> None:
        judge = MockJudge()
        assert isinstance(judge, Judge)

    def test_default_scoring_is_middling_threes(self, sample_task: Task) -> None:
        judge = MockJudge()
        result = judge.judge(sample_task, "any output")
        assert result.task_id == sample_task.id
        assert {s.dimension for s in result.scores} == {"correctness", "code_quality", "reasoning"}
        for s in result.scores:
            assert s.value == 3

    def test_custom_scoring_fn(self, sample_task: Task) -> None:
        def perfect(task: Task, output: str, dim: str) -> int:  # noqa: ARG001
            return 5

        judge = MockJudge(scoring_fn=perfect)
        result = judge.judge(sample_task, "any")
        assert all(s.value == 5 for s in result.scores)

    def test_dimension_specific_scoring(self, sample_task: Task) -> None:
        def by_dim(task: Task, output: str, dim: str) -> int:  # noqa: ARG001
            return {"correctness": 5, "code_quality": 3, "reasoning": 1}[dim]

        judge = MockJudge(scoring_fn=by_dim)
        result = judge.judge(sample_task, "any")
        scores = {s.dimension: s.value for s in result.scores}
        assert scores == {"correctness": 5, "code_quality": 3, "reasoning": 1}

    def test_to_score_rows_preserves_model_name(self, sample_task: Task) -> None:
        judge = MockJudge(model_name="custom-mock")
        result = judge.judge(sample_task, "any")
        rows = result.to_score_rows(run_id="R-1")
        assert all(r.judge_model == "custom-mock" for r in rows)

    def test_model_name_property(self) -> None:
        judge = MockJudge(model_name="mock-v2")
        assert judge.model_name == "mock-v2"
