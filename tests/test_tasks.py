"""Tests for swe_judge.tasks Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from swe_judge.tasks import (
    DimensionScore,
    HumanScore,
    JudgmentResult,
    Run,
    Score,
    Task,
    TestCase,
)


class TestTestCase:
    def test_accepts_expected_output_only(self) -> None:
        tc = TestCase(input="[1,2]", expected_output="1.5")
        assert tc.expected_exception is None

    def test_accepts_expected_exception_only(self) -> None:
        tc = TestCase(input="[]", expected_exception="ValueError")
        assert tc.expected_output is None

    def test_rejects_both_expectations(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            TestCase(input="[]", expected_output="0", expected_exception="ValueError")

    def test_rejects_neither_expectation(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            TestCase(input="[]")


class TestTask:
    def test_minimal_valid_task(self) -> None:
        t = Task(
            id="t1",
            category="bug_fix",
            difficulty=3,
            source="custom",
            prompt="fix it",
            reference_solution="done",
        )
        assert t.test_cases == []
        assert t.tags == []

    def test_difficulty_bounds(self) -> None:
        for bad in (0, 6, -1, 99):
            with pytest.raises(ValidationError):
                Task(
                    id="t1", category="bug_fix", difficulty=bad,  # type: ignore[arg-type]
                    source="custom", prompt="x", reference_solution="y",
                )

    def test_category_must_be_known(self) -> None:
        with pytest.raises(ValidationError):
            Task(
                id="t1", category="not_a_category",  # type: ignore[arg-type]
                difficulty=3, source="custom",
                prompt="x", reference_solution="y",
            )

    def test_round_trip_json(self, sample_task: Task) -> None:
        s = sample_task.model_dump_json()
        loaded = Task.model_validate_json(s)
        assert loaded == sample_task


class TestScore:
    def test_value_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Score(
                run_id="r1", task_id="t1", judge_model="j",
                dimension="correctness", value=6,
                rationale="x", anchor_matched="6 — none",
            )

    def test_rationale_required_nonempty(self) -> None:
        with pytest.raises(ValidationError):
            Score(
                run_id="r1", task_id="t1", judge_model="j",
                dimension="correctness", value=3,
                rationale="", anchor_matched="3 — Functional",
            )


class TestJudgmentResult:
    def test_to_score_rows_preserves_count(self) -> None:
        jr = JudgmentResult(
            task_id="t1",
            judge_model="judge-X",
            scores=[
                DimensionScore(dimension="correctness", value=4, rationale="r", anchor_matched="4"),
                DimensionScore(dimension="code_quality", value=3, rationale="r", anchor_matched="3"),
                DimensionScore(dimension="reasoning", value=5, rationale="r", anchor_matched="5"),
            ],
        )
        rows = jr.to_score_rows(run_id="RUN-1")
        assert len(rows) == 3
        assert {r.dimension for r in rows} == {"correctness", "code_quality", "reasoning"}
        assert all(r.run_id == "RUN-1" for r in rows)
        assert all(r.task_id == "t1" for r in rows)


class TestHumanScore:
    def test_round_trip(self) -> None:
        h = HumanScore(
            task_id="t1", dimension="correctness",
            value=4, rationale="solid",
            scorer="bjc",
        )
        loaded = HumanScore.model_validate(json.loads(h.model_dump_json()))
        assert loaded.task_id == h.task_id
        assert loaded.value == 4


class TestRun:
    def test_judge_models_must_be_nonempty(self) -> None:
        from datetime import datetime, timezone

        with pytest.raises(ValidationError):
            Run(
                id="r1", model_under_test="x",
                judge_models=[], dataset_version="abc",
                started_at=datetime.now(timezone.utc),
            )
