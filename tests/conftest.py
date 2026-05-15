"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from swe_judge.storage import Storage
from swe_judge.tasks import HumanScore, Score, Task, TestCase


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="01HXSAMPLE0000000000000001",
        category="bug_fix",
        difficulty=3,
        source="custom",
        prompt="Fix the off-by-one bug in this median function.",
        reference_solution="def median(xs):\n    ...",
        test_cases=[
            TestCase(input="[1, 2, 3, 4]", expected_output="2.5"),
        ],
        tags=["python", "off-by-one"],
        created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_tasks(sample_task: Task) -> list[Task]:
    return [
        sample_task,
        sample_task.model_copy(update={"id": "01HXSAMPLE0000000000000002"}),
        sample_task.model_copy(update={"id": "01HXSAMPLE0000000000000003"}),
    ]


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.sqlite")


@pytest.fixture
def perfect_agreement_scores() -> list[Score]:
    """Two judges, perfect agreement with VARIED values (so κ is defined).

    Cohen's κ is undefined when all observations share one value — you need
    variance in the score distribution for the chance-adjustment math to
    work. This fixture spreads scores across 2–5 with both judges identical.
    """
    # task_id -> dim -> agreed-upon score
    truth: dict[str, dict[str, int]] = {
        "t1": {"correctness": 5, "code_quality": 4, "reasoning": 3},
        "t2": {"correctness": 4, "code_quality": 3, "reasoning": 2},
        "t3": {"correctness": 3, "code_quality": 5, "reasoning": 4},
    }
    rows: list[Score] = []
    for task_id, dims in truth.items():
        for dim, value in dims.items():
            for judge in ["judge-A", "judge-B"]:
                rows.append(
                    Score(
                        run_id="R1",
                        task_id=task_id,
                        judge_model=judge,
                        dimension=dim,  # type: ignore[arg-type]
                        value=value,
                        rationale="—",
                        anchor_matched=f"{value} — agreed",
                    )
                )
    return rows


@pytest.fixture
def disagreement_scores() -> list[Score]:
    """Two judges with mixed agreement, designed for kappa < 1.0."""
    rows: list[Score] = []
    judge_a_values = [5, 4, 3, 2, 1, 5, 4, 3, 2]
    judge_b_values = [4, 4, 3, 3, 2, 5, 3, 3, 2]  # mostly close but not identical
    i = 0
    for task_id in ["t1", "t2", "t3"]:
        for dim in ["correctness", "code_quality", "reasoning"]:
            rows.append(
                Score(
                    run_id="R1",
                    task_id=task_id,
                    judge_model="judge-A",
                    dimension=dim,  # type: ignore[arg-type]
                    value=judge_a_values[i],
                    rationale="—",
                    anchor_matched=f"{judge_a_values[i]} — anchor",
                )
            )
            rows.append(
                Score(
                    run_id="R1",
                    task_id=task_id,
                    judge_model="judge-B",
                    dimension=dim,  # type: ignore[arg-type]
                    value=judge_b_values[i],
                    rationale="—",
                    anchor_matched=f"{judge_b_values[i]} — anchor",
                )
            )
            i += 1
    return rows


@pytest.fixture
def human_scores_for_agreement() -> list[HumanScore]:
    """Human scores aligned to the varied perfect_agreement_scores fixture."""
    truth: dict[str, dict[str, int]] = {
        "t1": {"correctness": 5, "code_quality": 4, "reasoning": 3},
        "t2": {"correctness": 4, "code_quality": 3, "reasoning": 2},
        "t3": {"correctness": 3, "code_quality": 5, "reasoning": 4},
    }
    rows: list[HumanScore] = []
    for task_id, dims in truth.items():
        for dim, value in dims.items():
            rows.append(
                HumanScore(
                    task_id=task_id,
                    dimension=dim,  # type: ignore[arg-type]
                    value=value,
                    rationale="—",
                    scorer="test-human",
                )
            )
    return rows
