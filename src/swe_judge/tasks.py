"""Core data models for swe-judge.

These Pydantic models are the canonical types for tasks, evaluation runs,
judge scores, and human ground-truth scores. They are intentionally minimal
and serializable to JSON / JSONL for portability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Category = Literal["bug_fix", "test_write", "code_review"]
"""SWE task categories included in v0.1 of the golden dataset."""

Dimension = Literal["correctness", "code_quality", "reasoning"]
"""Rubric dimensions scored by each judge in v0.1."""


# ---------------------------------------------------------------------------
# Test cases (executable validation, optional per task)
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    """A single executable validation case for a `bug_fix` or `test_write` task.

    Either `expected_output` or `expected_exception` must be set, not both.
    `code_review` tasks typically have an empty list of TestCases.
    """

    model_config = ConfigDict(extra="forbid")

    input: str = Field(description="Python expression evaluated as input to the function")
    expected_output: str | None = Field(default=None, description="Expected str(output)")
    expected_exception: str | None = Field(
        default=None, description="Expected exception class name, e.g. 'ValueError'"
    )

    @field_validator("expected_exception")
    @classmethod
    def _exactly_one_expectation(
        cls, v: str | None, info: object  # noqa: ARG003
    ) -> str | None:
        return v

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        """Enforce exactly one of expected_output / expected_exception is set."""
        has_out = self.expected_output is not None
        has_exc = self.expected_exception is not None
        if has_out == has_exc:
            raise ValueError(
                "TestCase requires exactly one of expected_output or expected_exception"
            )


# ---------------------------------------------------------------------------
# Tasks (the things being evaluated)
# ---------------------------------------------------------------------------


class Task(BaseModel):
    """A single golden-dataset task.

    The Task is shown to the model under test as `prompt`. The judges see
    the prompt + the model's output + the `reference_solution` and score
    across rubric dimensions.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="ULID-shaped identifier")
    category: Category
    difficulty: int = Field(ge=1, le=5, description="1=easy, 5=expert")
    source: str = Field(description="'custom' | 'swe-bench-lite' | 'humaneval-plus' | ...")
    prompt: str = Field(min_length=1, description="What is shown to the model under test")
    reference_solution: str = Field(min_length=1, description="Gold standard implementation")
    test_cases: list[TestCase] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Runs (an end-to-end evaluation of one model under test)
# ---------------------------------------------------------------------------


class Run(BaseModel):
    """A single evaluation run.

    A Run is "model X scored by judges Y, Z, W on dataset version D at time T".
    Each Run produces one Score row per (task, judge, dimension).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="ULID-shaped identifier")
    model_under_test: str = Field(description="e.g. 'claude-opus-4-7', 'gpt-5.2'")
    judge_models: list[str] = Field(min_length=1)
    dataset_version: str = Field(description="SHA256 of the golden-set JSONL")
    started_at: datetime
    completed_at: datetime | None = None
    config: dict[str, object] = Field(
        default_factory=dict,
        description="Rubric version, temperature, etc. — anything affecting reproducibility",
    )


# ---------------------------------------------------------------------------
# Scores (judge output)
# ---------------------------------------------------------------------------


class Score(BaseModel):
    """A single judge's score on (task, dimension) within a Run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    judge_model: str
    dimension: Dimension
    value: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, description="Judge's free-text justification")
    anchor_matched: str = Field(
        min_length=1, description="Which rubric anchor the judge cited (e.g. '4 — Strong')"
    )


# ---------------------------------------------------------------------------
# Judgment results (transient — what a Judge protocol returns before storage)
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """A score for a single rubric dimension produced by one judge."""

    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    value: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)
    anchor_matched: str = Field(min_length=1)


class JudgmentResult(BaseModel):
    """The full output of one judge scoring one task across all dimensions."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    judge_model: str
    scores: list[DimensionScore] = Field(min_length=1)

    def to_score_rows(self, run_id: str) -> list[Score]:
        """Convert this judgment into Score rows for persistence."""
        return [
            Score(
                run_id=run_id,
                task_id=self.task_id,
                judge_model=self.judge_model,
                dimension=ds.dimension,
                value=ds.value,
                rationale=ds.rationale,
                anchor_matched=ds.anchor_matched,
            )
            for ds in self.scores
        ]


# ---------------------------------------------------------------------------
# Human ground-truth scores (the 20-item subset)
# ---------------------------------------------------------------------------


class HumanScore(BaseModel):
    """A human-rater score on (task, dimension) for ground truth."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    dimension: Dimension
    value: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)
    scorer: str = Field(min_length=1, description="GitHub handle or initials")
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
