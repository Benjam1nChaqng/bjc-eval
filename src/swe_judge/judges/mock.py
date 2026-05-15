"""MockJudge — deterministic, API-free judge for tests and dry runs.

This is the workhorse for the test suite. Real judges call paid APIs and
are slow + non-deterministic; MockJudge lets us exercise everything else
(reliability calculations, storage, CLI orchestration) instantly.
"""

from __future__ import annotations

from collections.abc import Callable

from swe_judge.tasks import DimensionScore, JudgmentResult, Task


# A scoring function takes (task, model_output, dimension) and returns
# the integer score 1-5. Useful for parameterized fixtures.
ScoringFn = Callable[[Task, str, str], int]


def _default_scoring(task: Task, model_output: str, dimension: str) -> int:  # noqa: ARG001
    """Default canned scoring: every dimension gets a 3 (deliberately middling)."""
    return 3


class MockJudge:
    """A deterministic judge that returns scores from a provided callable.

    Implements the Judge protocol via duck typing. The judge() method
    never raises and never calls external APIs.
    """

    def __init__(
        self,
        model_name: str = "mock-judge-v1",
        scoring_fn: ScoringFn = _default_scoring,
        rationale: str = "Mock rationale (no real evaluation performed).",
        anchor: str = "3 — Adequate",
    ) -> None:
        self._model_name = model_name
        self._scoring_fn = scoring_fn
        self._rationale = rationale
        self._anchor = anchor

    @property
    def model_name(self) -> str:
        return self._model_name

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        dimensions = ["correctness", "code_quality", "reasoning"]
        scores = [
            DimensionScore(
                dimension=dim,  # type: ignore[arg-type]
                value=self._scoring_fn(task, model_output, dim),
                rationale=self._rationale,
                anchor_matched=self._anchor,
            )
            for dim in dimensions
        ]
        return JudgmentResult(
            task_id=task.id,
            judge_model=self._model_name,
            scores=scores,
        )
