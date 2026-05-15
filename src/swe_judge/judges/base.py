"""Judge protocol — the contract every concrete judge must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swe_judge.tasks import JudgmentResult, Task


class JudgeError(Exception):
    """Raised when a judge cannot produce a valid JudgmentResult.

    Examples:
      - Provider API returned an error or timed out
      - Provider returned malformed structured output that fails Zod-style validation
      - Required tool_use block was absent from the response
    """


@runtime_checkable
class Judge(Protocol):
    """A judge scores a single Task given a model output.

    Implementations must be safe to call concurrently — the runner will
    fan out N tasks across M judges in parallel.
    """

    @property
    def model_name(self) -> str:
        """Canonical model identifier, e.g. 'claude-opus-4-7'.

        Recorded on every Score row for analysis.
        """
        ...

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        """Score the model_output against the task across all rubric dimensions.

        Raises:
            JudgeError: if a valid JudgmentResult cannot be produced.
        """
        ...
