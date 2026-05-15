"""Runner — orchestrates a full evaluation across (tasks × judges).

Fans out every (Task, Judge) pair across a thread pool. Each judge call is
IO-bound on a network API, so threads (not processes) are the right primitive.

For v0.1 we evaluate a single model_under_test at a time. The "model output"
for each task is provided up front by the caller — generation of those
outputs is out of scope for the eval harness itself (the harness scores
existing outputs, it does not produce them).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, NamedTuple

import ulid

from swe_judge.judges.base import Judge, JudgeError
from swe_judge.tasks import JudgmentResult, Run, Score, Task


class JudgeFailure(NamedTuple):
    """A non-fatal failure of one (task, judge) pair."""

    task_id: str
    judge_model: str
    error: str


class RunResult(NamedTuple):
    """Aggregate result of a run — everything you need to print + persist."""

    run: Run
    scores: list[Score]
    failures: list[JudgeFailure]


def compute_dataset_version(tasks: Sequence[Task]) -> str:
    """SHA256 over the canonicalised task list — locks dataset identity."""
    canonical = json.dumps(
        [t.model_dump(mode="json") for t in tasks],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def run_evaluation(
    tasks: Sequence[Task],
    judges: Sequence[Judge],
    model_outputs: Mapping[str, str],
    model_under_test: str,
    max_workers: int = 6,
    on_progress: Callable[[str, str, JudgmentResult | JudgeFailure], None] | None = None,
    config: Mapping[str, object] | None = None,
) -> RunResult:
    """Run every judge against every task and produce a RunResult.

    Args:
        tasks: ordered list of golden tasks to score.
        judges: ensemble of judges (typically 3 for v0.1).
        model_outputs: mapping task_id -> the model_under_test's output for that task.
        model_under_test: canonical model identifier being evaluated.
        max_workers: thread pool size. 6 = 2 tasks per judge concurrently for 3 judges.
        on_progress: optional callback(task_id, judge_model, result_or_failure).
        config: extra metadata recorded on Run (rubric version, temperature, etc.)

    Returns:
        RunResult with the Run row, all Score rows, and any failures.

    Raises:
        ValueError: if model_outputs is missing entries for any task.
    """
    missing = [t.id for t in tasks if t.id not in model_outputs]
    if missing:
        raise ValueError(
            f"model_outputs missing entries for {len(missing)} task(s): {missing[:3]}..."
        )

    run = Run(
        id=str(ulid.new()),
        model_under_test=model_under_test,
        judge_models=[j.model_name for j in judges],
        dataset_version=compute_dataset_version(tasks),
        started_at=datetime.now(timezone.utc),
        config=dict(config) if config else {"rubric_version": "v1"},
    )

    pairs: list[tuple[Task, Judge]] = [(t, j) for t in tasks for j in judges]
    scores: list[Score] = []
    failures: list[JudgeFailure] = []

    def _one_pair(task: Task, judge: Judge) -> tuple[Task, Judge, JudgmentResult | JudgeError]:
        try:
            result = judge.judge(task, model_outputs[task.id])
            return task, judge, result
        except JudgeError as e:
            return task, judge, e
        except Exception as e:  # noqa: BLE001
            return task, judge, JudgeError(f"Unexpected error: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one_pair, t, j) for t, j in pairs]
        for fut in as_completed(futures):
            task, judge, outcome = fut.result()
            if isinstance(outcome, JudgmentResult):
                rows = outcome.to_score_rows(run.id)
                scores.extend(rows)
                if on_progress:
                    on_progress(task.id, judge.model_name, outcome)
            else:
                f = JudgeFailure(task_id=task.id, judge_model=judge.model_name, error=str(outcome))
                failures.append(f)
                if on_progress:
                    on_progress(task.id, judge.model_name, f)

    run = run.model_copy(update={"completed_at": datetime.now(timezone.utc)})
    return RunResult(run=run, scores=scores, failures=failures)
