"""Shared judge prompts.

All judges in the ensemble (Anthropic, OpenAI, Google) receive the SAME
system prompt. This is critical for fair comparison — model-specific prompt
tuning would invalidate the inter-judge agreement measurement.

The prompt enforces an evidence-first pattern: the judge cites the rubric
anchor, gives a rationale, THEN assigns the numeric score. This pattern is
documented to reduce position bias and self-enhancement bias in current
LLM-as-judge research (Ye et al., ICLR 2025; Adragna 2025).
"""

from __future__ import annotations

from swe_judge.rubrics.v1 import rubric_v1_full_text
from swe_judge.tasks import Task


JUDGE_SYSTEM_PROMPT = """You are a careful, calibrated evaluator of software-engineering work.

You will be shown:
1. A software-engineering TASK (the prompt given to the model under test)
2. A REFERENCE SOLUTION (the gold-standard answer)
3. A MODEL OUTPUT (what the model under test produced)

Your job is to score the MODEL OUTPUT against the rubric on three dimensions.
Each dimension is scored 1–5 according to the anchored levels below.

# Rubric

{rubric}

# Output requirements

For EACH of the three dimensions (correctness, code_quality, reasoning), you must:
1. Quote or paraphrase the specific anchor you matched (e.g. "4 — Strong").
2. Give a 1–3 sentence rationale referencing concrete elements of the model output.
3. Assign the integer score 1–5.

Do not assign half-scores. Do not skip dimensions. If a dimension is not
applicable to the task (e.g. reasoning quality for a one-line bug fix with
no explanation), score it conservatively against the anchor that best fits
what was actually produced — do not refuse to score.

# Calibration rules

- Be willing to give a 1 or 2 if warranted. Do not anchor to 3 to avoid
  controversy. Real engineering work has a wide quality distribution.
- The reference solution is one valid answer among possibly many. Do not
  penalize a model output for differing from the reference if it achieves
  the same correctness with comparable quality.
- Score on the work as written. Do not speculate about what the author
  "probably meant" if the code does not say it.
"""


def build_system_prompt() -> str:
    """Return the rendered system prompt with the v1 rubric inlined."""
    return JUDGE_SYSTEM_PROMPT.format(rubric=rubric_v1_full_text())


def build_user_message(task: Task, model_output: str) -> str:
    """Render the per-task user message shown to every judge."""
    return f"""# TASK

Category: {task.category}
Difficulty: {task.difficulty}/5
Source: {task.source}

## Prompt shown to the model under test

{task.prompt}

## Reference solution

{task.reference_solution}

## Model output to score

{model_output}

---

Score the MODEL OUTPUT above on correctness, code_quality, and reasoning
according to the rubric. Use the `submit_scores` tool to return your scores.
"""
