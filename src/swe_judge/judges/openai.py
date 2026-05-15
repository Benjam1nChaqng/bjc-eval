"""OpenAI judge — GPT-5.2 via function calling with forced tool_choice.

Implementation mirrors the Anthropic judge so the only difference between
ensemble members is the underlying model, not the prompt or schema.
"""

from __future__ import annotations

import json
import os

from swe_judge.judges.anthropic import JUDGE_TOOL_SCHEMA, TOOL_NAME
from swe_judge.judges.base import JudgeError
from swe_judge.prompts import build_system_prompt, build_user_message
from swe_judge.tasks import DimensionScore, JudgmentResult, Task


class OpenAIJudge:
    """GPT as a rubric judge.

    Uses function calling with `tool_choice={"type":"function", "function":{"name": ...}}`
    to force structured output matching JUDGE_TOOL_SCHEMA.
    """

    def __init__(
        self,
        model: str = "gpt-5.2",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI  # type: ignore[import-not-found]

        self._model = model
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_message(task, model_output)},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "description": "Submit the 3 rubric scores for this model output.",
                            "parameters": JUDGE_TOOL_SCHEMA,
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            )
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"OpenAI API call failed: {e}") from e

        try:
            tool_calls = response.choices[0].message.tool_calls
        except (AttributeError, IndexError) as e:
            raise JudgeError(f"OpenAI response shape unexpected: {e}") from e

        if not tool_calls:
            raise JudgeError(
                "OpenAI response had no tool_calls (tool_choice was forced — "
                "this should not happen)"
            )

        try:
            args = json.loads(tool_calls[0].function.arguments)
        except (json.JSONDecodeError, AttributeError) as e:
            raise JudgeError(f"OpenAI tool_call arguments were malformed: {e}") from e

        if not isinstance(args, dict) or "scores" not in args:
            raise JudgeError(f"OpenAI tool_call missing 'scores' key: {args!r}")

        try:
            scores = [DimensionScore(**s) for s in args["scores"]]
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not parse scores from OpenAI tool_call: {e}") from e

        return JudgmentResult(
            task_id=task.id,
            judge_model=self._model,
            scores=scores,
        )
