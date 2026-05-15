"""Anthropic judge — Claude Opus 4.7 via the Anthropic SDK with forced tool_use.

We use tool_use rather than free-text JSON because:
1. Claude's tool_use enforces the schema at decode time, not parse time.
2. Forced tool_choice eliminates the "model decided to chat instead of score" failure mode.
3. The schema is identical across providers (OpenAI, Google), which keeps
   the inter-judge comparison fair.
"""

from __future__ import annotations

import json
import os
from typing import Any

from swe_judge.judges.base import JudgeError
from swe_judge.prompts import build_system_prompt, build_user_message
from swe_judge.tasks import DimensionScore, JudgmentResult, Task

TOOL_NAME = "submit_scores"

# Shared tool schema across all three providers — single source of truth
# for what a valid judgment looks like.
JUDGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": ["correctness", "code_quality", "reasoning"],
                    },
                    "value": {"type": "integer", "minimum": 1, "maximum": 5},
                    "rationale": {"type": "string", "minLength": 1},
                    "anchor_matched": {"type": "string", "minLength": 1},
                },
                "required": ["dimension", "value", "rationale", "anchor_matched"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


class AnthropicJudge:
    """Claude as a rubric judge.

    Uses tool_use with `tool_choice={"type": "tool", "name": "submit_scores"}`
    to force Claude to return the JudgmentResult schema rather than chat.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        # Lazy import keeps tests/CI fast for users who don't have Anthropic installed.
        from anthropic import Anthropic  # type: ignore[import-not-found]

        self._model = model
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=build_system_prompt(),
                messages=[{"role": "user", "content": build_user_message(task, model_output)}],
                tools=[
                    {
                        "name": TOOL_NAME,
                        "description": "Submit the 3 rubric scores for this model output.",
                        "input_schema": JUDGE_TOOL_SCHEMA,
                    }
                ],
                tool_choice={"type": "tool", "name": TOOL_NAME},
            )
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Anthropic API call failed: {e}") from e

        tool_use_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_use_block is None:
            raise JudgeError(
                "Anthropic response did not contain a tool_use block "
                "(tool_choice was forced — this should not happen)"
            )

        raw_input = tool_use_block.input
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except json.JSONDecodeError as e:
                raise JudgeError(f"tool_use.input was a malformed JSON string: {e}") from e

        if not isinstance(raw_input, dict) or "scores" not in raw_input:
            raise JudgeError(f"tool_use.input missing 'scores' key: {raw_input!r}")

        try:
            scores = [DimensionScore(**s) for s in raw_input["scores"]]
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not parse scores from tool_use input: {e}") from e

        return JudgmentResult(
            task_id=task.id,
            judge_model=self._model,
            scores=scores,
        )
