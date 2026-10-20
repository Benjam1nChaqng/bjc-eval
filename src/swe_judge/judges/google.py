"""Google judge — Gemini 3 Pro via function calling.

Third member of the v0.1 ensemble. Uses Google's Gen AI SDK with
the same schema as the Anthropic and OpenAI judges.
"""

from __future__ import annotations

import json
import os
from typing import Any

from swe_judge.judges.anthropic import JUDGE_TOOL_SCHEMA, TOOL_NAME
from swe_judge.judges.base import JudgeError
from swe_judge.prompts import build_system_prompt, build_user_message
from swe_judge.tasks import DimensionScore, JudgmentResult, Task

# Retain the legacy function-schema compatibility filter during SDK migration.
# google.genai.types.Schema accepts more constraints than the old protobuf Schema,
# but local SDK acceptance does not prove backend/model support. Keep the wire
# contract unchanged until real Gemini runs justify relaxing individual keys.
# The shared schema is never mutated; DimensionScore still enforces score range,
# dimension literals, nonempty text, and forbidden extra score fields at parse time.
_GEMINI_INCOMPATIBLE_KEYS: frozenset[str] = frozenset(
    {
        # discovered via real-API smoke runs
        "additionalProperties",
        "minItems",
        "maxItems",
        "minimum",
        # defensive — same rejection mechanism, just haven't tripped yet
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "multipleOf",
        # JSON Schema composition (Pydantic emits these on Union types etc.)
        "$defs",
        "$ref",
        "allOf",
        "oneOf",
        "anyOf",
        "not",
        "const",
        "default",
    }
)


def _strip_gemini_incompatible_keys(node: Any) -> Any:
    """Return a deep copy of `node` with Gemini-incompatible keys removed."""
    if isinstance(node, dict):
        return {
            k: _strip_gemini_incompatible_keys(v)
            for k, v in node.items()
            if k not in _GEMINI_INCOMPATIBLE_KEYS
        }
    if isinstance(node, list):
        return [_strip_gemini_incompatible_keys(item) for item in node]
    return node


class GoogleJudge:
    """Gemini as a rubric judge.

    Uses the shared prompt and a compatibility-filtered copy of the shared
    tool schema. Function calls are returned for validation, never executed.
    """

    def __init__(
        self,
        model: str = "gemini-3-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._types = types
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=build_user_message(task, model_output),
                config=self._types.GenerateContentConfig(
                    system_instruction=build_system_prompt(),
                    temperature=self._temperature,
                    tools=[
                        self._types.Tool(
                            function_declarations=[
                                self._types.FunctionDeclaration(
                                    name=TOOL_NAME,
                                    description="Submit the 3 rubric scores for this model output.",
                                    parameters=_strip_gemini_incompatible_keys(JUDGE_TOOL_SCHEMA),
                                )
                            ]
                        )
                    ],
                    tool_config=self._types.ToolConfig(
                        function_calling_config=self._types.FunctionCallingConfig(
                            mode=self._types.FunctionCallingConfigMode.ANY,
                            allowed_function_names=[TOOL_NAME],
                        )
                    ),
                    automatic_function_calling=self._types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Google API call failed: {e}") from e

        try:
            # Text/thought parts can precede the call. Only inspect the first
            # candidate, matching the other judges' single-result contract.
            if not response.candidates or response.candidates[0].content is None:
                raise JudgeError("Google response shape unexpected: missing candidate content")
            parts = response.candidates[0].content.parts
            function_call = next(
                (
                    part.function_call
                    for part in (parts or [])
                    if part.function_call and part.function_call.name == TOOL_NAME
                ),
                None,
            )
        except (AttributeError, IndexError, TypeError) as e:
            raise JudgeError(f"Google response shape unexpected: {e}") from e

        if not function_call or function_call.name != TOOL_NAME:
            raise JudgeError(f"Google response did not contain '{TOOL_NAME}' function call")

        # The new SDK returns a dict; retain coercion and the JudgeError boundary.
        try:
            if function_call.args is None:
                raise ValueError("missing function arguments")
            args = dict(function_call.args)
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not coerce Google function_call.args: {e}") from e

        # Retain compatibility with scores encoded as a JSON string.
        if "scores" not in args:
            raise JudgeError(f"Google function_call missing 'scores' key: {args!r}")

        scores_raw = args["scores"]
        if isinstance(scores_raw, str):
            try:
                scores_raw = json.loads(scores_raw)
            except json.JSONDecodeError as e:
                raise JudgeError(f"Google 'scores' was a malformed JSON string: {e}") from e

        try:
            scores = [DimensionScore(**dict(s)) for s in scores_raw]
            return JudgmentResult(
                task_id=task.id,
                judge_model=self._model,
                scores=scores,
            )
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not parse scores from Google function_call: {e}") from e
