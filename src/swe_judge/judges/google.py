"""Google judge — Gemini 3 Pro via function calling.

Third member of the v0.1 ensemble. Uses Google's generative-ai SDK with
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

# Gemini's function-declaration schema is a strict subset of JSON Schema —
# it rejects these keys with `ValueError: Unknown field for Schema: <key>`
# before the model ever runs. Anthropic and OpenAI accept them, so we
# strip per-call here instead of mutating the shared JUDGE_TOOL_SCHEMA.
# Discovered incrementally during real-API smoke runs; the SDK reports
# the first incompatible key it finds, so each round of stripping
# exposes the next one. Sticking to the surgical set we've actually hit
# rather than guessing at the full Gemini blacklist.
_GEMINI_INCOMPATIBLE_KEYS: frozenset[str] = frozenset(
    {"additionalProperties", "$defs", "$ref", "minItems", "maxItems"}
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

    Note: Google's function-calling schema is JSON Schema with a few quirks
    (no `additionalProperties: false` in some SDK versions). We use the
    shared JUDGE_TOOL_SCHEMA and let the SDK adapt.
    """

    def __init__(
        self,
        model: str = "gemini-3-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self._genai = genai
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def judge(self, task: Task, model_output: str) -> JudgmentResult:
        gemini_schema = _strip_gemini_incompatible_keys(JUDGE_TOOL_SCHEMA)
        model = self._genai.GenerativeModel(
            model_name=self._model,
            system_instruction=build_system_prompt(),
            tools=[
                {
                    "function_declarations": [
                        {
                            "name": TOOL_NAME,
                            "description": "Submit the 3 rubric scores for this model output.",
                            "parameters": gemini_schema,
                        }
                    ]
                }
            ],
        )

        try:
            response = model.generate_content(
                build_user_message(task, model_output),
                generation_config={"temperature": self._temperature},
                tool_config={"function_calling_config": {"mode": "ANY"}},
            )
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Google API call failed: {e}") from e

        try:
            function_call = response.candidates[0].content.parts[0].function_call
        except (AttributeError, IndexError) as e:
            raise JudgeError(f"Google response shape unexpected: {e}") from e

        if not function_call or function_call.name != TOOL_NAME:
            raise JudgeError(
                f"Google response did not contain '{TOOL_NAME}' function call"
            )

        # Gemini returns function_call.args as a proto Mapping — coerce to dict
        try:
            args = dict(function_call.args)
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not coerce Google function_call.args: {e}") from e

        # Gemini sometimes nests scores under a Struct — handle both shapes
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
        except Exception as e:  # noqa: BLE001
            raise JudgeError(f"Could not parse scores from Google function_call: {e}") from e

        return JudgmentResult(
            task_id=task.id,
            judge_model=self._model,
            scores=scores,
        )
