"""Tests for the GoogleJudge — SDK-call contract only.

Mocks `google.generativeai` so they don't need an API key. Covers the
shape of the tool schema we pass to GenerativeModel — Gemini's function
declarations are a *subset* of JSON Schema (no additionalProperties,
no $defs, no $ref), and a stray key triggers `ValueError: Unknown field
for Schema: ...` before the model ever runs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from swe_judge.tasks import Task

GEMINI_INCOMPATIBLE_KEYS = {"additionalProperties", "$defs", "$ref"}


def _walk(node: Any) -> list[str]:
    """Yield every key found at any depth in a nested dict/list structure."""
    keys: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            keys.append(k)
            keys.extend(_walk(v))
    elif isinstance(node, list):
        for item in node:
            keys.extend(_walk(item))
    return keys


def _stub_gemini_response() -> MagicMock:
    """Minimal Gemini response with one valid function_call."""
    function_call = MagicMock()
    function_call.name = "submit_scores"
    function_call.args = {
        "scores": [
            {"dimension": "correctness", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
            {"dimension": "code_quality", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
            {"dimension": "reasoning", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
        ]
    }
    part = MagicMock()
    part.function_call = function_call
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _capture_schema(mock_model_cls: MagicMock) -> dict[str, Any]:
    """Pull the parameters dict out of the GenerativeModel(tools=...) call."""
    kwargs = mock_model_cls.call_args.kwargs
    return kwargs["tools"][0]["function_declarations"][0]["parameters"]


class TestGoogleJudgeSDKCall:
    def test_tool_schema_strips_additional_properties(
        self, sample_task: Task, mocker: pytest.MonkeyPatch
    ) -> None:
        """Gemini rejects `additionalProperties` with
        `ValueError: Unknown field for Schema: additionalProperties` — the
        schema we hand to GenerativeModel(tools=...) must not contain it
        at any nesting level."""
        mocker.patch("google.generativeai.configure")
        mock_model_cls = mocker.patch("google.generativeai.GenerativeModel")
        mock_model = mock_model_cls.return_value
        mock_model.generate_content.return_value = _stub_gemini_response()

        from swe_judge.judges.google import GoogleJudge

        judge = GoogleJudge(model="gemini-3-pro", api_key="test")
        judge.judge(sample_task, "any output")

        schema = _capture_schema(mock_model_cls)
        all_keys = _walk(schema)
        leaked = sorted(set(all_keys) & GEMINI_INCOMPATIBLE_KEYS)
        assert not leaked, (
            f"Gemini tool schema contains keys the SDK rejects: {leaked}. "
            f"Full key list at all depths: {sorted(set(all_keys))}"
        )

    def test_tool_schema_preserves_required_validation_fields(
        self, sample_task: Task, mocker: pytest.MonkeyPatch
    ) -> None:
        """The strip must not be over-eager — type, properties, required,
        enum, minimum/maximum/minItems/maxItems all stay."""
        mocker.patch("google.generativeai.configure")
        mock_model_cls = mocker.patch("google.generativeai.GenerativeModel")
        mock_model = mock_model_cls.return_value
        mock_model.generate_content.return_value = _stub_gemini_response()

        from swe_judge.judges.google import GoogleJudge

        judge = GoogleJudge(model="gemini-3-pro", api_key="test")
        judge.judge(sample_task, "any output")

        schema = _capture_schema(mock_model_cls)
        assert schema["type"] == "object"
        assert "scores" in schema["properties"]
        scores = schema["properties"]["scores"]
        assert scores["type"] == "array"
        assert scores["minItems"] == 3
        assert scores["maxItems"] == 3
        item = scores["items"]
        assert item["properties"]["dimension"]["enum"] == [
            "correctness",
            "code_quality",
            "reasoning",
        ]
        assert item["properties"]["value"]["minimum"] == 1
        assert item["properties"]["value"]["maximum"] == 5
        assert set(item["required"]) == {"dimension", "value", "rationale", "anchor_matched"}
