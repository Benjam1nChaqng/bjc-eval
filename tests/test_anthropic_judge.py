"""Tests for the AnthropicJudge — SDK-call contract only.

These tests mock the `anthropic.Anthropic` client so they do not require
an API key and are safe to run in CI. They cover the structure of the
SDK request, not the model's reasoning quality.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from swe_judge.tasks import Task


def _stub_anthropic_response() -> MagicMock:
    """Minimal Anthropic SDK response carrying one valid tool_use block."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.input = {
        "scores": [
            {"dimension": "correctness", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
            {"dimension": "code_quality", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
            {"dimension": "reasoning", "value": 4, "rationale": "ok", "anchor_matched": "4 — ok"},
        ]
    }
    response = MagicMock()
    response.content = [tool_use_block]
    return response


class TestAnthropicJudgeSDKCall:
    def test_does_not_pass_deprecated_temperature_to_sdk(
        self, sample_task: Task, mocker: pytest.MonkeyPatch
    ) -> None:
        """Opus 4.7 rejects `temperature` with HTTP 400 invalid_request_error.

        Regression guard: the SDK call must omit `temperature` entirely.
        """
        mock_anthropic_cls = mocker.patch("anthropic.Anthropic")
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _stub_anthropic_response()

        from swe_judge.judges.anthropic import AnthropicJudge

        judge = AnthropicJudge(model="claude-opus-4-7", api_key="test")
        judge.judge(sample_task, "any output")

        assert mock_client.messages.create.called
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "temperature" not in kwargs, (
            f"AnthropicJudge passed `temperature` to SDK kwargs: "
            f"{kwargs.get('temperature')!r}. Opus 4.7 deprecated this param "
            f"and rejects it with HTTP 400 invalid_request_error."
        )

    def test_forces_tool_choice_to_submit_scores(
        self, sample_task: Task, mocker: pytest.MonkeyPatch
    ) -> None:
        """tool_choice must be forced so Claude cannot free-form respond."""
        mock_anthropic_cls = mocker.patch("anthropic.Anthropic")
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create.return_value = _stub_anthropic_response()

        from swe_judge.judges.anthropic import AnthropicJudge

        judge = AnthropicJudge(model="claude-opus-4-7", api_key="test")
        judge.judge(sample_task, "any output")

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_scores"}
