"""Google judge contract tests: real SDK types, mocked network boundary."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from swe_judge.judges.anthropic import JUDGE_TOOL_SCHEMA, TOOL_NAME
from swe_judge.judges.base import JudgeError
from swe_judge.judges.google import (
    _GEMINI_INCOMPATIBLE_KEYS,
    GoogleJudge,
    _strip_gemini_incompatible_keys,
)
from swe_judge.prompts import build_system_prompt, build_user_message
from swe_judge.tasks import Task


def score_rows() -> list[dict[str, object]]:
    return [
        {"dimension": dim, "value": 4, "rationale": "ok", "anchor_matched": "4 — Strong"}
        for dim in ("correctness", "code_quality", "reasoning")
    ]


def response_with(args: dict[str, object], name: str = TOOL_NAME) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(text="Scoring complete"),
                        types.Part(function_call=types.FunctionCall(name=name, args=args)),
                    ]
                )
            )
        ]
    )


@pytest.fixture
def client() -> Iterator[MagicMock]:
    with patch("google.genai.Client", autospec=True) as constructor:
        client = constructor.return_value
        client.models.generate_content.return_value = response_with({"scores": score_rows()})
        yield client


def test_request_contract(client: MagicMock, sample_task: Task) -> None:
    before = copy.deepcopy(JUDGE_TOOL_SCHEMA)
    judge = GoogleJudge(model="gemini-test", api_key="test", temperature=0.25)
    result = judge.judge(sample_task, "output")
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == judge.model_name == "gemini-test"
    assert kwargs["contents"] == build_user_message(sample_task, "output")
    config = kwargs["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == build_system_prompt()
    assert config.temperature == 0.25
    assert config.automatic_function_calling.disable is True
    calling = config.tool_config.function_calling_config
    assert calling.mode == types.FunctionCallingConfigMode.ANY
    assert calling.allowed_function_names == [TOOL_NAME]
    tool = config.tools[0].function_declarations[0]
    assert tool.name == TOOL_NAME
    expected = _strip_gemini_incompatible_keys(JUDGE_TOOL_SCHEMA)
    assert tool.parameters == types.Schema.model_validate(expected)
    assert JUDGE_TOOL_SCHEMA == before
    assert result.task_id == sample_task.id
    assert result.judge_model == judge.model_name
    assert [s.model_dump() for s in result.scores] == score_rows()


@pytest.mark.parametrize("key", sorted(_GEMINI_INCOMPATIBLE_KEYS))
def test_schema_filter_recursive_and_nonmutating(key: str) -> None:
    schema = {
        "type": "object",
        key: False,
        "properties": {
            "scores": {
                "type": "array",
                "items": {"type": "integer", key: 1},
                "enum": [{key: 1, "description": "kept"}],
            }
        },
        "required": ["scores"],
    }
    before = copy.deepcopy(schema)
    filtered = _strip_gemini_incompatible_keys(schema)
    assert key not in filtered
    assert key not in filtered["properties"]["scores"]["items"]
    assert filtered["properties"]["scores"]["enum"] == [{"description": "kept"}]
    assert filtered["required"] == ["scores"]
    assert schema == before


@pytest.mark.parametrize("explicit,expected", [("explicit", "explicit"), (None, "environment")])
def test_api_key_precedence(
    monkeypatch: pytest.MonkeyPatch, explicit: str | None, expected: str
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "environment")
    with patch("google.genai.Client", autospec=True) as constructor:
        GoogleJudge(api_key=explicit)
        constructor.assert_called_once_with(api_key=expected)


def test_default_temperature(client: MagicMock, sample_task: Task) -> None:
    GoogleJudge(api_key="test").judge(sample_task, "output")
    assert client.models.generate_content.call_args.kwargs["config"].temperature == 0.0


def test_api_failure(client: MagicMock, sample_task: Task) -> None:
    error = RuntimeError("unavailable")
    client.models.generate_content.side_effect = error
    with pytest.raises(JudgeError, match="Google API call failed") as caught:
        GoogleJudge(api_key="test").judge(sample_task, "output")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    "response",
    [
        types.GenerateContentResponse(),
        types.GenerateContentResponse(candidates=[]),
        types.GenerateContentResponse(candidates=[types.Candidate()]),
        types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content())]),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[]))]
        ),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="No score")]))]
        ),
        response_with({"scores": score_rows()}, name="wrong_tool"),
    ],
)
def test_missing_call_is_judge_error(
    client: MagicMock, sample_task: Task, response: object
) -> None:
    client.models.generate_content.return_value = response
    with pytest.raises(JudgeError, match="Google response"):
        GoogleJudge(api_key="test").judge(sample_task, "output")


@pytest.mark.parametrize(
    "args,match",
    [
        (None, "Could not coerce"),
        ({}, "missing 'scores'"),
        ({"scores": "invalid JSON"}, "malformed JSON"),
        ({"scores": None}, "Could not parse scores"),
        ({"scores": []}, "Could not parse scores"),
        ({"scores": [dict(score_rows()[0], value=6)]}, "Could not parse scores"),
        ({"scores": [dict(score_rows()[0], dimension="unknown")]}, "Could not parse scores"),
        ({"scores": [dict(score_rows()[0], rationale="")]}, "Could not parse scores"),
        ({"scores": [dict(score_rows()[0], extra="bad")]}, "Could not parse scores"),
        ({"scores": [{"dimension": "correctness"}]}, "Could not parse scores"),
    ],
)
def test_malformed_scores(client: MagicMock, sample_task: Task, args: object, match: str) -> None:
    # Malformed args cannot always be built with SDK types: exercise the boundary directly.
    call = SimpleNamespace(name=TOOL_NAME, args=args)
    client.models.generate_content.return_value = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(function_call=call),
                    ]
                )
            )
        ]
    )
    with pytest.raises(JudgeError, match=match):
        GoogleJudge(api_key="test").judge(sample_task, "output")


def test_json_string_scores(client: MagicMock, sample_task: Task) -> None:
    client.models.generate_content.return_value = response_with(
        {"scores": json.dumps(score_rows())}
    )
    result = GoogleJudge(api_key="test").judge(sample_task, "output")
    assert [s.model_dump() for s in result.scores] == score_rows()


def test_sdk_serialization_without_network(
    sample_task: Task, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise SDK serialization/parsing, stubbing only its HTTP request boundary."""
    # Avoid inheriting workstation proxy/Vertex settings; no request leaves this test.
    for key in ("ALL_PROXY", "all_proxy", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(key, raising=False)
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Scoring complete"},
                        {"functionCall": {"name": TOOL_NAME, "args": {"scores": score_rows()}}},
                    ],
                }
            }
        ]
    }
    # The internal HTTP result changed across 1.x: provide both representations.
    http_result = SimpleNamespace(headers={}, json=payload, response_stream=[json.dumps(payload)])
    judge = GoogleJudge(api_key="test")
    with patch.object(
        judge._client._api_client, "_request", autospec=True, return_value=http_result
    ) as request:
        result = judge.judge(sample_task, "output")
    request.assert_called_once()
    wire = request.call_args.args[0].data
    assert wire["generationConfig"]["temperature"] == 0.0
    assert wire["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": [TOOL_NAME],
    }
    declaration = wire["tools"][0]["functionDeclarations"][0]
    assert declaration["name"] == TOOL_NAME
    schema = declaration["parameters"]
    assert schema["required"] == ["scores"]
    item = schema["properties"]["scores"]["items"]
    assert set(item["required"]) == {"dimension", "value", "rationale", "anchor_matched"}
    assert item["properties"]["dimension"]["enum"] == [
        "correctness",
        "code_quality",
        "reasoning",
    ]
    assert "minimum" not in item["properties"]["value"]
    assert "additionalProperties" not in schema
    assert [s.model_dump() for s in result.scores] == score_rows()
