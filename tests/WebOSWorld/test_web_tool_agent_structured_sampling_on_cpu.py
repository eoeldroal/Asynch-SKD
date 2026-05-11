from __future__ import annotations

from types import SimpleNamespace

import pytest

import verl.experimental.agent_loop.web_tool_agent_loop as web_tool_agent_loop_module
from verl.experimental.agent_loop.web_tool_agent_loop import WebOsGymToolAgentLoop


def _make_loop(
    *,
    rollout_name: str = "sglang",
    tool_parser_name: str = "qwen3_coder",
    custom: dict | None = None,
):
    loop = object.__new__(WebOsGymToolAgentLoop)
    loop.rollout_config = SimpleNamespace(name=rollout_name, custom=custom)
    loop.tool_parser_name = tool_parser_name
    return loop


def _bundled_tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "computer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["actions"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_build_generation_sampling_params_attaches_structural_tag_for_explicit_rl_path(monkeypatch):
    loop = _make_loop(custom={"enable_qwen3_coder_structured_output": True})
    base_sampling_params = {"temperature": 0.95, "top_p": 0.6, "top_k": -1}
    active_tool_schemas = _bundled_tool_schemas()
    builder_calls = []

    def _fake_builder(tool_schemas):
        builder_calls.append(tool_schemas)
        return '{"type": "structural_tag"}'

    monkeypatch.setattr(web_tool_agent_loop_module, "build_qwen_coder_structured_tag_json", _fake_builder)

    result = WebOsGymToolAgentLoop._build_generation_sampling_params(loop, base_sampling_params, active_tool_schemas)

    assert result == {
        "temperature": 0.95,
        "top_p": 0.6,
        "top_k": -1,
        "structural_tag": '{"type": "structural_tag"}',
        "ignore_eos": True,
    }
    assert base_sampling_params == {"temperature": 0.95, "top_p": 0.6, "top_k": -1}
    assert builder_calls == [active_tool_schemas]


@pytest.mark.parametrize(
    ("rollout_name", "tool_parser_name", "custom", "active_tool_schemas"),
    [
        ("vllm", "qwen3_coder", {"enable_qwen3_coder_structured_output": True}, _bundled_tool_schemas()),
        ("sglang", "hermes", {"enable_qwen3_coder_structured_output": True}, _bundled_tool_schemas()),
        ("sglang", "qwen3_coder", None, _bundled_tool_schemas()),
        ("sglang", "qwen3_coder", {"enable_qwen3_coder_structured_output": False}, _bundled_tool_schemas()),
        ("sglang", "qwen3_coder", {"enable_qwen3_coder_structured_output": True}, []),
    ],
)
def test_build_generation_sampling_params_skips_structural_tag_outside_explicit_rl_path(
    monkeypatch,
    rollout_name,
    tool_parser_name,
    custom,
    active_tool_schemas,
):
    loop = _make_loop(rollout_name=rollout_name, tool_parser_name=tool_parser_name, custom=custom)
    builder_calls = []

    def _fake_builder(tool_schemas):
        builder_calls.append(tool_schemas)
        return "unused"

    monkeypatch.setattr(web_tool_agent_loop_module, "build_qwen_coder_structured_tag_json", _fake_builder)

    result = WebOsGymToolAgentLoop._build_generation_sampling_params(
        loop,
        {"temperature": 1.0},
        active_tool_schemas,
    )

    assert result == {"temperature": 1.0}
    assert builder_calls == []
