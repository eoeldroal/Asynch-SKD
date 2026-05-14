import json
from pathlib import Path

import yaml
from xgrammar.grammar import Grammar

from verl.experimental.agent_loop.qwen_coder_structured_output import (
    build_qwen_coder_structured_tag_json,
)

def test_build_qwen_coder_structural_tag_json_uses_builtin_qwen_coder_suffix_only():
    tool_schemas = [
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
                            "minItems": 1,
                            "maxItems": 10,
                        }
                    },
                    "required": ["actions"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    raw = build_qwen_coder_structured_tag_json(tool_schemas)
    result = json.loads(raw)

    assert isinstance(raw, str)
    assert result["type"] == "structural_tag"
    assert result["format"]["type"] == "triggered_tags"
    assert result["format"]["triggers"] == ["<tool_call>\n<function="]
    assert result["format"]["at_least_one"] is False
    assert result["format"]["stop_after_first"] is False
    assert result["format"]["excludes"] == ["<think>", "</think>"]

    tag = result["format"]["tags"][0]
    assert tag["type"] == "tag"
    assert tag["begin"] == "<tool_call>\n<function=computer>\n"
    assert tag["end"] == "\n</function>\n</tool_call>"
    assert tag["content"]["type"] == "qwen_xml_parameter"

    grammar = str(Grammar.from_structural_tag(result))
    assert "triggered_tags" in grammar
    assert "loop_after_dispatch=true" in grammar
    assert "</think>\\n<tool_call>" not in grammar


def test_build_qwen_coder_structural_tag_json_rejects_non_bundled_tools():
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "CLICK",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": [],
                },
            },
        }
    ]

    try:
        build_qwen_coder_structured_tag_json(tool_schemas)
    except ValueError as exc:
        assert "computer" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-bundled tools")


def test_real_bundled_webgym_tool_config_builds_serialized_structural_tag():
    tool_config_path = Path(
        "/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml"
    )
    data = yaml.safe_load(tool_config_path.read_text())
    tool_schemas = [tool["tool_schema"] for tool in data["tools"]]

    raw = build_qwen_coder_structured_tag_json(tool_schemas)
    result = json.loads(raw)

    assert result["format"]["type"] == "triggered_tags"
    assert result["format"]["triggers"] == ["<tool_call>\n<function="]
    assert result["format"]["excludes"] == ["<think>", "</think>"]

    tag = result["format"]["tags"][0]
    assert tag["begin"] == "<tool_call>\n<function=computer>\n"
    assert tag["content"]["type"] == "qwen_xml_parameter"
