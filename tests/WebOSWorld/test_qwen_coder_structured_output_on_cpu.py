import json
from pathlib import Path

import yaml
from xgrammar.grammar import Grammar

from verl.experimental.agent_loop.qwen_coder_structured_output import (
    build_qwen_coder_structured_tag_json,
)


_EXPECTED_EXCLUDES = [
    "</think>",
    "<tool_call>",
    "<function=",
    "<parameter=",
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
]


def test_build_qwen_coder_structural_tag_json_serializes_reasoning_then_mandatory_tool_call():
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
    assert result["format"]["type"] == "sequence"

    prefix = result["format"]["elements"][0]
    assert prefix["type"] == "any_text"
    assert prefix["excludes"] == _EXPECTED_EXCLUDES

    tag = result["format"]["elements"][1]
    assert tag["type"] == "tag"
    assert tag["begin"] == "</think>\n<tool_call>\n<function=computer>\n"
    assert tag["end"] == "\n</function>\n</tool_call>"
    assert tag["content"]["type"] == "qwen_xml_parameter"

    grammar = str(Grammar.from_structural_tag(result))
    assert "sequence ::= ((any_text tag))" in grammar
    assert 'tag ::= (("</think>\\n<tool_call>\\n<function=computer>\\n"' in grammar
    assert "triggered_tags ::= ((" not in grammar


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

    prefix = result["format"]["elements"][0]
    assert prefix["type"] == "any_text"
    assert prefix["excludes"] == _EXPECTED_EXCLUDES

    tag = result["format"]["elements"][1]
    assert tag["begin"] == "</think>\n<tool_call>\n<function=computer>\n"
    assert tag["content"]["type"] == "qwen_xml_parameter"
