from __future__ import annotations

import json
from typing import Any

from xgrammar.structural_tag import (
    AnyTextFormat,
    QwenXMLParameterFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)


_BUNDLED_TOOL_NAME = "computer"
_QWEN_35_SPECIAL_AND_STRUCTURE_TOKENS = [
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


def _ensure_bundled_computer_tool(tool_schemas: list[dict[str, Any]]) -> None:
    if len(tool_schemas) != 1:
        raise ValueError(f"Expected exactly one bundled tool, got {len(tool_schemas)}")

    tool = tool_schemas[0]
    function = tool.get("function") or {}
    if function.get("name") != _BUNDLED_TOOL_NAME:
        raise ValueError(
            "Structured decoding for WebOSWorld RL requires the bundled 'computer' tool schema."
        )


def build_qwen_coder_structured_tag_json_deprecated(tool_schemas: list[dict[str, Any]]) -> str:
    """Deprecated SequenceFormat-based Qwen3.5-Coder structural tag builder.

    Kept only for comparison / rollback reference while the active path returns
    to the simpler builtin qwen_coder structural tag.
    """
    _ensure_bundled_computer_tool(tool_schemas)
    parameters = tool_schemas[0]["function"]["parameters"]

    tag = StructuralTag(format=SequenceFormat(elements=[
        AnyTextFormat(excludes=_QWEN_35_SPECIAL_AND_STRUCTURE_TOKENS),
        TagFormat(
            begin="</think>\n<tool_call>\n<function=computer>\n",
            content=QwenXMLParameterFormat(json_schema=parameters),
            end="\n</function>\n</tool_call>",
        ),
    ]))
    return json.dumps(tag.model_dump(exclude_none=True))
