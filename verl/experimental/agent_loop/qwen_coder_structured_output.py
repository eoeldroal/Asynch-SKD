from __future__ import annotations

import json
from typing import Any

from xgrammar import get_builtin_structural_tag


_BUNDLED_TOOL_NAME = "computer"


def _ensure_bundled_computer_tool(tool_schemas: list[dict[str, Any]]) -> None:
    if len(tool_schemas) != 1:
        raise ValueError(f"Expected exactly one bundled tool, got {len(tool_schemas)}")

    tool = tool_schemas[0]
    function = tool.get("function") or {}
    if function.get("name") != _BUNDLED_TOOL_NAME:
        raise ValueError(
            "Structured decoding for WebOSWorld RL requires the bundled 'computer' tool schema."
        )


def build_qwen_coder_structured_tag_json(tool_schemas: list[dict[str, Any]]) -> str:
    _ensure_bundled_computer_tool(tool_schemas)
    structural_tag = get_builtin_structural_tag(
        "qwen_coder", tools=tool_schemas, reasoning=False
    )
    return json.dumps(structural_tag.model_dump(exclude_none=True))
