from __future__ import annotations

import json
from typing import Any

from xgrammar.structural_tag import (
    QwenXMLParameterFormat,
    StructuralTag,
    TagFormat,
    TriggeredTagsFormat,
)


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
    """Build a structural_tag JSON for Qwen3.5-Coder WebOSWorld RL rollout.

    The Qwen3.5 chat template unconditionally appends ``<think>\\n`` to the
    prompt when ``add_generation_prompt=True``.  The grammar therefore must NOT
    force another ``<think>`` token at generation start.  Instead it assumes the
    opening tag is already in the prompt and constrains only what the model
    generates after it:

      [reasoning content]</think>
      <tool_call>\\n<function=computer>\\n
      <parameter=actions>\\n[JSON array]\\n</parameter>
      \\n</function>\\n</tool_call>
      <|im_end|>

    Key properties enforced by the grammar:
    - ``</think>`` is mandatory before any tool call (EOS blocked until then).
    - At least one ``<tool_call>`` block must be generated (``at_least_one=True``).
    - ``actions`` array must contain at least one object (``minItems: 1``).

    XGrammar requires the trigger to be a prefix of the tag ``begin`` string.
    The trigger ``</think>\\n<tool_call>\\n<function=`` therefore spans the closing
    think tag and the full tool-call opening up to the function name, with the
    ``begin`` completing it to ``computer>\\n``.  This guarantees ``</think>``
    appears in the output before any constrained tool-call block.
    """
    _ensure_bundled_computer_tool(tool_schemas)
    parameters = tool_schemas[0]["function"]["parameters"]

    # The chat template has already written <think>\n into the prompt.
    # The trigger begins with </think> so the model's free reasoning text is
    # unconstrained until it closes the think block.  XGrammar then forces the
    # remainder of begin ("computer>\n"), constrains the parameter content, and
    # closes with the end string.
    tag = StructuralTag(format=TriggeredTagsFormat(
        triggers=["</think>\n<tool_call>\n<function="],
        tags=[
            TagFormat(
                begin="</think>\n<tool_call>\n<function=computer>\n",
                content=QwenXMLParameterFormat(json_schema=parameters),
                end="\n</function>\n</tool_call>",
            )
        ],
        at_least_one=True,
        stop_after_first=True,
    ))
    return json.dumps(tag.model_dump(exclude_none=True))
