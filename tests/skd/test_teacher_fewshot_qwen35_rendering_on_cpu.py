from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.teacher_fewshot import load_teacher_fewshot_transcript


REPO_ROOT = Path(__file__).resolve().parents[2]
FEWSHOT_PATH = (
    REPO_ROOT
    / "WebOSWorld"
    / "webgym_rl"
    / "teacher_fewshot"
    / "prozilla_task_B_22_minimal"
    / "teacher_fewshot.json"
)
SYSTEM_PROMPT_PATH = REPO_ROOT / "WebOSWorld" / "webgym_rl" / "system_prompt_webgym_rl.txt"
TOOL_CONFIG_PATH = REPO_ROOT / "WebOSWorld" / "config" / "tool_config" / "webgym_rl_tool_config_bundled.yaml"
QWEN35_9B_PATH = Path(
    "/home/sogang_nlpy/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
TASK_PROMPT = "List the contents of the current home directory using the terminal."


def _load_tools() -> list[dict]:
    payload = yaml.safe_load(TOOL_CONFIG_PATH.read_text(encoding="utf-8"))
    return [tool["tool_schema"] for tool in payload["tools"]]


def test_teacher_fewshot_candidate_uses_structured_assistant_fields_only():
    messages, images = load_teacher_fewshot_transcript(FEWSHOT_PATH)

    assert images is not None
    assert len(images) == 6
    assert len(messages) == 13
    assert messages[0] == {"role": "user", "content": TASK_PROMPT}
    assert messages[1]["role"] == "tool"
    assert messages[-1]["role"] == "assistant"

    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    assert len(assistant_messages) == 6
    for message in assistant_messages:
        assert isinstance(message["content"], str)
        assert message["content"].strip()
        assert "reasoning_content" not in message
        assert "</think>" not in message["content"]
        assert "<tool_call>" not in message["content"]
        tool_calls = message["tool_calls"]
        assert isinstance(tool_calls, list) and len(tool_calls) == 1
        function = tool_calls[0]["function"]
        assert function["name"] == "computer"
        assert isinstance(function["arguments"], dict)
        assert isinstance(function["arguments"]["actions"], list)


@pytest.mark.skipif(not QWEN35_9B_PATH.exists(), reason="local Qwen3.5-9B snapshot not available")
def test_teacher_fewshot_candidate_renders_with_runtime_prompt_on_qwen35():
    messages, fewshot_images = load_teacher_fewshot_transcript(FEWSHOT_PATH)
    assert fewshot_images is not None

    runtime_image = fewshot_images[0]
    teacher_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")},
        *messages,
        {"role": "user", "content": TASK_PROMPT},
        {"role": "tool", "content": [{"type": "image"}]},
    ]
    tools = _load_tools()

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN35_9B_PATH), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(QWEN35_9B_PATH), trust_remote_code=True)

    tokenizer_text = tokenizer.apply_chat_template(
        teacher_messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    processor_text = processor.apply_chat_template(
        teacher_messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )

    assert tokenizer_text == processor_text
    assert tokenizer_text.count("<tool_response>") == 7
    assert tokenizer_text.count('<function=computer>') == 6
    assert "I need to open a terminal to list the contents of the home directory." in tokenizer_text
    assert "The terminal is now open and ready for input." in tokenizer_text
    assert '[{"action_type": "DONE"}]' in tokenizer_text
    assert '[{"action_type": "HOTKEY", "keys": ["ctrl", "alt", "t"]}]' in tokenizer_text
    assert tokenizer_text.rstrip().endswith("<|im_start|>assistant\n<think>")

    model_inputs = processor(
        text=[processor_text],
        images=[*fewshot_images, runtime_image],
        videos=None,
        return_tensors="pt",
        do_sample_frames=False,
    )
    prompt_ids = model_inputs["input_ids"][0].tolist()
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0
    assert "image_grid_thw" in model_inputs
    assert tuple(model_inputs["image_grid_thw"].shape) == (7, 3)
