from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verl.experimental.agent_loop.web_osgym_windowing import (
    format_previous_actions,
    normalize_web_osgym_steps,
    select_recent_web_osgym_steps,
)


@dataclass(frozen=True)
class WebOsGymPromptWindow:
    messages: list[dict[str, Any]]
    images: list[Any]
    selected_steps: list[dict[str, Any]]
    current_step_idx: int


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        if isinstance(converted, tuple):
            return list(converted)
        return [converted]
    return [value]


def _extract_instruction_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        text_blocks: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if text is not None:
                text_blocks.append(str(text))
        return "".join(text_blocks)
    return str(content)


def _extract_latest_user_instruction(base_messages: Any) -> str:
    for message in reversed(_coerce_list(base_messages)):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "user":
            continue
        return _extract_instruction_from_content(message.get("content"))
    return ""


def _build_prompt_text(instruction: str, previous_actions: str) -> str:
    return (
        "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous_actions}"
    )


def build_web_osgym_prompt_window(
    base_messages: Any,
    images: Any,
    steps: Any,
    *,
    history_n: int = 5,
    max_images_per_sample: int = 6,
) -> WebOsGymPromptWindow:
    normalized_steps = normalize_web_osgym_steps(steps)
    if not any(int(step["image_end"]) > int(step["image_start"]) for step in normalized_steps):
        raise ValueError("web_osgym_steps must include at least one observation step")

    current_step_idx = int(normalized_steps[-1]["step_idx"])
    selected_steps = select_recent_web_osgym_steps(
        normalized_steps,
        target_step_idx=current_step_idx,
        history_n=history_n,
        max_images_per_sample=max_images_per_sample,
    )
    current_step = selected_steps[-1]
    image_start = int(current_step["image_start"])
    image_end = int(current_step["image_end"])
    if image_end <= image_start:
        raise ValueError("current step must include at least one image")

    image_list = _coerce_list(images)
    current_images = image_list[image_start:image_end]
    if not current_images:
        raise ValueError("current step must include at least one image")

    system_messages = [
        dict(message)
        for message in _coerce_list(base_messages)
        if isinstance(message, Mapping) and message.get("role") == "system"
    ]
    instruction = _extract_latest_user_instruction(base_messages)
    previous_actions = format_previous_actions(
        [step for step in normalized_steps if int(step["step_idx"]) < current_step_idx]
    )

    content = [{"type": "image"} for _ in current_images]
    content.append({"type": "text", "text": _build_prompt_text(instruction, previous_actions)})

    return WebOsGymPromptWindow(
        messages=system_messages + [{"role": "user", "content": content}],
        images=current_images,
        selected_steps=[dict(step) for step in selected_steps],
        current_step_idx=current_step_idx,
    )
