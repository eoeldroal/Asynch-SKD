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
    image_indices: list[int]
    selected_steps: list[dict[str, Any]]
    current_step_idx: int
    old_summary_turn_indices: list[int]
    recent_observation_step_indices: list[int]
    recent_assistant_turn_indices: list[int]
    text_only_recent_step_count: int


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


def _coordinate_guidance() -> str:
    return "All action coordinates use a 1000x1000 screen coordinate system with origin at the top-left corner."


def _build_prompt_text(instruction: str, previous_actions: str, prompt_images: Sequence[Any]) -> str:
    del prompt_images
    return (
        "Please generate the next move according to the UI screenshot, instruction and previous actions.\n"
        f"{_coordinate_guidance()}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous_actions}"
    )


def _normalize_assistant_turns(value: Any) -> list[dict[str, Any]]:
    turns = []
    for item in _coerce_list(value):
        if not isinstance(item, Mapping):
            continue
        turns.append(
            {
                "assistant_turn": int(item.get("assistant_turn", len(turns) + 1)),
                "observation_step_idx": int(item.get("observation_step_idx", len(turns) + 1)),
                "response_text": str(item.get("response_text", "")),
                "actions": list(item.get("actions") or []),
            }
        )
    turns.sort(key=lambda item: (item["assistant_turn"], item["observation_step_idx"]))
    return turns


def _step_observation_text(step: Mapping[str, Any]) -> str | None:
    text = step.get("text")
    if text is None:
        return None
    text = str(text).strip()
    return text or None


def _observation_message(
    *,
    image_count: int,
    step_text: str | None,
    instruction_text: str | None,
) -> dict[str, Any]:
    content = [{"type": "image"} for _ in range(image_count)]
    text_parts: list[str] = []
    if instruction_text:
        text_parts.append(instruction_text)
    if step_text:
        observation_text = f"Observation:\n{step_text}"
        if instruction_text:
            text_parts.append(f"\n\n{observation_text}")
        else:
            text_parts.append(observation_text)
    if text_parts:
        content.append({"type": "text", "text": "".join(text_parts)})
    if not content:
        raise ValueError("observation message requires at least one image or one text block")
    return {"role": "user", "content": content}


def build_web_osgym_prompt_window(
    base_messages: Any,
    images: Any,
    steps: Any,
    assistant_turns: Any = None,
    *,
    history_n: int = 5,
    max_images_per_sample: int = 6,
) -> WebOsGymPromptWindow:
    normalized_steps = normalize_web_osgym_steps(steps)
    if not normalized_steps:
        raise ValueError("web_osgym_steps must include at least one observation step")

    current_step_idx = int(normalized_steps[-1]["step_idx"])
    selected_steps = select_recent_web_osgym_steps(
        normalized_steps,
        target_step_idx=current_step_idx,
        history_n=history_n,
        max_images_per_sample=max_images_per_sample,
    )
    if not selected_steps:
        raise ValueError("web_osgym_steps must include at least one observation step")

    normalized_turns = _normalize_assistant_turns(assistant_turns)
    recent_start_step_idx = int(selected_steps[0]["step_idx"])
    old_turns = [turn for turn in normalized_turns if int(turn["observation_step_idx"]) < recent_start_step_idx]
    recent_turns = [
        turn
        for turn in normalized_turns
        if recent_start_step_idx <= int(turn["observation_step_idx"]) < current_step_idx
    ]
    turn_by_step_idx = {int(turn["observation_step_idx"]): turn for turn in recent_turns}

    system_messages = [
        dict(message)
        for message in _coerce_list(base_messages)
        if isinstance(message, Mapping) and message.get("role") == "system"
    ]
    instruction = _extract_latest_user_instruction(base_messages)

    image_list = _coerce_list(images)
    prompt_images: list[Any] = []
    prompt_image_indices: list[int] = []
    text_only_recent_step_count = 0
    for step in selected_steps:
        image_start = int(step["image_start"])
        image_end = int(step["image_end"])
        step_image_indices = list(range(image_start, image_end))
        step_images = [image_list[idx] for idx in step_image_indices]
        prompt_images.extend(step_images)
        prompt_image_indices.extend(step_image_indices)

    instruction_prompt = _build_prompt_text(instruction, format_previous_actions(old_turns), prompt_images)
    messages: list[dict[str, Any]] = list(system_messages)

    for step_index, step in enumerate(selected_steps):
        image_start = int(step["image_start"])
        image_end = int(step["image_end"])
        step_image_indices = list(range(image_start, image_end))
        step_images = [image_list[idx] for idx in step_image_indices]

        step_text = _step_observation_text(step)
        if not step_images and step_text is None:
            raise ValueError(f"selected step {step['step_idx']} has neither image nor text")
        if not step_images and step_text is not None:
            text_only_recent_step_count += 1

        messages.append(
            _observation_message(
                image_count=len(step_images),
                step_text=step_text,
                instruction_text=instruction_prompt if step_index == 0 else None,
            )
        )

        step_idx = int(step["step_idx"])
        if step_idx == current_step_idx:
            continue

        assistant_turn = turn_by_step_idx.get(step_idx)
        if assistant_turn is None:
            raise ValueError(f"missing assistant turn for observation step {step_idx}")
        messages.append({"role": "assistant", "content": assistant_turn["response_text"]})

    return WebOsGymPromptWindow(
        messages=messages,
        images=prompt_images,
        image_indices=prompt_image_indices,
        selected_steps=[dict(step) for step in selected_steps],
        current_step_idx=current_step_idx,
        old_summary_turn_indices=[int(turn["assistant_turn"]) for turn in old_turns],
        recent_observation_step_indices=[int(step["step_idx"]) for step in selected_steps],
        recent_assistant_turn_indices=[int(turn["assistant_turn"]) for turn in recent_turns],
        text_only_recent_step_count=text_only_recent_step_count,
    )
