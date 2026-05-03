from __future__ import annotations

import pytest

from verl.experimental.agent_loop.web_osgym_prompt_window import build_web_osgym_prompt_window


def _final_text_block(prompt_window) -> str:
    return prompt_window.messages[-1]["content"][-1]["text"]


def test_build_web_osgym_prompt_window_uses_latest_image_and_prior_actions():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Open the profile menu"}],
        images=["image-0", "image-1"],
        steps=[
            {
                "step_idx": 1,
                "phase": "tool_observation",
                "actions": [{"action_type": "CLICK", "x": 12, "y": 34}],
                "image_start": 0,
                "image_end": 1,
            },
            {
                "step_idx": 2,
                "phase": "tool_observation",
                "image_start": 1,
                "image_end": 2,
            },
        ],
    )

    assert prompt_window.images == ["image-1"]
    assert prompt_window.current_step_idx == 2
    assert [step["step_idx"] for step in prompt_window.selected_steps] == [1, 2]
    assert prompt_window.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: Open the profile menu\n\nPrevious actions:\n1. CLICK(x=12, y=34)",
                },
            ],
        }
    ]


def test_build_web_osgym_prompt_window_initial_step_has_none_previous_actions():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Start from the home page"}],
        images=["initial-image"],
        steps=[
            {
                "step_idx": 1,
                "phase": "initial",
                "image_start": 0,
                "image_end": 1,
            }
        ],
    )

    assert prompt_window.images == ["initial-image"]
    assert prompt_window.current_step_idx == 1
    assert _final_text_block(prompt_window).endswith("Previous actions:\nNone")


def test_build_web_osgym_prompt_window_preserves_system_messages_before_user_prompt():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[
            {"role": "system", "content": "You are a precise browser agent."},
            {"role": "system", "content": "Reply with the next action only."},
            {"role": "user", "content": "Find the settings page"},
        ],
        images=["current-image"],
        steps=[{"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1}],
    )

    assert prompt_window.messages[:2] == [
        {"role": "system", "content": "You are a precise browser agent."},
        {"role": "system", "content": "Reply with the next action only."},
    ]
    assert prompt_window.messages[2]["role"] == "user"


def test_build_web_osgym_prompt_window_concatenates_text_blocks_from_latest_user_message():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[
            {"role": "user", "content": "Old instruction"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Open "},
                    {"type": "image"},
                    {"type": "text", "text": "the billing tab"},
                ],
            },
        ],
        images=["current-image"],
        steps=[{"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1}],
    )

    assert "Instruction: Open the billing tab" in _final_text_block(prompt_window)


def test_build_web_osgym_prompt_window_raises_when_no_steps_exist():
    with pytest.raises(ValueError, match="at least one observation step"):
        build_web_osgym_prompt_window(
            base_messages=[{"role": "user", "content": "Do the task"}],
            images=[],
            steps=[],
        )


def test_build_web_osgym_prompt_window_raises_when_current_step_has_no_image():
    with pytest.raises(ValueError, match="current step must include at least one image"):
        build_web_osgym_prompt_window(
            base_messages=[{"role": "user", "content": "Do the task"}],
            images=["old-image"],
            steps=[
                {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
                {"step_idx": 2, "phase": "action_only", "image_start": 1, "image_end": 1},
            ],
        )
