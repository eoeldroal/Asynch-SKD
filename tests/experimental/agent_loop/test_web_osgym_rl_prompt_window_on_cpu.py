from __future__ import annotations

import pytest
from PIL import Image

from verl.experimental.agent_loop.web_osgym_rl_prompt_window import build_web_osgym_prompt_window


def test_build_web_osgym_prompt_window_keeps_old_actions_only_outside_history_window():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[
            {"role": "system", "content": "You are a browser agent."},
            {"role": "user", "content": "Open settings"},
        ],
        images=["obs-1", "obs-2", "obs-3", "obs-4"],
        steps=[
            {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
            {"step_idx": 2, "phase": "tool_observation", "image_start": 1, "image_end": 2},
            {"step_idx": 3, "phase": "tool_observation", "image_start": 2, "image_end": 3},
            {"step_idx": 4, "phase": "tool_observation", "image_start": 3, "image_end": 4},
        ],
        assistant_turns=[
            {
                "assistant_turn": 1,
                "observation_step_idx": 1,
                "response_text": "A1",
                "actions": [{"action_type": "CLICK", "x": 10, "y": 20}],
            },
            {
                "assistant_turn": 2,
                "observation_step_idx": 2,
                "response_text": "A2",
                "actions": [{"action_type": "WAIT", "duration": 1}],
            },
            {
                "assistant_turn": 3,
                "observation_step_idx": 3,
                "response_text": "A3",
                "actions": [{"action_type": "CLICK", "x": 30, "y": 40}],
            },
        ],
        history_n=2,
        max_images_per_sample=6,
    )

    assert prompt_window.image_indices == [1, 2, 3]
    assert prompt_window.old_summary_turn_indices == [1]
    assert prompt_window.recent_observation_step_indices == [2, 3, 4]
    assert prompt_window.recent_assistant_turn_indices == [2, 3]

    first_user = prompt_window.messages[1]
    assert first_user["role"] == "user"
    assert first_user["content"][0] == {"type": "image"}
    assert "Previous actions:\nStep 1: CLICK(x=10, y=20)" in first_user["content"][-1]["text"]
    assert "WAIT" not in first_user["content"][-1]["text"]
    assert "CLICK(x=30, y=40)" not in first_user["content"][-1]["text"]


def test_build_web_osgym_prompt_window_emits_live_recent_user_assistant_chain():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Open settings"}],
        images=["obs-1", "obs-2", "obs-3"],
        steps=[
            {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
            {"step_idx": 2, "phase": "tool_observation", "image_start": 1, "image_end": 2},
            {"step_idx": 3, "phase": "tool_observation", "image_start": 2, "image_end": 3},
        ],
        assistant_turns=[
            {
                "assistant_turn": 1,
                "observation_step_idx": 1,
                "response_text": "A1",
                "actions": [{"action_type": "CLICK", "x": 10, "y": 20}],
            },
            {
                "assistant_turn": 2,
                "observation_step_idx": 2,
                "response_text": "A2",
                "actions": [{"action_type": "CLICK", "x": 30, "y": 40}],
            },
        ],
        history_n=2,
        max_images_per_sample=6,
    )

    assert prompt_window.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: Open settings\n\nPrevious actions:\nNone",
                },
            ],
        },
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": [{"type": "image"}]},
    ]


def test_build_web_osgym_prompt_window_keeps_text_only_recent_step_without_fake_image():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Recover after failure"}],
        images=["obs-1", "obs-3"],
        steps=[
            {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
            {
                "step_idx": 2,
                "phase": "tool_observation",
                "image_start": 1,
                "image_end": 1,
                "text": "Action failed: target was not focused",
            },
            {"step_idx": 3, "phase": "tool_observation", "image_start": 1, "image_end": 2},
        ],
        assistant_turns=[
            {
                "assistant_turn": 1,
                "observation_step_idx": 1,
                "response_text": "A1",
                "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
            },
            {
                "assistant_turn": 2,
                "observation_step_idx": 2,
                "response_text": "A2",
                "actions": [{"action_type": "CLICK", "x": 3, "y": 4}],
            },
        ],
        history_n=2,
        max_images_per_sample=6,
    )

    assert prompt_window.image_indices == [0, 1]
    assert prompt_window.text_only_recent_step_count == 1
    assert prompt_window.messages[2] == {
        "role": "user",
        "content": [{"type": "text", "text": "Observation:\nAction failed: target was not focused"}],
    }


def test_build_web_osgym_prompt_window_allows_text_only_current_step():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Recover after failure"}],
        images=["obs-1"],
        steps=[
            {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
            {
                "step_idx": 2,
                "phase": "tool_observation",
                "image_start": 1,
                "image_end": 1,
                "text": "Action failed: target was not focused",
            },
        ],
        assistant_turns=[
            {
                "assistant_turn": 1,
                "observation_step_idx": 1,
                "response_text": "A1",
                "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
            }
        ],
        history_n=1,
        max_images_per_sample=6,
    )

    assert prompt_window.image_indices == [0]
    assert prompt_window.messages[-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "Observation:\nAction failed: target was not focused"}],
    }


def test_build_web_osgym_prompt_window_preserves_system_messages_before_user_prompt():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[
            {"role": "system", "content": "You are a precise browser agent."},
            {"role": "system", "content": "Reply with the next action only."},
            {"role": "user", "content": "Find the settings page"},
        ],
        images=["current-image"],
        steps=[{"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1}],
        assistant_turns=[],
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
        assistant_turns=[],
    )

    assert "Instruction: Open the billing tab" in prompt_window.messages[-1]["content"][-1]["text"]


def test_build_web_osgym_prompt_window_raises_when_no_steps_exist():
    with pytest.raises(ValueError, match="at least one observation step"):
        build_web_osgym_prompt_window(
            base_messages=[{"role": "user", "content": "Do the task"}],
            images=[],
            steps=[],
            assistant_turns=[],
        )


def test_build_web_osgym_prompt_window_raises_when_selected_step_has_no_content():
    with pytest.raises(ValueError, match="has neither image nor text"):
        build_web_osgym_prompt_window(
            base_messages=[{"role": "user", "content": "Do the task"}],
            images=["old-image"],
            steps=[
                {"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1},
                {"step_idx": 2, "phase": "action_only", "image_start": 1, "image_end": 1},
            ],
            assistant_turns=[
                {
                    "assistant_turn": 1,
                    "observation_step_idx": 1,
                    "response_text": "A1",
                    "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
                }
            ],
        )


def test_build_web_osgym_prompt_window_does_not_include_coordinate_guidance_in_user_prompt():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Open settings"}],
        images=["obs-1"],
        steps=[{"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1}],
        assistant_turns=[],
    )

    first_text = prompt_window.messages[0]["content"][-1]["text"]
    assert "All action coordinates use a 1000x1000 screen coordinate system with origin at the top-left corner." not in first_text


def test_build_web_osgym_prompt_window_does_not_include_runtime_resolution_when_available():
    prompt_window = build_web_osgym_prompt_window(
        base_messages=[{"role": "user", "content": "Open settings"}],
        images=[Image.new("RGB", (1920, 1080), "white")],
        steps=[{"step_idx": 1, "phase": "initial", "image_start": 0, "image_end": 1}],
        assistant_turns=[],
    )

    first_text = prompt_window.messages[0]["content"][-1]["text"]
    assert "1000x1000" not in first_text
    assert "1920x1080" not in first_text
