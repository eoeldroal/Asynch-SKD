from __future__ import annotations

import numpy as np

from verl.experimental.agent_loop.web_osgym_windowing import (
    build_mini_step_image_spans,
    contiguous_one_spans,
    format_previous_actions,
    normalize_image_spans,
    normalize_web_osgym_steps,
    select_recent_web_osgym_steps,
)


def test_contiguous_one_spans_handles_simple_masks():
    assert contiguous_one_spans([1, 1, 0, 0, 1, 1, 1]) == [(0, 2), (4, 7)]
    assert contiguous_one_spans([0, 0, 0]) == []
    assert contiguous_one_spans([1, 0, 1]) == [(0, 1), (2, 3)]
    assert contiguous_one_spans([]) == []


def test_normalize_image_spans_handles_object_arrays_and_clamps_bounds():
    raw = np.array(
        [
            {"step_idx": "5", "image_start": -3, "image_end": 2, "terminal": "yes"},
            "skip-me",
            {"step_idx": 2, "image_start": -1, "image_end": -9, "terminal": 0},
        ],
        dtype=object,
    )

    assert normalize_image_spans(raw) == [
        {"step_idx": 2, "image_start": 0, "image_end": 0, "terminal": False},
        {"step_idx": 5, "image_start": 0, "image_end": 2, "terminal": True},
    ]


def test_normalize_web_osgym_steps_coerces_and_filters_fields():
    raw = np.array(
        [
            {
                "step_idx": "3",
                "assistant_turn": "2",
                "user_turn": 1,
                "phase": 7,
                "text_len": "11",
                "action_names": np.array(["CLICK", 2], dtype=object),
                "actions": [{"action_type": "CLICK", "x": "12"}, "skip", {"button": "left"}],
                "image_start": -5,
                "image_end": 2,
                "terminal": "true",
                "termination_reason": 9,
            },
            None,
            {
                "step_idx": 1,
                "assistant_turn": 0,
                "user_turn": "4",
                "phase": "observation",
                "text_len": 0,
                "action_names": "WAIT",
                "actions": {"action_type": "WAIT", "duration": 1},
                "image_start": -2,
                "image_end": 0,
                "terminal": 0,
                "termination_reason": None,
            },
        ],
        dtype=object,
    )

    assert normalize_web_osgym_steps(raw) == [
        {
            "step_idx": 1,
            "assistant_turn": 0,
            "user_turn": 4,
            "phase": "observation",
            "text": "",
            "text_len": 0,
            "action_names": ["WAIT"],
            "actions": [{"action_type": "WAIT", "duration": 1}],
            "image_start": 0,
            "image_end": 0,
            "terminal": False,
            "termination_reason": None,
        },
        {
            "step_idx": 3,
            "assistant_turn": 2,
            "user_turn": 1,
            "phase": "7",
            "text": "",
            "text_len": 11,
            "action_names": ["CLICK", "2"],
            "actions": [{"action_type": "CLICK", "x": "12"}, {"button": "left"}],
            "image_start": 0,
            "image_end": 2,
            "terminal": True,
            "termination_reason": "9",
        },
    ]


def test_build_mini_step_image_spans_keeps_only_image_bearing_steps():
    steps = normalize_web_osgym_steps(
        [
            {"step_idx": 1, "image_start": 0, "image_end": 0, "terminal": False},
            {"step_idx": 2, "image_start": 0, "image_end": 1, "terminal": False},
            {"step_idx": 3, "image_start": 1, "image_end": 3, "terminal": True},
        ]
    )

    assert build_mini_step_image_spans(steps) == [
        {"step_idx": 2, "image_start": 0, "image_end": 1, "terminal": False},
        {"step_idx": 3, "image_start": 1, "image_end": 3, "terminal": True},
    ]


def test_select_recent_web_osgym_steps_applies_target_history_and_image_cap():
    steps = normalize_web_osgym_steps(
        [
            {"step_idx": 1, "image_start": 0, "image_end": 1},
            {"step_idx": 2, "image_start": 1, "image_end": 1},
            {"step_idx": 3, "image_start": 1, "image_end": 2},
            {"step_idx": 4, "image_start": 2, "image_end": 4},
            {"step_idx": 5, "image_start": 4, "image_end": 4},
        ]
    )

    assert [step["step_idx"] for step in select_recent_web_osgym_steps(steps, target_step_idx=4, history_n=2)] == [
        2,
        3,
        4,
    ]
    assert [
        step["step_idx"]
        for step in select_recent_web_osgym_steps(steps, target_step_idx=4, history_n=3, max_images_per_sample=1)
    ] == [4]


def test_format_previous_actions_formats_numbered_actions_and_none():
    actions = [
        {"action_type": "CLICK", "x": 12, "y": 34, "button": "left"},
        {"action_type": "WAIT", "duration": 1},
    ]

    assert format_previous_actions(actions) == "Step 1: CLICK(x=12, y=34, button='left')\nStep 2: WAIT(duration=1)"
    assert format_previous_actions([]) == "None"


def test_format_previous_actions_flattens_nested_actions_from_normalized_steps():
    steps = normalize_web_osgym_steps(
        [
            {
                "step_idx": 1,
                "actions": [{"action_type": "CLICK", "x": 12, "y": 34, "button": "left"}],
            },
            {
                "step_idx": 2,
                "actions": [{"action_type": "WAIT", "duration": 1}],
            },
        ]
    )

    assert format_previous_actions(steps) == "Step 1: CLICK(x=12, y=34, button='left')\nStep 2: WAIT(duration=1)"
