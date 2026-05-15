import math

import pytest

from WebOSWorld.webgym_rl.create_webgym_rl_dataset import build_rows
from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl


def test_build_rows_uses_real_webgym_task_ids_and_prompts():
    tasks = [
        {"task_id": "form", "task_name": "Fill the form.", "website": "https://example.test/form"},
        {"task_id": "example_home", "task_name": "Open Example Domain.", "website": "https://example.com"},
    ]

    rows = build_rows(split="train", tasks=tasks, num_samples=3, agent_name="web_skd_agent")

    assert [row["extra_info"]["task_id"] for row in rows] == ["form", "example_home", "form"]
    assert rows[0]["data_source"] == "webgym_rl"
    assert rows[0]["agent_name"] == "web_skd_agent"
    assert rows[0]["reward_model"] == {"style": "webgym_rl", "ground_truth": "env_reward"}
    assert rows[0]["extra_info"]["tools_kwargs"]["web_osgym"]["create_kwargs"] == {
        "task_id": "form"
    }
    assert rows[0]["prompt"] == [{"role": "user", "content": "Fill the form."}]


def test_compute_score_webgym_rl_prefers_environment_reward():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={"web_osgym_env_reward_score": 1.0},
    )

    assert score["score"] == 1.0
    assert score["web_osgym_env_reward_score"] == 1.0
    assert score["web_osgym_format_reward"] == 0.0


def test_compute_score_webgym_rl_defaults_to_zero_without_environment_reward():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="DONE",
        ground_truth="env_reward",
        extra_info={},
    )

    assert score["score"] == 0.0
    assert score["web_osgym_env_reward_score"] == 0.0
    assert score["web_osgym_format_reward"] == 0.0


def test_compute_score_webgym_rl_adds_format_reward_with_first_valid_latency():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 1.0,
            "web_osgym_attempted_tool_calls": 2,
            "web_osgym_valid_tool_calls": 2,
            "web_osgym_first_valid_tool_call_index": 1,
        },
        format_reward_alpha=0.03,
        format_reward_tau=2.0,
    )

    assert score["score"] == 1.0 + 0.03
    assert score["web_osgym_format_reward"] == 1.0
    assert score["web_osgym_attempted_tool_calls"] == 2
    assert score["web_osgym_valid_tool_calls"] == 2
    assert score["web_osgym_first_valid_tool_call_index"] == 1


def test_compute_score_webgym_rl_penalizes_late_first_valid_tool_call():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_attempted_tool_calls": 4,
            "web_osgym_valid_tool_calls": 2,
            "web_osgym_first_valid_tool_call_index": 3,
            "web_osgym_termination_reason": "system_stop",
        },
        format_reward_alpha=1.0,
        format_reward_tau=2.0,
    )

    expected_format = (2.0 / 4.0) * math.exp(-1.0)
    assert score["score"] == expected_format
    assert score["web_osgym_format_reward"] == expected_format


def test_compute_score_webgym_rl_penalizes_budget_exhaustion():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_attempted_tool_calls": 6,
            "web_osgym_valid_tool_calls": 1,
            "web_osgym_first_valid_tool_call_index": 4,
            "web_osgym_termination_reason": "tool_response_budget_exhausted",
        },
        format_reward_alpha=1.0,
        format_reward_tau=2.0,
    )

    expected_format = (1.0 / 6.0) * math.exp(-1.5) - 0.15
    assert score["score"] == expected_format
    assert score["web_osgym_format_reward"] == expected_format


def test_compute_score_webgym_rl_zeroes_format_reward_without_tool_calls():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
        },
        format_reward_alpha=0.05,
        format_reward_tau=2.0,
    )

    assert score["score"] == 0.0
    assert score["web_osgym_format_reward"] == 0.0


def test_compute_score_webgym_rl_can_gate_format_reward_on_environment_progress():
    # Mirrors the observed reward-hack shape from fully async RL:
    # all tool calls are valid and first-valid is early, but env reward is still zero.
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_attempted_tool_calls": 8,
            "web_osgym_valid_tool_calls": 8,
            "web_osgym_first_valid_tool_call_index": 1,
            "web_osgym_termination_reason": "tool_response_budget_exhausted",
        },
        format_reward_alpha=0.5,
        format_reward_tau=2.0,
        format_reward_gate_by_env_score=True,
    )

    assert score["score"] == 0.0
    assert score["web_osgym_env_reward_score"] == 0.0
    assert score["web_osgym_format_reward"] == 0.0


def test_compute_score_webgym_rl_unlocks_gated_format_reward_after_environment_progress():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 1.0,
            "web_osgym_attempted_tool_calls": 8,
            "web_osgym_valid_tool_calls": 8,
            "web_osgym_first_valid_tool_call_index": 1,
            "web_osgym_termination_reason": "tool_response_budget_exhausted",
        },
        format_reward_alpha=0.5,
        format_reward_tau=2.0,
        format_reward_gate_by_env_score=True,
    )

    assert score["web_osgym_format_reward"] == pytest.approx(0.85)
    assert score["score"] == pytest.approx(1.425)


def test_compute_score_webgym_rl_decays_positive_format_reward_by_non_grounding_adjacency_ratio():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_attempted_tool_calls": 14,
            "web_osgym_valid_tool_calls": 14,
            "web_osgym_first_valid_tool_call_index": 1,
            "web_osgym_executed_action_count": 14,
            "web_osgym_non_grounding_adjacent_pair_count": 10,
        },
        format_reward_alpha=0.1,
        format_reward_tau=2.0,
    )

    expected_ratio = 10.0 / 13.0
    expected_raw_format = 1.0
    expected_effective_format = expected_raw_format * (1.0 - expected_ratio)
    assert score["web_osgym_raw_format_reward"] == pytest.approx(expected_raw_format)
    assert score["web_osgym_non_grounding_adjacency_ratio"] == pytest.approx(expected_ratio)
    assert score["web_osgym_format_reward"] == pytest.approx(expected_effective_format)
    assert score["score"] == pytest.approx(0.1 * expected_effective_format)


def test_compute_score_webgym_rl_keeps_negative_format_penalty_unchanged_under_repetition():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_attempted_tool_calls": 6,
            "web_osgym_valid_tool_calls": 1,
            "web_osgym_first_valid_tool_call_index": 4,
            "web_osgym_termination_reason": "tool_response_budget_exhausted",
            "web_osgym_executed_action_count": 12,
            "web_osgym_non_grounding_adjacent_pair_count": 11,
        },
        format_reward_alpha=0.1,
        format_reward_tau=2.0,
    )

    expected_raw_format = (1.0 / 6.0) * math.exp(-1.5) - 0.15
    assert expected_raw_format < 0.0
    assert score["web_osgym_raw_format_reward"] == pytest.approx(expected_raw_format)
    assert score["web_osgym_format_reward"] == pytest.approx(expected_raw_format)
    assert score["score"] == pytest.approx(0.1 * expected_raw_format)
