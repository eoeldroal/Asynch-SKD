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


def test_compute_score_webgym_rl_adds_normalized_format_reward():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 1.0,
            "web_osgym_attempted_tool_calls": 2,
            "web_osgym_valid_tool_calls": 2,
        },
        format_reward_alpha=0.03,
        format_reward_min_denominator=5,
    )

    assert score["score"] == 1.0 + 0.03 * (2.0 / 5.0)
    assert score["web_osgym_format_reward"] == 2.0 / 5.0
    assert score["web_osgym_attempted_tool_calls"] == 2
    assert score["web_osgym_valid_tool_calls"] == 2


def test_compute_score_webgym_rl_zeroes_format_reward_without_tool_calls():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 0.0,
        },
        format_reward_alpha=0.05,
        format_reward_min_denominator=5,
    )

    assert score["score"] == 0.0
    assert score["web_osgym_format_reward"] == 0.0
