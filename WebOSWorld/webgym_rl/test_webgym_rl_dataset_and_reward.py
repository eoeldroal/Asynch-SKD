from WebOSWorld.webgym_rl.create_webgym_rl_dataset import build_rows
from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl


def test_build_rows_uses_real_webgym_task_ids_and_prompts():
    tasks = [
        {"task_id": "form", "task_name": "Fill the form.", "website": "https://example.test/form"},
        {"task_id": "example_home", "task_name": "Open Example Domain.", "website": "https://example.com"},
    ]

    rows = build_rows(split="train", tasks=tasks, num_samples=3)

    assert [row["extra_info"]["task_id"] for row in rows] == ["form", "example_home", "form"]
    assert rows[0]["data_source"] == "webgym_rl"
    assert rows[0]["agent_name"] == "web_skd_agent"
    assert rows[0]["reward_model"] == {"style": "webgym_rl", "ground_truth": "env_reward"}
    assert rows[0]["extra_info"]["tools_kwargs"]["web_osgym"]["create_kwargs"] == {
        "task_id": "form"
    }
    assert rows[0]["prompt"][1]["content"] == "Fill the form."


def test_compute_score_webgym_rl_prefers_environment_reward():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant model text",
        ground_truth="env_reward",
        extra_info={"web_osgym_reward_score": 1.0},
    )

    assert score == 1.0


def test_compute_score_webgym_rl_defaults_to_zero_without_environment_reward():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="DONE",
        ground_truth="env_reward",
        extra_info={},
    )

    assert score == 0.0
