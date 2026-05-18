from pathlib import Path


def test_webgym_rl_launcher_uses_timestamped_rollout_log_dir():
    launcher = Path("/home/sogang_nlpy/verl/WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh")
    text = launcher.read_text(encoding="utf-8")

    assert 'RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"' in text
    assert 'ROLLOUT_DATA_DIR=/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_${RUN_TIMESTAMP}' in text
    assert "webgym_rl_tool_config_bundled.yaml" not in text
    assert "system_prompt_webgym_rl.txt" not in text
    assert "system_prompt_webgym_rl_action_named.txt" in text
    assert "webgym_rl_tool_config.yaml" in text
    assert 'WEBGYM_INITIAL_MODEL_PATH="${WEBGYM_INITIAL_MODEL_PATH:-/home/sogang_nlpy/verl/checkpoints/verl_async_skd_qwen35_webgym/qwen35_9b_to_27b_async_skd_webgym_fast_test_gdn_fix/global_step_5/actor/huggingface}"' in text
    assert 'WEBGYM_EXPERIMENT_NAME="${WEBGYM_EXPERIMENT_NAME:-qwen35_9b_step5_init_fully_async_webgym_tool}"' in text
    assert "web_osgym_window_enable=False" not in text
    assert "max_assistant_response_tokens=8192" in text
    assert "enable_qwen3_coder_structured_output=False" in text
    assert 'WEBGYM_LLM_JUDGE_ENABLE="${WEBGYM_LLM_JUDGE_ENABLE:-false}"' not in text
    assert 'WEBGYM_LLM_JUDGE_MODEL="${WEBGYM_LLM_JUDGE_MODEL:-gpt-5.4}"' not in text
    assert "reward.reward_manager.name=rate_limited" not in text
    assert "reward.num_workers=1" not in text
    assert "+reward.max_concurrent=1" not in text
    assert "+reward.timeout=300" not in text
    assert "llm_judge_enable=false" in text
    assert "llm_judge_model=gpt-5.4-mini" in text
    assert "llm_judge_only_zerogroup=true" in text
    assert "llm_judge_reasoning_effort=high" in text
    assert "llm_judge_image_detail=auto" in text
    assert "llm_judge_max_concurrency=6" in text
    assert "trainer.resume_mode=disable" in text
    assert '"trainer.experiment_name=${WEBGYM_EXPERIMENT_NAME}"' in text
    assert '"trainer.default_local_dir=/home/sogang_nlpy/verl/checkpoints/RL_main/${WEBGYM_EXPERIMENT_NAME}"' in text
