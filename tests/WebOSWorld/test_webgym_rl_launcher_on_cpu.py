from pathlib import Path


def test_webgym_rl_launcher_uses_timestamped_rollout_log_dir():
    launcher = Path("/home/sogang_nlpy/verl/WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh")
    text = launcher.read_text(encoding="utf-8")

    assert 'RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"' in text
    assert 'ROLLOUT_DATA_DIR=/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_${RUN_TIMESTAMP}' in text
