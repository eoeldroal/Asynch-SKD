from pathlib import Path


PROMPT_PATH = Path("/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/system_prompt_webgym_rl_action_named.txt")
VAL_SCRIPT_PATH = Path(
    "/home/sogang_nlpy/verl/WebOSWorld/val_run_qwen35_webgym_fully_async_rl_tool_veomni_action_named.sh"
)


def test_action_named_system_prompt_exists_and_uses_coordinate_guidance():
    text = PROMPT_PATH.read_text()

    assert "Provide x and y as separate integer fields" not in text
    assert "provide coordinate as a JSON array of two integers" in text
    assert "WAIT with an optional duration in seconds" in text
    assert "MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP, RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY, WAIT, DONE, and FAIL" in text


def test_action_named_val_script_points_at_action_named_assets_without_structured_output():
    text = VAL_SCRIPT_PATH.read_text()

    assert "WEBGYM_TOOL_CONFIG_PATH=/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml" in text
    assert "WEBGYM_SYSTEM_PROMPT_PATH=/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/system_prompt_webgym_rl_action_named.txt" in text
    assert "enable_qwen3_coder_structured_output=True" not in text
