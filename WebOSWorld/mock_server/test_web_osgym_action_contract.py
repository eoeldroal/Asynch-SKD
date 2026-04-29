from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from WebOSWorld.mock_server.web_osgym_mock_server import create_app


TOOL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "tool_config"
ACTIVE_TOOL_CONFIGS = {
    "webgym_rl_tool_config.yaml",
    "web_osgym_tool_config_webgym_rl.yaml",
}


def test_tool_config_dir_contains_only_real_and_mock_configs():
    assert {path.name for path in TOOL_CONFIG_DIR.glob("*.yaml")} == ACTIVE_TOOL_CONFIGS


@pytest.mark.parametrize(
    "config_name",
    sorted(ACTIVE_TOOL_CONFIGS),
)
def test_tool_configs_expose_official_computer_13_action_contract(config_name):
    config = yaml.safe_load((TOOL_CONFIG_DIR / config_name).read_text())
    expected_required = {
        "MOVE_TO": {"x", "y"},
        "CLICK": set(),
        "MOUSE_DOWN": set(),
        "MOUSE_UP": set(),
        "RIGHT_CLICK": set(),
        "DOUBLE_CLICK": set(),
        "DRAG_TO": {"x", "y"},
        "SCROLL": {"dx", "dy"},
        "TYPING": {"text"},
        "PRESS": {"key"},
        "KEY_DOWN": {"key"},
        "KEY_UP": {"key"},
        "HOTKEY": {"keys"},
        "WAIT": set(),
        "FAIL": set(),
        "DONE": set(),
    }

    schemas_by_tool_name = {
        tool["tool_schema"]["function"]["name"]: tool["tool_schema"]["function"] for tool in config["tools"]
    }

    assert set(schemas_by_tool_name) == set(expected_required)
    assert "computer" not in schemas_by_tool_name
    for action_type, required_fields in expected_required.items():
        schema = schemas_by_tool_name[action_type]
        parameters = schema["parameters"]
        assert set(parameters["required"]) == required_fields
        assert parameters["additionalProperties"] is False
        assert "description" in schema
        for property_schema in parameters["properties"].values():
            assert "description" not in property_schema

    for action_type in ["CLICK", "MOUSE_DOWN", "MOUSE_UP"]:
        assert schemas_by_tool_name[action_type]["parameters"]["properties"]["button"]["enum"] == [
            "left",
            "middle",
            "right",
        ]
    assert schemas_by_tool_name["CLICK"]["parameters"]["properties"]["num_clicks"]["minimum"] == 1


def test_mock_server_rejects_click_without_required_server_fields(tmp_path):
    client = TestClient(create_app(tmp_path / "requests.jsonl"))
    client.post("/", json={"session_id": 1, "task_id": "12345", "op": "start"}).raise_for_status()

    response = client.post(
        "/",
        json={
            "session_id": 1,
            "task_id": "12345",
            "op": "action",
            "actions": [{"action_type": "CLICK", "x": 10, "y": 20}],
        },
    )

    assert response.status_code == 422
    assert "button" in response.text
    assert "num_clicks" in response.text


def test_mock_server_accepts_click_with_required_server_fields(tmp_path):
    client = TestClient(create_app(tmp_path / "requests.jsonl"))
    client.post("/", json={"session_id": 1, "task_id": "12345", "op": "start"}).raise_for_status()

    response = client.post(
        "/",
        json={
            "session_id": 1,
            "task_id": "12345",
            "op": "action",
            "actions": [
                {
                    "action_type": "CLICK",
                    "button": "left",
                    "x": 10,
                    "y": 20,
                    "num_clicks": 1,
                }
            ],
        },
    )

    assert response.status_code == 200
