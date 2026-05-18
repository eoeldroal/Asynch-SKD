from pathlib import Path

import yaml


TOOL_CONFIG_PATH = Path("/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml")


def _load_tools():
    data = yaml.safe_load(TOOL_CONFIG_PATH.read_text())
    return data["tools"]


def _tool_by_name(name: str):
    for tool in _load_tools():
        function = tool["tool_schema"]["function"]
        if function["name"] == name:
            return function
    raise KeyError(name)


def test_action_named_tool_config_uses_coordinate_pair_for_pointer_actions():
    for tool_name, required in {
        "MOVE_TO": ["coordinate"],
        "DRAG_TO": ["coordinate"],
        "CLICK": [],
        "DOUBLE_CLICK": [],
        "RIGHT_CLICK": [],
    }.items():
        function = _tool_by_name(tool_name)
        properties = function["parameters"]["properties"]

        assert "coordinate" in properties
        assert "x" not in properties
        assert "y" not in properties
        coordinate = properties["coordinate"]
        assert coordinate["type"] == "array"
        assert coordinate["minItems"] == 2
        assert coordinate["maxItems"] == 2
        assert coordinate["items"]["type"] == "integer"
        assert coordinate["items"]["minimum"] == 0
        assert coordinate["items"]["maximum"] == 999
        assert "coordinate" in function["description"]
        assert "single integer field" not in function["description"]
        assert function["parameters"]["required"] == required


def test_action_named_tool_config_scroll_requires_only_dy():
    function = _tool_by_name("SCROLL")
    parameters = function["parameters"]
    properties = parameters["properties"]

    assert parameters["required"] == ["dy"]
    assert "dx" in properties
    assert "dy" in properties
    assert properties["dx"]["type"] == "integer"
    assert properties["dy"]["type"] == "integer"
    assert "Negative dy scrolls down" in function["description"]
    assert "positive dy scrolls up" in function["description"]


def test_action_named_tool_config_wait_accepts_optional_duration():
    function = _tool_by_name("WAIT")
    parameters = function["parameters"]
    properties = parameters["properties"]

    assert parameters["required"] == []
    assert "duration" in properties
    assert properties["duration"]["type"] == "number"
    assert properties["duration"]["minimum"] == 0.1
    assert properties["duration"]["maximum"] == 10.0


def test_action_named_tool_config_done_fail_have_empty_objects():
    for tool_name in ("DONE", "FAIL"):
        function = _tool_by_name(tool_name)
        parameters = function["parameters"]
        assert parameters["properties"] == {}
        assert parameters["required"] == []
