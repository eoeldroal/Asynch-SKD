import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from WebOSWorld.mock_server.web_osgym_mock_server import run_mock_server_in_thread
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.web_tool_agent_loop import WebOsGymToolAgentLoop
from verl.tools.utils.tool_registry import initialize_tools_from_config


def _write_tool_config(path: Path, base_url: str) -> None:
    path.write_text(
        f"""
tools:
  - class_name: "verl.tools.web_osgym_tool.WebOsGymTool"
    config:
      type: native
      base_url: "{base_url}"
      timeout: 10.0
      include_a11y: false
    tool_schema:
      type: "function"
      function:
        name: "computer"
        description: "Apply one or more low-level computer actions."
        parameters:
          type: "object"
          properties:
            actions:
              type: "array"
              items:
                type: "object"
                properties:
                  action_type:
                    type: "string"
                  x:
                    type: "integer"
                  y:
                    type: "integer"
                  text:
                    type: "string"
                required: ["action_type"]
          required: ["actions"]
""",
        encoding="utf-8",
    )


def _make_loop(tool):
    loop = object.__new__(WebOsGymToolAgentLoop)
    loop.tools = {"computer": tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.response_length = 128
    loop.max_assistant_turns = 4
    loop.max_user_turns = 4
    loop.processor = SimpleNamespace(image_processor=True)
    loop.tokenizer = SimpleNamespace()

    async def apply_chat_template(messages, **kwargs):
        return list(range(1, len(str(messages)) % 11 + 3))

    loop.apply_chat_template = apply_chat_template
    return loop


def _agent_data(tool):
    agent_data = AgentData(
        messages=[{"role": "user", "content": "complete task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="agent-request",
        tools_kwargs={"computer": {"create_kwargs": {"task_id": "12345"}}},
    )
    agent_data._active_tools = {"computer": tool}
    agent_data._active_tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    return agent_data


@pytest.mark.asyncio
async def test_web_tool_agent_uses_one_session_for_start_action_and_reward(tmp_path):
    log_path = tmp_path / "mock_web_osgym_requests.jsonl"
    server = run_mock_server_in_thread(host="127.0.0.1", port=0, log_path=log_path)
    loop = None
    agent_data = None
    try:
        config_path = tmp_path / "web_osgym_tool.yaml"
        _write_tool_config(config_path, server.base_url)
        tool = initialize_tools_from_config(config_path)[0]
        loop = _make_loop(tool)
        agent_data = _agent_data(tool)

        pending_state = await loop._handle_pending_state(agent_data, sampling_params={})
        assert pending_state == AgentState.GENERATING
        session_id = agent_data.extra_fields["web_osgym_session_id"]
        assert isinstance(session_id, int)

        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"CLICK","x":1,"y":2}]}'),
        ]
        action_state = await loop._handle_processing_tools_state(agent_data)
        assert action_state == AgentState.GENERATING

        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"DONE"}]}'),
        ]
        terminal_state = await loop._handle_processing_tools_state(agent_data)
        assert terminal_state == AgentState.TERMINATED
        assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0

        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        ops = [event["op"] for event in events]
        assert ops == ["start", "action", "action", "reward"]
        assert {event["session_id"] for event in events} == {session_id}
        assert {event["task_id"] for event in events} == {"12345"}
        assert events[1]["actions"][0]["action_type"] == "CLICK"
        assert events[2]["actions"][0]["action_type"] == "DONE"
    finally:
        if loop is not None and agent_data is not None:
            await loop._release_web_osgym_session(agent_data)
        server.shutdown()
