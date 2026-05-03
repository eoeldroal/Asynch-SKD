import asyncio
from types import SimpleNamespace

from PIL import Image

from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.web_tool_agent_loop import WebOsGymToolAgentLoop
from verl.tools.schemas import ToolResponse


class _FakeWebTool:
    name = "computer"

    def __init__(self):
        self.created = []
        self.executed = []
        self.released = []
        self.rewards = []
        self._instance_dict = {}
        self.tool_schema = SimpleNamespace(model_dump=lambda **_: {"function": {"name": "computer"}})

    async def create(self, *, task_id, request_id, include_a11y, **kwargs):
        self.created.append(
            {
                "task_id": task_id,
                "request_id": request_id,
                "include_a11y": include_a11y,
                **kwargs,
            }
        )
        self._instance_dict["instance-1"] = {
            "task_id": task_id,
            "request_id": request_id,
            "include_a11y": include_a11y,
            "reward": None,
        }
        return "instance-1", ToolResponse(
            text="A11Y_TREE:\nroot",
            image=[Image.new("RGB", (2, 2), "blue")],
        )

    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters, kwargs))
        return ToolResponse(text="done"), None, {
            "terminated": True,
            "termination_reason": "model_done",
            "action_count": 1,
        }

    async def execute_action_bundle(self, instance_id, actions, **kwargs):
        self.executed.append((instance_id, {"actions": actions}, kwargs))
        return ToolResponse(text="done"), None, {
            "terminated": True,
            "termination_reason": "model_done",
            "action_count": len(actions),
        }

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0

    async def release(self, instance_id):
        self.released.append(instance_id)
        self._instance_dict.pop(instance_id, None)


class _FakeTerminalObservationTool(_FakeWebTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters, kwargs))
        return ToolResponse(
            text="terminal screenshot should not become a new observation",
            image=[Image.new("RGB", (2, 2), "green")],
        ), None, {
            "terminated": True,
            "termination_reason": "model_done",
            "action_count": 1,
        }


class _FakeNonTerminalObservationTool(_FakeWebTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters, kwargs))
        return ToolResponse(text="non-terminal observation"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": 1,
        }


def _make_loop():
    loop = object.__new__(WebOsGymToolAgentLoop)
    loop.tools = {}
    loop.tool_schemas = []
    loop.response_length = 64
    loop.max_assistant_turns = 4
    loop.max_user_turns = 4
    loop.processor = SimpleNamespace(image_processor=True)
    loop.tokenizer = SimpleNamespace()
    loop.apply_chat_template_calls = []

    async def apply_chat_template(messages, **kwargs):
        loop.apply_chat_template_calls.append((messages, kwargs))
        return list(range(10, 10 + len(str(messages)) % 7 + 2))

    loop.apply_chat_template = apply_chat_template
    return loop


def _agent_data(tool):
    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="agent-request",
        tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345"}}},
    )
    agent_data._active_tools = {"computer": tool}
    agent_data._active_tool_schemas = [tool.tool_schema.model_dump()]
    return agent_data


def _action_agent_data(tool, action_name="CLICK"):
    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="agent-request",
        tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345"}}},
    )
    agent_data._active_tools = {action_name: tool}
    agent_data._active_tool_schemas = [tool.tool_schema.model_dump()]
    return agent_data


def test_web_tool_agent_keeps_tool_agent_layering():
    assert issubclass(WebOsGymToolAgentLoop, ToolAgentLoop)
    assert not issubclass(WebOsGymToolAgentLoop, SkdAgentLoop)


def test_generation_metadata_merge_keeps_zero_param_version_with_web_session_fields():
    tool = _FakeWebTool()
    agent_data = _agent_data(tool)
    agent_data.extra_fields.update(
        {
            "web_osgym_instance_id": "instance-1",
            "web_osgym_task_id": "12345",
            "web_osgym_session_id": 777,
            "web_osgym_include_a11y": False,
        }
    )

    ToolAgentLoop._merge_generation_extra_fields(
        agent_data,
        {
            "global_steps": 0,
            "min_global_steps": 0,
            "max_global_steps": 0,
        },
    )

    assert agent_data.extra_fields["web_osgym_session_id"] == 777
    assert agent_data.extra_fields["global_steps"] == 0
    assert agent_data.extra_fields["min_global_steps"] == 0
    assert agent_data.extra_fields["max_global_steps"] == 0

    ToolAgentLoop._merge_generation_extra_fields(
        agent_data,
        {
            "global_steps": 3,
            "min_global_steps": 3,
            "max_global_steps": 3,
        },
    )

    assert agent_data.extra_fields["global_steps"] == 3
    assert agent_data.extra_fields["min_global_steps"] == 0
    assert agent_data.extra_fields["max_global_steps"] == 3


def test_pending_starts_one_session_and_hides_visual_a11y_from_student_text():
    async def _run():
        loop = _make_loop()
        tool = _FakeWebTool()
        agent_data = _agent_data(tool)

        next_state = await loop._handle_pending_state(agent_data, sampling_params={})

        assert next_state == AgentState.GENERATING
        assert tool.created[0]["task_id"] == "12345"
        assert tool.created[0]["include_a11y"] is False
        assert agent_data.extra_fields["web_osgym_instance_id"] == "instance-1"
        assert agent_data.extra_fields["web_osgym_session_id"] == tool.created[0]["request_id"]
        assert agent_data.image_data and len(agent_data.image_data) == 1

        appended_message = agent_data.messages[-1]
        assert appended_message["role"] == "tool"
        assert appended_message["content"] == [{"type": "image"}]

    asyncio.run(_run())


def test_processing_reuses_session_and_sets_reward_score_on_terminal_action():
    async def _run():
        loop = _make_loop()
        tool = _FakeWebTool()
        agent_data = _agent_data(tool)
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"DONE"}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][0] == "instance-1"
        assert tool.executed[0][1]["actions"][0]["action_type"] == "DONE"
        assert tool.rewards == [("instance-1", {"termination_reason": "model_done"})]
        assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0
        assert agent_data.response_mask == []

    asyncio.run(_run())


def test_terminal_action_response_observation_is_not_committed():
    async def _run():
        loop = _make_loop()
        tool = _FakeTerminalObservationTool()
        agent_data = _agent_data(tool)
        agent_data.prompt_ids = [100, 101]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"DONE"}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert agent_data.prompt_ids == [100, 101]
        assert agent_data.messages == [{"role": "user", "content": "task"}]
        assert agent_data.image_data == []
        assert agent_data.response_mask == []
        assert tool.rewards == [("instance-1", {"termination_reason": "model_done"})]

    asyncio.run(_run())


def test_processing_accepts_action_named_tool_call():
    async def _run():
        loop = _make_loop()
        tool = _FakeWebTool()
        agent_data = _action_agent_data(tool, action_name="CLICK")
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments='{"x":1,"y":2}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][0] == "instance-1"
        assert tool.executed[0][1] == {"x": 1, "y": 2}
        assert tool.rewards == [("instance-1", {"termination_reason": "model_done"})]

    asyncio.run(_run())


def test_processing_bundles_multiple_action_named_tool_calls():
    async def _run():
        loop = _make_loop()
        tool = _FakeWebTool()
        agent_data = _action_agent_data(tool, action_name="CLICK")
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments='{"x":1,"y":2}'),
            FunctionCall(name="CLICK", arguments='{"x":3,"y":4}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][0] == "instance-1"
        assert tool.executed[0][1] == {
            "actions": [
                {"action_type": "CLICK", "x": 1, "y": 2},
                {"action_type": "CLICK", "x": 3, "y": 4},
            ]
        }
        assert agent_data.metrics["web_osgym/action_count"] == 2
        assert tool.rewards == [("instance-1", {"termination_reason": "model_done"})]

    asyncio.run(_run())


def test_tool_response_budget_exhaustion_fetches_reward_without_committing_observation():
    async def _run():
        loop = _make_loop()
        loop.response_length = 2
        tool = _FakeNonTerminalObservationTool()
        agent_data = _agent_data(tool)
        agent_data.response_mask = [1]
        agent_data.prompt_ids = [100]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"WAIT"}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert agent_data.prompt_ids == [100]
        assert agent_data.response_mask == [1]
        assert agent_data.extra_fields["web_osgym_termination_reason"] == "tool_response_budget_exhausted"
        assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0

    asyncio.run(_run())
