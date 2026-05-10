import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

from PIL import Image

from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.web_tool_agent_loop import WebOsGymToolAgentLoop, _WebOsGymGenerationInput
from verl.tools.schemas import ToolResponse
from verl.workers.rollout.replica import TokenOutput


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


class _FakePostprocessingWebTool(_FakeWebTool):
    def postprocess_tool_arguments(self, parameters):
        normalized = deepcopy(parameters)
        actions = normalized.get("actions") or []
        for action in actions:
            if action.get("action_type") == "PRESS" and action.get("key") == "enter":
                action["key"] = "Enter"
            if action.get("action_type") == "PRESS" and action.get("key") == "esc":
                action["key"] = "Escape"
        return normalized


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


class _FakeNonTerminalImageObservationTool(_FakeWebTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters, kwargs))
        return ToolResponse(
            text="A11Y_TREE:\nbutton",
            image=[Image.new("RGB", (3, 2), "red")],
        ), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": 1,
        }


class _FakeNonTerminalNormalizedActionTool(_FakeWebTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters, kwargs))
        return ToolResponse(text="normalized observation"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": 1,
            "web_osgym_actions": [{"action_type": "CLICK", "x": 1014, "y": 629, "button": "left", "num_clicks": 1}],
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


def test_record_web_osgym_step_initial_observation():
    loop = _make_loop()
    agent_data = _agent_data(_FakeWebTool())

    loop._record_web_osgym_step(
        agent_data,
        phase="initial",
        image_start=0,
        image_end=1,
        text_len=0,
        terminal=False,
        termination_reason=None,
        actions=None,
    )

    assert agent_data.extra_fields["web_osgym_steps"] == [
        {
            "step_idx": 1,
            "assistant_turn": 0,
            "user_turn": 0,
            "phase": "initial",
            "text": "",
            "text_len": 0,
            "action_names": [],
            "actions": [],
            "image_start": 0,
            "image_end": 1,
            "terminal": False,
            "termination_reason": None,
        }
    ]
    assert agent_data.extra_fields["mini_step_image_spans"] == [
        {
            "step_idx": 1,
            "image_start": 0,
            "image_end": 1,
            "terminal": False,
        }
    ]


def test_record_web_osgym_step_action_observation_preserves_actions():
    loop = _make_loop()
    agent_data = _agent_data(_FakeWebTool())
    agent_data.assistant_turns = 2
    agent_data.user_turns = 1
    actions = [{"action_type": "CLICK", "x": 12, "y": 34}]

    loop._record_web_osgym_step(
        agent_data,
        phase="tool_observation",
        image_start=1,
        image_end=2,
        text_len=7,
        terminal=False,
        termination_reason=None,
        actions=actions,
    )

    recorded_step = agent_data.extra_fields["web_osgym_steps"][0]
    assert recorded_step["assistant_turn"] == 2
    assert recorded_step["user_turn"] == 1
    assert recorded_step["phase"] == "tool_observation"
    assert recorded_step["text"] == ""
    assert recorded_step["text_len"] == 7
    assert recorded_step["action_names"] == ["CLICK"]
    assert recorded_step["actions"] == actions
    assert recorded_step["image_start"] == 1
    assert recorded_step["image_end"] == 2
    assert recorded_step["terminal"] is False
    assert recorded_step["termination_reason"] is None
    assert agent_data.extra_fields["mini_step_image_spans"] == [
        {
            "step_idx": recorded_step["step_idx"],
            "image_start": 1,
            "image_end": 2,
            "terminal": False,
        }
    ]


def test_windowed_generation_uses_harness_prompt_window():
    class _FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return TokenOutput(
                token_ids=[101, 102],
                log_probs=[-0.1, -0.2],
                extra_fields={"global_steps": 0},
            )

    class _FakeToolParser:
        def __init__(self):
            self.calls = []

        async def extract_tool_calls(self, response_ids, tools):
            self.calls.append((response_ids, tools))
            return "", [FunctionCall(name="computer", arguments="{}")]

    async def _run():
        loop = _make_loop()
        loop.rollout_config = SimpleNamespace(
            name="vllm",
            custom={},
            multi_turn=SimpleNamespace(
                web_osgym_window_enable=True,
                web_osgym_window_history_n=1,
                web_osgym_window_max_images_per_sample=6,
            )
        )
        loop.server_manager = _FakeServerManager()
        loop.tool_parser = _FakeToolParser()
        loop.tokenizer.decode = lambda ids, skip_special_tokens=False: "A2"

        agent_data = _agent_data(_FakeWebTool())
        agent_data._web_osgym_base_messages = [
            {"role": "system", "content": "Use the computer carefully."},
            {"role": "user", "content": "Open the settings page"},
        ]
        agent_data.prompt_ids = list(range(100))
        agent_data.image_data = ["old-image", "current-image"]
        agent_data.extra_fields["web_osgym_steps"] = [
            {
                "step_idx": 1,
                "phase": "tool_observation",
                "text": "",
                "actions": [{"action_type": "CLICK", "x": 12, "y": 34}],
                "image_start": 0,
                "image_end": 1,
            },
            {
                "step_idx": 2,
                "phase": "tool_observation",
                "text": "",
                "actions": [],
                "image_start": 1,
                "image_end": 2,
            },
        ]
        agent_data.extra_fields["web_osgym_assistant_turns"] = [
            {
                "assistant_turn": 1,
                "observation_step_idx": 1,
                "response_text": "A1",
                "actions": [{"action_type": "CLICK", "x": 12, "y": 34}],
            }
        ]

        state = await loop._handle_generating_state(agent_data, {})

        assert state == AgentState.PROCESSING_TOOLS
        generate_call = loop.server_manager.calls[0]
        assert generate_call["prompt_ids"] != list(range(100))
        assert generate_call["image_data"] == ["old-image", "current-image"]
        assert generate_call["video_data"] is None
        prompt_messages, prompt_kwargs = loop.apply_chat_template_calls[-1]
        assert prompt_kwargs["images"] == ["old-image", "current-image"]
        assert prompt_messages[0] == {"role": "system", "content": "Use the computer carefully."}
        prompt_text = prompt_messages[1]["content"][-1]["text"]
        assert "Instruction: Open the settings page" in prompt_text
        assert "Previous actions:\nNone" in prompt_text
        assert prompt_messages[2] == {"role": "assistant", "content": "A1"}
        assert prompt_messages[3] == {"role": "user", "content": [{"type": "image"}]}
        assert agent_data.prompt_ids[-2:] == [101, 102]
        assert agent_data.response_mask[-2:] == [1, 1]
        assert agent_data.metrics["web_osgym/window_active"] == 1
        assert agent_data.metrics["web_osgym/window_step_count"] == 2
        assert agent_data.metrics["web_osgym/window_image_count"] == 2
        assert agent_data.metrics["web_osgym/window_old_summary_turn_count"] == 0
        assert agent_data.metrics["web_osgym/window_recent_observation_step_count"] == 2
        assert agent_data.metrics["web_osgym/window_recent_assistant_turn_count"] == 1
        assert agent_data.metrics["web_osgym/window_text_only_recent_step_count"] == 0
        window = agent_data.extra_fields["web_osgym_generation_windows"][0]
        assert window["prompt_image_indices"] == [0, 1]
        assert window["selected_step_indices"] == [1, 2]
        assert window["old_summary_turn_indices"] == []
        assert window["recent_observation_step_indices"] == [1, 2]
        assert window["recent_assistant_turn_indices"] == [1]
        assert window["text_only_recent_step_count"] == 0
        assert agent_data.extra_fields["web_osgym_assistant_turns"][-1]["observation_step_idx"] == 2
        assert agent_data.extra_fields["web_osgym_assistant_turns"][-1]["response_text"] == "A2"

    asyncio.run(_run())


def test_generation_uses_compact_server_prompt_ids_for_image_bearing_sglang_requests():
    class _FakeServerManager:
        def __init__(self):
            self.calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            return TokenOutput(
                token_ids=[201],
                log_probs=[-0.1],
                num_preempted=0,
                stop_reason=None,
                extra_fields={"global_steps": 0},
            )

    class _FakeToolParser:
        async def extract_tool_calls(self, response_ids, tools):
            del response_ids, tools
            return "", []

    async def _run():
        loop = _make_loop()
        loop.tool_parser_name = "noop"
        loop.rollout_config = SimpleNamespace(
            name="sglang",
            skip_tokenizer_init=False,
            custom={},
            multi_turn=SimpleNamespace(
                web_osgym_window_enable=True,
                web_osgym_window_history_n=1,
                web_osgym_window_max_images_per_sample=6,
            ),
        )
        loop.server_manager = _FakeServerManager()
        loop.tool_parser = _FakeToolParser()

        expanded_prompt_ids = [301, 302, 303, 304]
        compact_prompt_ids = [41, 42]

        async def _fake_build_generation_inputs(agent_data):
            del agent_data
            return _WebOsGymGenerationInput(
                prompt_ids=expanded_prompt_ids,
                server_prompt_ids=compact_prompt_ids,
                images=["image-1"],
                videos=None,
                window_used=True,
                image_indices=[0],
                selected_step_indices=[1],
                old_summary_turn_indices=[],
                recent_observation_step_indices=[1],
                recent_assistant_turn_indices=[],
                text_only_recent_step_count=0,
            )

        loop._build_web_osgym_generation_inputs = _fake_build_generation_inputs

        agent_data = _agent_data(_FakeWebTool())
        agent_data.prompt_ids = [100]
        agent_data.extra_fields["web_osgym_reward_score"] = 0.0

        state = await loop._handle_generating_state(agent_data, {})

        assert state == AgentState.TERMINATED
        generate_call = loop.server_manager.calls[0]
        assert generate_call["prompt_ids"] == compact_prompt_ids
        assert generate_call["image_data"] == ["image-1"]
        assert agent_data.extra_fields["web_osgym_generation_windows"][0]["prompt_ids"] == expanded_prompt_ids
        assert agent_data.extra_fields["web_osgym_generation_windows"][0]["window_used"] is True
        assert agent_data.prompt_ids == [100, 201]
        assert agent_data.response_mask == [1]

    asyncio.run(_run())


def test_non_window_generation_builds_compact_server_prompt_ids_for_image_bearing_sglang_requests():
    async def _run():
        loop = _make_loop()
        loop.rollout_config = SimpleNamespace(
            name="sglang",
            skip_tokenizer_init=False,
            custom={},
            multi_turn=SimpleNamespace(
                web_osgym_window_enable=False,
                web_osgym_window_history_n=1,
                web_osgym_window_max_images_per_sample=6,
            ),
        )

        compact_prompt_ids = [41, 42]

        async def _fake_build_server_prompt_ids(*, messages, images, tools):
            del messages, tools
            assert images == ["image-1"]
            return compact_prompt_ids

        loop._build_server_prompt_ids = _fake_build_server_prompt_ids

        agent_data = _agent_data(_FakeWebTool())
        agent_data.messages = [{"role": "user", "content": "task"}]
        agent_data.prompt_ids = [301, 302, 303, 304]
        agent_data.image_data = ["image-1"]

        generation_inputs = await loop._build_web_osgym_generation_inputs(agent_data)

        assert generation_inputs.prompt_ids == [301, 302, 303, 304]
        assert generation_inputs.server_prompt_ids == compact_prompt_ids
        assert generation_inputs.images == ["image-1"]
        assert generation_inputs.window_used is False

    asyncio.run(_run())


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
        assert agent_data.extra_fields["web_osgym_steps"] == [
            {
                "step_idx": 1,
                "assistant_turn": 0,
                "user_turn": 0,
                "phase": "initial",
                "text": "",
                "text_len": 0,
                "action_names": [],
                "actions": [],
                "image_start": 0,
                "image_end": 1,
                "terminal": False,
                "termination_reason": None,
            }
        ]
        assert agent_data.extra_fields["mini_step_image_spans"] == [
            {
                "step_idx": 1,
                "image_start": 0,
                "image_end": 1,
                "terminal": False,
            }
        ]

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
        agent_data.extra_fields["web_osgym_steps"] = []
        agent_data.extra_fields["mini_step_image_spans"] = []
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
        assert agent_data.extra_fields["web_osgym_steps"] == []
        assert agent_data.extra_fields["mini_step_image_spans"] == []

    asyncio.run(_run())


def test_terminal_action_named_response_observation_is_not_committed_for_fail():
    async def _run():
        loop = _make_loop()
        tool = _FakeTerminalObservationTool()
        agent_data = _action_agent_data(tool, action_name="FAIL")
        agent_data.prompt_ids = [100, 101]
        agent_data.extra_fields["web_osgym_steps"] = []
        agent_data.extra_fields["mini_step_image_spans"] = []
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
            FunctionCall(name="FAIL", arguments='{"reason":"submitted failure"}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][0] == "instance-1"
        assert tool.executed[0][1] == {"reason": "submitted failure"}
        assert agent_data.prompt_ids == [100, 101]
        assert agent_data.messages == [{"role": "user", "content": "task"}]
        assert agent_data.image_data == []
        assert agent_data.response_mask == []
        assert tool.rewards == [("instance-1", {"termination_reason": "model_done"})]
        assert agent_data.extra_fields["web_osgym_steps"] == []
        assert agent_data.extra_fields["mini_step_image_spans"] == []

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


def test_processing_postprocesses_single_bundled_computer_tool_call():
    async def _run():
        loop = _make_loop()
        tool = _FakePostprocessingWebTool()
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
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"PRESS","key":"enter"}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][1] == {"actions": [{"action_type": "PRESS", "key": "Enter"}]}

    asyncio.run(_run())


def test_processing_postprocesses_bundled_action_named_tool_calls():
    async def _run():
        loop = _make_loop()
        tool = _FakePostprocessingWebTool()
        agent_data = _action_agent_data(tool, action_name="PRESS")
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
            FunctionCall(name="PRESS", arguments='{"key":"enter"}'),
            FunctionCall(name="PRESS", arguments='{"key":"esc"}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.TERMINATED
        assert tool.executed[0][1] == {
            "actions": [
                {"action_type": "PRESS", "key": "Enter"},
                {"action_type": "PRESS", "key": "Escape"},
            ]
        }

    asyncio.run(_run())


def test_processing_respects_max_parallel_calls_for_web_osgym_bundling():
    async def _run():
        loop = _make_loop()
        loop.max_parallel_calls = 1
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
        assert tool.executed[0][1] == {"x": 1, "y": 2}
        assert agent_data.metrics["web_osgym/action_count"] == 1

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


def test_processing_records_non_terminal_image_observation_step():
    async def _run():
        loop = _make_loop()
        tool = _FakeNonTerminalImageObservationTool()
        agent_data = _agent_data(tool)

        next_state = await loop._handle_pending_state(agent_data, sampling_params={})
        assert next_state == AgentState.GENERATING

        agent_data.assistant_turns = 1
        agent_data.prompt_ids = [100]
        agent_data.response_mask = []
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"CLICK","x":1,"y":2}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.GENERATING
        assert len(agent_data.image_data) == 2
        assert agent_data.extra_fields["web_osgym_steps"] == [
            {
                "step_idx": 1,
                "assistant_turn": 0,
                "user_turn": 0,
                "phase": "initial",
                "text": "",
                "text_len": 0,
                "action_names": [],
                "actions": [],
                "image_start": 0,
                "image_end": 1,
                "terminal": False,
                "termination_reason": None,
            },
            {
                "step_idx": 2,
                "assistant_turn": 1,
                "user_turn": 0,
                "phase": "tool_observation",
                "text": "",
                "text_len": 0,
                "action_names": ["CLICK"],
                "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
                "image_start": 1,
                "image_end": 2,
                "terminal": False,
                "termination_reason": None,
            },
        ]
        assert agent_data.extra_fields["mini_step_image_spans"] == [
            {
                "step_idx": 1,
                "image_start": 0,
                "image_end": 1,
                "terminal": False,
            },
            {
                "step_idx": 2,
                "image_start": 1,
                "image_end": 2,
                "terminal": False,
            },
        ]

    asyncio.run(_run())


def test_processing_updates_previous_action_history_with_postprocessed_actions():
    async def _run():
        loop = _make_loop()
        tool = _FakeNonTerminalNormalizedActionTool()
        agent_data = _agent_data(tool)
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
                "web_osgym_steps": [
                    {
                        "step_idx": 1,
                        "assistant_turn": 0,
                        "user_turn": 0,
                        "phase": "initial",
                        "text": "",
                        "text_len": 0,
                        "action_names": [],
                        "actions": [],
                        "image_start": 0,
                        "image_end": 1,
                        "terminal": False,
                        "termination_reason": None,
                    }
                ],
                "web_osgym_assistant_turns": [
                    {
                        "assistant_turn": 1,
                        "observation_step_idx": 1,
                        "response_start": 0,
                        "response_end": 3,
                        "response_text": "A1",
                        "actions": [{"action_type": "CLICK", "x": "[528, 584]"}],
                    }
                ],
            }
        )
        agent_data.extra_fields["web_osgym_assistant_turns"] = [
            {
                "assistant_turn": 1,
                "user_turn": 0,
                "observation_step_idx": 1,
                "response_start": 0,
                "response_end": 3,
                "response_text": "Open the button panel",
                "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
            }
        ]
        agent_data.assistant_turns = 1
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.prompt_ids = [100]
        agent_data.response_mask = []
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"CLICK","x":"[528, 584]"}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.GENERATING
        assert agent_data.extra_fields["web_osgym_assistant_turns"][-1]["actions"] == [
            {"action_type": "CLICK", "x": 1014, "y": 629, "button": "left", "num_clicks": 1}
        ]

    asyncio.run(_run())


def test_finalize_output_records_rollout_backprop_unit_trace():
    loop = _make_loop()
    agent_data = _agent_data(_FakeWebTool())
    agent_data.prompt_ids = [1, 2, 3, 4]
    agent_data.response_mask = [1, 0]
    agent_data.response_logprobs = [0.1, 0.0]
    agent_data.image_data = [Image.new("RGB", (2, 2), "blue")]
    agent_data.extra_fields["web_osgym_steps"] = [
        {
            "step_idx": 1,
            "assistant_turn": 0,
            "user_turn": 0,
            "phase": "initial",
            "text": "",
            "text_len": 0,
            "action_names": [],
            "actions": [],
            "image_start": 0,
            "image_end": 1,
            "terminal": False,
            "termination_reason": None,
        }
    ]
    agent_data.extra_fields["mini_step_image_spans"] = [
        {"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False}
    ]

    output = loop._finalize_web_agent_output(agent_data)

    assert output.extra_fields["web_osgym_unit_trace"] == {
        "rollout_context": "full_accumulated_prompt",
        "backprop_context": "full_agent_loop_output",
        "harness_prompt_window": "metadata_available_not_active",
        "window_history_n": 5,
        "window_max_images_per_sample": 6,
        "window_fallback_count": 0,
        "generation_window_count": 0,
        "step_count": 1,
        "image_span_count": 1,
        "window_old_summary_turn_count": 0,
        "window_recent_observation_step_count": 0,
        "window_recent_assistant_turn_count": 0,
        "window_text_only_recent_step_count": 0,
        "window_prompt_image_count": 0,
    }
    assert agent_data.metrics["web_osgym/step_count"] == 1
    assert agent_data.metrics["web_osgym/image_span_count"] == 1


def test_web_osgym_tool_trace_dumps_tool_call_result_and_image(monkeypatch, tmp_path):
    class _FakeImageObservationTool(_FakeWebTool):
        async def execute(self, instance_id, parameters, **kwargs):
            self.executed.append((instance_id, parameters, kwargs))
            return ToolResponse(
                text="A11Y_TREE:\nbutton",
                image=[Image.new("RGB", (3, 2), "red")],
            ), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 1,
            }

    async def _run():
        monkeypatch.setenv("WEB_OSGYM_TOOL_TRACE_DIR", str(tmp_path))
        loop = _make_loop()
        tool = _FakeImageObservationTool()
        agent_data = _agent_data(tool)
        agent_data.prompt_ids = [100]
        agent_data.extra_fields["web_osgym_sample_uid"] = "uid-123"
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 777,
                "web_osgym_include_a11y": False,
                "global_steps": 0,
                "web_osgym_generation_windows": [
                    {
                        "prompt_ids": [11, 12],
                        "prompt_image_indices": [4, 5],
                        "old_summary_turn_indices": [1],
                        "recent_observation_step_indices": [2, 3],
                        "recent_assistant_turn_indices": [2],
                        "text_only_recent_step_count": 1,
                    }
                ],
            }
        )
        agent_data.extra_fields["web_osgym_assistant_turns"] = [
            {
                "assistant_turn": 1,
                "user_turn": 0,
                "observation_step_idx": 1,
                "response_start": 0,
                "response_end": 3,
                "response_text": "Open the button panel",
                "actions": [{"action_type": "CLICK", "x": 1, "y": 2}],
            }
        ]
        agent_data.assistant_turns = 1
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 777,
            "include_a11y": False,
            "reward": None,
        }
        agent_data.tool_calls = [
            FunctionCall(name="computer", arguments='{"actions":[{"action_type":"CLICK","x":1,"y":2}]}'),
        ]

        next_state = await loop._handle_processing_tools_state(agent_data)

        assert next_state == AgentState.GENERATING
        session_dir = tmp_path / "12345___uid-123___global_step_0___777"
        trajectory_path = session_dir / "trajectory.jsonl"
        assert trajectory_path.exists()
        events = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
        assert len(events) == 1
        event = events[0]
        assert event["session_id"] == 777
        assert event["task_id"] == "12345"
        assert event["sample_uid"] == "uid-123"
        assert event["model_output_text"] == "Open the button panel"
        assert event["tool_calls_raw"] == ['{"actions":[{"action_type":"CLICK","x":1,"y":2}]}']
        assert event["tool_calls_parsed"] == [{"actions": [{"action_type": "CLICK", "x": 1, "y": 2}]}]
        assert event["actions"] == [{"action_type": "CLICK", "x": 1, "y": 2}]
        assert event["result"]["terminated"] is False
        assert event["prompt_window"] == {
            "prompt_image_indices": [4, 5],
            "old_summary_turn_indices": [1],
            "recent_observation_step_indices": [2, 3],
            "recent_assistant_turn_indices": [2],
            "text_only_recent_step_count": 1,
        }
        assert event["observation_text"] == "A11Y_TREE:\nbutton"
        assert event["images"][0]["width"] == 3
        assert event["images"][0]["height"] == 2
        image_path = session_dir / event["image_paths"][0]
        assert image_path.exists()

    asyncio.run(_run())


def test_web_osgym_summary_writes_session_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("WEB_OSGYM_TOOL_TRACE_DIR", str(tmp_path))
    loop = _make_loop()
    agent_data = _agent_data(_FakeWebTool())
    agent_data.request_id = "req-1"
    agent_data.assistant_turns = 3
    agent_data.user_turns = 2
    agent_data.extra_fields.update(
        {
            "web_osgym_task_id": "prozilla_explorer_11",
            "web_osgym_session_id": 987,
            "web_osgym_sample_uid": "uid-xyz",
            "global_steps": 3,
            "web_osgym_reward_score": 1.0,
            "web_osgym_termination_reason": "model_done",
            "web_osgym_trajectory_counts": {"invalid_action_count": 2, "parse_error_count": 1, "event_count": 4},
        }
    )

    loop._write_web_osgym_summary(agent_data)

    summary_path = tmp_path / "prozilla_explorer_11___uid-xyz___global_step_3___987" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["task_id"] == "prozilla_explorer_11"
    assert summary["sample_uid"] == "uid-xyz"
    assert summary["global_step"] == 3
    assert summary["session_id"] == 987
    assert summary["reward_score"] == 1.0
    assert summary["termination_reason"] == "model_done"
    assert summary["invalid_action_count"] == 2
    assert summary["parse_error_count"] == 1
    assert summary["event_count"] == 4
    assert summary["completed"] is True
    assert summary["has_reward"] is True


def test_decode_response_text_preserves_special_tokens():
    loop = _make_loop()
    captured = {}

    def _decode(ids, skip_special_tokens=False):
        captured["skip_special_tokens"] = skip_special_tokens
        return "readable text"

    loop.tokenizer.decode = _decode

    result = loop._decode_response_text([1, 2, 3])

    assert result == "readable text"
    assert captured["skip_special_tokens"] is False
