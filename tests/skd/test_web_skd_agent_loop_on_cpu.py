import unittest
from copy import deepcopy
from unittest.mock import patch

import verl.experimental.agent_loop.web_skd_agent_loop as web_skd_agent_loop_module
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.web_skd_agent_loop import WebSkdAgentLoop
from verl.tools.base_tool import ToolResponse


class _FakeTool:
    name = "computer"
    tool_schema = None

    def __init__(self):
        self.created = []
        self.executed = []
        self.rewards = []
        self._instance_dict = {}

    async def create(self, **kwargs):
        self.created.append(kwargs)
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
        }
        return "instance-1", ToolResponse(text="A11Y_TREE:\nroot", image=["start-image"])

    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="At failed_action_index 0, action Failed. Reason: target field was not focused"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": len(parameters["actions"]),
        }

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0

    def restore_instance(self, instance_id, **kwargs):
        self._instance_dict[instance_id] = dict(kwargs)


class _ImageFakeTool(_FakeTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="A11Y_TREE:\nroot", image=["image-1"]), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": len(parameters["actions"]),
        }


class _TerminalImageFakeTool(_FakeTool):
    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="A11Y_TREE:\nterminal", image=["terminal-image"]), None, {
            "terminated": True,
            "termination_reason": "model_done",
            "action_count": len(parameters["actions"]),
        }


class _ActionFakeTool(_FakeTool):
    name = "CLICK"

    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="At failed_action_index 0, action Failed. Reason: target field was not focused"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": 1,
        }

    async def execute_action_bundle(self, instance_id, actions, **kwargs):
        self.executed.append((instance_id, {"actions": actions}))
        return ToolResponse(text="At failed_action_index 0, action Failed. Reason: target field was not focused"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": len(actions),
        }


def _build_loop():
    loop = WebSkdAgentLoop.__new__(WebSkdAgentLoop)
    loop.tools = {"computer": _FakeTool()}
    loop.tool_schemas = []
    loop.teacher_key = "data_source"
    loop.response_length = 64
    loop.loss_top_k = 4
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 4096
    loop.tool_response_truncate_side = "left"
    loop.teacher_system_prompt = None
    loop.teacher_server_manager = None
    loop.prompt_length = 64
    loop.tool_parser_name = "qwen3_coder"
    loop.processor = None
    loop.apply_chat_template_kwargs = {}

    async def _fake_apply_server_chat_template(messages, **kwargs):
        return [21, 22]

    loop._apply_server_chat_template = _fake_apply_server_chat_template
    return loop


class TestWebSkdAgentLoop(unittest.IsolatedAsyncioTestCase):
    def test_web_skd_agent_is_still_skd(self):
        self.assertTrue(issubclass(WebSkdAgentLoop, SkdAgentLoop))

    def test_resolve_request_prompt_inputs_from_agent_state_uses_current_state(self):
        loop = _build_loop()
        loop.teacher_system_prompt = "teacher-system"

        agent_data = AgentData(
            messages=[{"role": "user", "content": "student-task"}],
            image_data=["image-1"],
            video_data=[],
            metrics={},
            request_id="req-resolve",
            tools_kwargs={},
        )
        agent_data.extra_fields["web_osgym_teacher_messages"] = [{"role": "user", "content": "teacher-task"}]
        agent_data.extra_fields["teacher_prompt_ids"] = [101, 102, 103]

        student_messages, teacher_messages, teacher_prompt_ids, image_data = (
            loop._resolve_request_prompt_inputs_from_agent_state(agent_data)
        )

        self.assertEqual(student_messages, [{"role": "user", "content": "student-task"}])
        self.assertEqual(teacher_messages[0], {"role": "system", "content": "teacher-system"})
        self.assertEqual(teacher_messages[1], {"role": "user", "content": "teacher-task"})
        self.assertEqual(teacher_prompt_ids, [101, 102, 103])
        self.assertEqual(image_data, ["image-1"])

    def test_tool_message_has_one_marker_per_image(self):
        loop = _build_loop()

        message = loop._build_tool_message("obs", ["image-1", "image-2"])

        self.assertEqual(
            message,
            {
                "role": "tool",
                "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "obs"}],
            },
        )

    async def test_pending_requests_a11y_but_only_teacher_prompt_gets_a11y(self):
        loop = _build_loop()

        async def _fake_apply_chat_template(messages, **kwargs):
            if any("A11Y_TREE" in str(m.get("content")) for m in messages):
                return [7, 8, 9]
            return [1, 2, 3]

        loop.apply_chat_template = _fake_apply_chat_template
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345", "request_id": 101}}},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []

        state = await WebSkdAgentLoop._handle_pending_state(loop, agent_data, {})

        self.assertEqual(state, AgentState.GENERATING)
        self.assertTrue(loop.tools["computer"].created[0]["include_a11y"])
        self.assertNotIn("A11Y_TREE", str(agent_data.messages))
        self.assertTrue(agent_data.extra_fields["web_osgym_teacher_observation_text"].startswith("A11Y_TREE"))
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], [7, 8, 9])
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 1)

    async def test_pending_discards_whole_start_bundle_when_tokenization_fails(self):
        loop = _build_loop()

        async def _failing_apply_chat_template(messages, **kwargs):
            raise RuntimeError("tokenization failed")

        loop.apply_chat_template = _failing_apply_chat_template
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345", "request_id": 101}}},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []

        with self.assertRaisesRegex(RuntimeError, "tokenization failed"):
            await WebSkdAgentLoop._handle_pending_state(loop, agent_data, {})

        self.assertEqual(agent_data.messages, [{"role": "user", "content": "task"}])
        self.assertEqual(agent_data.image_data, [])
        self.assertEqual(agent_data.prompt_ids, [])
        self.assertNotIn("server_prompt_ids", agent_data.extra_fields)
        self.assertNotIn("teacher_prompt_ids", agent_data.extra_fields)
        self.assertNotIn("teacher_server_prompt_ids", agent_data.extra_fields)

    async def test_processing_tools_keeps_error_text_for_student_but_not_a11y(self):
        loop = _build_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2},{"action_type":"CLICK","x":3,"y":4}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 0)
        self.assertIn("failed_action_index", str(agent_data.messages[-1]["content"]))
        self.assertEqual(agent_data.metrics["web_osgym/action_count"], 2)
        self.assertEqual(len(agent_data.extra_fields["teacher_ids_list"]), len(agent_data.response_mask))
        self.assertEqual(len(agent_data.extra_fields["teacher_logprobs_list"]), len(agent_data.response_mask))

    async def test_processing_tools_updates_teacher_sglang_prefix_surplus_for_image_observation(self):
        loop = _build_loop()
        loop.tools = {"computer": _ImageFakeTool()}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            images = kwargs.get("images") or []
            return [11, 12] + [90] * (3 * len(images))

        async def _fake_apply_server_chat_template(messages, **kwargs):
            if messages and messages[-1]["role"] == "tool":
                return [1, 2, 3, 21, 22]
            return [21, 22]

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_apply_server_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-image-surplus",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                    "web_osgym_include_a11y": True,
                    "teacher_prompt_ids": [1, 2, 3],
                    "teacher_server_prompt_ids": [1, 2, 3],
                    "server_prompt_ids": [1, 2, 3],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 3)

    async def test_processing_tools_does_not_count_teacher_only_text_gap_as_sglang_surplus(self):
        loop = _build_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12, 13, 14]

        async def _fake_apply_server_chat_template(messages, **kwargs):
            return [21, 22]

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_apply_server_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-teacher-text-gap",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3, 41]
        agent_data.response_ids = [41]
        agent_data.response_mask = [1]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3, 41],
                "teacher_server_prompt_ids": [1, 2, 3, 41],
                "server_prompt_ids": [1, 2, 3, 41],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [[41, 410, 411, 0]],
                "teacher_logprobs_list": [[-1.0, -1.1, -1.2, 0.0]],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "mini_step_image_spans": [{"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 0)

    async def test_processing_tools_accepts_action_named_tool_call(self):
        loop = _build_loop()
        tool = _ActionFakeTool()
        loop.tools = {"CLICK": tool}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": True,
            "reward": None,
        }
        agent_data.tool_calls = [
            type("Call", (), {"name": "CLICK", "arguments": '{"x":1,"y":2}'})()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(tool.executed[0], ("instance-1", {"x": 1, "y": 2}))
        self.assertEqual(agent_data.metrics["web_osgym/action_count"], 1)

    async def test_processing_tools_bundles_multiple_action_named_tool_calls(self):
        loop = _build_loop()
        loop.max_parallel_calls = 2
        tool = _ActionFakeTool()
        loop.tools = {"CLICK": tool}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": True,
            "reward": None,
        }
        agent_data.tool_calls = [
            type("Call", (), {"name": "CLICK", "arguments": '{"x":1,"y":2}'})(),
            type("Call", (), {"name": "CLICK", "arguments": '{"x":3,"y":4}'})(),
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(
            tool.executed[0],
            (
                "instance-1",
                {
                    "actions": [
                        {"action_type": "CLICK", "x": 1, "y": 2},
                        {"action_type": "CLICK", "x": 3, "y": 4},
                    ]
                },
            ),
        )
        self.assertEqual(agent_data.metrics["web_osgym/action_count"], 2)

    async def test_processing_tools_begin_trace_includes_tool_call_summary(self):
        loop = _build_loop()
        tool = _ActionFakeTool()
        loop.tools = {"CLICK": tool}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": True,
            "reward": None,
        }
        agent_data.tool_calls = [
            type("Call", (), {"name": "CLICK", "arguments": '{"x":1,"y":2}'})(),
            type("Call", (), {"name": "CLICK", "arguments": '{"x":3,"y":4}'})(),
        ]
        trace_calls = []

        def _capture_trace(event_name, **kwargs):
            trace_calls.append((event_name, kwargs))

        with patch.object(web_skd_agent_loop_module, "_trace_async_skd", side_effect=_capture_trace):
            state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        begin_event = next(kwargs for event_name, kwargs in trace_calls if event_name == "web_skd.tool_processing_begin")
        self.assertEqual(begin_event["tool_calls_len"], 2)
        self.assertEqual(begin_event["tool_call_names"], ["CLICK", "CLICK"])

    async def test_processing_tools_rebuilds_server_prompt_stream_from_current_messages(self):
        loop = _build_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], [21, 22])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], [21, 22])

    async def test_processing_tools_rebuilds_teacher_prompt_streams_for_image_boundary(self):
        loop = _build_loop()
        loop.tools = {"computer": _ImageFakeTool()}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12]

        loop.apply_chat_template = _fake_apply_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]
        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], [21, 22])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], [21, 22])
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], [1, 2, 3, 11, 12])
        self.assertEqual(agent_data.image_data, ["image-1"])

    async def test_processing_tools_discards_whole_observation_bundle_on_response_cutoff(self):
        loop = _build_loop()
        loop.tools = {"computer": _ImageFakeTool()}
        loop.response_length = 4
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, **kwargs):
            return [11, 12, 13, 14]

        async def _fake_apply_server_chat_template(messages, **kwargs):
            return [31, 32]

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_apply_server_chat_template

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-1",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
            }
        )
        before_messages = deepcopy(agent_data.messages)
        before_extra = deepcopy(agent_data.extra_fields)
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(agent_data.messages, before_messages)
        self.assertEqual(agent_data.prompt_ids, [1, 2, 3])
        self.assertEqual(agent_data.response_mask, [])
        self.assertEqual(agent_data.image_data, [])
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], before_extra["teacher_prompt_ids"])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], before_extra["teacher_server_prompt_ids"])
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], before_extra["server_prompt_ids"])
        self.assertEqual(
            agent_data.extra_fields["web_osgym_teacher_messages"], before_extra["web_osgym_teacher_messages"]
        )
        self.assertEqual(agent_data.extra_fields["web_osgym_termination_reason"], "tool_response_budget_exhausted")
        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)

    async def test_tool_observation_commit_is_atomic(self):
        loop = _build_loop()
        loop.tools = {"computer": _ImageFakeTool()}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, images=None, **kwargs):
            del kwargs
            if messages and messages[0]["role"] == "tool" and images:
                return [11, 12, 13, 14]
            return [21, 22]

        async def _fake_apply_server_chat_template(messages, **kwargs):
            del kwargs
            if messages and messages[-1]["role"] == "tool":
                return [31, 32]
            return [41, 42]

        async def _force_teacher_guard(*args, **kwargs):
            del args, kwargs
            return True

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_apply_server_chat_template
        loop._terminate_if_teacher_prefix_overflows = _force_teacher_guard

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=["initial-image"],
            video_data=[],
            metrics={},
            request_id="req-atomic-guard",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3, 41]
        agent_data.response_ids = [41]
        agent_data.response_mask = [1]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3, 41],
                "teacher_server_prompt_ids": [1, 2, 3, 41],
                "server_prompt_ids": [1, 2, 3, 41],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [[41, 410, 411, 0]],
                "teacher_logprobs_list": [[-1.0, -1.1, -1.2, 0.0]],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "mini_step_image_spans": [{"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]
        before_messages = deepcopy(agent_data.messages)
        before_image_data = deepcopy(agent_data.image_data)
        before_prompt_ids = list(agent_data.prompt_ids)
        before_response_mask = list(agent_data.response_mask)
        before_extra = deepcopy(agent_data.extra_fields)

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(agent_data.messages, before_messages)
        self.assertEqual(agent_data.image_data, before_image_data)
        self.assertEqual(agent_data.prompt_ids, before_prompt_ids)
        self.assertEqual(agent_data.response_mask, before_response_mask)
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], before_extra["server_prompt_ids"])
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], before_extra["teacher_prompt_ids"])
        self.assertEqual(
            agent_data.extra_fields["teacher_server_prompt_ids"],
            before_extra["teacher_server_prompt_ids"],
        )
        self.assertEqual(
            agent_data.extra_fields["web_osgym_teacher_messages"],
            before_extra["web_osgym_teacher_messages"],
        )
        self.assertEqual(agent_data.extra_fields["mini_step_image_spans"], before_extra["mini_step_image_spans"])
        self.assertEqual(agent_data.extra_fields["teacher_ids_list"], before_extra["teacher_ids_list"])
        self.assertEqual(
            agent_data.extra_fields["teacher_logprobs_list"],
            before_extra["teacher_logprobs_list"],
        )

    async def test_processing_tools_requires_completed_assistant_turn(self):
        loop = _build_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-pending-turn",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3, 41]
        agent_data.response_ids = [41]
        agent_data.response_mask = [1]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3, 41],
                "teacher_server_prompt_ids": [1, 2, 3, 41],
                "server_prompt_ids": [1, 2, 3, 41],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [[41, 410, 411, 0]],
                "teacher_logprobs_list": [[-1.0, -1.1, -1.2, 0.0]],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "skd_pending_turn_state": {
                    "tokens": [41],
                    "teacher_ids_rows": [[41, 410, 411, 0]],
                    "teacher_logprobs_rows": [[-1.0, -1.1, -1.2, 0.0]],
                    "raw_chunk": [41],
                    "verified_chunk": [41],
                },
                "skd_pending_turn_chunks": 1,
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        with self.assertRaisesRegex(ValueError, "completed assistant turn"):
            await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(loop.tools["computer"].executed, [])

    async def test_processing_tools_rebuilds_teacher_history_from_canonical_messages(self):
        loop = _build_loop()
        loop.tools = {"computer": _ImageFakeTool()}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fake_apply_chat_template(messages, images=None, videos=None, remove_system_prompt=False, tools=None):
            del videos, remove_system_prompt, tools
            if messages and messages[0]["role"] == "tool" and images:
                content = messages[0]["content"]
                if any(item.get("type") == "text" for item in content):
                    return [71, 72, 73, 74]
                return [61, 62, 63, 64]
            return [51, 52]

        async def _fake_server_template(messages, tools=None, remove_system_prompt=False):
            del tools, remove_system_prompt
            if len(messages) == 2 and messages[0] == {"role": "user", "content": "task"} and messages[-1]["role"] == "tool":
                content = messages[-1]["content"]
                if any(item.get("type") == "text" for item in content):
                    return [1, 2, 3, 91, 92]
                return [1, 2, 3, 81, 82]
            return [999]

        async def _never_overflow(*args, **kwargs):
            del args, kwargs
            return False

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_server_template
        loop._terminate_if_teacher_prefix_overflows = _never_overflow

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-rebuild-teacher-history",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [777],
                "server_prompt_ids": [888],
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}',
                },
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(
            agent_data.extra_fields["web_osgym_teacher_messages"],
            [
                {"role": "user", "content": "task"},
                {"role": "tool", "content": [{"type": "image"}, {"type": "text", "text": "A11Y_TREE:\nroot"}]},
            ],
        )
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], [1, 2, 3, 81, 82])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], [1, 2, 3, 91, 92])

    async def test_processing_tools_does_not_commit_terminal_action_response_observation(self):
        loop = _build_loop()
        loop.tools = {"computer": _TerminalImageFakeTool()}
        loop._build_teacher_messages = lambda messages: deepcopy(messages)

        async def _fail_if_template_called(*args, **kwargs):
            del args, kwargs
            raise AssertionError("terminal action response must not be templated")

        loop.apply_chat_template = _fail_if_template_called
        loop._apply_server_chat_template = _fail_if_template_called

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=["initial-image"],
            video_data=[],
            metrics={},
            request_id="req-terminal",
            tools_kwargs={},
        )
        agent_data._active_tools = loop.tools
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_ids = [41, 42]
        agent_data.response_mask = [1, 1]
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "server_prompt_ids": [1, 2, 3],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [[1, 0, 0, 0], [2, 0, 0, 0]],
                "teacher_logprobs_list": [[-1.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "mini_step_image_spans": [{"step_idx": 1, "image_start": 0, "image_end": 1}],
            }
        )
        agent_data.tool_calls = [
            type(
                "Call",
                (),
                {
                    "name": "computer",
                    "arguments": '{"actions":[{"action_type":"DONE"}]}',
                },
            )()
        ]
        before_messages = deepcopy(agent_data.messages)
        before_image_data = deepcopy(agent_data.image_data)
        before_prompt_ids = list(agent_data.prompt_ids)
        before_response_mask = list(agent_data.response_mask)
        before_extra = deepcopy(agent_data.extra_fields)

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(agent_data.messages, before_messages)
        self.assertEqual(agent_data.image_data, before_image_data)
        self.assertEqual(agent_data.prompt_ids, before_prompt_ids)
        self.assertEqual(agent_data.response_mask, before_response_mask)
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], before_extra["server_prompt_ids"])
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], before_extra["teacher_prompt_ids"])
        self.assertEqual(
            agent_data.extra_fields["teacher_server_prompt_ids"],
            before_extra["teacher_server_prompt_ids"],
        )
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 0)
        self.assertEqual(agent_data.extra_fields["web_osgym_teacher_messages"], before_extra["web_osgym_teacher_messages"])
        self.assertEqual(agent_data.extra_fields["mini_step_image_spans"], before_extra["mini_step_image_spans"])
        self.assertEqual(agent_data.extra_fields["web_osgym_termination_reason"], "model_done")
        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)

    async def test_system_stop_fetches_reward_on_skd_loop(self):
        loop = _build_loop()

        async def _base_generating(self, agent_data, sampling_params, ignore_termination=False, stop_after_skd_chunk=False):
            return AgentState.TERMINATED

        original = SkdAgentLoop._handle_generating_state
        SkdAgentLoop._handle_generating_state = _base_generating
        try:
            agent_data = AgentData(
                messages=[],
                image_data=[],
                video_data=[],
                metrics={},
                request_id="req-1",
                tools_kwargs={},
            )
            agent_data._active_tools = loop.tools
            agent_data.extra_fields.update(
                {
                    "web_osgym_instance_id": "instance-1",
                    "web_osgym_task_id": "12345",
                    "web_osgym_session_id": 101,
                    "web_osgym_include_a11y": True,
                    "teacher_prompt_ids": [],
                    "web_osgym_teacher_messages": [],
                }
            )

            state = await WebSkdAgentLoop._handle_generating_state(loop, agent_data, {})
        finally:
            SkdAgentLoop._handle_generating_state = original

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)

    async def test_student_observation_commit_appends_compact_delta_ids_for_image_boundary(self):
        loop = _build_loop()
        loop.loop = __import__("asyncio").get_running_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)
        loop.teacher_server_manager = None

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-student-image-boundary",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": _ImageFakeTool()}
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "server_prompt_ids": [1, 2, 3],
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "task-1",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
            }
        )

        async def _fake_apply_chat_template(messages, images=None, videos=None, remove_system_prompt=False, tools=None):
            del videos, remove_system_prompt, tools
            if messages and messages[0]["role"] == "tool" and images:
                content = messages[0]["content"]
                if any(item.get("type") == "text" for item in content):
                    return [71, 72, 73, 74]
                return [61, 62, 63, 64]
            return [51, 52]

        async def _fake_server_template(messages, tools=None, remove_system_prompt=False):
            del tools, remove_system_prompt
            if messages and messages[-1]["role"] == "tool":
                content = messages[-1]["content"]
                if any(item.get("type") == "text" for item in content):
                    return [1, 2, 3, 91, 92]
                return [1, 2, 3, 81, 82]
            return [41]

        async def _never_overflow(*args, **kwargs):
            del args, kwargs
            return False

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_server_template
        loop._terminate_if_teacher_prefix_overflows = _never_overflow

        agent_data.tool_calls = [
            type(
                "ToolCall",
                (),
                {"name": "computer", "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}'},
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.messages[-1], {"role": "tool", "content": [{"type": "image"}]})
        self.assertEqual(agent_data.image_data, ["image-1"])
        self.assertEqual(agent_data.prompt_ids, [1, 2, 3, 61, 62, 63, 64])
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], [1, 2, 3, 81, 82])
        self.assertEqual(agent_data.response_mask, [0, 0, 0, 0])
        self.assertEqual(agent_data.user_turns, 1)
        self.assertEqual(
            agent_data.extra_fields["mini_step_image_spans"],
            [{"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False}],
        )
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], [1, 2, 3, 71, 72, 73, 74])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], [1, 2, 3, 91, 92])
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 2)

    async def test_teacher_observation_commit_appends_canonical_delta_ids(self):
        loop = _build_loop()
        loop.loop = __import__("asyncio").get_running_loop()
        loop._build_teacher_messages = lambda messages: deepcopy(messages)
        loop.teacher_server_manager = None

        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="req-canonical-append",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": _ImageFakeTool()}
        agent_data._active_tool_schemas = []
        agent_data.prompt_ids = [1, 2, 3]
        agent_data.response_mask = []
        agent_data.extra_fields.update(
            {
                "server_prompt_ids": [1, 2, 3],
                "teacher_prompt_ids": [1, 2, 3],
                "teacher_server_prompt_ids": [1, 2, 3],
                "teacher_sglang_prefix_surplus": 0,
                "teacher_ids_list": [],
                "teacher_logprobs_list": [],
                "web_osgym_teacher_messages": [{"role": "user", "content": "task"}],
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "task-1",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
            }
        )

        async def _fake_apply_chat_template(messages, images=None, videos=None, remove_system_prompt=False, tools=None):
            del videos, remove_system_prompt, tools
            if messages and messages[0]["role"] == "tool" and images:
                return [71, 72, 73, 74]
            return [61, 62]

        async def _fake_server_template(messages, tools=None, remove_system_prompt=False):
            del tools, remove_system_prompt
            if messages and messages[-1]["role"] == "tool":
                return [1, 2, 3, 81, 82]
            return [91]

        async def _never_overflow(*args, **kwargs):
            del args, kwargs
            return False

        loop.apply_chat_template = _fake_apply_chat_template
        loop._apply_server_chat_template = _fake_server_template
        loop._terminate_if_teacher_prefix_overflows = _never_overflow

        agent_data.tool_calls = [
            type(
                "ToolCall",
                (),
                {"name": "computer", "arguments": '{"actions":[{"action_type":"CLICK","x":1,"y":2}]}'},
            )()
        ]

        state = await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(state, AgentState.GENERATING)
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], [1, 2, 3, 71, 72, 73, 74])
        self.assertEqual(agent_data.extra_fields["teacher_server_prompt_ids"], [1, 2, 3, 81, 82])
        self.assertEqual(agent_data.extra_fields["teacher_sglang_prefix_surplus"], 2)
