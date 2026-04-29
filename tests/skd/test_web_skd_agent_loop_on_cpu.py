import unittest
from copy import deepcopy

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


class _ActionFakeTool(_FakeTool):
    name = "CLICK"

    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="At failed_action_index 0, action Failed. Reason: target field was not focused"), None, {
            "terminated": False,
            "termination_reason": None,
            "action_count": 1,
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

    async def _fake_apply_server_chat_template(messages, **kwargs):
        return [21, 22]

    loop._apply_server_chat_template = _fake_apply_server_chat_template
    return loop


class TestWebSkdAgentLoop(unittest.IsolatedAsyncioTestCase):
    def test_web_skd_agent_is_still_skd(self):
        self.assertTrue(issubclass(WebSkdAgentLoop, SkdAgentLoop))

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
        self.assertIn("failed_action_index", str(agent_data.messages[-1]["content"]))
        self.assertEqual(agent_data.metrics["web_osgym/action_count"], 2)
        self.assertEqual(len(agent_data.extra_fields["teacher_ids_list"]), len(agent_data.response_mask))
        self.assertEqual(len(agent_data.extra_fields["teacher_logprobs_list"]), len(agent_data.response_mask))

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

    async def test_processing_tools_requires_existing_server_prompt_stream(self):
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

        with self.assertRaisesRegex(ValueError, "server_prompt_ids"):
            await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

    async def test_processing_tools_requires_teacher_streams_before_committing_bundle(self):
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
        before_messages = deepcopy(agent_data.messages)
        before_prompt_ids = list(agent_data.prompt_ids)

        with self.assertRaisesRegex(ValueError, "teacher_server_prompt_ids"):
            await WebSkdAgentLoop._handle_processing_tools_state(loop, agent_data)

        self.assertEqual(agent_data.messages, before_messages)
        self.assertEqual(agent_data.prompt_ids, before_prompt_ids)
        self.assertEqual(agent_data.response_mask, [])
        self.assertEqual(agent_data.image_data, [])
        self.assertEqual(agent_data.extra_fields["server_prompt_ids"], [1, 2, 3])
        self.assertEqual(agent_data.extra_fields["teacher_prompt_ids"], [1, 2, 3])
        self.assertEqual(agent_data.extra_fields["web_osgym_teacher_messages"], [{"role": "user", "content": "task"}])

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
                }
            )

            state = await WebSkdAgentLoop._handle_generating_state(loop, agent_data, {})
        finally:
            SkdAgentLoop._handle_generating_state = original

        self.assertEqual(state, AgentState.TERMINATED)
        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)
