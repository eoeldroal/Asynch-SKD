import asyncio
import json
import unittest

from PIL import Image

from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_parser import ToolParseError
from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymRemoteError
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.base_tool import ToolResponse
from verl.tools.web_osgym_tool import WebOsGymTool


class _FakeTool:
    name = "computer"
    tool_schema = None

    def __init__(self):
        self.created = []
        self.rewards = []
        self.restored = []
        self.detached_reward_requests = []
        self._instance_dict = {}

    async def create(self, **kwargs):
        self.created.append(kwargs)
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
            "screen_width": 1920,
            "screen_height": 1080,
        }
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0

    def request_reward_detached(self, *, request_id: int, task_id: str):
        self.detached_reward_requests.append((request_id, task_id))
        return asyncio.create_task(asyncio.sleep(0), name=f"reward-close-{request_id}")

    def restore_instance(self, instance_id, **kwargs):
        self.restored.append((instance_id, kwargs))
        self._instance_dict[instance_id] = dict(kwargs)


class _ActionClient:
    def __init__(self):
        self.calls = []

    async def action(self, **kwargs):
        self.calls.append(kwargs)

        class _Response:
            status = "ok"
            text = "next"
            image = None

        return _Response()


class _RetryingFakeTool(_FakeTool):
    def __init__(self, *, failures_before_success: int):
        super().__init__()
        self.failures_before_success = failures_before_success
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        self.created.append(kwargs)
        if self.attempts <= self.failures_before_success:
            raise WebOsGymRemoteError(
                op="start",
                session_id=kwargs["request_id"],
                task_id=kwargs["task_id"],
                error_type="fail_request_handle",
                message=f"attempt {self.attempts} failed",
            )
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
            "screen_width": 1920,
            "screen_height": 1080,
        }
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])


class TestWebOsGymLoopMixin(unittest.IsolatedAsyncioTestCase):
    def test_tool_parse_error_feedback_uses_payload_only_example(self):
        loop = WebOsGymLoopMixin()

        feedback = loop._build_tool_parse_error_feedback(
            ToolParseError(kind="actions_json_malformed", message="the actions JSON is malformed.")
        )

        self.assertIn("Retry with exactly one corrected tool call.", feedback)
        self.assertIn("Example actions payload only:", feedback)
        self.assertIn('[{"action_type":"CLICK","coordinate":[621,680]}]', feedback)
        self.assertNotIn("Below is an example of a valid tool call format:", feedback)
        self.assertNotIn("<tool_call>\n<function=computer>", feedback)

    def test_tool_parse_error_feedback_uses_action_named_rules_when_active_tools_are_not_bundled(self):
        loop = WebOsGymLoopMixin()

        feedback = loop._build_tool_parse_error_feedback(
            ToolParseError(kind="unknown_parse_error", message="the tool call could not be parsed."),
            active_tool_names=["CLICK", "SCROLL", "DONE"],
        )

        self.assertIn("The function name must be one of: CLICK, SCROLL, DONE.", feedback)
        self.assertIn("Example function call only:", feedback)
        self.assertIn("<function=CLICK>", feedback)
        self.assertIn("<parameter=coordinate>", feedback)
        self.assertIn("[621, 680]", feedback)
        self.assertNotIn("<parameter=actions>", feedback)
        self.assertNotIn("The function name must be `computer`.", feedback)

    async def test_bundle_web_osgym_tool_calls_preserves_action_named_arguments(self):
        click_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "CLICK",
                    "description": "CLICK action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "button": {"type": "string", "enum": ["left", "middle", "right"]},
                        },
                        "required": [],
                    },
                },
            }
        )
        typing_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "TYPING",
                    "description": "TYPING action.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        )
        click_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=click_schema)
        typing_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=typing_schema)
        loop = WebOsGymLoopMixin()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {
            "CLICK": click_tool,
            "TYPING": typing_tool,
        }
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments=json.dumps({"coordinate": [10, 20], "button": "left"})),
            FunctionCall(name="TYPING", arguments=json.dumps({"text": "hello"})),
        ]

        bundled_args, error_response = loop._bundle_web_osgym_tool_calls(agent_data)

        self.assertIsNone(error_response)
        self.assertEqual(
            bundled_args,
            {
                "actions": [
                    {"action_type": "CLICK", "coordinate": [10, 20], "button": "left"},
                    {"action_type": "TYPING", "text": "hello"},
                ]
            },
        )

    async def test_execute_web_osgym_tool_call_absorbs_bracketed_x_pair_in_action_named_lane(self):
        click_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "CLICK",
                    "description": "CLICK action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "button": {"type": "string", "enum": ["left", "middle", "right"]},
                        },
                        "required": [],
                    },
                },
            }
        )
        click_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=click_schema)
        click_tool.client = _ActionClient()
        click_tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
            "cursor_x": None,
            "cursor_y": None,
            "screen_width": 1000,
            "screen_height": 1000,
        }

        loop = WebOsGymLoopMixin()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"CLICK": click_tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments=json.dumps({"x": "[19, 974]"})),
        ]

        tool_response, _, result = await loop._execute_web_osgym_tool_calls(agent_data)

        self.assertEqual(tool_response.text, "next")
        self.assertFalse(result["invalid_action"])
        sent_action = click_tool.client.calls[0]["actions"][0]
        self.assertEqual(sent_action.action_type, "CLICK")
        self.assertEqual(sent_action.x, 19)
        self.assertEqual(sent_action.y, 974)

    async def test_execute_web_osgym_tool_call_absorbs_coordinate_field_in_action_named_lane(self):
        click_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "CLICK",
                    "description": "CLICK action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "button": {"type": "string", "enum": ["left", "middle", "right"]},
                        },
                        "required": [],
                    },
                },
            }
        )
        click_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=click_schema)
        click_tool.client = _ActionClient()
        click_tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
            "cursor_x": None,
            "cursor_y": None,
            "screen_width": 1000,
            "screen_height": 1000,
        }

        loop = WebOsGymLoopMixin()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"CLICK": click_tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments=json.dumps({"coordinate": [19, 974]})),
        ]

        tool_response, _, result = await loop._execute_web_osgym_tool_calls(agent_data)

        self.assertEqual(tool_response.text, "next")
        self.assertFalse(result["invalid_action"])
        sent_action = click_tool.client.calls[0]["actions"][0]
        self.assertEqual(sent_action.action_type, "CLICK")
        self.assertEqual(sent_action.x, 19)
        self.assertEqual(sent_action.y, 974)

    async def test_execute_web_osgym_tool_call_absorbs_scalar_xy_into_coordinate_in_action_named_lane(self):
        click_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "CLICK",
                    "description": "CLICK action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "button": {"type": "string", "enum": ["left", "middle", "right"]},
                        },
                        "required": [],
                    },
                },
            }
        )
        click_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=click_schema)
        click_tool.client = _ActionClient()
        click_tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
            "cursor_x": None,
            "cursor_y": None,
            "screen_width": 1000,
            "screen_height": 1000,
        }

        loop = WebOsGymLoopMixin()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"CLICK": click_tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )
        agent_data.tool_calls = [
            FunctionCall(name="CLICK", arguments=json.dumps({"x": 19, "y": 974})),
        ]

        tool_response, _, result = await loop._execute_web_osgym_tool_calls(agent_data)

        self.assertEqual(tool_response.text, "next")
        self.assertFalse(result["invalid_action"])
        sent_action = click_tool.client.calls[0]["actions"][0]
        self.assertEqual(sent_action.action_type, "CLICK")
        self.assertEqual(sent_action.x, 19)
        self.assertEqual(sent_action.y, 974)

    async def test_execute_web_osgym_tool_call_normalizes_wait_timeout_alias_in_action_named_lane(self):
        wait_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "WAIT",
                    "description": "WAIT action.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "number"},
                        },
                        "required": [],
                    },
                },
            }
        )
        wait_tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=wait_schema)
        wait_tool.client = _ActionClient()
        wait_tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
            "cursor_x": None,
            "cursor_y": None,
            "screen_width": 1000,
            "screen_height": 1000,
        }

        loop = WebOsGymLoopMixin()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"WAIT": wait_tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )
        agent_data.tool_calls = [
            FunctionCall(name="WAIT", arguments=json.dumps({"timeout": "2"})),
        ]

        tool_response, _, result = await loop._execute_web_osgym_tool_calls(agent_data)

        self.assertEqual(tool_response.text, "next")
        self.assertFalse(result["invalid_action"])
        self.assertEqual(result["action_count"], 2)
        self.assertEqual([action.action_type for action in wait_tool.client.calls[0]["actions"]], ["WAIT", "WAIT"])

    async def test_start_session_stores_instance_id_and_observation(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(agent_data.extra_fields["web_osgym_instance_id"], "instance-1")
        self.assertEqual(agent_data.extra_fields["web_osgym_session_id"], 101)
        self.assertEqual(agent_data.extra_fields["web_osgym_screen_width"], 1920)
        self.assertEqual(agent_data.extra_fields["web_osgym_screen_height"], 1080)
        self.assertEqual(response.text, "initial-observation")

    async def test_start_session_reads_shared_web_osgym_create_kwargs(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "shared-task"}}},
        )
        agent_data._active_tools = {"CLICK": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.created[0]["task_id"], "shared-task")

    async def test_start_session_retries_until_third_attempt_succeeds(self):
        loop = WebOsGymLoopMixin()
        tool = _RetryingFakeTool(failures_before_success=2)
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "retry-task"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.attempts, 3)
        self.assertEqual(len(tool.created), 3)
        self.assertEqual(
            [kwargs["request_id"] for kwargs in tool.created],
            [101, 101, 101],
        )
        self.assertEqual(agent_data.extra_fields["web_osgym_instance_id"], "instance-1")
        self.assertEqual(response.text, "initial-observation")

    async def test_start_session_raises_after_three_failed_attempts(self):
        loop = WebOsGymLoopMixin()
        tool = _RetryingFakeTool(failures_before_success=3)
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "retry-task"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        with self.assertRaises(WebOsGymRemoteError):
            await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.attempts, 3)
        self.assertNotIn("web_osgym_instance_id", agent_data.extra_fields)

    async def test_finalize_with_reward_stores_reward_once(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_trajectory_dir": "/tmp/trajectory/session-101",
                "web_osgym_include_a11y": False,
                "web_osgym_trajectory_counts": {
                    "attempted_tool_call_count": 2,
                    "valid_tool_call_count": 2,
                    "first_valid_tool_call_index": 1,
                    "executed_action_count": 8,
                    "non_grounding_adjacent_pair_count": 6,
                },
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
        }

        await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")

        self.assertEqual(agent_data.extra_fields["web_osgym_env_reward_score"], 1.0)
        self.assertEqual(agent_data.extra_fields["web_osgym_attempted_tool_calls"], 2)
        self.assertEqual(agent_data.extra_fields["web_osgym_valid_tool_calls"], 2)
        self.assertEqual(agent_data.extra_fields["web_osgym_first_valid_tool_call_index"], 1)
        self.assertTrue(agent_data.extra_fields["web_osgym_reward_requested"])
        self.assertEqual(
            agent_data.extra_fields["reward_extra_info"],
            {
                "request_id": "loop-req",
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_trajectory_dir": "/tmp/trajectory/session-101",
                "web_osgym_attempted_tool_calls": 2,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 2,
                "web_osgym_executed_action_count": 8,
                "web_osgym_non_grounding_adjacent_pair_count": 6,
                "web_osgym_termination_reason": "system_stop",
            },
        )
        self.assertNotIn("web_osgym_reward_score", agent_data.extra_fields)
        self.assertNotIn("web_osgym_format_reward", agent_data.extra_fields)
        self.assertEqual(len(tool.rewards), 1)
        self.assertEqual(tool.detached_reward_requests, [])

    async def test_request_web_osgym_reward_best_effort_sends_once_for_unfetched_session(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )

        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")
        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")

        self.assertEqual(tool.detached_reward_requests, [(101, "12345")])
        self.assertTrue(agent_data.extra_fields["web_osgym_reward_requested"])
        self.assertEqual(agent_data.extra_fields["web_osgym_termination_reason"], "system_stop")

    def test_ensure_session_restores_missing_local_instance_state(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "web_osgym_cursor_x": 7,
                "web_osgym_cursor_y": 8,
                "web_osgym_screen_width": 1920,
                "web_osgym_screen_height": 1080,
            }
        )

        loop._ensure_web_osgym_session(agent_data)

        self.assertEqual(
            tool.restored,
            [
                (
                    "instance-1",
                    {
                        "task_id": "12345",
                        "request_id": 101,
                        "include_a11y": True,
                        "reward": None,
                        "cursor_x": 7,
                        "cursor_y": 8,
                        "screen_width": 1920,
                        "screen_height": 1080,
                    },
                )
            ],
        )
