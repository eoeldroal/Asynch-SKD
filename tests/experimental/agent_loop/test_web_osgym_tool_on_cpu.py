import unittest

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.web_osgym_tool import WebOsGymTool


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "computer",
                "description": "Apply one or more low-level computer actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "description": "One or more Computer 13 actions.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action_type": {"type": "string", "description": "Computer 13 action type."},
                                    "x": {"type": "integer", "description": "Screen x coordinate."},
                                    "y": {"type": "integer", "description": "Screen y coordinate."},
                                    "text": {"type": "string", "description": "Typing payload."},
                                },
                                "required": ["action_type"],
                            },
                        },
                    },
                    "required": ["actions"],
                },
            },
        }
    )


def _action_tool_schema(action_name: str, properties: dict | None = None, required: list[str] | None = None):
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": action_name,
                "description": f"{action_name} action.",
                "parameters": {
                    "type": "object",
                    "properties": properties or {},
                    "required": required or [],
                },
            },
        }
    )


class TestWebOsGymTool(unittest.IsolatedAsyncioTestCase):
    def test_tool_schema_preserves_nested_items(self):
        schema = _tool_schema().model_dump(exclude_unset=True, exclude_none=True)
        self.assertIn("items", schema["function"]["parameters"]["properties"]["actions"])

    async def test_tool_create_starts_session_and_stores_session_request_id(self):
        class _FakeClient:
            async def start(self, **kwargs):
                class _Response:
                    text = "A11Y_TREE:\nroot"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()

        instance_id, response = await tool.create(task_id="12345", request_id=101, include_a11y=True)

        self.assertEqual(response.text, "A11Y_TREE:\nroot")
        self.assertEqual(tool._instance_dict[instance_id]["request_id"], 101)
        self.assertEqual(tool._instance_dict[instance_id]["task_id"], "12345")

    async def test_tool_execute_uses_same_session_request_id(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "A11Y_TREE:\nnext"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, reward, metrics = await tool.execute(
            "i1",
            {"actions": [{"action_type": "CLICK", "x": 1, "y": 2}]},
        )

        self.assertEqual(response.text, "A11Y_TREE:\nnext")
        self.assertIsNone(reward)
        self.assertFalse(metrics["terminated"])
        self.assertEqual(metrics["action_count"], 1)
        self.assertEqual(seen["request_id"], 101)
        self.assertEqual(seen["task_id"], "12345")

    async def test_tool_execute_logs_final_server_payload_before_action_request(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        with self.assertLogs("verl.tools.web_osgym_tool", level="INFO") as logs:
            await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 1, "y": 2}]})

        log_text = "\n".join(logs.output)
        self.assertIn("[WebOsGymTool][ServerPayload]", log_text)
        self.assertIn("request_id=101", log_text)
        self.assertIn("task_id='12345'", log_text)
        self.assertIn("'action_type': 'CLICK'", log_text)
        self.assertEqual(seen["request_id"], 101)

    async def test_action_named_tool_execute_wraps_arguments_as_single_action(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(
            config={"base_url": "http://env"},
            tool_schema=_action_tool_schema(
                "CLICK",
                properties={
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "middle", "right"]},
                    "num_clicks": {"type": "integer"},
                },
            ),
        )
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, reward, metrics = await tool.execute("i1", {"x": 1, "y": 2})

        self.assertEqual(response.text, "next")
        self.assertIsNone(reward)
        self.assertEqual(metrics["action_count"], 1)
        action = seen["actions"][0]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.x, 1)
        self.assertEqual(action.y, 2)
        self.assertEqual(action.button, "left")
        self.assertEqual(action.num_clicks, 1)

    async def test_action_named_tool_rejects_backend_actions_argument(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(
            config={"base_url": "http://env"},
            tool_schema=_action_tool_schema("DONE"),
        )
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, reward, metrics = await tool.execute("i1", {"actions": [{"action_type": "DONE"}]})

        self.assertIn("DONE does not accept parameter(s): actions", response.text)
        self.assertIsNone(reward)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_marks_done_fail_as_terminal_without_fetching_reward(self):
        class _FakeClient:
            async def action(self, **kwargs):
                class _Response:
                    text = "done"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        _, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "DONE"}]})
        self.assertTrue(metrics["terminated"])
        self.assertEqual(metrics["termination_reason"], "model_done")

    async def test_tool_execute_preserves_multi_action_payload(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        await tool.execute(
            "i1",
            {
                "actions": [
                    {"action_type": "MOVE_TO", "x": 1, "y": 2},
                    {"action_type": "CLICK", "x": 1, "y": 2},
                ]
            },
        )

        self.assertEqual(len(seen["actions"]), 2)
        self.assertEqual(seen["actions"][0].action_type, "MOVE_TO")
        self.assertEqual(seen["actions"][1].action_type, "CLICK")

    async def test_tool_execute_defaults_click_button_and_num_clicks(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 10, "y": 20}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.button, "left")
        self.assertEqual(action.num_clicks, 1)
        self.assertEqual(action.x, 10)
        self.assertEqual(action.y, 20)

    async def test_tool_execute_uses_current_cursor_for_click_without_coordinates(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        await tool.execute(
            "i1",
            {"actions": [{"action_type": "MOVE_TO", "x": 10, "y": 20}, {"action_type": "CLICK"}]},
        )

        action = seen["actions"][1]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.x, 10)
        self.assertEqual(action.y, 20)
        self.assertEqual(action.button, "left")
        self.assertEqual(action.num_clicks, 1)

    async def test_tool_execute_rejects_coordinate_free_click_before_cursor_is_known(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "CLICK"}]})

        self.assertIn("CLICK omitted x/y", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_uses_current_cursor_for_double_click_without_coordinates(self):
        seen = {}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        await tool.execute(
            "i1",
            {"actions": [{"action_type": "MOVE_TO", "x": 5, "y": 6}, {"action_type": "DOUBLE_CLICK"}]},
        )

        action = seen["actions"][1]
        self.assertEqual(action.action_type, "DOUBLE_CLICK")
        self.assertEqual(action.x, 5)
        self.assertEqual(action.y, 6)

    async def test_tool_execute_restores_cursor_from_agent_extra_fields(self):
        seen = {}

        class _AgentData:
            extra_fields = {"web_osgym_cursor_x": 7, "web_osgym_cursor_y": 8}

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        await tool.execute("i1", {"actions": [{"action_type": "CLICK"}]}, agent_data=_AgentData())

        action = seen["actions"][0]
        self.assertEqual(action.x, 7)
        self.assertEqual(action.y, 8)

    async def test_tool_execute_rejects_coordinate_free_double_click_before_cursor_is_known(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "DOUBLE_CLICK"}]})

        self.assertIn("DOUBLE_CLICK omitted x/y", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_rejects_backend_unsupported_action_before_http(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "RIGHT_CLICK", "x": 1, "y": 2}]})

        self.assertIn("not supported by the current WebGym backend", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_returns_observation_for_invalid_action_item(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, reward, metrics = await tool.execute("i1", {"actions": ["DONE"]})

        self.assertIn("Invalid Web/OSGym action payload", response.text)
        self.assertIsNone(reward)
        self.assertFalse(metrics["terminated"])
        self.assertTrue(metrics["invalid_action"])
        self.assertEqual(metrics["action_count"], 0)
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_returns_observation_for_non_object_arguments(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        response, reward, metrics = await tool.execute("i1", ["DONE"])

        self.assertIn("Web/OSGym tool arguments must be an object", response.text)
        self.assertIsNone(reward)
        self.assertFalse(metrics["terminated"])
        self.assertTrue(metrics["invalid_action"])
        self.assertEqual(metrics["action_count"], 0)
        self.assertFalse(tool.client.action_called)

    async def test_tool_calc_reward_uses_existing_session_request_id(self):
        class _FakeClient:
            async def reward(self, **kwargs):
                assert kwargs["request_id"] == 101
                assert kwargs["task_id"] == "12345"
                return 1.0

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

        reward = await tool.calc_reward("i1")
        self.assertEqual(reward, 1.0)
