import unittest

from PIL import Image

from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymRemoteError
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.web_osgym_tool import WebOsGymTool


def _tool_schema(*, min_items: int | None = None, max_items: int | None = None) -> OpenAIFunctionToolSchema:
    actions_schema = {
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
    }
    if min_items is not None:
        actions_schema["minItems"] = min_items
    if max_items is not None:
        actions_schema["maxItems"] = max_items

    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "computer",
                "description": "Apply one or more low-level computer actions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": actions_schema,
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
    @staticmethod
    def _instance_state(**overrides):
        state = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
            "screen_width": 1000,
            "screen_height": 1000,
        }
        state.update(overrides)
        return state

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

    async def test_tool_create_stores_screen_dimensions_from_initial_image(self):
        class _FakeClient:
            async def start(self, **kwargs):
                class _Response:
                    text = "A11Y_TREE:\nroot"
                    image = Image.new("RGB", (1920, 1080), "white")

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()

        instance_id, _ = await tool.create(task_id="12345", request_id=101, include_a11y=True)

        self.assertEqual(tool._instance_dict[instance_id]["screen_width"], 1920)
        self.assertEqual(tool._instance_dict[instance_id]["screen_height"], 1080)

    async def test_tool_create_does_not_restore_instance_when_start_fails(self):
        class _FakeClient:
            async def start(self, **kwargs):
                raise WebOsGymRemoteError(
                    op="start",
                    session_id=101,
                    task_id="12345",
                    error_type="gateway_busy",
                    message="Timed out waiting for gateway capacity.",
                )

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()

        with self.assertRaises(WebOsGymRemoteError):
            await tool.create(task_id="12345", request_id=101, include_a11y=True)

        self.assertEqual(tool._instance_dict, {})

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 10, "y": 20}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.button, "left")
        self.assertEqual(action.num_clicks, 1)
        self.assertEqual(action.x, 10)
        self.assertEqual(action.y, 20)

    async def test_tool_execute_allows_right_click_button_in_click_action(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 10, "y": 20, "button": "right"}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.button, "right")
        self.assertEqual(action.num_clicks, 1)

    async def test_tool_execute_allows_middle_click_button_in_click_action(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 10, "y": 20, "button": "middle"}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "CLICK")
        self.assertEqual(action.button, "middle")
        self.assertEqual(action.num_clicks, 1)

    async def test_tool_execute_normalizes_press_key_alias(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "PRESS", "key": "enter"}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "PRESS")
        self.assertEqual(action.key, "Enter")

    async def test_tool_execute_normalizes_hotkey_key_aliases(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "HOTKEY", "keys": ["ctrl", "right"]}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "HOTKEY")
        self.assertEqual(action.keys, ["Control", "ArrowRight"])

    async def test_tool_execute_normalizes_hotkey_command_alias_to_control(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "HOTKEY", "keys": ["cmd", "a"]}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "HOTKEY")
        self.assertEqual(action.keys, ["Control", "a"])

    async def test_tool_execute_splits_hotkey_combo_string(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "HOTKEY", "keys": ["ctrl+a"]}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "HOTKEY")
        self.assertEqual(action.keys, ["Control", "a"])

    async def test_tool_execute_splits_hotkey_combo_string_with_spaces(self):
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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "HOTKEY", "keys": ["cmd + shift + t"]}]})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "HOTKEY")
        self.assertEqual(action.keys, ["Control", "Shift", "t"])

    async def test_action_named_tool_execute_normalizes_press_key_alias(self):
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
                "PRESS",
                properties={
                    "key": {"type": "string"},
                },
                required=["key"],
            ),
        )
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"key": "esc"})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "PRESS")
        self.assertEqual(action.key, "Escape")

    async def test_action_named_tool_execute_normalizes_meta_alias_to_control(self):
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
                "PRESS",
                properties={
                    "key": {"type": "string"},
                },
                required=["key"],
            ),
        )
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"key": "meta"})

        action = seen["actions"][0]
        self.assertEqual(action.action_type, "PRESS")
        self.assertEqual(action.key, "Control")

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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

        response, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "CLICK"}]})

        self.assertIn("CLICK omitted x/y", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_rejects_action_bundles_above_schema_max_items(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema(min_items=1, max_items=2))
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        response, _, metrics = await tool.execute(
            "i1",
            {
                "actions": [
                    {"action_type": "WAIT"},
                    {"action_type": "WAIT"},
                    {"action_type": "WAIT"},
                ]
            },
        )

        self.assertIn("between 1 and 2 actions", response.text)
        self.assertIn("got 3", response.text)
        self.assertIn("No actions were executed", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertEqual(metrics["action_count"], 0)
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
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

        await tool.execute("i1", {"actions": [{"action_type": "CLICK"}]}, agent_data=_AgentData())

        action = seen["actions"][0]
        self.assertEqual(action.x, 7)
        self.assertEqual(action.y, 8)

    async def test_tool_execute_restores_screen_dimensions_from_agent_extra_fields(self):
        seen = {}

        class _AgentData:
            extra_fields = {
                "web_osgym_screen_width": 1920,
                "web_osgym_screen_height": 1080,
            }

        class _FakeClient:
            async def action(self, **kwargs):
                seen.update(kwargs)

                class _Response:
                    text = "next"
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state(screen_width=None, screen_height=None)

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 528, "y": 582}]}, agent_data=_AgentData())

        action = seen["actions"][0]
        self.assertEqual(action.x, 1014)
        self.assertEqual(action.y, 629)

    async def test_tool_execute_rejects_coordinate_free_double_click_before_cursor_is_known(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        response, _, metrics = await tool.execute("i1", {"actions": [{"action_type": "DOUBLE_CLICK"}]})

        self.assertIn("DOUBLE_CLICK omitted x/y", response.text)
        self.assertTrue(metrics["invalid_action"])
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_returns_observation_for_gateway_error_response(self):
        class _FakeClient:
            async def action(self, **kwargs):
                class _Response:
                    status = "error"
                    error_type = "fail_request_handle"
                    message = "WEBGYM-RL failed to handle request."
                    text = None
                    image = None

                return _Response()

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        response, reward, metrics = await tool.execute(
            "i1", {"actions": [{"action_type": "SCROLL", "dx": 0, "dy": -10}]}
        )

        self.assertIn("Web/OSGym environment error", response.text)
        self.assertIn("WEBGYM-RL failed to handle request.", response.text)
        self.assertIsNone(reward)
        self.assertFalse(metrics["terminated"])
        self.assertTrue(metrics["invalid_action"])
        self.assertEqual(metrics["action_count"], 1)

    async def test_tool_execute_returns_observation_for_invalid_action_item(self):
        class _FakeClient:
            def __init__(self):
                self.action_called = False

            async def action(self, **kwargs):
                self.action_called = True

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

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
        tool._instance_dict["i1"] = self._instance_state()

        response, reward, metrics = await tool.execute("i1", ["DONE"])

        self.assertIn("Web/OSGym tool arguments must be an object", response.text)
        self.assertIsNone(reward)
        self.assertFalse(metrics["terminated"])
        self.assertTrue(metrics["invalid_action"])
        self.assertEqual(metrics["action_count"], 0)
        self.assertFalse(tool.client.action_called)

    async def test_tool_execute_scales_1000_grid_coordinates_to_screen_pixels(self):
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
        tool._instance_dict["i1"] = self._instance_state(screen_width=1920, screen_height=1080)

        await tool.execute("i1", {"actions": [{"action_type": "CLICK", "x": 528, "y": 582}]})

        action = seen["actions"][0]
        self.assertEqual(action.x, 1014)
        self.assertEqual(action.y, 629)

    async def test_tool_execute_uses_top_left_origin_for_1000_grid(self):
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
        tool._instance_dict["i1"] = self._instance_state(screen_width=1920, screen_height=1080)

        await tool.execute("i1", {"actions": [{"action_type": "MOVE_TO", "x": 0, "y": 0}, {"action_type": "CLICK"}]})

        move_action = seen["actions"][0]
        click_action = seen["actions"][1]
        self.assertEqual((move_action.x, move_action.y), (0, 0))
        self.assertEqual((click_action.x, click_action.y), (0, 0))

    async def test_tool_calc_reward_uses_existing_session_request_id(self):
        class _FakeClient:
            async def reward(self, **kwargs):
                assert kwargs["request_id"] == 101
                assert kwargs["task_id"] == "12345"
                return 1.0

        tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
        tool.client = _FakeClient()
        tool._instance_dict["i1"] = self._instance_state()

        reward = await tool.calc_reward("i1")
        self.assertEqual(reward, 1.0)
