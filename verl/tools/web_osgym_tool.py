import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Optional
from uuid import uuid4

import httpx
from pydantic import ValidationError

from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymAction, WebOsGymClient
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 1


def _image_size(image: Any) -> tuple[int | None, int | None]:
    size = getattr(image, "size", None)
    if not size or len(size) != 2:
        return None, None
    width, height = size
    return int(width), int(height)


class WebOsGymTool(BaseTool):
    COMPUTER_13_ACTIONS = {
        "MOVE_TO",
        "CLICK",
        "MOUSE_DOWN",
        "MOUSE_UP",
        "RIGHT_CLICK",
        "DOUBLE_CLICK",
        "DRAG_TO",
        "SCROLL",
        "TYPING",
        "PRESS",
        "KEY_DOWN",
        "KEY_UP",
        "HOTKEY",
        "WAIT",
        "DONE",
        "FAIL",
    }
    BACKEND_UNSUPPORTED_ACTIONS = {"MOUSE_DOWN", "MOUSE_UP", "RIGHT_CLICK", "DRAG_TO"}

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.client = WebOsGymClient(base_url=config["base_url"], timeout=config.get("timeout", 30.0))
        self.include_a11y = config.get("include_a11y", False)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def restore_instance(
        self,
        instance_id: str,
        *,
        task_id: str,
        request_id: int,
        include_a11y: bool,
        reward: float | None = None,
        cursor_x: int | None = None,
        cursor_y: int | None = None,
    ) -> None:
        self._instance_dict[instance_id] = {
            "task_id": task_id,
            "request_id": request_id,
            "include_a11y": include_a11y,
            "reward": reward,
            "cursor_x": cursor_x,
            "cursor_y": cursor_y,
        }

    async def create(
        self,
        instance_id: Optional[str] = None,
        *,
        task_id: str,
        request_id: int,
        include_a11y: bool | None = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        del kwargs
        instance_id = instance_id or str(uuid4())
        include_a11y = self.include_a11y if include_a11y is None else include_a11y
        response = await self.client.start(request_id=request_id, task_id=task_id, include_a11y=include_a11y)
        self.restore_instance(
            instance_id,
            task_id=task_id,
            request_id=request_id,
            include_a11y=include_a11y,
            reward=None,
        )
        image = [response.image] if response.image is not None else None
        response_kwargs = {"text": response.text}
        if image is not None:
            response_kwargs["image"] = image
        return instance_id, ToolResponse(**response_kwargs)

    @staticmethod
    def _require_coordinates(action: WebOsGymAction, cursor_x: int | None, cursor_y: int | None) -> tuple[int, int]:
        has_x = action.x is not None
        has_y = action.y is not None
        if has_x != has_y:
            raise ValueError(f"{action.action_type} requires both x and y when either coordinate is provided")
        if has_x and has_y:
            return int(action.x), int(action.y)
        if cursor_x is None or cursor_y is None:
            raise ValueError(
                f"{action.action_type} omitted x/y, but no current cursor position is known. "
                "Provide x/y or call MOVE_TO first."
            )
        return int(cursor_x), int(cursor_y)

    @staticmethod
    def _require_field(action: WebOsGymAction, field_name: str):
        value = getattr(action, field_name)
        if value is None:
            raise ValueError(f"{action.action_type} requires field '{field_name}'")
        return value

    def _is_action_named_tool(self) -> bool:
        return self.name in self.COMPUTER_13_ACTIONS

    def _validate_action_tool_parameters(self, parameters: Mapping[str, Any]) -> None:
        schema_parameters = self.tool_schema.function.parameters
        allowed_parameters = set(schema_parameters.properties)
        extra_parameters = sorted(set(parameters) - allowed_parameters)
        if extra_parameters:
            raise ValueError(f"{self.name} does not accept parameter(s): {', '.join(extra_parameters)}")

        missing_parameters = sorted(set(schema_parameters.required or []) - set(parameters))
        if missing_parameters:
            raise ValueError(f"{self.name} requires parameter(s): {', '.join(missing_parameters)}")

        for parameter_name, value in parameters.items():
            property_schema = schema_parameters.properties.get(parameter_name)
            if property_schema is None:
                continue
            enum_values = property_schema.enum
            if enum_values is not None and value not in enum_values:
                raise ValueError(
                    f"{self.name}.{parameter_name} must be one of {enum_values}, got {value!r}"
                )

    def _normalize_actions(
        self, actions: list[WebOsGymAction], state: dict[str, Any]
    ) -> tuple[list[WebOsGymAction], int | None, int | None]:
        cursor_x = state.get("cursor_x")
        cursor_y = state.get("cursor_y")
        normalized = []

        for action in actions:
            action_type = action.action_type
            if action_type in self.BACKEND_UNSUPPORTED_ACTIONS:
                raise ValueError(
                    f"{action_type} is part of Computer 13 but is not supported by the current WebGym backend"
                )

            payload = action.model_dump(exclude_none=True)
            if action_type == "MOVE_TO":
                x = self._require_field(action, "x")
                y = self._require_field(action, "y")
                cursor_x, cursor_y = int(x), int(y)

            elif action_type == "CLICK":
                x, y = self._require_coordinates(action, cursor_x, cursor_y)
                button = action.button or "left"
                if button.lower() != "left":
                    raise ValueError(f"CLICK only supports button='left', got {button!r}")
                num_clicks = 1 if action.num_clicks is None else int(action.num_clicks)
                if num_clicks < 1:
                    raise ValueError(f"CLICK requires num_clicks >= 1, got {num_clicks}")
                payload.update({"button": "left", "x": x, "y": y, "num_clicks": num_clicks})
                cursor_x, cursor_y = x, y

            elif action_type == "DOUBLE_CLICK":
                x, y = self._require_coordinates(action, cursor_x, cursor_y)
                payload.update({"x": x, "y": y})
                cursor_x, cursor_y = x, y

            elif action_type == "SCROLL":
                self._require_field(action, "dx")
                self._require_field(action, "dy")

            elif action_type == "TYPING":
                self._require_field(action, "text")

            elif action_type in {"PRESS", "KEY_DOWN", "KEY_UP"}:
                self._require_field(action, "key")

            elif action_type == "HOTKEY":
                keys = self._require_field(action, "keys")
                if not keys:
                    raise ValueError("HOTKEY requires at least one key")

            elif action_type not in {"WAIT", "DONE", "FAIL"}:
                raise ValueError(f"Unsupported action_type: {action_type}")

            normalized.append(WebOsGymAction(**payload))

        return normalized, cursor_x, cursor_y

    def _parse_raw_actions(
        self, raw_actions: Any, state: dict[str, Any]
    ) -> tuple[list[WebOsGymAction], int | None, int | None]:
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("Web/OSGym bundled action tool requires a non-empty actions list")

        actions = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                raise ValueError(
                    f"Web/OSGym actions[{index}] must be an object matching the action schema, "
                    f"got {type(raw_action).__name__}"
                )
            actions.append(WebOsGymAction(**raw_action))

        terminal_actions = [action for action in actions if action.action_type in {"DONE", "FAIL"}]
        if terminal_actions and len(actions) != 1:
            raise ValueError("DONE/FAIL must be sent as a standalone action list")
        return self._normalize_actions(actions, state)

    def _parse_actions(
        self, parameters: dict[str, Any], state: dict[str, Any]
    ) -> tuple[list[WebOsGymAction], int | None, int | None]:
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Web/OSGym tool arguments must be an object, got {type(parameters).__name__}")

        if self._is_action_named_tool():
            self._validate_action_tool_parameters(parameters)
            raw_actions = [{"action_type": self.name, **parameters}]
        else:
            raw_actions = parameters.get("actions")

        return self._parse_raw_actions(raw_actions, state)

    def _restore_cursor_from_agent_data(self, state: dict[str, Any], agent_data: Any) -> dict[str, Any] | None:
        extra_fields = getattr(agent_data, "extra_fields", None)
        if isinstance(extra_fields, dict):
            if state.get("cursor_x") is None and "web_osgym_cursor_x" in extra_fields:
                state["cursor_x"] = extra_fields["web_osgym_cursor_x"]
            if state.get("cursor_y") is None and "web_osgym_cursor_y" in extra_fields:
                state["cursor_y"] = extra_fields["web_osgym_cursor_y"]
            return extra_fields
        return None

    async def _send_actions(
        self,
        instance_id: str,
        actions: list[WebOsGymAction],
        *,
        cursor_x: int | None,
        cursor_y: int | None,
        extra_fields: dict[str, Any] | None,
        agent_request_id: str | None = None,
    ) -> tuple[ToolResponse, float | None, dict]:
        state = self._instance_dict[instance_id]
        final_actions = [action.model_dump(exclude_none=True) for action in actions]
        logger.warning(
            "[WebOsGymTool][ServerPayload] op=action request_id=%r task_id=%r include_a11y=%r actions=%s",
            state["request_id"],
            state["task_id"],
            state["include_a11y"],
            final_actions,
        )
        _trace_async_skd(
            "web_tool.action_http_begin",
            agent_request_id=agent_request_id,
            tool_request_id=state["request_id"],
            task_id=state["task_id"],
            include_a11y=state["include_a11y"],
            action_count=len(actions),
            cursor_x=cursor_x,
            cursor_y=cursor_y,
        )

        try:
            http_t0 = time.monotonic()
            response = await self.client.action(
                request_id=state["request_id"],
                task_id=state["task_id"],
                include_a11y=state["include_a11y"],
                actions=actions,
            )
            http_ms = (time.monotonic() - http_t0) * 1000
        except httpx.HTTPStatusError as exc:
            _trace_async_skd(
                "web_tool.action_http_error",
                agent_request_id=agent_request_id,
                tool_request_id=state["request_id"],
                status_code=exc.response.status_code,
                error_type=type(exc).__name__,
            )
            if 400 <= exc.response.status_code < 500:
                return ToolResponse(text=f"Invalid Web/OSGym action payload rejected by environment: {exc}"), None, {
                    "terminated": False,
                    "termination_reason": None,
                    "action_count": 0,
                    "invalid_action": True,
                }
            raise
        except httpx.HTTPError as exc:
            _trace_async_skd(
                "web_tool.action_http_error",
                agent_request_id=agent_request_id,
                tool_request_id=state["request_id"],
                status_code=None,
                error_type=type(exc).__name__,
            )
            raise
        _trace_async_skd(
            "web_tool.action_http_done",
            agent_request_id=agent_request_id,
            tool_request_id=state["request_id"],
            elapsed_ms=round(http_ms, 1),
            status=response.status,
            text_len=len(response.text or ""),
            has_image=response.image_b64 is not None,
            image_b64_len=len(response.image_b64 or ""),
        )
        state["cursor_x"] = cursor_x
        state["cursor_y"] = cursor_y
        if isinstance(extra_fields, dict):
            extra_fields["web_osgym_cursor_x"] = cursor_x
            extra_fields["web_osgym_cursor_y"] = cursor_y
        image_t0 = time.monotonic()
        decoded_image = response.image
        image_ms = (time.monotonic() - image_t0) * 1000
        width, height = _image_size(decoded_image)
        _trace_async_skd(
            "web_tool.image_decode_done",
            agent_request_id=agent_request_id,
            tool_request_id=state["request_id"],
            elapsed_ms=round(image_ms, 1),
            has_image=decoded_image is not None,
            image_width=width,
            image_height=height,
        )
        image = [decoded_image] if decoded_image is not None else None
        terminal_action = actions[0] if len(actions) == 1 and actions[0].action_type in {"DONE", "FAIL"} else None
        terminated = terminal_action is not None
        termination_reason = None
        if terminal_action and terminal_action.action_type == "DONE":
            termination_reason = "model_done"
        elif terminal_action and terminal_action.action_type == "FAIL":
            termination_reason = "model_fail"
        response_kwargs = {"text": response.text}
        if image is not None:
            response_kwargs["image"] = image
        _trace_async_skd(
            "web_tool.action_complete",
            agent_request_id=agent_request_id,
            tool_request_id=state["request_id"],
            action_count=len(actions),
            terminated=terminated,
            termination_reason=termination_reason,
            response_text_len=len(response.text or ""),
            image_count=_safe_len(image),
        )
        return ToolResponse(**response_kwargs), None, {
            "terminated": terminated,
            "termination_reason": termination_reason,
            "action_count": len(actions),
        }

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        agent_data = kwargs.get("agent_data")
        state = self._instance_dict[instance_id]
        extra_fields = self._restore_cursor_from_agent_data(state, agent_data)
        agent_request_id = getattr(agent_data, "request_id", None)
        try:
            actions, cursor_x, cursor_y = self._parse_actions(parameters, state)
        except (TypeError, ValueError, ValidationError) as exc:
            return ToolResponse(text=f"Invalid Web/OSGym action payload: {exc}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }

        return await self._send_actions(
            instance_id,
            actions,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            extra_fields=extra_fields,
            agent_request_id=agent_request_id,
        )

    @rollout_trace_op
    async def execute_action_bundle(
        self, instance_id: str, actions: list[dict[str, Any]], **kwargs
    ) -> tuple[ToolResponse, float | None, dict]:
        agent_data = kwargs.get("agent_data")
        state = self._instance_dict[instance_id]
        extra_fields = self._restore_cursor_from_agent_data(state, agent_data)
        agent_request_id = getattr(agent_data, "request_id", None)
        try:
            parsed_actions, cursor_x, cursor_y = self._parse_raw_actions(actions, state)
        except (TypeError, ValueError, ValidationError) as exc:
            return ToolResponse(text=f"Invalid Web/OSGym action payload: {exc}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }

        return await self._send_actions(
            instance_id,
            parsed_actions,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            extra_fields=extra_fields,
            agent_request_id=agent_request_id,
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        del kwargs
        state = self._instance_dict[instance_id]
        if state["reward"] is None:
            state["reward"] = await self.client.reward(request_id=state["request_id"], task_id=state["task_id"])
        return float(state["reward"])

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)
