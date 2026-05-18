import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import Mapping
from copy import deepcopy
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


def _scale_relative_coordinate(value: int, screen_extent: int, axis_name: str) -> int:
    if not 0 <= value <= 999:
        raise ValueError(f"{axis_name} must be within the 1000x1000 coordinate grid, got {value}")
    if screen_extent <= 0:
        raise ValueError(f"screen {axis_name} extent must be positive, got {screen_extent}")
    return round(value * (screen_extent - 1) / 999)


_PLAYWRIGHT_KEY_ALIASES = {
    "enter": "Enter",
    "return": "Enter",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "space": "Space",
    "spacebar": "Space",
    "backspace": "Backspace",
    "bksp": "Backspace",
    "delete": "Delete",
    "del": "Delete",
    "suppr": "Delete",
    "insert": "Insert",
    "ins": "Insert",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pgup": "PageUp",
    "pagedown": "PageDown",
    "pgdn": "PageDown",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "ctrl": "Control",
    "control": "Control",
    "controlormeta": "Control",
    "alt": "Alt",
    "option": "Alt",
    "shift": "Shift",
    "cmd": "Control",
    "command": "Control",
    "meta": "Meta",
    "super": "Meta",
    "win": "Meta",
    "windows": "Meta",
}


_FUNCTION_KEY_PATTERN = re.compile(r"^f([1-9]|1[0-2])$", re.IGNORECASE)


def _is_combo_key_string(value: Any) -> bool:
    return isinstance(value, str) and "+" in value and len([part for part in value.split("+") if part.strip()]) > 1


def _normalize_playwright_key_alias(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    lowered = stripped.lower()
    alias = _PLAYWRIGHT_KEY_ALIASES.get(lowered)
    if alias is not None:
        return alias
    if _FUNCTION_KEY_PATTERN.fullmatch(lowered):
        return lowered.upper()
    return stripped


def _canonicalize_hotkey_part(value: Any) -> Any:
    normalized = _normalize_playwright_key_alias(value)
    if isinstance(normalized, str) and len(normalized) == 1 and normalized.isalpha():
        return normalized.upper()
    return normalized


def _normalize_hotkey_keys(value: Any) -> Any:
    if not isinstance(value, list):
        return value

    normalized_keys = []
    for key in value:
        if isinstance(key, str):
            parts = [part.strip() for part in key.split("+")]
            if len(parts) > 1 and all(parts):
                normalized_keys.extend(_canonicalize_hotkey_part(part) for part in parts)
                continue
        normalized_keys.append(_canonicalize_hotkey_part(key))
    return normalized_keys


_COORD_ACTIONS = frozenset({"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "MOVE_TO", "DRAG_TO"})
_ACTION_TYPE_ALIASES = {
    "LEFT_CLICK": "CLICK",
    "left_click": "CLICK",
    "left": "CLICK",
}


def _coerce_legacy_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _coerce_wait_duration_seconds(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration if duration - int(duration) != 0 else int(duration)


def _normalize_web_osgym_action_payload(raw_action: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(raw_action)
    action_type = normalized.get("action_type")
    if isinstance(action_type, str):
        normalized["action_type"] = _ACTION_TYPE_ALIASES.get(action_type, action_type)
        action_type = normalized["action_type"]

    if action_type == "WAIT":
        duration = _coerce_wait_duration_seconds(normalized.get("duration"))
        if duration is not None:
            normalized["duration"] = duration
        for alias in ("timeout", "delay"):
            if alias not in normalized:
                continue
            alias_value = normalized.pop(alias)
            alias_duration = _coerce_wait_duration_seconds(alias_value)
            if duration is None:
                normalized["duration"] = alias_duration if alias_duration is not None else alias_value
                duration = alias_duration

    if action_type in _COORD_ACTIONS and "coordinate" not in normalized:
        x = normalized.get("x")
        y = normalized.get("y")
        if isinstance(x, list) and len(x) >= 2:
            normalized["coordinate"] = [int(x[0]), int(x[1])]
            normalized.pop("x", None)
            normalized.pop("y", None)
        elif x is not None and y is not None:
            normalized["coordinate"] = [int(x), int(y)]
            normalized.pop("x", None)
            normalized.pop("y", None)

    if action_type in {"PRESS", "KEY_DOWN", "KEY_UP"} and "key" in normalized:
        normalized["key"] = _normalize_playwright_key_alias(normalized.get("key"))
    elif action_type == "HOTKEY" and isinstance(normalized.get("keys"), list):
        normalized["keys"] = _normalize_hotkey_keys(normalized.get("keys"))
    return normalized


def _extract_coordinate_pair_like(value: Any) -> tuple[int, int] | None:
    if isinstance(value, list) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except Exception:
                return None
            if isinstance(parsed, list) and len(parsed) >= 2:
                return int(parsed[0]), int(parsed[1])
    return None


def _expand_web_osgym_action_payloads(raw_action: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalize_web_osgym_action_payload(raw_action)
    action_type = normalized.get("action_type")
    enter = _coerce_legacy_bool(normalized.pop("enter", None))
    normalized.pop("clear", None)

    if action_type == "WAIT":
        duration = _coerce_wait_duration_seconds(normalized.get("duration"))
        if duration is not None:
            normalized.pop("duration", None)
            wait_count = max(1, min(10, math.ceil(float(duration))))
            return [{"action_type": "WAIT"} for _ in range(wait_count)]

    if action_type == "TYPING" and enter is True:
        return [normalized, {"action_type": "PRESS", "key": _normalize_playwright_key_alias("enter")}]
    return [normalized]


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        timeout = float(config.get("timeout", 30.0))
        minimum_safe_action_timeout = config.get("minimum_safe_action_timeout")
        if minimum_safe_action_timeout is not None:
            minimum_safe_action_timeout = float(minimum_safe_action_timeout)
            if timeout < minimum_safe_action_timeout:
                raise ValueError(
                    "WebOsGym client timeout must be >= minimum_safe_action_timeout to avoid "
                    "reward-close requests racing an in-flight action."
                )
        self.client = WebOsGymClient(base_url=config["base_url"], timeout=timeout)
        self.minimum_safe_action_timeout = minimum_safe_action_timeout
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
        screen_width: int | None = None,
        screen_height: int | None = None,
    ) -> None:
        self._instance_dict[instance_id] = {
            "task_id": task_id,
            "request_id": request_id,
            "include_a11y": include_a11y,
            "reward": reward,
            "cursor_x": cursor_x,
            "cursor_y": cursor_y,
            "screen_width": screen_width,
            "screen_height": screen_height,
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
        screen_width, screen_height = _image_size(response.image)
        self.restore_instance(
            instance_id,
            task_id=task_id,
            request_id=request_id,
            include_a11y=include_a11y,
            reward=None,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        image = [response.image] if response.image is not None else None
        response_kwargs = {"text": response.text}
        if image is not None:
            response_kwargs["image"] = image
        return instance_id, ToolResponse(**response_kwargs)

    @staticmethod
    def _scale_xy_to_screen(state: dict[str, Any], x: int, y: int) -> tuple[int, int]:
        screen_width = state.get("screen_width")
        screen_height = state.get("screen_height")
        if not isinstance(screen_width, int) or not isinstance(screen_height, int):
            raise ValueError("screen dimensions are unknown; cannot project 1000x1000 coordinates to pixels")
        return (
            _scale_relative_coordinate(int(x), screen_width, "x"),
            _scale_relative_coordinate(int(y), screen_height, "y"),
        )

    @staticmethod
    def _require_coordinates(action: WebOsGymAction, cursor_x: int | None, cursor_y: int | None) -> tuple[int, int]:
        if action.coordinate is not None:
            if len(action.coordinate) < 2:
                raise ValueError(
                    f"{action.action_type} coordinate must have at least 2 elements, got {action.coordinate}"
                )
            return action.coordinate[0], action.coordinate[1]
        has_x = action.x is not None
        has_y = action.y is not None
        if has_x != has_y:
            raise ValueError(f"{action.action_type} requires both x and y when either coordinate is provided")
        if has_x and has_y:
            return int(action.x), int(action.y)
        if cursor_x is None or cursor_y is None:
            raise ValueError(
                f"{action.action_type} omitted coordinate, but no current cursor position is known. "
                "Provide coordinate or call MOVE_TO first."
            )
        return int(cursor_x), int(cursor_y)

    @staticmethod
    def _require_field(action: WebOsGymAction, field_name: str):
        value = getattr(action, field_name)
        if value is None:
            raise ValueError(f"{action.action_type} requires field '{field_name}'")
        return value

    @staticmethod
    def _normalize_mouse_button(action_type: str, button: str | None) -> str:
        normalized_button = (button or "left").lower()
        if normalized_button not in {"left", "middle", "right"}:
            raise ValueError(
                f"{action_type} button must be one of 'left', 'middle', or 'right', got {button!r}"
            )
        return normalized_button

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

    def _get_action_count_bounds(self) -> tuple[int | None, int | None]:
        if self._is_action_named_tool():
            return None, None
        actions_schema = self.tool_schema.function.parameters.properties.get("actions")
        if actions_schema is None:
            return None, None
        schema_extra = getattr(actions_schema, "model_extra", None) or {}
        min_items = _coerce_optional_int(schema_extra.get("minItems"))
        max_items = _coerce_optional_int(schema_extra.get("maxItems"))
        return min_items, max_items

    def _validate_action_count(self, action_count: int) -> None:
        min_items, max_items = self._get_action_count_bounds()
        if min_items is None and max_items is None:
            if action_count < 1:
                raise ValueError("Web/OSGym bundled action tool requires a non-empty actions list")
            return

        min_ok = min_items is None or action_count >= min_items
        max_ok = max_items is None or action_count <= max_items
        if min_ok and max_ok:
            return

        if min_items is not None and max_items is not None:
            range_text = f"between {min_items} and {max_items}"
        elif min_items is not None:
            range_text = f"at least {min_items}"
        else:
            range_text = f"at most {max_items}"
        raise ValueError(
            f"Web/OSGym bundled action tool requires {range_text} actions, got {action_count}. "
            "No actions were executed."
        )

    def postprocess_tool_arguments(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, Mapping):
            return parameters

        normalized = deepcopy(dict(parameters))
        if self._is_action_named_tool():
            normalized_action = _normalize_web_osgym_action_payload({"action_type": self.name, **normalized})
            normalized = {key: value for key, value in normalized_action.items() if key != "action_type"}
            if self.name in _COORD_ACTIONS:
                allowed_parameters = set(self.tool_schema.function.parameters.properties)
                uses_coordinate_schema = "coordinate" in allowed_parameters and not {"x", "y"} & allowed_parameters

                coordinate = normalized.get("coordinate")
                pair = _extract_coordinate_pair_like(coordinate)
                if pair is None:
                    pair = _extract_coordinate_pair_like(normalized.get("x"))
                if pair is None:
                    x = normalized.get("x")
                    y = normalized.get("y")
                    if x is not None and y is not None:
                        pair = int(x), int(y)

                if pair is not None:
                    if uses_coordinate_schema:
                        normalized["coordinate"] = [pair[0], pair[1]]
                        normalized.pop("x", None)
                        normalized.pop("y", None)
                    else:
                        normalized["x"], normalized["y"] = pair
                        normalized.pop("coordinate", None)
            return normalized

        raw_actions = normalized.get("actions")
        if not isinstance(raw_actions, list):
            return normalized
        expanded_actions = []
        for action in raw_actions:
            if isinstance(action, Mapping):
                expanded_actions.extend(_expand_web_osgym_action_payloads(action))
            else:
                expanded_actions.append(action)
        normalized["actions"] = expanded_actions
        return normalized

    def _normalize_actions(
        self, actions: list[WebOsGymAction], state: dict[str, Any]
    ) -> tuple[list[WebOsGymAction], int | None, int | None]:
        cursor_x = state.get("cursor_x")
        cursor_y = state.get("cursor_y")
        normalized = []

        for action in actions:
            action_type = action.action_type
            payload = action.model_dump(exclude_none=True)
            if action_type == "MOVE_TO":
                if action.coordinate is not None:
                    x, y = action.coordinate[0], action.coordinate[1]
                else:
                    x = self._require_field(action, "x")
                    y = self._require_field(action, "y")
                scaled_x, scaled_y = self._scale_xy_to_screen(state, int(x), int(y))
                payload.pop("coordinate", None)
                payload.update({"x": scaled_x, "y": scaled_y})
                cursor_x, cursor_y = scaled_x, scaled_y

            elif action_type == "CLICK":
                x, y = self._require_coordinates(action, cursor_x, cursor_y)
                button = self._normalize_mouse_button(action_type, action.button)
                num_clicks = 1 if action.num_clicks is None else int(action.num_clicks)
                if num_clicks < 1:
                    raise ValueError(f"CLICK requires num_clicks >= 1, got {num_clicks}")
                click_x, click_y = int(x), int(y)
                if action.coordinate is not None or (action.x is not None and action.y is not None):
                    click_x, click_y = self._scale_xy_to_screen(state, click_x, click_y)
                payload.pop("coordinate", None)
                payload.update({"button": button, "x": click_x, "y": click_y, "num_clicks": num_clicks})
                cursor_x, cursor_y = click_x, click_y

            elif action_type in {"MOUSE_DOWN", "MOUSE_UP"}:
                payload.update({"button": self._normalize_mouse_button(action_type, action.button)})

            elif action_type == "RIGHT_CLICK":
                x, y = self._require_coordinates(action, cursor_x, cursor_y)
                click_x, click_y = int(x), int(y)
                if action.coordinate is not None or (action.x is not None and action.y is not None):
                    click_x, click_y = self._scale_xy_to_screen(state, click_x, click_y)
                payload.pop("coordinate", None)
                payload.update({"x": click_x, "y": click_y})
                cursor_x, cursor_y = click_x, click_y

            elif action_type == "DOUBLE_CLICK":
                x, y = self._require_coordinates(action, cursor_x, cursor_y)
                click_x, click_y = int(x), int(y)
                if action.coordinate is not None or (action.x is not None and action.y is not None):
                    click_x, click_y = self._scale_xy_to_screen(state, click_x, click_y)
                payload.pop("coordinate", None)
                payload.update({"x": click_x, "y": click_y})
                cursor_x, cursor_y = click_x, click_y

            elif action_type == "DRAG_TO":
                if action.coordinate is not None:
                    x, y = action.coordinate[0], action.coordinate[1]
                else:
                    x = self._require_field(action, "x")
                    y = self._require_field(action, "y")
                drag_x, drag_y = self._scale_xy_to_screen(state, int(x), int(y))
                payload.pop("coordinate", None)
                payload.update({"x": drag_x, "y": drag_y})
                cursor_x, cursor_y = drag_x, drag_y

            elif action_type == "SCROLL":
                dx = 0 if action.dx is None else int(action.dx)
                dy = int(self._require_field(action, "dy"))
                payload.update({"dx": -dx, "dy": -dy})

            elif action_type == "TYPING":
                self._require_field(action, "text")

            elif action_type in {"PRESS", "KEY_DOWN", "KEY_UP"}:
                key = self._require_field(action, "key")
                if _is_combo_key_string(key):
                    raise ValueError(f"{action_type} expects a single key name, got combo string {key!r}")

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
        if not isinstance(raw_actions, list):
            raise ValueError("Web/OSGym bundled action tool requires an actions list")
        expanded_raw_actions = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                raise ValueError(
                    f"Web/OSGym actions[{index}] must be an object matching the action schema, "
                    f"got {type(raw_action).__name__}"
                )
            expanded_raw_actions.extend(_expand_web_osgym_action_payloads(raw_action))

        self._validate_action_count(len(expanded_raw_actions))

        actions = []
        for raw_action in expanded_raw_actions:
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
            if state.get("screen_width") is None and "web_osgym_screen_width" in extra_fields:
                state["screen_width"] = extra_fields["web_osgym_screen_width"]
            if state.get("screen_height") is None and "web_osgym_screen_height" in extra_fields:
                state["screen_height"] = extra_fields["web_osgym_screen_height"]
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
        response_status = getattr(response, "status", "ok")
        _trace_async_skd(
            "web_tool.action_http_done",
            agent_request_id=agent_request_id,
            tool_request_id=state["request_id"],
            elapsed_ms=round(http_ms, 1),
            status=response_status,
            text_len=len(response.text or ""),
            has_image=getattr(response, "image_b64", None) is not None,
            image_b64_len=len(getattr(response, "image_b64", None) or ""),
        )
        if response_status != "ok":
            error_message = (
                getattr(response, "message", None)
                or getattr(response, "text", None)
                or "Web/OSGym environment returned an error response."
            )
            error_type = getattr(response, "error_type", None) or "unknown"
            return ToolResponse(text=f"Web/OSGym environment error ({error_type}): {error_message}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": len(actions),
                "invalid_action": True,
                "web_osgym_error_type": error_type,
                "web_osgym_actions": final_actions,
            }
        state["cursor_x"] = cursor_x
        state["cursor_y"] = cursor_y
        if isinstance(extra_fields, dict):
            extra_fields["web_osgym_cursor_x"] = cursor_x
            extra_fields["web_osgym_cursor_y"] = cursor_y
        image_t0 = time.monotonic()
        decoded_image = response.image
        image_ms = (time.monotonic() - image_t0) * 1000
        width, height = _image_size(decoded_image)
        if width is not None:
            state["screen_width"] = width
        if height is not None:
            state["screen_height"] = height
        if isinstance(extra_fields, dict):
            if state.get("screen_width") is not None:
                extra_fields["web_osgym_screen_width"] = state["screen_width"]
            if state.get("screen_height") is not None:
                extra_fields["web_osgym_screen_height"] = state["screen_height"]
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
            "invalid_action": False,
            "web_osgym_actions": final_actions,
        }

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float | None, dict]:
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

    def request_reward_detached(self, *, request_id: int, task_id: str) -> asyncio.Task:
        async def _request_reward() -> None:
            try:
                await self.client.reward(request_id=request_id, task_id=task_id)
            except Exception:
                logger.warning(
                    "[WebOsGymTool] best-effort reward close failed request_id=%r task_id=%r",
                    request_id,
                    task_id,
                    exc_info=True,
                )

        return asyncio.create_task(_request_reward(), name=f"web_osgym_reward_close_{request_id}")

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)
