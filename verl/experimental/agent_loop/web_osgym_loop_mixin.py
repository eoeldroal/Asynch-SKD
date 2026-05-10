from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.tool_parser import ToolParseError
from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymRemoteError
from verl.tools.schemas import ToolResponse


class WebOsGymLoopMixin:
    shared_tools_kwargs_key = "web_osgym"
    legacy_bundled_tool_name = "computer"
    web_osgym_start_max_attempts = 3
    web_osgym_start_retry_delay_sec = 1.0

    @staticmethod
    def _tool_parse_error_rules() -> list[str]:
        return [
            "Output only well-formed tool call blocks. Do not include extra chat text before, between, or after them.",
            "Each tool call must use `<tool_call> ... </tool_call>`.",
            "Each function block must use `<function=...> ... </function>`.",
            "Each parameter block must use `<parameter=...> ... </parameter>`.",
            "The function name must be `computer`.",
            "The value inside `<parameter=actions>` must be a valid JSON array.",
            "Keep brackets and braces balanced.",
            "Do not nest `<parameter=...>` tags inside the `actions` JSON.",
            "`x` and `y` must be single integers, not lists or tuples.",
            "If you use `DONE` or `FAIL`, it must be the only action in that `actions` array.",
        ]

    def _build_tool_parse_error_feedback(self, parse_error: ToolParseError) -> str:
        lines = [
            f"Invalid tool call format: {parse_error.message}",
            "",
            "Below is an example of a valid tool call format:",
            "",
            "<tool_call>",
            "<function=computer>",
            "<parameter=actions>",
            '[{"action_type":"CLICK","x":621,"y":680}]',
            "</parameter>",
            "</function>",
            "</tool_call>",
            "",
            "Rules:",
        ]
        lines.extend(f"- {rule}" for rule in self._tool_parse_error_rules())
        return "\n".join(lines)

    def _get_active_tool(self, agent_data, tool_name: str | None = None):
        active_tools = getattr(agent_data, "_active_tools", {})
        if tool_name is not None and tool_name in active_tools:
            return active_tools[tool_name]
        if self.legacy_bundled_tool_name in active_tools:
            return active_tools[self.legacy_bundled_tool_name]
        if active_tools:
            return next(iter(active_tools.values()))
        raise KeyError(f"No active Web/OSGym tool is available for {tool_name or self.shared_tools_kwargs_key!r}")

    def _allocate_web_osgym_session_id(self) -> int:
        return uuid4().int % (2**31 - 1) or 1

    def _get_create_kwargs(self, agent_data) -> dict:
        tools_kwargs = agent_data.tools_kwargs or {}
        preferred_keys = [self.shared_tools_kwargs_key, self.legacy_bundled_tool_name]
        for key in preferred_keys:
            kwargs = tools_kwargs.get(key)
            if isinstance(kwargs, dict) and "create_kwargs" in kwargs:
                return deepcopy(kwargs.get("create_kwargs") or {})

        active_tools = getattr(agent_data, "_active_tools", {})
        for key in active_tools:
            kwargs = tools_kwargs.get(key)
            if isinstance(kwargs, dict) and "create_kwargs" in kwargs:
                return deepcopy(kwargs.get("create_kwargs") or {})
        return {}

    def _ensure_web_osgym_session(self, agent_data, tool_name: str | None = None) -> None:
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        tool = self._get_active_tool(agent_data, tool_name)
        instance_dict = getattr(tool, "_instance_dict", {})
        if instance_id in instance_dict:
            return
        restore = getattr(tool, "restore_instance", None)
        if restore is None:
            raise ValueError("Tool does not support restoring a persistent web_osgym session")
        restore(
            instance_id,
            task_id=agent_data.extra_fields["web_osgym_task_id"],
            request_id=agent_data.extra_fields["web_osgym_session_id"],
            include_a11y=agent_data.extra_fields["web_osgym_include_a11y"],
            reward=agent_data.extra_fields.get("web_osgym_reward_score"),
            cursor_x=agent_data.extra_fields.get("web_osgym_cursor_x"),
            cursor_y=agent_data.extra_fields.get("web_osgym_cursor_y"),
            screen_width=agent_data.extra_fields.get("web_osgym_screen_width"),
            screen_height=agent_data.extra_fields.get("web_osgym_screen_height"),
        )

    def _bundle_web_osgym_tool_calls(self, agent_data, tool_calls=None) -> tuple[dict | None, ToolResponse | None]:
        active_tools = getattr(agent_data, "_active_tools", {})
        actions = []
        if tool_calls is None:
            tool_calls = agent_data.tool_calls

        for tool_call in tool_calls:
            if tool_call.name not in active_tools:
                available = list(active_tools.keys())
                return None, ToolResponse(text=f"Unknown function '{tool_call.name}'. Available tools: {available}")

            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                return None, ToolResponse(text=f"Invalid JSON in arguments for '{tool_call.name}': {exc}")

            if not isinstance(tool_args, Mapping):
                return None, ToolResponse(
                    text=f"Invalid arguments for '{tool_call.name}': expected an object, got {type(tool_args).__name__}"
                )

            if tool_call.name == self.legacy_bundled_tool_name:
                raw_actions = tool_args.get("actions")
                if not isinstance(raw_actions, list):
                    return None, ToolResponse(
                        text=f"Invalid arguments for '{tool_call.name}': expected a non-empty actions list"
                    )
                actions.extend(raw_actions)
            else:
                # The model sees Computer 13 as action-named tools, while the
                # Web/OSGym server still receives one ordered actions list.
                actions.append({"action_type": tool_call.name, **tool_args})

        bundled_args = {"actions": actions}
        tool = self._get_active_tool(agent_data)
        postprocess_tool_arguments = getattr(tool, "postprocess_tool_arguments", None)
        if callable(postprocess_tool_arguments):
            bundled_args = postprocess_tool_arguments(bundled_args)
        return bundled_args, None

    async def _execute_web_osgym_tool_calls(self, agent_data):
        max_parallel_calls = getattr(self, "max_parallel_calls", None)
        if not max_parallel_calls:
            max_parallel_calls = len(agent_data.tool_calls)
        selected_tool_calls = agent_data.tool_calls[: max_parallel_calls]
        tool_call = selected_tool_calls[0]
        active_tools = getattr(agent_data, "_active_tools", {})
        if tool_call.name not in active_tools:
            available = list(active_tools.keys())
            return ToolResponse(text=f"Unknown function '{tool_call.name}'. Available tools: {available}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }
        self._ensure_web_osgym_session(agent_data, tool_call.name)

        if len(selected_tool_calls) == 1:
            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                return ToolResponse(text=f"Invalid JSON in arguments for '{tool_call.name}': {exc}"), None, {
                    "terminated": False,
                    "termination_reason": None,
                    "action_count": 0,
                    "invalid_action": True,
                }

            tool = self._get_active_tool(agent_data, tool_call.name)
            postprocess_tool_arguments = getattr(tool, "postprocess_tool_arguments", None)
            if callable(postprocess_tool_arguments):
                tool_args = postprocess_tool_arguments(tool_args)
            instance_id = agent_data.extra_fields["web_osgym_instance_id"]
            return await tool.execute(instance_id, tool_args, agent_data=agent_data)

        bundled_args, error_response = self._bundle_web_osgym_tool_calls(agent_data, tool_calls=selected_tool_calls)
        if error_response is not None:
            return error_response, None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }

        tool = self._get_active_tool(agent_data, tool_call.name)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        execute_bundle = getattr(tool, "execute_action_bundle", None)
        if execute_bundle is not None:
            return await execute_bundle(instance_id, bundled_args["actions"], agent_data=agent_data)
        return await tool.execute(instance_id, bundled_args, agent_data=agent_data)

    async def _start_web_osgym_session(self, agent_data, *, include_a11y: bool):
        tool = self._get_active_tool(agent_data)
        create_kwargs = self._get_create_kwargs(agent_data)
        task_id = create_kwargs.get("task_id") or agent_data.extra_fields.get("task_id")
        if task_id is None:
            raise ValueError("Web/OS gym session requires task_id in tools_kwargs.create_kwargs or agent_data.extra_fields")
        session_id = int(agent_data.extra_fields.get("web_osgym_session_id") or self._allocate_web_osgym_session_id())
        last_error = None
        for attempt in range(1, self.web_osgym_start_max_attempts + 1):
            try:
                instance_id, start_response = await tool.create(
                    task_id=task_id,
                    request_id=session_id,
                    include_a11y=include_a11y,
                )
                break
            except (WebOsGymRemoteError, httpx.HTTPError) as exc:
                last_error = exc
                if attempt >= self.web_osgym_start_max_attempts:
                    raise
                await asyncio.sleep(self.web_osgym_start_retry_delay_sec)
        else:
            raise RuntimeError("web_osgym start retry loop exited unexpectedly") from last_error
        agent_data.extra_fields["web_osgym_instance_id"] = instance_id
        agent_data.extra_fields["web_osgym_task_id"] = task_id
        agent_data.extra_fields["web_osgym_session_id"] = session_id
        agent_data.extra_fields["web_osgym_include_a11y"] = include_a11y
        instance_state = getattr(tool, "_instance_dict", {}).get(instance_id, {})
        if instance_state.get("screen_width") is not None:
            agent_data.extra_fields["web_osgym_screen_width"] = instance_state["screen_width"]
        if instance_state.get("screen_height") is not None:
            agent_data.extra_fields["web_osgym_screen_height"] = instance_state["screen_height"]
        return start_response

    async def _finalize_with_web_osgym_reward(self, agent_data, termination_reason: str) -> None:
        if agent_data.extra_fields.get("web_osgym_reward_fetched"):
            return
        self._ensure_web_osgym_session(agent_data)
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        reward = await tool.calc_reward(instance_id, termination_reason=termination_reason)
        agent_data.extra_fields["web_osgym_reward_fetched"] = True
        agent_data.extra_fields["web_osgym_termination_reason"] = termination_reason
        agent_data.extra_fields["web_osgym_reward_score"] = float(reward)
        reward_extra_info = agent_data.extra_fields.get("reward_extra_info") or {}
        if not isinstance(reward_extra_info, dict):
            reward_extra_info = {}
        agent_data.extra_fields["reward_extra_info"] = {
            **reward_extra_info,
            "web_osgym_reward_score": float(reward),
            "web_osgym_termination_reason": termination_reason,
        }

    async def _release_web_osgym_session(self, agent_data) -> None:
        instance_id = agent_data.extra_fields.get("web_osgym_instance_id")
        if instance_id is None:
            return
        tool = self._get_active_tool(agent_data)
        await tool.release(instance_id)
