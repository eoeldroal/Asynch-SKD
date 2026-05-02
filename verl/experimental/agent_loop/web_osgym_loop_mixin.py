from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

from verl.tools.schemas import ToolResponse

_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    fields = {"pid": os.getpid(), "mono_ns": time.monotonic_ns(), **fields}
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


class WebOsGymLoopMixin:
    shared_tools_kwargs_key = "web_osgym"
    legacy_bundled_tool_name = "computer"

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
        )

    def _bundle_web_osgym_tool_calls(self, agent_data) -> tuple[dict | None, ToolResponse | None]:
        active_tools = getattr(agent_data, "_active_tools", {})
        actions = []
        _trace_async_skd(
            "web_osgym_mixin.bundle_begin",
            request_id=getattr(agent_data, "request_id", None),
            tool_calls_len=len(agent_data.tool_calls),
            active_tool_names=sorted(active_tools.keys()),
        )
        bundle_t0 = time.monotonic()

        for tool_call in agent_data.tool_calls:
            if tool_call.name not in active_tools:
                available = list(active_tools.keys())
                _trace_async_skd(
                    "web_osgym_mixin.bundle_unknown_tool",
                    request_id=getattr(agent_data, "request_id", None),
                    tool_name=tool_call.name,
                    available=available,
                )
                return None, ToolResponse(text=f"Unknown function '{tool_call.name}'. Available tools: {available}")

            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                _trace_async_skd(
                    "web_osgym_mixin.bundle_json_error",
                    request_id=getattr(agent_data, "request_id", None),
                    tool_name=tool_call.name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                return None, ToolResponse(text=f"Invalid JSON in arguments for '{tool_call.name}': {exc}")

            if not isinstance(tool_args, Mapping):
                _trace_async_skd(
                    "web_osgym_mixin.bundle_invalid_args",
                    request_id=getattr(agent_data, "request_id", None),
                    tool_name=tool_call.name,
                    args_type=type(tool_args).__name__,
                )
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

        bundle_ms = (time.monotonic() - bundle_t0) * 1000
        _trace_async_skd(
            "web_osgym_mixin.bundle_done",
            request_id=getattr(agent_data, "request_id", None),
            elapsed_ms=round(bundle_ms, 1),
            action_count=len(actions),
        )
        return {"actions": actions}, None

    async def _execute_web_osgym_tool_calls(self, agent_data):
        tool_call = agent_data.tool_calls[0]
        active_tools = getattr(agent_data, "_active_tools", {})
        _trace_async_skd(
            "web_osgym_mixin.execute_begin",
            request_id=getattr(agent_data, "request_id", None),
            first_tool_name=tool_call.name,
            tool_calls_len=len(agent_data.tool_calls),
            active_tool_names=sorted(active_tools.keys()),
        )
        if tool_call.name not in active_tools:
            available = list(active_tools.keys())
            _trace_async_skd(
                "web_osgym_mixin.execute_unknown_tool",
                request_id=getattr(agent_data, "request_id", None),
                tool_name=tool_call.name,
                available=available,
            )
            return ToolResponse(text=f"Unknown function '{tool_call.name}'. Available tools: {available}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }
        ensure_t0 = time.monotonic()
        self._ensure_web_osgym_session(agent_data, tool_call.name)
        _trace_async_skd(
            "web_osgym_mixin.ensure_session_done",
            request_id=getattr(agent_data, "request_id", None),
            elapsed_ms=round((time.monotonic() - ensure_t0) * 1000, 1),
            tool_name=tool_call.name,
            instance_id=agent_data.extra_fields.get("web_osgym_instance_id"),
        )

        if len(agent_data.tool_calls) == 1:
            try:
                json_t0 = time.monotonic()
                tool_args = json.loads(tool_call.arguments)
                _trace_async_skd(
                    "web_osgym_mixin.single_json_done",
                    request_id=getattr(agent_data, "request_id", None),
                    elapsed_ms=round((time.monotonic() - json_t0) * 1000, 1),
                    tool_name=tool_call.name,
                    arg_keys=sorted(tool_args.keys()) if isinstance(tool_args, Mapping) else None,
                )
            except (json.JSONDecodeError, TypeError) as exc:
                _trace_async_skd(
                    "web_osgym_mixin.single_json_error",
                    request_id=getattr(agent_data, "request_id", None),
                    tool_name=tool_call.name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                return ToolResponse(text=f"Invalid JSON in arguments for '{tool_call.name}': {exc}"), None, {
                    "terminated": False,
                    "termination_reason": None,
                    "action_count": 0,
                    "invalid_action": True,
                }

            tool = self._get_active_tool(agent_data, tool_call.name)
            instance_id = agent_data.extra_fields["web_osgym_instance_id"]
            _trace_async_skd(
                "web_osgym_mixin.single_execute_call_begin",
                request_id=getattr(agent_data, "request_id", None),
                tool_name=tool_call.name,
                instance_id=instance_id,
            )
            return await tool.execute(instance_id, tool_args, agent_data=agent_data)

        bundled_args, error_response = self._bundle_web_osgym_tool_calls(agent_data)
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
            _trace_async_skd(
                "web_osgym_mixin.bundle_execute_call_begin",
                request_id=getattr(agent_data, "request_id", None),
                tool_name=tool_call.name,
                instance_id=instance_id,
                action_count=len(bundled_args["actions"]),
            )
            return await execute_bundle(instance_id, bundled_args["actions"], agent_data=agent_data)
        _trace_async_skd(
            "web_osgym_mixin.bundle_execute_fallback_begin",
            request_id=getattr(agent_data, "request_id", None),
            tool_name=tool_call.name,
            instance_id=instance_id,
            action_count=len(bundled_args["actions"]),
        )
        return await tool.execute(instance_id, bundled_args, agent_data=agent_data)

    async def _start_web_osgym_session(self, agent_data, *, include_a11y: bool):
        tool = self._get_active_tool(agent_data)
        create_kwargs = self._get_create_kwargs(agent_data)
        task_id = create_kwargs.get("task_id") or agent_data.extra_fields.get("task_id")
        if task_id is None:
            raise ValueError("Web/OS gym session requires task_id in tools_kwargs.create_kwargs or agent_data.extra_fields")
        session_id = int(agent_data.extra_fields.get("web_osgym_session_id") or self._allocate_web_osgym_session_id())
        _trace_async_skd(
            "web_osgym_mixin.start_session_begin",
            request_id=getattr(agent_data, "request_id", None),
            session_id=session_id,
            task_id=task_id,
            include_a11y=include_a11y,
        )
        create_t0 = time.monotonic()
        instance_id, start_response = await tool.create(
            task_id=task_id,
            request_id=session_id,
            include_a11y=include_a11y,
        )
        create_ms = (time.monotonic() - create_t0) * 1000
        _trace_async_skd(
            "web_osgym_mixin.start_session_create_done",
            request_id=getattr(agent_data, "request_id", None),
            elapsed_ms=round(create_ms, 1),
            session_id=session_id,
            task_id=task_id,
            instance_id=instance_id,
            response_text_len=len(start_response.text or ""),
            image_count=0 if start_response.image is None else 1,
        )
        agent_data.extra_fields["web_osgym_instance_id"] = instance_id
        agent_data.extra_fields["web_osgym_task_id"] = task_id
        agent_data.extra_fields["web_osgym_session_id"] = session_id
        agent_data.extra_fields["web_osgym_include_a11y"] = include_a11y
        return start_response

    async def _finalize_with_web_osgym_reward(self, agent_data, termination_reason: str) -> None:
        if agent_data.extra_fields.get("web_osgym_reward_fetched"):
            return
        _trace_async_skd(
            "web_osgym_mixin.finalize_reward_begin",
            request_id=getattr(agent_data, "request_id", None),
            termination_reason=termination_reason,
        )
        self._ensure_web_osgym_session(agent_data)
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        reward_t0 = time.monotonic()
        reward = await tool.calc_reward(instance_id, termination_reason=termination_reason)
        reward_ms = (time.monotonic() - reward_t0) * 1000
        agent_data.extra_fields["web_osgym_reward_fetched"] = True
        agent_data.extra_fields["web_osgym_termination_reason"] = termination_reason
        agent_data.extra_fields["web_osgym_reward_score"] = float(reward)
        _trace_async_skd(
            "web_osgym_mixin.finalize_reward_done",
            request_id=getattr(agent_data, "request_id", None),
            elapsed_ms=round(reward_ms, 1),
            termination_reason=termination_reason,
            reward=float(reward),
        )

    async def _release_web_osgym_session(self, agent_data) -> None:
        instance_id = agent_data.extra_fields.get("web_osgym_instance_id")
        if instance_id is None:
            return
        tool = self._get_active_tool(agent_data)
        _trace_async_skd(
            "web_osgym_mixin.release_session_begin",
            request_id=getattr(agent_data, "request_id", None),
            instance_id=instance_id,
        )
        await tool.release(instance_id)
        _trace_async_skd(
            "web_osgym_mixin.release_session_done",
            request_id=getattr(agent_data, "request_id", None),
            instance_id=instance_id,
        )
