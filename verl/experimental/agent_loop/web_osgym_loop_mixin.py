from __future__ import annotations

from copy import deepcopy
from uuid import uuid4


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

    async def _start_web_osgym_session(self, agent_data, *, include_a11y: bool):
        tool = self._get_active_tool(agent_data)
        create_kwargs = self._get_create_kwargs(agent_data)
        task_id = create_kwargs.get("task_id") or agent_data.extra_fields.get("task_id")
        if task_id is None:
            raise ValueError("Web/OS gym session requires task_id in tools_kwargs.create_kwargs or agent_data.extra_fields")
        session_id = int(agent_data.extra_fields.get("web_osgym_session_id") or self._allocate_web_osgym_session_id())
        instance_id, start_response = await tool.create(
            task_id=task_id,
            request_id=session_id,
            include_a11y=include_a11y,
        )
        agent_data.extra_fields["web_osgym_instance_id"] = instance_id
        agent_data.extra_fields["web_osgym_task_id"] = task_id
        agent_data.extra_fields["web_osgym_session_id"] = session_id
        agent_data.extra_fields["web_osgym_include_a11y"] = include_a11y
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

    async def _release_web_osgym_session(self, agent_data) -> None:
        instance_id = agent_data.extra_fields.get("web_osgym_instance_id")
        if instance_id is None:
            return
        tool = self._get_active_tool(agent_data)
        await tool.release(instance_id)
