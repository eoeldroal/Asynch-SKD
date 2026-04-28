from __future__ import annotations

from copy import deepcopy
from uuid import uuid4


class WebOsGymLoopMixin:
    web_osgym_tool_name = "computer"

    def _get_active_tool(self, agent_data):
        active_tools = getattr(agent_data, "_active_tools", {})
        return active_tools[self.web_osgym_tool_name]

    def _allocate_web_osgym_session_id(self) -> int:
        return uuid4().int % (2**31 - 1) or 1

    def _get_create_kwargs(self, agent_data) -> dict:
        kwargs = agent_data.tools_kwargs.get(self.web_osgym_tool_name, {})
        return deepcopy(kwargs.get("create_kwargs", {}))

    def _ensure_web_osgym_session(self, agent_data) -> None:
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        tool = self._get_active_tool(agent_data)
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
