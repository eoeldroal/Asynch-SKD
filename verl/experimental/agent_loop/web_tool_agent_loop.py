from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("web_tool_agent")
class WebOsGymToolAgentLoop(WebOsGymLoopMixin, ToolAgentLoop):
    """ToolAgentLoop variant with a persistent Web/OSGym environment session.

    The generic ``ToolAgentLoop`` creates and releases a tool instance for each
    tool call. Web/OSGym trajectories need the opposite lifecycle: one session
    is created before the first model token, all Web/OSGym actions are applied to
    that same session, and the environment reward is fetched once at trajectory
    termination. This class keeps the normal ToolAgentLoop generation semantics
    while specializing only the Web/OSGym environment boundary.
    """

    def _split_env_observation(self, env_text: str | None, image_data: list[Any] | None) -> str:
        if not env_text:
            return ""
        if image_data:
            return ""

        # Visual observations are carried by screenshots. Image-less responses
        # are actionable feedback, typically malformed action or capture-failure
        # text, so the student must see them to recover.
        return env_text

    @staticmethod
    def _normalize_image_data(image_data: Any) -> list[Any] | None:
        if image_data is None:
            return None
        if isinstance(image_data, list):
            return [image for image in image_data if image is not None] or None
        return [image_data]

    @staticmethod
    def _prospective_image_data(agent_data: AgentData, new_images: list[Any] | None) -> list[Any] | None:
        image_data = list(agent_data.image_data) if agent_data.image_data is not None else None
        if new_images:
            if image_data is None:
                image_data = []
            image_data.extend(new_images)
        return image_data

    def _extend_image_data(self, agent_data: AgentData, image_data: list[Any] | None) -> None:
        if not image_data:
            return
        if agent_data.image_data is None:
            agent_data.image_data = []
        elif not isinstance(agent_data.image_data, list):
            agent_data.image_data = [agent_data.image_data]
        agent_data.image_data.extend(image_data)

    def _build_tool_message(self, tool_response_text: str | None, image_data: list[Any] | None) -> dict[str, Any]:
        if image_data:
            content = [{"type": "image"} for _ in image_data]
            if tool_response_text:
                content.append({"type": "text", "text": tool_response_text})
            return {"role": "tool", "content": content}
        return {"role": "tool", "content": tool_response_text or ""}

    async def _init_web_agent_data(self, **kwargs) -> AgentData:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )

        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        return agent_data

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        del sampling_params
        start_response = await self._start_web_osgym_session(agent_data, include_a11y=False)
        image_data = self._normalize_image_data(start_response.image)
        student_obs = self._split_env_observation(start_response.text, image_data)

        messages = deepcopy(agent_data.messages)
        if student_obs or image_data:
            messages.append(self._build_tool_message(student_obs, image_data))

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            messages,
            tools=schemas,
            images=self._prospective_image_data(agent_data, image_data),
            videos=agent_data.video_data,
        )

        self._extend_image_data(agent_data, image_data)
        agent_data.messages = messages
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        tool_call = agent_data.tool_calls[0]
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        result: dict[str, Any] = {
            "terminated": False,
            "termination_reason": None,
            "action_count": 0,
        }
        if tool_call.name not in active_tools:
            available = list(active_tools.keys())
            tool_response = ToolResponse(text=f"Unknown function '{tool_call.name}'. Available tools: {available}")
            result["invalid_action"] = True
        else:
            self._ensure_web_osgym_session(agent_data, tool_call.name)
            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                tool_response = ToolResponse(text=f"Invalid JSON in arguments for '{tool_call.name}': {exc}")
                result["invalid_action"] = True
            else:
                with simple_timer("tool_calls", agent_data.metrics):
                    tool = self._get_active_tool(agent_data, tool_call.name)
                    instance_id = agent_data.extra_fields["web_osgym_instance_id"]
                    tool_response, _, result = await tool.execute(instance_id, tool_args, agent_data=agent_data)

        agent_data.metrics["web_osgym/action_count"] = result.get("action_count", 0)
        if result.get("invalid_action"):
            agent_data.metrics["web_osgym/invalid_action"] = 1

        image_data = self._normalize_image_data(tool_response.image)
        student_obs = self._split_env_observation(tool_response.text, image_data)

        if not student_obs and not image_data:
            if result.get("terminated"):
                await self._finalize_with_web_osgym_reward(
                    agent_data,
                    termination_reason=result.get("termination_reason") or "model_done",
                )
                return AgentState.TERMINATED
            return AgentState.GENERATING

        message = self._build_tool_message(student_obs, image_data)
        response_ids = await self.apply_chat_template(
            [message],
            images=image_data,
            videos=None,
            remove_system_prompt=True,
        )

        # The environment observation is an atomic bundle. If admitting it would
        # exceed the response budget, end the trajectory and fetch reward without
        # partially committing text/image state.
        if response_ids and len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason="tool_response_budget_exhausted",
            )
            return AgentState.TERMINATED

        self._extend_image_data(agent_data, image_data)
        agent_data.messages.append(message)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        if result.get("terminated"):
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason=result.get("termination_reason") or "model_done",
            )
            return AgentState.TERMINATED

        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        next_state = await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
        )
        if next_state == AgentState.TERMINATED and "web_osgym_reward_score" not in agent_data.extra_fields:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        return next_state

    def _finalize_web_agent_output(self, agent_data: AgentData) -> AgentLoopOutput:
        if agent_data.response_mask:
            response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
            prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        else:
            response_ids = []
            prompt_ids = list(agent_data.prompt_ids)
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        reward_score = output.extra_fields.get("web_osgym_reward_score")
        if reward_score is not None:
            output.reward_score = float(reward_score)
        return output

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        agent_data = await self._init_web_agent_data(**kwargs)
        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    logger.error("Invalid state: %s", state)
                    state = AgentState.TERMINATED
            return self._finalize_web_agent_output(agent_data)
        finally:
            await self._release_web_osgym_session(agent_data)
