from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids


@register("web_skd_agent")
class WebSkdAgentLoop(WebOsGymLoopMixin, SkdAgentLoop):
    def _split_env_observation(self, env_text: str | None, image_data: list[Any] | None) -> tuple[str, str]:
        if not env_text:
            return "", ""
        if image_data:
            return "", env_text

        # Image-less Web/OSGym responses are treated as non-visual environment
        # feedback, not teacher-only accessibility context. In practice this is
        # how the environment reports action failures or screenshot capture
        # failures: the student must see that text to recover on the next turn,
        # while the teacher still sees the same feedback for alignment. Normal
        # visual observations keep text teacher-only because that text is the
        # privileged a11y/auxiliary channel.
        return env_text, env_text

    def _extend_image_data(self, agent_data: AgentData, image_data: list[Any] | None) -> None:
        if not image_data:
            return
        if agent_data.image_data is None:
            agent_data.image_data = []
        agent_data.image_data.extend(image_data)

    def _build_tool_message(self, tool_response_text: str | None, image_data: list[Any] | None):
        if image_data:
            content = [{"type": "image"}]
            if tool_response_text:
                content.append({"type": "text", "text": tool_response_text})
            return {"role": "tool", "content": content}
        return {"role": "tool", "content": tool_response_text or ""}

    async def _recompute_teacher_prompt_ids(self, agent_data: AgentData) -> list[int]:
        teacher_messages = deepcopy(agent_data.extra_fields.get("web_osgym_teacher_messages", []))
        teacher_messages = self._build_teacher_messages(teacher_messages)
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self.apply_chat_template(
            teacher_messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )

    async def _apply_server_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        *,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        """Tokenize the chat for SGLang's multimodal server contract.

        ``AgentLoopBase.apply_chat_template`` uses the HF processor when one is
        available. For Qwen VL processors that expands one logical image marker
        into many image token ids, which is the right representation for local
        training tensors. SGLang's token-in/token-out multimodal endpoint,
        however, expects the unexpanded chat-template ids plus the actual images
        in ``image_data``. Keeping this server-side representation separate
        prevents real Web/OSGym screenshots from being counted as hundreds of
        independent images during teacher logprob requests.
        """
        tokenized = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        prompt_ids = normalize_token_ids(tokenized)
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

    async def _recompute_server_prompt_ids(self, agent_data: AgentData, messages: list[dict]) -> list[int]:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self._apply_server_chat_template(messages, tools=schemas)

    async def _recompute_teacher_server_prompt_ids(self, agent_data: AgentData) -> list[int]:
        teacher_messages = deepcopy(agent_data.extra_fields.get("web_osgym_teacher_messages", []))
        teacher_messages = self._build_teacher_messages(teacher_messages)
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self._apply_server_chat_template(teacher_messages, tools=schemas)

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        del sampling_params
        start_response = await self._start_web_osgym_session(agent_data, include_a11y=True)
        student_obs, teacher_obs = self._split_env_observation(start_response.text, start_response.image)
        self._extend_image_data(agent_data, start_response.image)

        base_messages = deepcopy(agent_data.messages)
        teacher_messages = deepcopy(base_messages)

        if student_obs or start_response.image:
            agent_data.messages.append(self._build_tool_message(student_obs, start_response.image))
        if teacher_obs or start_response.image:
            teacher_messages.append(self._build_tool_message(teacher_obs, start_response.image))

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        agent_data.prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.extra_fields["server_prompt_ids"] = await self._recompute_server_prompt_ids(
            agent_data, agent_data.messages
        )
        agent_data.extra_fields["web_osgym_teacher_messages"] = teacher_messages
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs
        agent_data.extra_fields["teacher_prompt_ids"] = await self._recompute_teacher_prompt_ids(agent_data)
        agent_data.extra_fields["teacher_server_prompt_ids"] = await self._recompute_teacher_server_prompt_ids(agent_data)
        return AgentState.GENERATING

    def _restore_partial_state(self, partial_state):
        agent_data, next_state = super()._restore_partial_state(partial_state)
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas
        return agent_data, next_state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        tool_call = agent_data.tool_calls[0]
        tool_args = json.loads(tool_call.arguments)
        self._ensure_web_osgym_session(agent_data)
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        tool_response, _, result = await tool.execute(instance_id, tool_args, agent_data=agent_data)
        agent_data.metrics["web_osgym/action_count"] = result.get("action_count", 0)

        self._extend_image_data(agent_data, tool_response.image)
        student_obs, teacher_obs = self._split_env_observation(tool_response.text, tool_response.image)

        appended_len = 0
        if student_obs or tool_response.image:
            student_message = self._build_tool_message(student_obs, tool_response.image)
            agent_data.messages.append(student_message)
            response_ids = await self.apply_chat_template(
                [student_message],
                images=tool_response.image if tool_response.image else None,
                videos=None,
                remove_system_prompt=True,
            )
            server_response_ids = await self._apply_server_chat_template(
                [student_message],
                remove_system_prompt=True,
            )
            appended_len = len(response_ids)
            agent_data.prompt_ids += response_ids
            agent_data.extra_fields.setdefault("server_prompt_ids", list(agent_data.prompt_ids[:-appended_len]))
            agent_data.extra_fields["server_prompt_ids"].extend(server_response_ids)
            agent_data.response_mask += [0] * appended_len
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * appended_len
            agent_data.user_turns += 1

        teacher_messages = deepcopy(agent_data.extra_fields.get("web_osgym_teacher_messages", []))
        if teacher_obs or tool_response.image:
            teacher_message = self._build_tool_message(teacher_obs, tool_response.image)
            teacher_messages.append(teacher_message)
            teacher_response_ids = await self.apply_chat_template(
                [teacher_message],
                images=tool_response.image if tool_response.image else None,
                videos=None,
                remove_system_prompt=True,
            )
            teacher_server_response_ids = await self._apply_server_chat_template(
                [teacher_message],
                remove_system_prompt=True,
            )
            teacher_prompt_ids = agent_data.extra_fields.setdefault("teacher_prompt_ids", [])
            agent_data.extra_fields.setdefault("teacher_server_prompt_ids", list(teacher_prompt_ids))
            teacher_prompt_ids.extend(teacher_response_ids)
            agent_data.extra_fields["teacher_server_prompt_ids"].extend(teacher_server_response_ids)
        agent_data.extra_fields["web_osgym_teacher_messages"] = teacher_messages
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs

        self._append_dummy_teacher_rows(agent_data, appended_len)
        self._assert_teacher_alignment(agent_data)
        if appended_len > 0:
            self._increment_skd_prefix_stats(agent_data, env_units=1, tokens=appended_len)

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
        stop_after_skd_chunk: bool = False,
    ) -> AgentState:
        next_state = await super()._handle_generating_state(
            agent_data,
            sampling_params,
            ignore_termination=ignore_termination,
            stop_after_skd_chunk=stop_after_skd_chunk,
        )
        if next_state == AgentState.TERMINATED and "web_osgym_reward_score" not in agent_data.extra_fields:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        return next_state

    def _finalize_boundary_agent_output(self, agent_data: AgentData) -> AgentLoopOutput:
        output = super()._finalize_boundary_agent_output(agent_data)
        reward_score = output.extra_fields.get("web_osgym_reward_score")
        if reward_score is not None:
            output.reward_score = float(reward_score)
        return output

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        agent_data = await self._init_boundary_agent_data(**kwargs)
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
                    state = AgentState.TERMINATED
            return self._finalize_boundary_agent_output(agent_data)
        finally:
            await self._release_web_osgym_session(agent_data)
            await self._release_teacher_sticky_session(agent_data.request_id)

    async def run_until_exportable_boundary(
        self,
        sampling_params: dict[str, Any],
        *,
        sample_id: str,
        logical_step: int,
        source_type: str,
        partial_state=None,
        **kwargs: Any,
    ):
        if partial_state is None:
            agent_data = await self._init_boundary_agent_data(**kwargs)
            state = AgentState.PENDING
        else:
            agent_data, state = self._restore_partial_state(partial_state)

        try:
            next_state = await self._run_until_exportable_boundary(agent_data, state, sampling_params)
            if next_state == AgentState.TERMINATED:
                await self._release_teacher_sticky_session(agent_data.request_id)
                await self._release_web_osgym_session(agent_data)
                return self._finalize_boundary_agent_output(agent_data)

            return self._export_partial_state(
                agent_data,
                next_state,
                sample_id=sample_id,
                logical_step=logical_step,
                source_type=source_type,
            )
        except Exception:
            await self._release_teacher_sticky_session(agent_data.request_id)
            await self._release_web_osgym_session(agent_data)
            raise

    async def run_from_partial_to_completion(
        self,
        sampling_params: dict[str, Any],
        *,
        partial_state,
    ) -> AgentLoopOutput:
        agent_data, state = self._restore_partial_state(partial_state)
        parent_request_id = agent_data.request_id
        agent_data.extra_fields["parent_request_id"] = parent_request_id
        agent_data.request_id = uuid4().hex
        try:
            await self._run_until_terminated(agent_data, state, sampling_params)
            return self._finalize_boundary_agent_output(agent_data)
        finally:
            await self._release_web_osgym_session(agent_data)
            for request_id in (agent_data.request_id, parent_request_id):
                await self._release_teacher_sticky_session(request_id)
