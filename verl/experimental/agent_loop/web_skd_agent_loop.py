from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.skd_agent_loop import (
    _SKD_PENDING_TURN_CHUNKS,
    SkdAgentLoop,
    SkdTurnChunkState,
    _safe_len,
    _trace_async_skd,
)
from verl.experimental.agent_loop.tool_agent_loop import (
    _TOOL_PARSE_ERROR_RETRY_COUNT_KEY,
    AgentData,
    AgentState,
)
from verl.experimental.agent_loop.tool_parser import ToolParseError
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
            content = [{"type": "image"} for _ in image_data]
            if tool_response_text:
                content.append({"type": "text", "text": tool_response_text})
            return {"role": "tool", "content": content}
        return {"role": "tool", "content": tool_response_text or ""}

    def _require_prompt_stream(self, agent_data: AgentData, key: str) -> list[int]:
        stream = agent_data.extra_fields.get(key)
        if not isinstance(stream, list):
            raise ValueError(f"WebSKD requires {key} to be initialized before committing tool observations")
        return stream

    @staticmethod
    def _require_teacher_messages(agent_data: AgentData) -> list[dict]:
        teacher_messages = agent_data.extra_fields.get("web_osgym_teacher_messages")
        if not isinstance(teacher_messages, list) or not teacher_messages:
            raise ValueError("WebSKD requires web_osgym_teacher_messages to be initialized before request rebuild")
        return teacher_messages

    @staticmethod
    def _prospective_image_data(agent_data: AgentData, new_images: list[Any] | None) -> list[Any] | None:
        image_data = list(agent_data.image_data) if agent_data.image_data is not None else None
        if new_images:
            if image_data is None:
                image_data = []
            image_data.extend(new_images)
        return image_data

    @staticmethod
    def _multimodal_prefix_surplus_delta(
        expanded_ids: list[int],
        server_ids: list[int],
        image_data: list[Any] | None,
    ) -> int:
        """Return only the multimodal expansion gap for a committed observation.

        Teacher-only text can make the teacher prefix longer than the student
        prefix, but it is still real SGLang input and must not be treated as a
        coordinate surplus. Only observations that actually carry images can
        increase the SGLang-expanded coordinate offset.
        """
        if not image_data:
            return 0
        surplus = len(expanded_ids) - len(server_ids)
        if surplus < 0:
            raise ValueError(
                "WebSKD multimodal expansion gap must be non-negative: "
                f"expanded_ids={len(expanded_ids)} server_ids={len(server_ids)}"
            )
        return surplus

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
        _trace_async_skd(
            "web_skd.server_chat_template_begin",
            message_count=len(messages),
            tool_count=_safe_len(tools),
            remove_system_prompt=remove_system_prompt,
        )
        template_t0 = time.monotonic()
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
        template_ms = (time.monotonic() - template_t0) * 1000
        normalize_t0 = time.monotonic()
        prompt_ids = normalize_token_ids(tokenized)
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        normalize_ms = (time.monotonic() - normalize_t0) * 1000
        _trace_async_skd(
            "web_skd.server_chat_template_done",
            elapsed_ms=round(template_ms + normalize_ms, 1),
            template_ms=round(template_ms, 1),
            normalize_ms=round(normalize_ms, 1),
            prompt_len=len(prompt_ids),
            message_count=len(messages),
            tool_count=_safe_len(tools),
            remove_system_prompt=remove_system_prompt,
        )
        return prompt_ids

    async def _recompute_server_prompt_ids(self, agent_data: AgentData, messages: list[dict]) -> list[int]:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self._apply_server_chat_template(messages, tools=schemas)

    async def _recompute_teacher_server_prompt_ids(
        self,
        agent_data: AgentData,
        teacher_messages: list[dict],
    ) -> list[int]:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self._apply_server_chat_template(teacher_messages, tools=schemas)

    async def _recompute_teacher_prompt_ids(
        self,
        agent_data: AgentData,
        teacher_messages: list[dict],
        image_data: list[Any] | None,
    ) -> list[int]:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        return await self.apply_chat_template(
            teacher_messages,
            tools=schemas,
            images=image_data,
            videos=agent_data.video_data,
        )

    async def _build_request_prompt_views(
        self,
        agent_data: AgentData,
        *,
        student_messages: list[dict],
        teacher_messages: list[dict],
        teacher_prompt_ids: list[int],
        image_data: list[Any] | None,
    ) -> tuple[list[int], list[int], int]:
        server_prompt_ids = await self._recompute_server_prompt_ids(agent_data, student_messages)
        teacher_server_prompt_ids = await self._recompute_teacher_server_prompt_ids(agent_data, teacher_messages)
        teacher_sglang_prefix_surplus = self._multimodal_prefix_surplus_delta(
            teacher_prompt_ids,
            teacher_server_prompt_ids,
            image_data,
        )
        return server_prompt_ids, teacher_server_prompt_ids, teacher_sglang_prefix_surplus

    async def _build_request_prompt_views_from_turn_state(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> tuple[list[int], list[int], list[int], int]:
        student_messages, teacher_messages, _committed_teacher_prompt_ids, image_data = (
            self._resolve_request_prompt_inputs_from_agent_state(agent_data)
        )
        server_prompt_ids = await self._recompute_server_prompt_ids(agent_data, student_messages)
        committed_teacher_prompt_ids = await self._recompute_teacher_prompt_ids(
            agent_data,
            teacher_messages,
            image_data,
        )
        teacher_server_prompt_ids = await self._recompute_teacher_server_prompt_ids(
            agent_data,
            teacher_messages,
        )
        teacher_sglang_prefix_surplus = self._multimodal_prefix_surplus_delta(
            committed_teacher_prompt_ids,
            teacher_server_prompt_ids,
            image_data,
        )
        turn_tokens = list(turn_state.tokens)
        teacher_prompt_ids = list(committed_teacher_prompt_ids) + turn_tokens
        return (
            list(server_prompt_ids) + turn_tokens,
            teacher_prompt_ids,
            list(teacher_server_prompt_ids) + turn_tokens,
            teacher_sglang_prefix_surplus,
        )

    async def _build_student_request_prompt_ids(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> list[int]:
        student_request_prompt_ids, _, _, _ = await self._build_request_prompt_views_from_turn_state(
            agent_data,
            turn_state,
        )
        return student_request_prompt_ids

    async def _build_teacher_verify_request_view(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> tuple[list[int], list[int], int]:
        _, teacher_prompt_ids, teacher_server_prompt_ids, teacher_sglang_prefix_surplus = (
            await self._build_request_prompt_views_from_turn_state(agent_data, turn_state)
        )
        return teacher_prompt_ids, teacher_server_prompt_ids, teacher_sglang_prefix_surplus

    def _resolve_request_prompt_inputs_from_agent_state(
        self,
        agent_data: AgentData,
    ) -> tuple[list[dict], list[dict], list[int], list[Any] | None]:
        student_messages = list(agent_data.messages)
        teacher_messages = deepcopy(self._require_teacher_messages(agent_data))
        teacher_messages = self._build_teacher_messages(deepcopy(teacher_messages))
        teacher_prompt_ids = self._require_prompt_stream(agent_data, "teacher_prompt_ids")
        image_data = agent_data.image_data
        return student_messages, teacher_messages, teacher_prompt_ids, image_data

    def _assert_processing_tools_turn_is_committed(self, agent_data: AgentData) -> None:
        pending_turn_state = self._get_pending_turn_state(agent_data)
        pending_turn_chunks = int(agent_data.extra_fields.get(_SKD_PENDING_TURN_CHUNKS, 0))
        if pending_turn_state.tokens or pending_turn_chunks:
            raise ValueError("WebSKD PROCESSING_TOOLS requires a completed assistant turn before tool execution")

    def _resolve_tool_processing_commit_inputs(
        self,
        agent_data: AgentData,
    ) -> tuple[list[dict], list[dict], list[int]]:
        student_messages = list(agent_data.messages)
        teacher_messages = deepcopy(self._require_teacher_messages(agent_data))
        teacher_prompt_ids = list(self._require_prompt_stream(agent_data, "teacher_prompt_ids"))
        return student_messages, teacher_messages, teacher_prompt_ids

    def _validate_teacher_state_for_partial(self, agent_data: AgentData) -> None:
        self._require_teacher_messages(agent_data)
        self._require_prompt_stream(agent_data, "teacher_prompt_ids")

    async def _handle_tool_parse_error(self, agent_data: AgentData, parse_error: ToolParseError) -> AgentState:
        retry_count = int(agent_data.extra_fields.get(_TOOL_PARSE_ERROR_RETRY_COUNT_KEY, 0))
        if retry_count >= self.max_tool_parse_error_retries:
            return AgentState.TERMINATED

        feedback_text = self._build_tool_parse_error_feedback(parse_error)
        student_obs, teacher_obs = self._split_env_observation(feedback_text, None)
        student_message = self._build_tool_message(student_obs, None)
        teacher_message = self._build_tool_message(teacher_obs, None)

        response_ids = await self.apply_chat_template(
            [student_message],
            images=None,
            videos=None,
            remove_system_prompt=True,
        )
        teacher_response_ids = await self.apply_chat_template(
            [teacher_message],
            images=None,
            videos=None,
            remove_system_prompt=True,
        )
        if response_ids and len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="tool_response_budget_exhausted")
            return AgentState.TERMINATED

        next_student_messages, next_teacher_messages, committed_teacher_prompt_ids = (
            self._resolve_tool_processing_commit_inputs(agent_data)
        )
        next_student_messages.append(student_message)
        next_teacher_messages.append(teacher_message)
        next_teacher_prompt_ids = list(committed_teacher_prompt_ids)
        next_teacher_prompt_ids.extend(teacher_response_ids)

        (
            next_server_prompt_ids,
            next_teacher_server_prompt_ids,
            next_teacher_sglang_prefix_surplus,
        ) = await self._build_request_prompt_views(
            agent_data,
            student_messages=next_student_messages,
            teacher_messages=self._build_teacher_messages(deepcopy(next_teacher_messages)),
            teacher_prompt_ids=next_teacher_prompt_ids,
            image_data=agent_data.image_data,
        )
        if next_teacher_server_prompt_ids is not None and await self._terminate_if_teacher_prefix_overflows(
            agent_data,
            prefix_len=len(next_teacher_server_prompt_ids),
            stage="tool_parse_error",
        ):
            return AgentState.TERMINATED

        appended_len = len(response_ids)
        next_prompt_ids = list(agent_data.prompt_ids) + response_ids
        next_response_mask = list(agent_data.response_mask) + ([0] * appended_len)
        next_response_logprobs = list(agent_data.response_logprobs)
        if next_response_logprobs:
            next_response_logprobs.extend([0.0] * appended_len)

        next_teacher_ids_list = None
        next_teacher_logprobs_list = None
        if "teacher_ids_list" in agent_data.extra_fields or "teacher_logprobs_list" in agent_data.extra_fields:
            if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
                raise ValueError("WebSKD teacher alignment requires both teacher_ids_list and teacher_logprobs_list")
            assert self.loss_top_k is not None and self.loss_top_k > 0, "SKD dummy rows require distillation topk > 0"
            next_teacher_ids_list = [list(row) for row in agent_data.extra_fields["teacher_ids_list"]]
            next_teacher_logprobs_list = [list(row) for row in agent_data.extra_fields["teacher_logprobs_list"]]
            next_teacher_ids_list.extend([[0] * self.loss_top_k for _ in range(appended_len)])
            next_teacher_logprobs_list.extend([[0.0] * self.loss_top_k for _ in range(appended_len)])

        agent_data.metrics["tool_parse_error"] = 1
        agent_data.extra_fields[_TOOL_PARSE_ERROR_RETRY_COUNT_KEY] = retry_count + 1
        agent_data.messages = next_student_messages
        agent_data.prompt_ids = next_prompt_ids
        agent_data.response_mask = next_response_mask
        if next_response_logprobs:
            agent_data.response_logprobs = next_response_logprobs
        agent_data.user_turns += 1
        agent_data.extra_fields["web_osgym_teacher_messages"] = next_teacher_messages
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs
        agent_data.extra_fields["teacher_prompt_ids"] = next_teacher_prompt_ids
        agent_data.extra_fields.pop("server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_sglang_prefix_surplus", None)
        if next_teacher_ids_list is not None and next_teacher_logprobs_list is not None:
            agent_data.extra_fields["teacher_ids_list"] = next_teacher_ids_list
            agent_data.extra_fields["teacher_logprobs_list"] = next_teacher_logprobs_list
        return AgentState.GENERATING

    async def _terminate_if_teacher_prefix_overflows(
        self,
        agent_data: AgentData,
        *,
        prefix_len: int,
        stage: str,
    ) -> bool:
        """Terminate before committing a teacher prefix that cannot be verified further."""
        teacher_routing_key = agent_data.extra_fields.get("teacher_routing_key")
        teacher_overflow, teacher_max_model_len, teacher_required_len = self._teacher_future_verify_overflows(
            prefix_len=prefix_len,
            routing_key=teacher_routing_key,
        )
        if not teacher_overflow:
            return False

        termination_reason = "teacher_context_exhausted"
        agent_data.extra_fields["skd_termination_reason"] = termination_reason
        agent_data.extra_fields["skd_teacher_context_required_len"] = teacher_required_len
        agent_data.extra_fields["skd_teacher_max_model_len"] = teacher_max_model_len
        agent_data.extra_fields["skd_teacher_context_exhausted_stage"] = stage
        warnings.warn(
            "[WebSKD] terminating before committing teacher observation because teacher context would overflow: "
            f"req={agent_data.request_id} "
            f"stage={stage} "
            f"prefix_len={prefix_len} "
            f"required_len={teacher_required_len} "
            f"max_model_len={teacher_max_model_len}",
            stacklevel=1,
        )
        await self._finalize_with_web_osgym_reward(agent_data, termination_reason=termination_reason)
        return True

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        del sampling_params
        _trace_async_skd(
            "web_skd.pending_begin",
            request_id=agent_data.request_id,
            prompt_len=len(agent_data.prompt_ids),
            image_count=_safe_len(agent_data.image_data),
        )
        start_t0 = time.monotonic()
        start_response = await self._start_web_osgym_session(agent_data, include_a11y=True)
        start_ms = (time.monotonic() - start_t0) * 1000
        _trace_async_skd(
            "web_skd.pending_start_session_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(start_ms, 1),
            response_text_len=len(start_response.text or ""),
            image_count=_safe_len(start_response.image),
        )
        student_obs, teacher_obs = self._split_env_observation(start_response.text, start_response.image)
        next_image_data = self._prospective_image_data(agent_data, start_response.image)
        _trace_async_skd(
            "web_skd.pending_observation_split_done",
            request_id=agent_data.request_id,
            student_obs_len=len(student_obs or ""),
            teacher_obs_len=len(teacher_obs or ""),
            start_image_count=_safe_len(start_response.image),
            next_image_count=_safe_len(next_image_data),
        )

        base_messages = deepcopy(agent_data.messages)
        student_messages = deepcopy(base_messages)
        teacher_messages = deepcopy(base_messages)

        if student_obs or start_response.image:
            student_messages.append(self._build_tool_message(student_obs, start_response.image))
        if teacher_obs or start_response.image:
            teacher_messages.append(self._build_tool_message(teacher_obs, start_response.image))

        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        _trace_async_skd(
            "web_skd.pending_student_template_begin",
            request_id=agent_data.request_id,
            message_count=len(student_messages),
            tool_count=_safe_len(schemas),
            image_count=_safe_len(next_image_data),
        )
        student_template_t0 = time.monotonic()
        prompt_ids = await self.apply_chat_template(
            student_messages,
            tools=schemas,
            images=next_image_data,
            videos=agent_data.video_data,
        )
        student_template_ms = (time.monotonic() - student_template_t0) * 1000
        _trace_async_skd(
            "web_skd.pending_student_template_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(student_template_ms, 1),
            prompt_len=len(prompt_ids),
            image_count=_safe_len(next_image_data),
        )
        teacher_template_messages = self._build_teacher_messages(deepcopy(teacher_messages))
        _trace_async_skd(
            "web_skd.pending_teacher_template_begin",
            request_id=agent_data.request_id,
            message_count=len(teacher_template_messages),
            tool_count=_safe_len(schemas),
            image_count=_safe_len(next_image_data),
        )
        teacher_template_t0 = time.monotonic()
        teacher_prompt_ids = await self.apply_chat_template(
            teacher_template_messages,
            tools=schemas,
            images=next_image_data,
            videos=agent_data.video_data,
        )
        teacher_template_ms = (time.monotonic() - teacher_template_t0) * 1000
        _trace_async_skd(
            "web_skd.pending_teacher_template_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(teacher_template_ms, 1),
            teacher_prompt_len=len(teacher_prompt_ids),
            image_count=_safe_len(next_image_data),
        )
        commit_t0 = time.monotonic()
        image_start = 0
        image_end = _safe_len(next_image_data)
        agent_data.image_data = next_image_data
        agent_data.messages = student_messages
        agent_data.prompt_ids = prompt_ids
        if image_end > image_start:
            agent_data.extra_fields["mini_step_image_spans"] = [
                {
                    "step_idx": 1,
                    "image_start": image_start,
                    "image_end": image_end,
                    "terminal": False,
                }
            ]
        agent_data.extra_fields["web_osgym_teacher_messages"] = teacher_messages
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs
        agent_data.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
        agent_data.extra_fields.pop("server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_sglang_prefix_surplus", None)
        commit_ms = (time.monotonic() - commit_t0) * 1000
        _trace_async_skd(
            "web_skd.pending_commit_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(commit_ms, 1),
            prompt_len=len(agent_data.prompt_ids),
            teacher_prompt_len=len(teacher_prompt_ids),
            image_count=_safe_len(agent_data.image_data),
        )
        return AgentState.GENERATING

    def _restore_partial_state(self, partial_state):
        agent_data, next_state = super()._restore_partial_state(partial_state)
        agent_data._active_tools = self.tools
        agent_data._active_tool_schemas = self.tool_schemas
        return agent_data, next_state

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        self._assert_processing_tools_turn_is_committed(agent_data)
        tool_call_names = [getattr(tool_call, "name", None) for tool_call in agent_data.tool_calls]
        _trace_async_skd(
            "web_skd.tool_processing_begin",
            request_id=agent_data.request_id,
            response_len=len(agent_data.response_mask),
            image_count=_safe_len(agent_data.image_data),
            user_turns=agent_data.user_turns,
            assistant_turns=agent_data.assistant_turns,
            tool_calls_len=len(agent_data.tool_calls),
            tool_call_names=tool_call_names,
        )
        tool_t0 = time.monotonic()
        _trace_async_skd(
            "web_skd.tool_processing_execute_begin",
            request_id=agent_data.request_id,
            tool_calls_len=len(agent_data.tool_calls),
            tool_call_names=tool_call_names,
        )
        tool_response, _, result = await self._execute_web_osgym_tool_calls(agent_data)
        tool_ms = (time.monotonic() - tool_t0) * 1000
        agent_data.metrics["web_osgym/action_count"] = result.get("action_count", 0)
        if result.get("invalid_action"):
            agent_data.metrics["web_osgym/invalid_action"] = 1
        _trace_async_skd(
            "web_skd.tool_processing_tool_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(tool_ms, 1),
            action_count=result.get("action_count", 0),
            invalid_action=result.get("invalid_action", False),
            terminated=result.get("terminated", False),
            termination_reason=result.get("termination_reason"),
            response_text_len=len(tool_response.text or ""),
            image_count=_safe_len(tool_response.image),
        )

        if result.get("terminated"):
            termination_reason = result.get("termination_reason") or "model_done"
            _trace_async_skd(
                "web_skd.tool_processing_terminated",
                request_id=agent_data.request_id,
                appended_len=0,
                response_len=len(agent_data.response_mask),
                termination_reason=termination_reason,
                image_count=_safe_len(agent_data.image_data),
            )
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason=termination_reason,
            )
            return AgentState.TERMINATED

        student_obs, teacher_obs = self._split_env_observation(tool_response.text, tool_response.image)
        image_data = tool_response.image if tool_response.image else None
        _trace_async_skd(
            "web_skd.tool_processing_split_done",
            request_id=agent_data.request_id,
            student_obs_len=len(student_obs or ""),
            teacher_obs_len=len(teacher_obs or ""),
            response_image_count=_safe_len(tool_response.image),
            committed_image_count=_safe_len(agent_data.image_data),
        )

        student_message = None
        response_ids: list[int] = []
        if student_obs or image_data:
            student_message = self._build_tool_message(student_obs, image_data)
            student_template_t0 = time.monotonic()
            _trace_async_skd(
                "web_skd.tool_processing_student_template_begin",
                request_id=agent_data.request_id,
                image_count=_safe_len(image_data),
                text_len=len(student_obs or ""),
            )
            response_ids = await self.apply_chat_template(
                [student_message],
                images=image_data,
                videos=None,
                remove_system_prompt=True,
            )
            student_template_ms = (time.monotonic() - student_template_t0) * 1000
            _trace_async_skd(
                "web_skd.tool_processing_student_template_done",
                request_id=agent_data.request_id,
                template_ms=round(student_template_ms, 1),
                response_ids_len=len(response_ids),
                image_count=_safe_len(image_data),
                text_len=len(student_obs or ""),
            )

        teacher_message = None
        teacher_response_ids: list[int] = []
        if teacher_obs or image_data:
            teacher_message = self._build_tool_message(teacher_obs, image_data)
            teacher_template_t0 = time.monotonic()
            _trace_async_skd(
                "web_skd.tool_processing_teacher_template_begin",
                request_id=agent_data.request_id,
                image_count=_safe_len(image_data),
                text_len=len(teacher_obs or ""),
            )
            teacher_response_ids = await self.apply_chat_template(
                [teacher_message],
                images=image_data,
                videos=None,
                remove_system_prompt=True,
            )
            teacher_template_ms = (time.monotonic() - teacher_template_t0) * 1000
            _trace_async_skd(
                "web_skd.tool_processing_teacher_template_done",
                request_id=agent_data.request_id,
                template_ms=round(teacher_template_ms, 1),
                response_ids_len=len(teacher_response_ids),
                image_count=_safe_len(image_data),
                text_len=len(teacher_obs or ""),
            )

        # Treat the environment observation as one atomic bundle. The local
        # prompt ids, server prompt ids, teacher prompt ids, teacher messages,
        # response mask, and image_data must either all advance together or all
        # remain unchanged; otherwise Qwen-VL postprocessing can see image
        # tensors whose image tokens were never admitted into the final
        # sequence.
        if response_ids and len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            _trace_async_skd(
                "web_skd.tool_processing_budget_exhausted",
                request_id=agent_data.request_id,
                response_len=len(agent_data.response_mask),
                appended_response_ids_len=len(response_ids),
                response_limit=self.response_length,
            )
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="tool_response_budget_exhausted")
            return AgentState.TERMINATED

        (
            next_student_messages,
            next_teacher_messages,
            committed_teacher_prompt_ids,
        ) = self._resolve_tool_processing_commit_inputs(agent_data)
        if student_message is not None:
            next_student_messages.append(student_message)

        if teacher_message is not None:
            next_teacher_messages.append(teacher_message)

        next_teacher_prompt_ids = list(committed_teacher_prompt_ids)
        if teacher_message is not None and teacher_response_ids:
            next_teacher_prompt_ids.extend(teacher_response_ids)

        next_image_data = self._prospective_image_data(agent_data, image_data)

        next_server_prompt_ids: list[int] | None = None
        next_teacher_server_prompt_ids: list[int] | None = None
        next_teacher_sglang_prefix_surplus: int | None = None
        if student_message is not None or teacher_message is not None:
            rebuild_t0 = time.monotonic()
            (
                next_server_prompt_ids,
                next_teacher_server_prompt_ids,
                next_teacher_sglang_prefix_surplus,
            ) = await self._build_request_prompt_views(
                agent_data,
                student_messages=next_student_messages,
                teacher_messages=self._build_teacher_messages(deepcopy(next_teacher_messages)),
                teacher_prompt_ids=next_teacher_prompt_ids,
                image_data=next_image_data,
            )
            _trace_async_skd(
                "web_skd.tool_processing_rebuild_request_views_done",
                request_id=agent_data.request_id,
                elapsed_ms=round((time.monotonic() - rebuild_t0) * 1000, 1),
                server_prompt_len=len(next_server_prompt_ids),
                teacher_server_prompt_len=len(next_teacher_server_prompt_ids),
                teacher_sglang_prefix_surplus=next_teacher_sglang_prefix_surplus,
            )

        if next_teacher_server_prompt_ids is not None and await self._terminate_if_teacher_prefix_overflows(
            agent_data,
            prefix_len=len(next_teacher_server_prompt_ids),
            stage="tool_observation",
        ):
            return AgentState.TERMINATED

        appended_len = 0
        if student_message is not None:
            appended_len = len(response_ids)
        next_prompt_ids = list(agent_data.prompt_ids) + response_ids
        next_response_mask = list(agent_data.response_mask) + ([0] * appended_len)
        next_response_logprobs = list(agent_data.response_logprobs)
        if next_response_logprobs:
            next_response_logprobs.extend([0.0] * appended_len)
        next_user_turns = agent_data.user_turns + (1 if student_message is not None else 0)

        next_mini_step_image_spans = None
        if image_data:
            image_start = _safe_len(agent_data.image_data)
            image_end = _safe_len(next_image_data)
            next_mini_step_image_spans = list(agent_data.extra_fields.get("mini_step_image_spans") or [])
            next_mini_step_image_spans.append(
                {
                    "step_idx": int(agent_data.assistant_turns) + 1,
                    "image_start": image_start,
                    "image_end": image_end,
                    "terminal": False,
                }
            )

        next_teacher_ids_list = None
        next_teacher_logprobs_list = None
        if "teacher_ids_list" in agent_data.extra_fields or "teacher_logprobs_list" in agent_data.extra_fields:
            if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
                raise ValueError("WebSKD teacher alignment requires both teacher_ids_list and teacher_logprobs_list")
            assert self.loss_top_k is not None and self.loss_top_k > 0, "SKD dummy rows require distillation topk > 0"
            next_teacher_ids_list = [list(row) for row in agent_data.extra_fields["teacher_ids_list"]]
            next_teacher_logprobs_list = [list(row) for row in agent_data.extra_fields["teacher_logprobs_list"]]
            next_teacher_ids_list.extend([[0] * self.loss_top_k for _ in range(appended_len)])
            next_teacher_logprobs_list.extend([[0.0] * self.loss_top_k for _ in range(appended_len)])
            if len(next_teacher_ids_list) != len(next_response_mask):
                raise AssertionError(
                    f"[SKD] teacher_ids_list length {len(next_teacher_ids_list)} != response_mask length {len(next_response_mask)}"
                )
            if len(next_teacher_logprobs_list) != len(next_response_mask):
                raise AssertionError(
                    "[SKD] teacher_logprobs_list length "
                    f"{len(next_teacher_logprobs_list)} != response_mask length {len(next_response_mask)}"
                )

        commit_t0 = time.monotonic()
        agent_data.messages = next_student_messages
        agent_data.image_data = next_image_data
        agent_data.prompt_ids = next_prompt_ids
        agent_data.response_mask = next_response_mask
        if next_response_logprobs:
            agent_data.response_logprobs = next_response_logprobs
        agent_data.user_turns = next_user_turns
        # Keep the raw teacher conversation as the durable state. Compact
        # request ids and the teacher verify offset are rebuilt from this
        # canonical state at request time instead of living as durable state.
        agent_data.extra_fields["web_osgym_teacher_messages"] = next_teacher_messages
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs
        agent_data.extra_fields["teacher_prompt_ids"] = next_teacher_prompt_ids
        agent_data.extra_fields.pop("server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_server_prompt_ids", None)
        agent_data.extra_fields.pop("teacher_sglang_prefix_surplus", None)
        if next_mini_step_image_spans is not None:
            agent_data.extra_fields["mini_step_image_spans"] = next_mini_step_image_spans
        if next_teacher_ids_list is not None and next_teacher_logprobs_list is not None:
            agent_data.extra_fields["teacher_ids_list"] = next_teacher_ids_list
            agent_data.extra_fields["teacher_logprobs_list"] = next_teacher_logprobs_list
        if appended_len > 0:
            self._increment_skd_prefix_stats(agent_data, env_units=1, tokens=appended_len)
        commit_ms = (time.monotonic() - commit_t0) * 1000
        _trace_async_skd(
            "web_skd.tool_processing_commit_bundle_done",
            request_id=agent_data.request_id,
            elapsed_ms=round(commit_ms, 1),
            appended_len=appended_len,
            prompt_len=len(agent_data.prompt_ids),
            response_len=len(agent_data.response_mask),
            image_count=_safe_len(agent_data.image_data),
            teacher_prompt_len=len(next_teacher_prompt_ids),
            teacher_server_prompt_len=len(next_teacher_server_prompt_ids) if next_teacher_server_prompt_ids is not None else None,
            teacher_sglang_prefix_surplus=next_teacher_sglang_prefix_surplus,
        )

        _trace_async_skd(
            "web_skd.tool_processing_commit_done",
            request_id=agent_data.request_id,
            appended_len=appended_len,
            response_len=len(agent_data.response_mask),
            prompt_len=len(agent_data.prompt_ids),
            image_count=_safe_len(agent_data.image_data),
            teacher_sglang_prefix_surplus=next_teacher_sglang_prefix_surplus,
            terminated=result.get("terminated", False),
            termination_reason=result.get("termination_reason"),
        )
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
                    raise ValueError(f"Invalid state while running WebSKD loop: {state}")
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
