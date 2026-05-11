from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.qwen_coder_structured_output import build_qwen_coder_structured_tag_json
from verl.experimental.agent_loop.web_osgym_trajectory_logger import WebOsGymTrajectoryLogger
from verl.experimental.agent_loop.tool_agent_loop import (
    _TOOL_PARSE_ERROR_RETRY_COUNT_KEY,
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import ToolParseError
from verl.experimental.agent_loop.web_osgym_rl_prompt_window import build_web_osgym_prompt_window
from verl.experimental.agent_loop.web_osgym_windowing import build_mini_step_image_spans
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.utils.chat_template import apply_chat_template
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class _WebOsGymGenerationInput:
    prompt_ids: list[int]
    server_prompt_ids: list[int] | None
    images: list[Any] | None
    videos: Any | None
    window_used: bool
    image_indices: list[int]
    selected_step_indices: list[int]
    old_summary_turn_indices: list[int]
    recent_observation_step_indices: list[int]
    recent_assistant_turn_indices: list[int]
    text_only_recent_step_count: int


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

    def _record_web_osgym_step(
        self,
        agent_data: AgentData,
        *,
        phase: str,
        image_start: int,
        image_end: int,
        text: str | None = None,
        text_len: int | None = None,
        terminal: bool,
        termination_reason: str | None,
        actions: list[dict[str, Any]] | None,
    ) -> None:
        normalized_actions = [dict(action) for action in (actions or []) if isinstance(action, dict)]
        step_text = str(text or "")
        if text_len is None:
            text_len = len(step_text)
        steps = list(agent_data.extra_fields.get("web_osgym_steps") or [])
        steps.append(
            {
                "step_idx": len(steps) + 1,
                "assistant_turn": int(agent_data.assistant_turns),
                "user_turn": int(agent_data.user_turns),
                "phase": phase,
                "text": step_text,
                "text_len": int(text_len),
                "action_names": [
                    str(action_name)
                    for action_name in (
                        action.get("action_type") or action.get("name") for action in normalized_actions
                    )
                    if action_name is not None
                ],
                "actions": normalized_actions,
                "image_start": int(image_start),
                "image_end": int(image_end),
                "terminal": bool(terminal),
                "termination_reason": termination_reason,
            }
        )
        agent_data.extra_fields["web_osgym_steps"] = steps
        agent_data.extra_fields["mini_step_image_spans"] = build_mini_step_image_spans(steps)

    def _record_web_osgym_assistant_turn(
        self,
        agent_data: AgentData,
        *,
        observation_step_idx: int,
        response_start: int,
        response_end: int,
        response_text: str,
        actions: list[dict[str, Any]],
    ) -> None:
        turns = list(agent_data.extra_fields.get("web_osgym_assistant_turns") or [])
        turns.append(
            {
                "assistant_turn": len(turns) + 1,
                "observation_step_idx": int(observation_step_idx),
                "response_start": int(response_start),
                "response_end": int(response_end),
                "response_text": response_text,
                "actions": [dict(action) for action in actions if isinstance(action, dict)],
            }
        )
        agent_data.extra_fields["web_osgym_assistant_turns"] = turns

    def _update_latest_web_osgym_assistant_turn_actions(
        self,
        agent_data: AgentData,
        *,
        actions: list[dict[str, Any]],
    ) -> None:
        turns = list(agent_data.extra_fields.get("web_osgym_assistant_turns") or [])
        if not turns:
            return
        turns[-1] = {
            **turns[-1],
            "actions": [dict(action) for action in actions if isinstance(action, dict)],
        }
        agent_data.extra_fields["web_osgym_assistant_turns"] = turns

    def _record_web_osgym_unit_trace(self, agent_data: AgentData) -> None:
        steps = list(agent_data.extra_fields.get("web_osgym_steps") or [])
        image_spans = list(agent_data.extra_fields.get("mini_step_image_spans") or [])
        window_enabled = self._web_osgym_window_enabled()
        generation_windows = list(agent_data.extra_fields.get("web_osgym_generation_windows") or [])
        latest_window = generation_windows[-1] if generation_windows else {}
        unit_trace = {
            "rollout_context": "windowed_prompt" if window_enabled else "full_accumulated_prompt",
            "backprop_context": "windowed_generation_rows" if window_enabled else "full_agent_loop_output",
            "harness_prompt_window": "active" if window_enabled else "metadata_available_not_active",
            "window_history_n": self._web_osgym_window_history_n(),
            "window_max_images_per_sample": self._web_osgym_window_max_images_per_sample(),
            "window_fallback_count": agent_data.metrics.get("web_osgym/window_fallback_count", 0),
            "generation_window_count": len(generation_windows),
            "step_count": len(steps),
            "image_span_count": len(image_spans),
            "window_old_summary_turn_count": len(latest_window.get("old_summary_turn_indices") or []),
            "window_recent_observation_step_count": len(latest_window.get("recent_observation_step_indices") or []),
            "window_recent_assistant_turn_count": len(latest_window.get("recent_assistant_turn_indices") or []),
            "window_text_only_recent_step_count": int(latest_window.get("text_only_recent_step_count", 0)),
            "window_prompt_image_count": len(latest_window.get("prompt_image_indices") or []),
        }
        agent_data.extra_fields["web_osgym_unit_trace"] = unit_trace
        agent_data.metrics["web_osgym/step_count"] = len(steps)
        agent_data.metrics["web_osgym/image_span_count"] = len(image_spans)
        agent_data.metrics["web_osgym/generation_window_count"] = len(generation_windows)
        if os.getenv("WEB_OSGYM_UNIT_TRACE"):
            logger.warning("[WebOsGymTool][UnitTrace] %s", unit_trace)

    @staticmethod
    def _trace_tool_calls(tool_calls) -> list[dict[str, Any]]:
        traced = []
        for tool_call in tool_calls:
            item = {"name": tool_call.name, "arguments": tool_call.arguments}
            try:
                item["parsed_arguments"] = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                item["parsed_arguments"] = None
            traced.append(item)
        return traced

    def _trace_actions_from_tool_calls(self, agent_data: AgentData) -> list[dict[str, Any]]:
        actions = []
        for tool_call in agent_data.tool_calls:
            try:
                tool_args = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(tool_args, dict):
                continue
            if tool_call.name == self.legacy_bundled_tool_name:
                raw_actions = tool_args.get("actions")
                if isinstance(raw_actions, list):
                    actions.extend(action for action in raw_actions if isinstance(action, dict))
            else:
                actions.append({"action_type": tool_call.name, **tool_args})
        return actions

    def _web_osgym_window_enabled(self) -> bool:
        multi_turn = getattr(getattr(self, "rollout_config", None), "multi_turn", None)
        return bool(getattr(multi_turn, "web_osgym_window_enable", False))

    def _web_osgym_window_history_n(self) -> int:
        multi_turn = getattr(getattr(self, "rollout_config", None), "multi_turn", None)
        return int(getattr(multi_turn, "web_osgym_window_history_n", 5))

    def _web_osgym_window_max_images_per_sample(self) -> int | None:
        multi_turn = getattr(getattr(self, "rollout_config", None), "multi_turn", None)
        value = getattr(multi_turn, "web_osgym_window_max_images_per_sample", 6)
        return None if value is None else int(value)

    def _build_generation_sampling_params(
        self,
        base_sampling_params: dict[str, Any],
        active_tool_schemas: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        params = dict(base_sampling_params)
        rollout_custom = getattr(self.rollout_config, "custom", None) or {}
        structured_output_enabled = bool(rollout_custom.get("enable_qwen3_coder_structured_output", False))
        if self.rollout_config.name != "sglang":
            return params
        if self.tool_parser_name != "qwen3_coder":
            return params
        if not structured_output_enabled:
            return params
        if not active_tool_schemas:
            return params
        params["structural_tag"] = build_qwen_coder_structured_tag_json(active_tool_schemas)
        params["ignore_eos"] = True
        return params

    def _should_use_server_prompt_ids(self, images: list[Any] | None) -> bool:
        return (
            self.rollout_config.name == "sglang"
            and not getattr(self.rollout_config, "skip_tokenizer_init", False)
            and bool(images)
        )

    async def _apply_server_chat_template(self, messages: list[dict], tools: list[dict] | None = None) -> list[int]:
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
        return normalize_token_ids(tokenized)

    async def _build_server_prompt_ids(
        self,
        *,
        messages: list[dict],
        images: list[Any] | None,
        tools: list[dict] | None,
    ) -> list[int] | None:
        if not self._should_use_server_prompt_ids(images):
            return None
        server_prompt_ids = await self._apply_server_chat_template(messages, tools=tools)
        if not server_prompt_ids:
            raise ValueError("Server prompt ids unexpectedly empty for multimodal SGLang generation")
        return server_prompt_ids

    async def _build_web_osgym_generation_inputs(
        self,
        agent_data: AgentData,
    ) -> _WebOsGymGenerationInput:
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        if not self._web_osgym_window_enabled():
            server_prompt_ids = await self._build_server_prompt_ids(
                messages=agent_data.messages,
                images=agent_data.image_data,
                tools=schemas,
            )
            image_count = len(agent_data.image_data or [])
            return _WebOsGymGenerationInput(
                agent_data.prompt_ids,
                server_prompt_ids,
                agent_data.image_data,
                agent_data.video_data,
                False,
                list(range(image_count)),
                [],
                [],
                [],
                [],
                0,
            )

        steps = agent_data.extra_fields.get("web_osgym_steps") or []
        if not steps:
            server_prompt_ids = await self._build_server_prompt_ids(
                messages=agent_data.messages,
                images=agent_data.image_data,
                tools=schemas,
            )
            image_count = len(agent_data.image_data or [])
            return _WebOsGymGenerationInput(
                agent_data.prompt_ids,
                server_prompt_ids,
                agent_data.image_data,
                agent_data.video_data,
                False,
                list(range(image_count)),
                [],
                [],
                [],
                [],
                0,
            )

        try:
            prompt_window = build_web_osgym_prompt_window(
                base_messages=getattr(agent_data, "_web_osgym_base_messages", agent_data.messages),
                images=agent_data.image_data,
                steps=steps,
                assistant_turns=agent_data.extra_fields.get("web_osgym_assistant_turns"),
                history_n=self._web_osgym_window_history_n(),
                max_images_per_sample=self._web_osgym_window_max_images_per_sample(),
            )
        except ValueError as exc:
            agent_data.metrics["web_osgym/window_fallback_count"] = (
                agent_data.metrics.get("web_osgym/window_fallback_count", 0) + 1
            )
            logger.warning("Falling back to full Web/OSGym prompt window: %s", exc)
            image_count = len(agent_data.image_data or [])
            return _WebOsGymGenerationInput(
                agent_data.prompt_ids,
                None,
                agent_data.image_data,
                agent_data.video_data,
                False,
                list(range(image_count)),
                [],
                [],
                [],
                [],
                0,
            )

        prompt_ids = await self.apply_chat_template(
            prompt_window.messages,
            tools=schemas,
            images=prompt_window.images,
            videos=None,
        )
        server_prompt_ids = await self._build_server_prompt_ids(
            messages=prompt_window.messages,
            images=prompt_window.images,
            tools=schemas,
        )
        agent_data.metrics["web_osgym/window_active"] = 1
        agent_data.metrics["web_osgym/window_step_count"] = len(prompt_window.selected_steps)
        agent_data.metrics["web_osgym/window_image_count"] = len(prompt_window.images)
        agent_data.metrics["web_osgym/window_old_summary_turn_count"] = len(prompt_window.old_summary_turn_indices)
        agent_data.metrics["web_osgym/window_recent_observation_step_count"] = len(
            prompt_window.recent_observation_step_indices
        )
        agent_data.metrics["web_osgym/window_recent_assistant_turn_count"] = len(
            prompt_window.recent_assistant_turn_indices
        )
        agent_data.metrics["web_osgym/window_text_only_recent_step_count"] = (
            prompt_window.text_only_recent_step_count
        )
        return _WebOsGymGenerationInput(
            prompt_ids,
            server_prompt_ids,
            prompt_window.images,
            None,
            True,
            prompt_window.image_indices,
            [int(step.get("step_idx", 0)) for step in prompt_window.selected_steps],
            list(prompt_window.old_summary_turn_indices),
            list(prompt_window.recent_observation_step_indices),
            list(prompt_window.recent_assistant_turn_indices),
            int(prompt_window.text_only_recent_step_count),
        )

    def _record_web_osgym_generation_window(
        self,
        agent_data: AgentData,
        *,
        generation_inputs: _WebOsGymGenerationInput,
        response_start: int,
        response_end: int,
    ) -> None:
        if not self._web_osgym_window_enabled():
            return
        windows = list(agent_data.extra_fields.get("web_osgym_generation_windows") or [])
        windows.append(
            {
                "assistant_turn": int(agent_data.assistant_turns),
                "response_start": int(response_start),
                "response_end": int(response_end),
                "prompt_ids": list(generation_inputs.prompt_ids),
                "prompt_token_count": len(generation_inputs.prompt_ids),
                "window_used": bool(generation_inputs.window_used),
                "image_indices": list(generation_inputs.image_indices),
                "prompt_image_indices": list(generation_inputs.image_indices),
                "selected_step_indices": list(generation_inputs.selected_step_indices),
                "old_summary_turn_indices": list(generation_inputs.old_summary_turn_indices),
                "recent_observation_step_indices": list(generation_inputs.recent_observation_step_indices),
                "recent_assistant_turn_indices": list(generation_inputs.recent_assistant_turn_indices),
                "text_only_recent_step_count": int(generation_inputs.text_only_recent_step_count),
                "history_n": self._web_osgym_window_history_n(),
                "max_images_per_sample": self._web_osgym_window_max_images_per_sample(),
            }
        )
        agent_data.extra_fields["web_osgym_generation_windows"] = windows

    def _decode_response_text(self, token_ids: list[int]) -> str:
        decode = getattr(self.tokenizer, "decode", None)
        if callable(decode):
            return str(decode(token_ids, skip_special_tokens=False)).strip()
        return ""

    def _latest_web_osgym_assistant_turn(self, agent_data: AgentData) -> dict[str, Any] | None:
        turns = list(agent_data.extra_fields.get("web_osgym_assistant_turns") or [])
        if not turns:
            return None
        return dict(turns[-1])

    def _get_web_osgym_trajectory_logger(self, agent_data: AgentData) -> WebOsGymTrajectoryLogger | None:
        trace_dir_value = os.getenv("WEB_OSGYM_TOOL_TRACE_DIR") or agent_data.extra_fields.get(
            "web_osgym_tool_trace_dir"
        )
        if not trace_dir_value:
            return None
        return WebOsGymTrajectoryLogger(Path(trace_dir_value))

    @staticmethod
    def _coerce_web_osgym_log_global_step(value: Any) -> int | None:
        try:
            step = int(value)
        except (TypeError, ValueError):
            return None
        return step if step >= 0 else None

    def _resolve_web_osgym_log_global_step(self, agent_data: AgentData) -> int:
        extra_fields = agent_data.extra_fields
        for key in ("web_osgym_log_global_step", "global_steps", "min_global_steps", "max_global_steps"):
            resolved = self._coerce_web_osgym_log_global_step(extra_fields.get(key))
            if resolved is not None:
                extra_fields["web_osgym_log_global_step"] = resolved
                return resolved
        extra_fields["web_osgym_log_global_step"] = 0
        return 0

    def _get_web_osgym_session_dir(self, agent_data: AgentData) -> Path | None:
        logger_obj = self._get_web_osgym_trajectory_logger(agent_data)
        if logger_obj is None:
            return None
        cached_dir = agent_data.extra_fields.get("web_osgym_trajectory_dir")
        if cached_dir:
            return Path(cached_dir)
        session_dir = logger_obj.session_dir(
            task_id=agent_data.extra_fields.get("web_osgym_task_id"),
            sample_uid=agent_data.extra_fields.get("web_osgym_sample_uid"),
            global_step=self._resolve_web_osgym_log_global_step(agent_data),
            session_id=agent_data.extra_fields.get("web_osgym_session_id"),
        )
        agent_data.extra_fields["web_osgym_trajectory_dir"] = str(session_dir)
        return session_dir

    def _current_prompt_window(self, agent_data: AgentData) -> dict[str, Any]:
        generation_window = (agent_data.extra_fields.get("web_osgym_generation_windows") or [{}])[-1]
        return {
            "prompt_image_indices": list(generation_window.get("prompt_image_indices") or []),
            "old_summary_turn_indices": list(generation_window.get("old_summary_turn_indices") or []),
            "recent_observation_step_indices": list(generation_window.get("recent_observation_step_indices") or []),
            "recent_assistant_turn_indices": list(generation_window.get("recent_assistant_turn_indices") or []),
            "text_only_recent_step_count": int(generation_window.get("text_only_recent_step_count", 0)),
        }

    def _append_web_osgym_initial_observation(
        self, agent_data: AgentData, *, observation_text: str | None, image_data: list[Any] | None
    ) -> None:
        logger_obj = self._get_web_osgym_trajectory_logger(agent_data)
        session_dir = self._get_web_osgym_session_dir(agent_data)
        if logger_obj is None or session_dir is None:
            return
        image_records = logger_obj.write_images(session_dir, assistant_turn=0, user_turn=0, images=image_data)
        logger_obj.append_event(
            session_dir,
            {
                "event_type": "initial_observation",
                "request_id": agent_data.request_id,
                "session_id": agent_data.extra_fields.get("web_osgym_session_id"),
                "task_id": agent_data.extra_fields.get("web_osgym_task_id"),
                "sample_uid": agent_data.extra_fields.get("web_osgym_sample_uid"),
                "assistant_turn": 0,
                "user_turn": 0,
                "observation_text": observation_text,
                "image_paths": [record["path"] for record in image_records],
                "images": image_records,
            },
        )

    def _append_web_osgym_assistant_event(
        self,
        agent_data: AgentData,
        *,
        tool_response=None,
        result: dict[str, Any] | None = None,
        image_data: list[Any] | None = None,
        parse_error: ToolParseError | None = None,
    ) -> None:
        latest_turn = self._latest_web_osgym_assistant_turn(agent_data)
        logger_obj = self._get_web_osgym_trajectory_logger(agent_data)
        session_dir = self._get_web_osgym_session_dir(agent_data)
        if latest_turn is None or logger_obj is None or session_dir is None:
            return
        image_records = logger_obj.write_images(
            session_dir,
            assistant_turn=int(latest_turn.get("assistant_turn", agent_data.assistant_turns)),
            user_turn=int(agent_data.user_turns),
            images=image_data,
        )
        traced_tool_calls = self._trace_tool_calls(agent_data.tool_calls)
        event = {
            "event_type": "assistant_turn",
            "request_id": agent_data.request_id,
            "session_id": agent_data.extra_fields.get("web_osgym_session_id"),
            "task_id": agent_data.extra_fields.get("web_osgym_task_id"),
            "sample_uid": agent_data.extra_fields.get("web_osgym_sample_uid"),
            "instance_id": agent_data.extra_fields.get("web_osgym_instance_id"),
            "assistant_turn": latest_turn.get("assistant_turn"),
            "user_turn": latest_turn.get("user_turn"),
            "observation_step_idx": latest_turn.get("observation_step_idx"),
            "model_output_text": latest_turn.get("response_text"),
            "tool_call_count": len(agent_data.tool_calls),
            "tool_calls_raw": [item.get("arguments") for item in traced_tool_calls],
            "tool_calls_parsed": [item.get("parsed_arguments") for item in traced_tool_calls],
            "actions": (result or {}).get("web_osgym_actions")
            or latest_turn.get("actions")
            or self._trace_actions_from_tool_calls(agent_data),
            "result": {
                "terminated": (result or {}).get("terminated"),
                "termination_reason": (result or {}).get("termination_reason"),
                "invalid_action": bool((result or {}).get("invalid_action")),
                "action_count": (result or {}).get("action_count"),
                "error_type": (result or {}).get("web_osgym_error_type"),
            },
            "parse_error": parse_error.model_dump() if parse_error is not None else None,
            "prompt_window": self._current_prompt_window(agent_data),
            "observation_text": getattr(tool_response, "text", None),
            "image_paths": [record["path"] for record in image_records],
            "images": image_records,
        }
        logger_obj.append_event(session_dir, event)
        counts = dict(agent_data.extra_fields.get("web_osgym_trajectory_counts") or {})
        counts["event_count"] = int(counts.get("event_count", 0)) + 1
        if event["result"]["invalid_action"]:
            counts["invalid_action_count"] = int(counts.get("invalid_action_count", 0)) + 1
        if parse_error is not None:
            counts["parse_error_count"] = int(counts.get("parse_error_count", 0)) + 1
        agent_data.extra_fields["web_osgym_trajectory_counts"] = counts

    def _write_web_osgym_summary(self, agent_data: AgentData, *, fatal_error: str | None = None) -> None:
        logger_obj = self._get_web_osgym_trajectory_logger(agent_data)
        session_dir = self._get_web_osgym_session_dir(agent_data)
        if logger_obj is None or session_dir is None:
            return
        counts = dict(agent_data.extra_fields.get("web_osgym_trajectory_counts") or {})
        logger_obj.write_summary(
            session_dir,
            {
                "task_id": agent_data.extra_fields.get("web_osgym_task_id"),
                "sample_uid": agent_data.extra_fields.get("web_osgym_sample_uid"),
                "global_step": self._resolve_web_osgym_log_global_step(agent_data),
                "session_id": agent_data.extra_fields.get("web_osgym_session_id"),
                "request_id": agent_data.request_id,
                "reward_score": agent_data.extra_fields.get("web_osgym_reward_score"),
                "termination_reason": agent_data.extra_fields.get("web_osgym_termination_reason"),
                "num_turns": agent_data.user_turns + agent_data.assistant_turns + 1,
                "invalid_action_count": int(counts.get("invalid_action_count", 0)),
                "parse_error_count": int(counts.get("parse_error_count", 0)),
                "event_count": int(counts.get("event_count", 0)),
                "completed": fatal_error is None,
                "has_reward": "web_osgym_reward_score" in agent_data.extra_fields,
                "fatal_error": fatal_error,
            },
        )

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
        agent_data._web_osgym_base_messages = deepcopy(messages)
        if kwargs.get("uid") is not None:
            agent_data.extra_fields["web_osgym_sample_uid"] = str(kwargs.get("uid"))
        elif kwargs.get("index") is not None:
            agent_data.extra_fields["web_osgym_sample_uid"] = f"index_{kwargs.get('index')}"
        trajectory_global_step = self._coerce_web_osgym_log_global_step(kwargs.get("_trajectory_global_step"))
        if trajectory_global_step is not None:
            agent_data.extra_fields["web_osgym_log_global_step"] = trajectory_global_step

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

        image_start = len(agent_data.image_data) if agent_data.image_data else 0
        self._extend_image_data(agent_data, image_data)
        image_end = len(agent_data.image_data) if agent_data.image_data else image_start
        agent_data.messages = messages
        agent_data.prompt_ids = prompt_ids
        if student_obs or image_data:
            self._record_web_osgym_step(
                agent_data,
                phase="initial",
                image_start=image_start,
                image_end=image_end,
                text=student_obs,
                text_len=len(student_obs or ""),
                terminal=False,
                termination_reason=None,
                actions=[],
            )
            self._append_web_osgym_initial_observation(
                agent_data,
                observation_text=student_obs,
                image_data=image_data,
            )
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        with simple_timer("tool_calls", agent_data.metrics):
            tool_response, _, result = await self._execute_web_osgym_tool_calls(agent_data)

        agent_data.metrics["web_osgym/action_count"] = result.get("action_count", 0)
        if result.get("invalid_action"):
            agent_data.metrics["web_osgym/invalid_action"] = 1

        processed_actions = result.get("web_osgym_actions")
        if processed_actions is None:
            processed_actions = [] if result.get("invalid_action") else self._trace_actions_from_tool_calls(agent_data)
        self._update_latest_web_osgym_assistant_turn_actions(agent_data, actions=processed_actions)

        image_data = self._normalize_image_data(tool_response.image)
        try:
            self._append_web_osgym_assistant_event(
                agent_data,
                tool_response=tool_response,
                result=result,
                image_data=image_data,
            )
        except Exception:
            logger.warning("Failed to write Web/OSGym trajectory event", exc_info=True)

        if result.get("terminated"):
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason=result.get("termination_reason") or "model_done",
            )
            return AgentState.TERMINATED

        student_obs = self._split_env_observation(tool_response.text, image_data)

        if not student_obs and not image_data:
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

        image_start = len(agent_data.image_data) if agent_data.image_data else 0
        self._extend_image_data(agent_data, image_data)
        image_end = len(agent_data.image_data) if agent_data.image_data else image_start
        self._record_web_osgym_step(
            agent_data,
            phase="tool_observation",
            image_start=image_start,
            image_end=image_end,
            text=student_obs,
            text_len=len(student_obs or ""),
            terminal=False,
            termination_reason=None,
            actions=processed_actions,
        )
        agent_data.messages.append(message)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        return AgentState.GENERATING

    async def _handle_tool_parse_error(self, agent_data: AgentData, parse_error: ToolParseError) -> AgentState:
        retry_count = int(agent_data.extra_fields.get(_TOOL_PARSE_ERROR_RETRY_COUNT_KEY, 0))
        if retry_count >= self.max_tool_parse_error_retries:
            return AgentState.TERMINATED

        feedback_text = self._build_tool_parse_error_feedback(parse_error)
        message = self._build_tool_message(feedback_text, None)
        response_ids = await self.apply_chat_template(
            [message],
            images=None,
            videos=None,
            remove_system_prompt=True,
        )
        if response_ids and len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason="tool_response_budget_exhausted",
            )
            return AgentState.TERMINATED

        agent_data.metrics["tool_parse_error"] = 1
        agent_data.extra_fields[_TOOL_PARSE_ERROR_RETRY_COUNT_KEY] = retry_count + 1
        image_start = len(agent_data.image_data) if agent_data.image_data else 0
        image_end = len(agent_data.image_data) if agent_data.image_data else image_start
        self._record_web_osgym_step(
            agent_data,
            phase="tool_parse_error",
            image_start=image_start,
            image_end=image_end,
            text=feedback_text,
            text_len=len(feedback_text),
            terminal=False,
            termination_reason=None,
            actions=[],
        )
        agent_data.messages.append(message)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        generation_inputs = await self._build_web_osgym_generation_inputs(agent_data)
        active_tool_schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        effective_sampling_params = self._build_generation_sampling_params(sampling_params, active_tool_schemas)
        request_prompt_ids = generation_inputs.prompt_ids
        if generation_inputs.server_prompt_ids is not None:
            request_prompt_ids = generation_inputs.server_prompt_ids
        elif generation_inputs.window_used and self._should_use_server_prompt_ids(generation_inputs.images):
            raise ValueError(
                "Windowed multimodal SGLang generation expected server prompt ids but none were built."
            )
        response_start = len(agent_data.response_mask)
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=request_prompt_ids,
                sampling_params=effective_sampling_params,
                image_data=generation_inputs.images,
                video_data=generation_inputs.videos,
            )
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        self._merge_generation_extra_fields(agent_data, output.extra_fields)

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if agent_data.response_ids:
            self._record_web_osgym_generation_window(
                agent_data,
                generation_inputs=generation_inputs,
                response_start=response_start,
                response_end=len(agent_data.response_mask),
            )

        if output.routed_experts is not None and not generation_inputs.window_used:
            agent_data.routed_experts = output.routed_experts

        latest_step_idx = int((agent_data.extra_fields.get("web_osgym_steps") or [{}])[-1].get("step_idx", 0))
        self._record_web_osgym_assistant_turn(
            agent_data,
            observation_step_idx=latest_step_idx,
            response_start=response_start,
            response_end=len(agent_data.response_mask),
            response_text=self._decode_response_text(agent_data.response_ids),
            actions=[],
        )

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            self._append_web_osgym_assistant_event(
                agent_data,
                result={
                    "terminated": True,
                    "termination_reason": "system_stop",
                    "invalid_action": False,
                    "action_count": 0,
                    "web_osgym_error_type": None,
                },
            )
            next_state = AgentState.TERMINATED
        elif self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            self._append_web_osgym_assistant_event(
                agent_data,
                result={
                    "terminated": True,
                    "termination_reason": "system_stop",
                    "invalid_action": False,
                    "action_count": 0,
                    "web_osgym_error_type": None,
                },
            )
            next_state = AgentState.TERMINATED
        elif self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            self._append_web_osgym_assistant_event(
                agent_data,
                result={
                    "terminated": True,
                    "termination_reason": "system_stop",
                    "invalid_action": False,
                    "action_count": 0,
                    "web_osgym_error_type": None,
                },
            )
            next_state = AgentState.TERMINATED
        else:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            tools = [tool.tool_schema for tool in active_tools.values()]
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
            parse_error = getattr(self.tool_parser, "last_parse_error", None)
            if agent_data.tool_calls:
                self._update_latest_web_osgym_assistant_turn_actions(
                    agent_data,
                    actions=self._trace_actions_from_tool_calls(agent_data),
                )
                next_state = AgentState.PROCESSING_TOOLS
            elif parse_error is not None:
                self._append_web_osgym_assistant_event(agent_data, parse_error=parse_error)
                next_state = await self._handle_tool_parse_error(agent_data, parse_error)
            else:
                self._append_web_osgym_assistant_event(
                    agent_data,
                    result={
                        "terminated": True,
                        "termination_reason": "system_stop",
                        "invalid_action": False,
                        "action_count": 0,
                        "web_osgym_error_type": None,
                    },
                )
                next_state = AgentState.TERMINATED

        if next_state == AgentState.TERMINATED and "web_osgym_reward_score" not in agent_data.extra_fields:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        return next_state

    def _finalize_web_agent_output(self, agent_data: AgentData) -> AgentLoopOutput:
        self._record_web_osgym_unit_trace(agent_data)
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
        fatal_error: str | None = None
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
        except Exception as exc:
            fatal_error = str(exc)
            raise
        finally:
            try:
                self._write_web_osgym_summary(agent_data, fatal_error=fatal_error)
            except Exception:
                logger.warning("Failed to write Web/OSGym trajectory summary", exc_info=True)
            await self._release_web_osgym_session(agent_data)
