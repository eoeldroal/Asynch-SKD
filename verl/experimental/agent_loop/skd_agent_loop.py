# Copyright 2025 DDAI Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Speculative Knowledge Distillation (SKD) Agent Loop.

Extends ToolAgentLoop by overriding the generation phase:
instead of generating a full sequence at once, generates in chunks,
verifies each chunk against the Teacher model's top-K,
and replaces rejected tokens with Teacher's top-1.

Teacher logprobs are accumulated during chunk verification,
eliminating the need for a separate teacher logprob computation in postprocessing.
"""

from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import logging
import os
from pathlib import Path
import time
import warnings
from typing import Any
from uuid import uuid4

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    register,
    rollout_trace_op,
)
from verl.experimental.agent_loop.teacher_fewshot import load_teacher_fewshot_transcript
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.async_skd.events import emit_async_skd_event, get_async_skd_event_context
from verl.experimental.async_skd.state import SkdPartialState
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# SKD debug logging: set VERL_SKD_DEBUG=1 to enable per-chunk diagnostics.
# VERL_SKD_DEBUG=2 for token-level alignment verification (first 3 samples per batch).
_SKD_DEBUG = int(os.getenv("VERL_SKD_DEBUG", "0"))
_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))
_SKD_CHUNK_TRACE = int(os.getenv("VERL_ASYNC_SKD_CHUNK_TRACE", "0"))
_SKD_CHUNK_TRACE_TOPK = int(os.getenv("VERL_ASYNC_SKD_CHUNK_TRACE_TOPK", "5"))
_SKD_CHUNK_TRACE_MAX_TOKENS = int(os.getenv("VERL_ASYNC_SKD_CHUNK_TRACE_MAX_TOKENS", "64"))
_SKD_CHUNK_LIVE_TAIL_CHARS = int(os.getenv("VERL_ASYNC_SKD_CHUNK_LIVE_TAIL_CHARS", "256"))
_SKD_PENDING_TURN_STATE = "skd_pending_turn_state"
_SKD_PENDING_TURN_CHUNKS = "skd_pending_turn_chunks"


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    fields = {"pid": os.getpid(), "mono_ns": time.monotonic_ns(), **fields}
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


def _chunk_live_log_path() -> str | None:
    path = os.getenv("VERL_ASYNC_SKD_CHUNK_LIVE_LOG")
    return path if path else None


def _decode_chunk_text(tokenizer: Any, ids: list[int]) -> str:
    if not ids:
        return ""
    for kwargs in (
        {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        {"skip_special_tokens": False},
        {},
    ):
        try:
            return tokenizer.decode(ids, **kwargs)
        except TypeError:
            continue
    return " ".join(str(token_id) for token_id in ids)


def _tail_text(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def _count_special_marker(text: str, marker: str) -> int:
    return text.count(marker)


def _append_chunk_live_log(
    *,
    tokenizer: Any,
    request_id: str,
    chunk_idx: int,
    student_ms: int,
    teacher_ms: int,
    chunk: list[int],
    new_tokens: list[int],
    accepted: int,
    rejected: int,
    response_len: int,
    eos_in_new_tokens: bool,
) -> None:
    path = _chunk_live_log_path()
    if path is None:
        return

    raw_text = _decode_chunk_text(tokenizer, chunk)
    verified_text = _decode_chunk_text(tokenizer, new_tokens)
    record = {
        "ts": time.time(),
        "pid": os.getpid(),
        "request_id": request_id,
        "chunk_idx": chunk_idx,
        "student_ms": student_ms,
        "teacher_ms": teacher_ms,
        "accepted": accepted,
        "rejected": rejected,
        "new_tokens": len(new_tokens),
        "response_len": response_len,
        "eos_in_new_tokens": eos_in_new_tokens,
        "raw_text_tail": _tail_text(raw_text, _SKD_CHUNK_LIVE_TAIL_CHARS),
        "verified_text_tail": _tail_text(verified_text, _SKD_CHUNK_LIVE_TAIL_CHARS),
        "im_start_count": _count_special_marker(verified_text, "<|im_start|>"),
        "im_end_count": _count_special_marker(verified_text, "<|im_end|>"),
        "endoftext_count": _count_special_marker(verified_text, "<|endoftext|>"),
        **get_async_skd_event_context(),
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _token_edges(ids: list[int], *, head: int = 8, tail: int = 8) -> dict[str, list[int]]:
    values = list(ids)
    if len(values) <= head + tail:
        return {"all": values}
    return {
        "head": values[:head],
        "tail": values[-tail:],
    }


@dataclass(frozen=True)
class SkdTeacherLogprobRange:
    """Named coordinate contract for teacher prompt-logprob delta requests."""

    server_logical_start_len: int
    sglang_logprob_start_len: int
    expected_logprob_rows: int
    teacher_sglang_prefix_surplus: int


@dataclass
class SkdTurnChunkState:
    """Mutable assistant-turn buffer kept outside the committed rollout state."""

    tokens: list[int]
    teacher_ids_rows: list[list[int]]
    teacher_logprobs_rows: list[list[float]]
    raw_chunk: list[int]
    verified_chunk: list[int]

    def to_payload(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "teacher_ids_rows": [list(row) for row in self.teacher_ids_rows],
            "teacher_logprobs_rows": [list(row) for row in self.teacher_logprobs_rows],
            "raw_chunk": list(self.raw_chunk),
            "verified_chunk": list(self.verified_chunk),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "SkdTurnChunkState":
        if payload is None:
            return cls(tokens=[], teacher_ids_rows=[], teacher_logprobs_rows=[], raw_chunk=[], verified_chunk=[])
        required_keys = {"tokens", "teacher_ids_rows", "teacher_logprobs_rows", "raw_chunk", "verified_chunk"}
        missing_keys = required_keys.difference(payload)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Invalid SKD pending turn state payload: missing keys: {missing}")
        return cls(
            tokens=list(payload["tokens"]),
            teacher_ids_rows=[list(row) for row in payload["teacher_ids_rows"]],
            teacher_logprobs_rows=[list(row) for row in payload["teacher_logprobs_rows"]],
            raw_chunk=list(payload["raw_chunk"]),
            verified_chunk=list(payload["verified_chunk"]),
        )


def _build_teacher_logprob_range(
    *,
    teacher_server_prompt_len: int,
    teacher_sglang_prefix_surplus: int,
    chunk_len: int,
) -> SkdTeacherLogprobRange:
    """Build the teacher range in logical and SGLang-expanded coordinates.

    ``teacher_server_prompt_len`` is the real teacher prefix length sent through
    SGLang's token-in multimodal API. It includes teacher-only system prompts,
    a11y trees, and teacher-only tool-result text. ``teacher_sglang_prefix_surplus``
    is only the cumulative multimodal expansion offset that SGLang applies
    internally before interpreting ``logprob_start_len``.
    """
    if teacher_server_prompt_len <= 0:
        raise ValueError(f"teacher_server_prompt_len must be positive, got {teacher_server_prompt_len}.")
    if teacher_sglang_prefix_surplus < 0:
        raise ValueError(
            f"teacher_sglang_prefix_surplus must be non-negative, got {teacher_sglang_prefix_surplus}."
        )
    if chunk_len < 0:
        raise ValueError(f"chunk_len must be non-negative, got {chunk_len}.")

    server_logical_start_len = teacher_server_prompt_len - 1
    sglang_logprob_start_len = server_logical_start_len + teacher_sglang_prefix_surplus
    return SkdTeacherLogprobRange(
        server_logical_start_len=server_logical_start_len,
        sglang_logprob_start_len=sglang_logprob_start_len,
        expected_logprob_rows=chunk_len,
        teacher_sglang_prefix_surplus=teacher_sglang_prefix_surplus,
    )


def _teacher_sglang_prefix_surplus_from_fields(extra_fields: dict[str, Any], *, has_multimodal: bool) -> int:
    """Return the explicitly tracked teacher multimodal expansion surplus."""
    tracked = extra_fields.get("teacher_sglang_prefix_surplus")
    if tracked is None:
        if has_multimodal:
            raise ValueError("teacher_sglang_prefix_surplus is required for multimodal teacher verification.")
        raise ValueError("teacher_sglang_prefix_surplus is required for teacher verification.")
    tracked_int = int(tracked)
    if tracked_int < 0:
        raise ValueError(f"teacher_sglang_prefix_surplus must be non-negative, got {tracked_int}.")
    return tracked_int


@register("skd_agent")
class SkdAgentLoop(ToolAgentLoop):
    """Agent loop with Speculative Knowledge Distillation.

    Inherits ToolAgentLoop's state machine (PENDING → GENERATING → PROCESSING_TOOLS → TERMINATED)
    and overrides only _handle_generating_state to implement SKD chunk-based generation with
    Teacher verification.
    """

    # Class-level counter for debug logging (only first N samples get token-level logs)
    _debug_sample_count = 0

    def __init__(self, *args, teacher_server_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_server_manager = teacher_server_manager

        # SKD config — read from distillation config if available, otherwise defaults
        distillation_config = self.config.get("distillation", {})
        skd_config = distillation_config.get("skd", {})
        self.skd_chunk_size = skd_config.get("chunk_size", 1024)
        self.skd_verify_top_k = skd_config.get("verify_top_k", 25)
        self.max_chunks_per_sample = skd_config.get("max_chunks_per_sample", 60)
        self.teacher_system_prompt_path = skd_config.get("teacher_system_prompt_path")
        self.teacher_fewshot_path = skd_config.get("teacher_fewshot_path")
        self.teacher_key = distillation_config.get("teacher_key", "data_source")

        # Loss top-K for teacher logprobs accumulation (from distillation_loss config)
        loss_config = distillation_config.get("distillation_loss", {})
        self.loss_top_k = loss_config.get("topk", 128)
        self.teacher_system_prompt = self._load_teacher_system_prompt()
        self.teacher_fewshot_messages, self.teacher_fewshot_images = self._load_teacher_fewshot()

        if self.teacher_server_manager is None:
            raise ValueError("SkdAgentLoop requires teacher_server_manager for teacher verification")

    def _load_teacher_system_prompt(self) -> str | None:
        """Load teacher-only system prompt once at worker initialization."""
        if not self.teacher_system_prompt_path:
            return None
        prompt_path = Path(self.teacher_system_prompt_path).expanduser()
        teacher_text = prompt_path.read_text(encoding="utf-8").strip()
        return teacher_text or None

    def _load_teacher_fewshot(self) -> tuple[list[dict[str, Any]], list[Any] | None]:
        if not self.teacher_fewshot_path:
            return [], None
        messages, images = load_teacher_fewshot_transcript(self.teacher_fewshot_path)
        return messages, images

    def _teacher_images_with_runtime(self, runtime_images: list[Any] | None) -> list[Any] | None:
        teacher_fewshot_images = getattr(self, "teacher_fewshot_images", None)
        if not teacher_fewshot_images:
            return runtime_images
        merged = list(teacher_fewshot_images)
        if runtime_images:
            merged.extend(runtime_images)
        return merged

    def _safe_decode_token(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except Exception:
            return f"<tok:{int(token_id)}>"

    def _build_teacher_chunk_debug_rows(
        self,
        *,
        chunk: list[int],
        new_tokens: list[int],
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        rejection_pos: int | None,
    ) -> list[dict[str, Any]]:
        limit = min(len(chunk), max(_SKD_CHUNK_TRACE_MAX_TOKENS, 0))
        topk = min(max(_SKD_CHUNK_TRACE_TOPK, 1), int(teacher_ids.shape[1]) if teacher_ids.dim() > 1 else 1)
        rows: list[dict[str, Any]] = []
        for idx in range(limit):
            teacher_top_ids = teacher_ids[idx, :topk].tolist() if idx < teacher_ids.shape[0] else []
            teacher_top_logprobs = (
                teacher_logprobs[idx, :topk].tolist() if idx < teacher_logprobs.shape[0] else []
            )
            final_token_id = int(new_tokens[idx]) if idx < len(new_tokens) else None
            rows.append(
                {
                    "position": idx,
                    "student_token_id": int(chunk[idx]),
                    "student_token_text": self._safe_decode_token(int(chunk[idx])),
                    "teacher_top_ids": [int(token_id) for token_id in teacher_top_ids],
                    "teacher_top_tokens": [self._safe_decode_token(int(token_id)) for token_id in teacher_top_ids],
                    "teacher_top_logprobs": [float(logprob) for logprob in teacher_top_logprobs],
                    "in_verify_top_k": bool(int(chunk[idx]) in teacher_top_ids[: self.skd_verify_top_k]),
                    "final_token_id": final_token_id,
                    "final_token_text": self._safe_decode_token(final_token_id) if final_token_id is not None else None,
                    "committed": idx < len(new_tokens),
                    "replaced": rejection_pos is not None and idx == rejection_pos,
                }
            )
        return rows

    def _emit_teacher_chunk_debug_event(
        self,
        *,
        event_name: str,
        request_id: str,
        chunk_idx: int,
        chunk: list[int],
        new_tokens: list[int],
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        rejection_pos: int | None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        if _SKD_CHUNK_TRACE <= 0:
            return
        payload = {
            "request_id": request_id,
            "chunk_idx": chunk_idx,
            "chunk_len": len(chunk),
            "new_tokens_len": len(new_tokens),
            "rejection_pos": rejection_pos,
            "rows": self._build_teacher_chunk_debug_rows(
                chunk=chunk,
                new_tokens=new_tokens,
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                rejection_pos=rejection_pos,
            ),
        }
        if extra_fields:
            payload.update(extra_fields)
        emit_async_skd_event(event_name, **payload)

    def _log_teacher_chunk_debug_console(
        self,
        *,
        request_id: str,
        chunk_idx: int,
        chunk: list[int],
        new_tokens: list[int],
        teacher_ids: torch.Tensor,
        teacher_logprobs: torch.Tensor,
        rejection_pos: int | None,
    ) -> None:
        if _SKD_DEBUG < 2:
            return
        try:
            rows = self._build_teacher_chunk_debug_rows(
                chunk=chunk,
                new_tokens=new_tokens,
                teacher_ids=teacher_ids,
                teacher_logprobs=teacher_logprobs,
                rejection_pos=rejection_pos,
            )
            lines = [
                f"[SKD_CHUNK] req={request_id} chunk={chunk_idx} chunk_len={len(chunk)} "
                f"new_tokens_len={len(new_tokens)} rejection_pos={rejection_pos}"
            ]
            for row in rows:
                top_pairs = ", ".join(
                    f"{tok}@{logprob:.4f}"
                    for tok, logprob in zip(row["teacher_top_tokens"], row["teacher_top_logprobs"], strict=True)
                )
                lines.append(
                    f"  pos={row['position']} student={row['student_token_text']}({row['student_token_id']}) "
                    f"committed={row['committed']} replaced={row['replaced']} "
                    f"final={row['final_token_text']}({row['final_token_id']}) top={top_pairs}"
                )
            warnings.warn("\n".join(lines), stacklevel=2)
        except Exception as exc:
            warnings.warn(f"[SKD_CHUNK] logging error: {exc}", stacklevel=2)

    def _build_teacher_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge teacher-only system guidance into the initial conversation."""
        teacher_messages = deepcopy(messages)
        teacher_system_prompt = getattr(self, "teacher_system_prompt", None)
        if teacher_system_prompt:
            if teacher_messages and teacher_messages[0].get("role") == "system":
                content = teacher_messages[0].get("content")
                if isinstance(content, str):
                    merged = content.rstrip()
                    if merged:
                        merged = f"{merged}\n\n{teacher_system_prompt}"
                    else:
                        merged = teacher_system_prompt
                    teacher_messages[0]["content"] = merged
            else:
                teacher_messages.insert(0, {"role": "system", "content": teacher_system_prompt})

        teacher_fewshot_messages = getattr(self, "teacher_fewshot_messages", None) or []
        if teacher_fewshot_messages:
            insert_at = 1 if teacher_messages and teacher_messages[0].get("role") == "system" else 0
            teacher_messages[insert_at:insert_at] = deepcopy(teacher_fewshot_messages)
        return teacher_messages

    async def _init_boundary_agent_data(self, **kwargs: Any) -> AgentData:
        """Create AgentData for SKD boundary execution without changing ToolAgentLoop."""
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
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
        routing_key = kwargs.get(getattr(self, "teacher_key", "data_source"))
        if routing_key is not None:
            if hasattr(routing_key, "item"):
                routing_key = routing_key.item()
            agent_data.extra_fields["teacher_routing_key"] = routing_key
        teacher_replica_id = kwargs.get("teacher_replica_id")
        if teacher_replica_id is not None:
            if hasattr(teacher_replica_id, "item"):
                teacher_replica_id = teacher_replica_id.item()
            agent_data.extra_fields["teacher_replica_id"] = teacher_replica_id
        agent_data.extra_fields["raw_prompt"] = deepcopy(kwargs["raw_prompt"])
        _trace_async_skd(
            "loop.init_boundary_agent_data",
            request_id=request_id,
            teacher_replica_id=agent_data.extra_fields.get("teacher_replica_id"),
            teacher_routing_key=agent_data.extra_fields.get("teacher_routing_key"),
            raw_prompt_len=len(messages),
            kwargs_keys=sorted(kwargs.keys()),
        )
        return agent_data

    def _finalize_boundary_agent_output(self, agent_data: AgentData) -> AgentLoopOutput:
        """Build AgentLoopOutput from AgentData using ToolAgentLoop-compatible semantics."""
        committed_response_len = len(agent_data.response_mask)
        if committed_response_len > 0:
            response_ids = list(agent_data.prompt_ids[-committed_response_len:])
            prompt_ids = list(agent_data.prompt_ids[: len(agent_data.prompt_ids) - committed_response_len])
        else:
            response_ids = []
            prompt_ids = list(agent_data.prompt_ids)
        response_mask = list(agent_data.response_mask)
        response_logprobs = list(agent_data.response_logprobs) if agent_data.response_logprobs else None
        finalized_extra_fields = deepcopy(agent_data.extra_fields)

        pending_turn_state = self._get_pending_turn_state(agent_data)
        visible_pending_assistant_turns = 1 if pending_turn_state.tokens else 0
        if pending_turn_state.tokens:
            response_ids.extend(pending_turn_state.tokens)
            response_mask.extend([1] * len(pending_turn_state.tokens))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(pending_turn_state.tokens))
            if "teacher_ids_list" not in finalized_extra_fields or "teacher_logprobs_list" not in finalized_extra_fields:
                raise ValueError("SKD requires teacher row lists before finalizing a pending assistant turn")
            finalized_extra_fields["teacher_ids_list"] = list(finalized_extra_fields["teacher_ids_list"]) + [
                list(row) for row in pending_turn_state.teacher_ids_rows
            ]
            finalized_extra_fields["teacher_logprobs_list"] = list(
                finalized_extra_fields["teacher_logprobs_list"]
            ) + [list(row) for row in pending_turn_state.teacher_logprobs_rows]
        if "teacher_ids_list" not in finalized_extra_fields or "teacher_logprobs_list" not in finalized_extra_fields:
            raise ValueError("SKD requires teacher row lists in finalized output")
        if len(finalized_extra_fields["teacher_ids_list"]) != len(response_mask):
            raise ValueError(
                "SKD finalized output teacher_ids_list must align with response_mask: "
                f"teacher_rows={len(finalized_extra_fields['teacher_ids_list'])} response_mask={len(response_mask)}"
            )
        if len(finalized_extra_fields["teacher_logprobs_list"]) != len(response_mask):
            raise ValueError(
                "SKD finalized output teacher_logprobs_list must align with response_mask: "
                f"teacher_rows={len(finalized_extra_fields['teacher_logprobs_list'])} response_mask={len(response_mask)}"
            )

        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs is not None else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + visible_pending_assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=finalized_extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.setdefault("parent_request_id", None)
        return output

    async def _release_teacher_sticky_session(self, request_id: str) -> None:
        if self.teacher_server_manager is None:
            return
        release = getattr(self.teacher_server_manager, "release_sticky_session", None)
        if release is None:
            return
        result = release(request_id)
        if inspect.isawaitable(result):
            await result

    @rollout_trace_op
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
                    raise ValueError(f"Invalid state while running SKD loop: {state}")
            return self._finalize_boundary_agent_output(agent_data)
        finally:
            await self._release_teacher_sticky_session(agent_data.request_id)

    def _append_student_prompt_delta_to_teacher_stream(self, agent_data: AgentData, prev_prompt_len: int) -> None:
        prompt_delta = agent_data.prompt_ids[prev_prompt_len:]
        if not prompt_delta:
            return

        teacher_prompt_ids = self._require_teacher_prompt_stream(agent_data, "teacher_prompt_ids")
        teacher_prompt_ids.extend(prompt_delta)

        teacher_server_prompt_ids = self._require_teacher_prompt_stream(agent_data, "teacher_server_prompt_ids")
        teacher_server_prompt_ids.extend(prompt_delta)

    def _append_dummy_teacher_rows(self, agent_data: AgentData, count: int) -> None:
        """Keep SKD teacher targets aligned with response_mask for tool/user spans."""
        if count <= 0:
            return
        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            raise ValueError("SKD requires teacher_ids_list and teacher_logprobs_list before appending dummy rows")
        assert self.loss_top_k is not None and self.loss_top_k > 0, "SKD dummy rows require distillation topk > 0"

        teacher_ids_list = agent_data.extra_fields["teacher_ids_list"]
        teacher_logprobs_list = agent_data.extra_fields["teacher_logprobs_list"]
        teacher_ids_list.extend([[0] * self.loss_top_k for _ in range(count)])
        teacher_logprobs_list.extend([[0.0] * self.loss_top_k for _ in range(count)])

    def _validate_teacher_state_for_partial(self, agent_data: AgentData) -> None:
        self._require_teacher_prompt_stream(agent_data, "teacher_prompt_ids")
        self._require_teacher_prompt_stream(agent_data, "teacher_server_prompt_ids")
        _teacher_sglang_prefix_surplus_from_fields(
            agent_data.extra_fields,
            has_multimodal=bool(_safe_len(agent_data.image_data) or _safe_len(agent_data.video_data)),
        )

    @staticmethod
    def _require_teacher_prompt_stream(agent_data: AgentData, key: str) -> list[int]:
        stream = agent_data.extra_fields.get(key)
        if not isinstance(stream, list):
            raise ValueError(f"SKD requires {key} to be initialized before teacher verification")
        return stream

    def _get_pending_turn_state(self, agent_data: AgentData) -> SkdTurnChunkState:
        payload = agent_data.extra_fields.get(_SKD_PENDING_TURN_STATE)
        state = SkdTurnChunkState.from_payload(payload)
        if len(state.teacher_ids_rows) != len(state.tokens):
            raise ValueError(
                "Invalid SKD pending turn state: "
                f"teacher_ids_rows={len(state.teacher_ids_rows)} tokens={len(state.tokens)}"
            )
        if len(state.teacher_logprobs_rows) != len(state.tokens):
            raise ValueError(
                "Invalid SKD pending turn state: "
                f"teacher_logprobs_rows={len(state.teacher_logprobs_rows)} tokens={len(state.tokens)}"
            )
        return state

    def _set_pending_turn_state(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
        *,
        pending_chunks: int,
        clear_response_ids: bool = True,
    ) -> None:
        if turn_state.tokens:
            agent_data.extra_fields[_SKD_PENDING_TURN_STATE] = turn_state.to_payload()
            agent_data.extra_fields[_SKD_PENDING_TURN_CHUNKS] = int(pending_chunks)
            agent_data.response_ids = list(turn_state.tokens)
            return

        agent_data.extra_fields.pop(_SKD_PENDING_TURN_STATE, None)
        agent_data.extra_fields.pop(_SKD_PENDING_TURN_CHUNKS, None)
        if clear_response_ids:
            agent_data.response_ids = []

    async def _build_request_prompt_views_from_turn_state(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> tuple[list[int], list[int], list[int], int]:
        student_request_prompt_ids = await self._build_student_request_prompt_ids(
            agent_data,
            turn_state,
        )
        teacher_prompt_ids, teacher_server_prompt_ids, teacher_sglang_prefix_surplus = (
            await self._build_teacher_verify_request_view(agent_data, turn_state)
        )
        return (
            student_request_prompt_ids,
            teacher_prompt_ids,
            teacher_server_prompt_ids,
            teacher_sglang_prefix_surplus,
        )

    async def _build_student_request_prompt_ids(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> list[int]:
        return list(agent_data.prompt_ids) + list(turn_state.tokens)

    async def _build_teacher_verify_request_view(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
    ) -> tuple[list[int], list[int], int]:
        committed_teacher_prompt_ids = self._require_teacher_prompt_stream(agent_data, "teacher_prompt_ids")
        teacher_prompt_ids = list(committed_teacher_prompt_ids) + list(turn_state.tokens)
        committed_teacher_server_prompt_ids = self._require_teacher_prompt_stream(
            agent_data,
            "teacher_server_prompt_ids",
        )
        teacher_server_prompt_ids = list(committed_teacher_server_prompt_ids) + list(turn_state.tokens)
        has_teacher_multimodal = bool(_safe_len(agent_data.image_data) or _safe_len(agent_data.video_data))
        teacher_sglang_prefix_surplus = _teacher_sglang_prefix_surplus_from_fields(
            agent_data.extra_fields,
            has_multimodal=has_teacher_multimodal,
        )
        return teacher_prompt_ids, teacher_server_prompt_ids, teacher_sglang_prefix_surplus

    def _commit_pending_turn_state(
        self,
        agent_data: AgentData,
        turn_state: SkdTurnChunkState,
        *,
        pending_chunks: int,
        finalize_assistant_turn: bool,
    ) -> None:
        agent_data.response_ids = list(turn_state.tokens)
        if not turn_state.tokens:
            turn_state.raw_chunk = []
            turn_state.verified_chunk = []
            self._set_pending_turn_state(agent_data, turn_state, pending_chunks=0)
            return

        teacher_ids_list = agent_data.extra_fields.setdefault("teacher_ids_list", [])
        teacher_logprobs_list = agent_data.extra_fields.setdefault("teacher_logprobs_list", [])
        assert len(turn_state.teacher_ids_rows) == len(turn_state.tokens), (
            "[SKD] pending teacher_ids_rows must stay token aligned: "
            f"rows={len(turn_state.teacher_ids_rows)} tokens={len(turn_state.tokens)}"
        )
        assert len(turn_state.teacher_logprobs_rows) == len(turn_state.tokens), (
            "[SKD] pending teacher_logprobs_rows must stay token aligned: "
            f"rows={len(turn_state.teacher_logprobs_rows)} tokens={len(turn_state.tokens)}"
        )

        agent_data.prompt_ids.extend(turn_state.tokens)
        agent_data.response_mask.extend([1] * len(turn_state.tokens))
        teacher_ids_list.extend([list(row) for row in turn_state.teacher_ids_rows])
        teacher_logprobs_list.extend([list(row) for row in turn_state.teacher_logprobs_rows])
        for field_name in ("teacher_prompt_ids", "teacher_server_prompt_ids"):
            prompt_ids = agent_data.extra_fields.get(field_name)
            if prompt_ids is not None:
                prompt_ids.extend(turn_state.tokens)
        agent_data.response_ids = list(turn_state.tokens)
        if finalize_assistant_turn:
            agent_data.assistant_turns += 1
        self._increment_skd_prefix_stats(agent_data, gen_chunks=pending_chunks, tokens=len(turn_state.tokens))
        turn_state.tokens = []
        turn_state.teacher_ids_rows = []
        turn_state.teacher_logprobs_rows = []
        turn_state.raw_chunk = []
        turn_state.verified_chunk = []
        self._set_pending_turn_state(agent_data, turn_state, pending_chunks=0, clear_response_ids=False)
        self._assert_teacher_alignment(agent_data)

    def _teacher_max_model_len(self, routing_key: Any = None) -> int | None:
        """Return the configured teacher context limit when the manager exposes it."""
        if self.teacher_server_manager is None:
            return None
        get_max_model_len = getattr(self.teacher_server_manager, "max_model_len_for_routing_key", None)
        if get_max_model_len is not None:
            max_model_len = get_max_model_len(routing_key)
            return int(max_model_len) if max_model_len is not None else None

        # Test and probe managers may not implement the formal accessor yet.
        # Fall back to the same public config shape used by the production
        # manager instead of assuming a specific concrete class.
        teacher_model_configs = getattr(self.teacher_server_manager, "teacher_model_configs", None)
        if not teacher_model_configs:
            return None
        if len(teacher_model_configs) == 1:
            teacher_config = next(iter(teacher_model_configs.values()))
        elif routing_key in teacher_model_configs:
            teacher_config = teacher_model_configs[routing_key]
        else:
            return None
        max_model_len = getattr(getattr(teacher_config, "inference", None), "max_model_len", None)
        return int(max_model_len) if max_model_len is not None else None

    def _teacher_request_overflows(
        self,
        *,
        sequence_len: int,
        routing_key: Any = None,
    ) -> tuple[bool, int | None, int]:
        """Check SGLang teacher request length before sending it.

        The teacher logprob request still asks SGLang to generate one token
        while returning prompt logprobs, so the server-side budget is
        ``len(prompt_ids) + 1 <= max_model_len``.
        """
        required_len = sequence_len + 1
        max_model_len = self._teacher_max_model_len(routing_key)
        if max_model_len is None:
            return False, None, required_len
        return required_len > max_model_len, max_model_len, required_len

    def _teacher_future_verify_overflows(
        self,
        *,
        prefix_len: int,
        routing_key: Any = None,
        min_future_chunk_len: int = 1,
    ) -> tuple[bool, int | None, int]:
        """Check whether a committed teacher prefix leaves room for any next verified token."""
        return self._teacher_request_overflows(
            sequence_len=prefix_len + min_future_chunk_len,
            routing_key=routing_key,
        )

    @staticmethod
    def _current_multi_modal_data(agent_data: AgentData) -> dict[str, Any]:
        """Build current multimodal context for student/teacher rollout calls."""
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        return multi_modal_data

    def _increment_skd_prefix_stats(
        self,
        agent_data: AgentData,
        *,
        gen_chunks: int = 0,
        env_units: int = 0,
        tokens: int = 0,
    ) -> None:
        """Track committed prefix size without changing rollout semantics."""
        extra_fields = agent_data.extra_fields
        extra_fields["skd_committed_gen_chunks"] = extra_fields.get("skd_committed_gen_chunks", 0) + gen_chunks
        extra_fields["skd_committed_env_units"] = extra_fields.get("skd_committed_env_units", 0) + env_units
        extra_fields["skd_committed_prefix_tokens"] = extra_fields.get("skd_committed_prefix_tokens", 0) + tokens

    def _record_rollout_version_from_output(self, agent_data: AgentData, output: Any) -> None:
        """Track min/max rollout model versions reported by the inference engine."""
        extra = getattr(output, "extra_fields", None) or {}
        version_values = []
        for key in ("min_global_steps", "global_steps", "max_global_steps"):
            value = extra.get(key)
            if value is None:
                continue
            try:
                version_values.append(int(value))
            except (TypeError, ValueError):
                continue
        if not version_values:
            return

        current_min = agent_data.extra_fields.get("rollout_min_version")
        current_max = agent_data.extra_fields.get("rollout_max_version")
        new_min = min(version_values) if current_min is None else min(int(current_min), min(version_values))
        new_max = max(version_values) if current_max is None else max(int(current_max), max(version_values))
        agent_data.extra_fields["rollout_min_version"] = new_min
        agent_data.extra_fields["rollout_max_version"] = new_max
        agent_data.extra_fields.setdefault("rollout_birth_version", new_min)

    def _is_qwen_hermes_exportable_assistant_prefix(self, agent_data: AgentData) -> bool:
        """Return whether current assistant prefix may be exported before the next chunk.

        Tool parsing still only happens after EOS in ``_handle_generating_state``.
        A closed ``</tool_call>`` without EOS is therefore just a resumable
        generation prefix, not an executable tool call.
        """
        return True

    def _can_export_partial_state(self, agent_data: AgentData, next_state: AgentState) -> bool:
        """Return whether the current trajectory can be snapshotted for resume."""
        if next_state != AgentState.GENERATING:
            return False

        teacher_ids_list = agent_data.extra_fields.get("teacher_ids_list")
        teacher_logprobs_list = agent_data.extra_fields.get("teacher_logprobs_list")
        if teacher_ids_list is None or teacher_logprobs_list is None:
            return False
        if len(agent_data.response_mask) != len(teacher_ids_list):
            return False
        if len(agent_data.response_mask) != len(teacher_logprobs_list):
            return False
        return self._is_qwen_hermes_exportable_assistant_prefix(agent_data)

    def _export_partial_state(
        self,
        agent_data: AgentData,
        next_state: AgentState,
        *,
        sample_id: str,
        logical_step: int,
        source_type: str,
    ) -> SkdPartialState:
        """Export an unfinished trajectory snapshot at a committed-unit boundary."""
        self._validate_teacher_state_for_partial(agent_data)
        if not self._can_export_partial_state(agent_data, next_state):
            raise ValueError(
                "Cannot export SKD partial state: "
                f"next_state={next_state}, "
                f"response_len={len(agent_data.response_mask)}, "
                f"teacher_rows={len(agent_data.extra_fields.get('teacher_ids_list', []))}"
            )
        self._assert_teacher_alignment(agent_data)

        extra_fields = deepcopy(agent_data.extra_fields)
        partial = SkdPartialState(
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            agent_state=next_state.value,
            request_id=agent_data.request_id,
            tools_kwargs=deepcopy(agent_data.tools_kwargs),
            messages=deepcopy(agent_data.messages),
            prompt_ids=list(agent_data.prompt_ids),
            teacher_prompt_ids=list(self._require_teacher_prompt_stream(agent_data, "teacher_prompt_ids")),
            response_ids=list(agent_data.response_ids),
            response_mask=list(agent_data.response_mask),
            response_logprobs=list(agent_data.response_logprobs),
            assistant_turns=agent_data.assistant_turns,
            user_turns=agent_data.user_turns,
            tool_rewards=list(agent_data.tool_rewards),
            turn_scores=list(agent_data.turn_scores),
            rollout_birth_version=extra_fields.get("rollout_birth_version"),
            rollout_min_version=extra_fields.get("rollout_min_version"),
            rollout_max_version=extra_fields.get("rollout_max_version"),
            committed_gen_chunks=int(extra_fields.get("skd_committed_gen_chunks", 0)),
            committed_env_units=int(extra_fields.get("skd_committed_env_units", 0)),
            committed_prefix_tokens=int(extra_fields.get("skd_committed_prefix_tokens", 0)),
            metrics=deepcopy(agent_data.metrics),
            extra_fields=extra_fields,
            image_data=deepcopy(agent_data.image_data),
            video_data=deepcopy(agent_data.video_data),
        )
        _trace_async_skd(
            "loop.export_partial_state",
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            request_id=agent_data.request_id,
            teacher_replica_id=partial.extra_fields.get("teacher_replica_id"),
            teacher_routing_key=partial.extra_fields.get("teacher_routing_key"),
            response_len=len(partial.response_mask),
            committed_gen_chunks=partial.committed_gen_chunks,
            committed_prefix_tokens=partial.committed_prefix_tokens,
        )
        return partial

    def _restore_partial_state(self, partial_state: SkdPartialState) -> tuple[AgentData, AgentState]:
        """Restore a previously exported SKD partial snapshot."""
        try:
            next_state = AgentState(partial_state.agent_state)
        except ValueError as exc:
            raise ValueError(f"Invalid SKD partial state agent_state={partial_state.agent_state!r}") from exc

        if next_state != AgentState.GENERATING:
            raise ValueError(f"Cannot restore SKD partial state into unsupported next_state={next_state}")

        agent_data = AgentData(
            messages=deepcopy(partial_state.messages),
            image_data=deepcopy(partial_state.image_data),
            video_data=deepcopy(partial_state.video_data),
            metrics=deepcopy(partial_state.metrics),
            request_id=partial_state.request_id,
            tools_kwargs=deepcopy(partial_state.tools_kwargs),
        )
        agent_data.prompt_ids = list(partial_state.prompt_ids)
        agent_data.response_ids = list(partial_state.response_ids)
        agent_data.response_mask = list(partial_state.response_mask)
        agent_data.response_logprobs = list(partial_state.response_logprobs)
        agent_data.assistant_turns = partial_state.assistant_turns
        agent_data.user_turns = partial_state.user_turns
        agent_data.tool_rewards = list(partial_state.tool_rewards)
        agent_data.turn_scores = list(partial_state.turn_scores)
        agent_data.extra_fields = deepcopy(partial_state.extra_fields)

        # Keep the structured dataclass fields authoritative for restore.  The
        # same values are mirrored into extra_fields because the existing SKD
        # loss reconstruction path consumes them from there.
        agent_data.extra_fields["teacher_prompt_ids"] = list(partial_state.teacher_prompt_ids)
        if partial_state.rollout_birth_version is not None:
            agent_data.extra_fields["rollout_birth_version"] = partial_state.rollout_birth_version
        if partial_state.rollout_min_version is not None:
            agent_data.extra_fields["rollout_min_version"] = partial_state.rollout_min_version
        if partial_state.rollout_max_version is not None:
            agent_data.extra_fields["rollout_max_version"] = partial_state.rollout_max_version
        agent_data.extra_fields["skd_committed_gen_chunks"] = partial_state.committed_gen_chunks
        agent_data.extra_fields["skd_committed_env_units"] = partial_state.committed_env_units
        agent_data.extra_fields["skd_committed_prefix_tokens"] = partial_state.committed_prefix_tokens
        self._validate_teacher_state_for_partial(agent_data)

        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            raise ValueError("Invalid SKD partial state: missing teacher row lists in extra_fields")
        pending_turn_state = self._get_pending_turn_state(agent_data)
        pending_chunks = int(agent_data.extra_fields.get(_SKD_PENDING_TURN_CHUNKS, 0))
        self._set_pending_turn_state(
            agent_data,
            pending_turn_state,
            pending_chunks=pending_chunks,
        )
        self._assert_teacher_alignment(agent_data)
        _trace_async_skd(
            "loop.restore_partial_state",
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=partial_state.source_type,
            request_id=partial_state.request_id,
            teacher_replica_id=agent_data.extra_fields.get("teacher_replica_id"),
            teacher_routing_key=agent_data.extra_fields.get("teacher_routing_key"),
            response_len=len(agent_data.response_mask),
            committed_gen_chunks=partial_state.committed_gen_chunks,
            committed_prefix_tokens=partial_state.committed_prefix_tokens,
        )
        return agent_data, next_state

    async def _run_until_exportable_boundary(
        self,
        agent_data: AgentData,
        state: AgentState,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        """Advance a trajectory until it is completed or safe to export.

        This is the cooperative pause driver for lookahead execution.  It never
        interrupts inside an SKD chunk, teacher verification, tool execution, or
        dummy-row append.  It returns only at ``TERMINATED`` or at a boundary
        accepted by ``_can_export_partial_state``.
        """
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
                continue
            if state == AgentState.GENERATING:
                state = await self._handle_generating_state(
                    agent_data,
                    sampling_params,
                    stop_after_skd_chunk=True,
                )
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                raise ValueError(f"Invalid AgentState while advancing SKD boundary: {state}")

            if self._can_export_partial_state(agent_data, state):
                return state

        return state

    async def _run_until_terminated(
        self,
        agent_data: AgentData,
        state: AgentState,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        """Advance a restored SKD trajectory to terminal state without boundary pausing."""
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                raise ValueError(f"Invalid AgentState while completing SKD partial: {state}")

        return state

    async def run_until_exportable_boundary(
        self,
        sampling_params: dict[str, Any],
        *,
        sample_id: str,
        logical_step: int,
        source_type: str,
        partial_state: SkdPartialState | None = None,
        **kwargs: Any,
    ) -> AgentLoopOutput | SkdPartialState:
        """Run a fresh or resumed SKD trajectory until completion or exportable boundary."""
        if partial_state is None:
            agent_data = await self._init_boundary_agent_data(**kwargs)
            state = AgentState.PENDING
        else:
            agent_data, state = self._restore_partial_state(partial_state)

        try:
            next_state = await self._run_until_exportable_boundary(agent_data, state, sampling_params)
            if next_state == AgentState.TERMINATED:
                await self._release_teacher_sticky_session(agent_data.request_id)
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
            raise

    async def run_from_partial_to_completion(
        self,
        sampling_params: dict[str, Any],
        *,
        partial_state: SkdPartialState,
    ) -> AgentLoopOutput:
        """Resume a partial SKD trajectory and run it to terminal completion."""
        agent_data, state = self._restore_partial_state(partial_state)
        parent_request_id = agent_data.request_id
        agent_data.extra_fields["parent_request_id"] = parent_request_id
        agent_data.request_id = uuid4().hex
        _trace_async_skd(
            "loop.resume_request_rebind",
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=partial_state.source_type,
            parent_request_id=parent_request_id,
            new_request_id=agent_data.request_id,
            teacher_replica_id=agent_data.extra_fields.get("teacher_replica_id"),
            teacher_routing_key=agent_data.extra_fields.get("teacher_routing_key"),
        )
        try:
            await self._run_until_terminated(agent_data, state, sampling_params)
            return self._finalize_boundary_agent_output(agent_data)
        finally:
            for request_id in (agent_data.request_id, parent_request_id):
                await self._release_teacher_sticky_session(request_id)

    def _assert_teacher_alignment(self, agent_data: AgentData) -> None:
        """Validate that response_mask and teacher rows stay response-token aligned."""
        if "teacher_ids_list" not in agent_data.extra_fields or "teacher_logprobs_list" not in agent_data.extra_fields:
            return

        teacher_ids_list = agent_data.extra_fields["teacher_ids_list"]
        teacher_logprobs_list = agent_data.extra_fields["teacher_logprobs_list"]
        response_len = len(agent_data.response_mask)
        assert len(teacher_ids_list) == response_len, (
            f"[SKD] teacher_ids_list length {len(teacher_ids_list)} != response_mask length {response_len}"
        )
        assert len(teacher_logprobs_list) == response_len, (
            f"[SKD] teacher_logprobs_list length {len(teacher_logprobs_list)} != response_mask length {response_len}"
        )

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Initialize separate student and teacher prompt streams."""
        del sampling_params
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids

        teacher_messages = self._build_teacher_messages(agent_data.messages)
        teacher_images = self._teacher_images_with_runtime(agent_data.image_data)
        if teacher_messages == agent_data.messages:
            teacher_prompt_ids = list(prompt_ids)
        else:
            teacher_prompt_ids = await self.apply_chat_template(
                teacher_messages,
                tools=schemas,
                images=teacher_images,
                videos=agent_data.video_data,
            )
        agent_data.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
        agent_data.extra_fields["teacher_server_prompt_ids"] = list(teacher_prompt_ids)
        agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 0
        return AgentState.GENERATING

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        prev_prompt_len = len(agent_data.prompt_ids)
        prev_response_len = len(agent_data.response_mask)
        next_state = await super()._handle_processing_tools_state(agent_data)
        self._append_student_prompt_delta_to_teacher_stream(agent_data, prev_prompt_len)
        appended_len = len(agent_data.response_mask) - prev_response_len
        self._append_dummy_teacher_rows(agent_data, appended_len)
        self._assert_teacher_alignment(agent_data)
        if appended_len > 0:
            self._increment_skd_prefix_stats(agent_data, env_units=1, tokens=appended_len)
        return next_state

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
        stop_after_skd_chunk: bool = False,
    ) -> AgentState:
        """SKD chunk-based generation with Teacher verification.

        Replaces ToolAgentLoop's single-shot generation with:
        1. Student generates a chunk of tokens
        2. Teacher verifies each token against its top-K
        3. First rejected token is replaced with Teacher's top-1
        4. Repeat from the replacement point
        5. Teacher logprobs are accumulated during verification (no separate pass needed)
        """
        if self.teacher_server_manager is None:
            raise ValueError("SKD requires teacher_server_manager for teacher verification")

        sample_start_time = time.monotonic()

        # Initialize SKD metrics
        skd_metrics = agent_data.metrics.setdefault("skd", {
            "accept_count": 0,
            "reject_count": 0,
            "chunk_count": 0,
            "student_gen_ms": 0.0,
            "teacher_verify_ms": 0.0,
        })

        # Initialize teacher logprobs accumulation lists in extra_fields
        teacher_ids_list = agent_data.extra_fields.setdefault("teacher_ids_list", [])
        teacher_logprobs_list = agent_data.extra_fields.setdefault("teacher_logprobs_list", [])

        # Keep the unfinished assistant turn outside the committed rollout
        # state so partial export/restore can carry it directly.
        turn_state = self._get_pending_turn_state(agent_data)
        pending_turn_chunks = int(agent_data.extra_fields.get(_SKD_PENDING_TURN_CHUNKS, 0))

        # Debug: token-level alignment logging for first N samples
        SkdAgentLoop._debug_sample_count += 1
        do_token_debug = _SKD_DEBUG >= 2 and SkdAgentLoop._debug_sample_count <= 3
        termination_reason = "unknown"
        chunk_output = None

        initial_prompt_len = len(agent_data.prompt_ids)

        # === SKD chunk loop ===
        with simple_timer("skd_generate", agent_data.metrics):
            while True:
                current_response_len = len(agent_data.response_mask) + len(turn_state.tokens)
                remaining_budget = self.response_length - current_response_len
                if remaining_budget <= 0:
                    termination_reason = "budget_exhausted"
                    break
                actual_chunk_size = min(self.skd_chunk_size, remaining_budget)
                (
                    student_request_prompt_ids,
                    teacher_prompt_ids,
                    teacher_server_prompt_ids,
                    teacher_sglang_prefix_surplus,
                ) = await self._build_request_prompt_views_from_turn_state(agent_data, turn_state)

                # 1. Student generates a chunk
                next_chunk_idx = skd_metrics["chunk_count"] + 1
                _trace_async_skd(
                    "loop.student_generate_begin",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    server_prompt_len=len(student_request_prompt_ids),
                    response_len=current_response_len,
                    max_tokens=actual_chunk_size,
                    image_count=_safe_len(agent_data.image_data),
                    video_count=_safe_len(agent_data.video_data),
                )
                request_input_kind = "input_ids"
                _trace_async_skd(
                    "loop.student_generate_request_view",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    request_input_kind=request_input_kind,
                    server_prompt_len=len(student_request_prompt_ids),
                    image_count=_safe_len(agent_data.image_data),
                )
                chunk_t0 = time.monotonic()
                with simple_timer("skd_student_chunk", agent_data.metrics):
                    _trace_async_skd(
                        "loop.student_generate_await_begin",
                        request_id=agent_data.request_id,
                        chunk_idx=next_chunk_idx,
                        request_input_kind=request_input_kind,
                        server_prompt_len=len(student_request_prompt_ids),
                        max_tokens=actual_chunk_size,
                        image_count=_safe_len(agent_data.image_data),
                        video_count=_safe_len(agent_data.video_data),
                    )
                    try:
                        student_sampling_params = {**sampling_params, "max_tokens": actual_chunk_size}
                        if "stop_token_ids" not in student_sampling_params:
                            eos_token_id = self.tokenizer.eos_token_id
                            if eos_token_id is not None:
                                student_sampling_params["stop_token_ids"] = (
                                    list(eos_token_id) if isinstance(eos_token_id, list) else [eos_token_id]
                                )
                        chunk_output = await self.server_manager.generate(
                            request_id=agent_data.request_id,
                            prompt_ids=student_request_prompt_ids,
                            sampling_params=student_sampling_params,
                            image_data=agent_data.image_data,
                            video_data=agent_data.video_data,
                        )
                    except Exception as exc:
                        student_error_ms = (time.monotonic() - chunk_t0) * 1000
                        _trace_async_skd(
                            "loop.student_generate_await_error",
                            request_id=agent_data.request_id,
                            chunk_idx=next_chunk_idx,
                            elapsed_ms=round(student_error_ms, 1),
                            error_type=type(exc).__name__,
                            error=repr(exc),
                        )
                        raise
                    _trace_async_skd(
                        "loop.student_generate_await_done",
                        request_id=agent_data.request_id,
                        chunk_idx=next_chunk_idx,
                        elapsed_ms=round((time.monotonic() - chunk_t0) * 1000, 1),
                        output_len=len(chunk_output.token_ids),
                        stop_reason=chunk_output.stop_reason,
                    )
                self._record_rollout_version_from_output(agent_data, chunk_output)
                chunk = chunk_output.token_ids
                student_ms = (time.monotonic() - chunk_t0) * 1000
                skd_metrics["student_gen_ms"] += student_ms
                _trace_async_skd(
                    "loop.student_generate_done",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    elapsed_ms=round(student_ms, 1),
                    output_len=len(chunk),
                    stop_reason=chunk_output.stop_reason,
                    server_prompt_len=len(student_request_prompt_ids),
                )

                if not chunk:
                    termination_reason = "empty_chunk"
                    break

                # Track num_preempted (same pattern as ToolAgentLoop)
                if agent_data.metrics.get("num_preempted") is None:
                    agent_data.metrics["num_preempted"] = (
                        chunk_output.num_preempted if chunk_output.num_preempted is not None else -1
                    )
                else:
                    agent_data.metrics["num_preempted"] += (
                        chunk_output.num_preempted if chunk_output.num_preempted is not None else 0
                    )

                # 2. Teacher verification
                teacher_t0 = time.monotonic()
                with simple_timer("skd_teacher_verify", agent_data.metrics):
                    teacher_replica_id = agent_data.extra_fields.get("teacher_replica_id")
                    teacher_routing_key = agent_data.extra_fields.get("teacher_routing_key")
                    bind_sticky_request = getattr(self.teacher_server_manager, "bind_sticky_request", None)
                    _trace_async_skd(
                        "loop.teacher_bind_attempt",
                        request_id=agent_data.request_id,
                        teacher_replica_id=teacher_replica_id,
                        teacher_routing_key=teacher_routing_key,
                        teacher_prompt_len=len(teacher_prompt_ids),
                        teacher_server_prompt_len=len(teacher_server_prompt_ids),
                        teacher_mm_prefix_surplus=max(len(teacher_prompt_ids) - len(teacher_server_prompt_ids), 0),
                        chunk_len=len(chunk),
                    )
                    if (
                        teacher_replica_id is not None
                        and bind_sticky_request is not None
                    ):
                        result = bind_sticky_request(
                            routing_key=teacher_routing_key,
                            request_id=agent_data.request_id,
                            server_id=str(teacher_replica_id),
                        )
                        if inspect.isawaitable(result):
                            await result
                        _trace_async_skd(
                            "loop.teacher_bind_applied",
                            request_id=agent_data.request_id,
                            teacher_replica_id=teacher_replica_id,
                            teacher_routing_key=teacher_routing_key,
                        )
                    verify_sequence = teacher_server_prompt_ids + chunk
                    teacher_overflow, teacher_max_model_len, teacher_required_len = self._teacher_request_overflows(
                        sequence_len=len(verify_sequence),
                        routing_key=teacher_routing_key,
                    )
                    if teacher_overflow:
                        termination_reason = "teacher_context_exhausted"
                        agent_data.extra_fields["skd_termination_reason"] = termination_reason
                        agent_data.extra_fields["skd_teacher_context_required_len"] = teacher_required_len
                        agent_data.extra_fields["skd_teacher_max_model_len"] = teacher_max_model_len
                        _trace_async_skd(
                            "loop.teacher_context_exhausted_before_verify",
                            request_id=agent_data.request_id,
                            teacher_replica_id=teacher_replica_id,
                            teacher_routing_key=teacher_routing_key,
                            teacher_server_prompt_len=len(teacher_server_prompt_ids),
                            chunk_len=len(chunk),
                            required_len=teacher_required_len,
                            max_model_len=teacher_max_model_len,
                        )
                        warnings.warn(
                            "[SKD] terminating before teacher verify because teacher context would overflow: "
                            f"req={agent_data.request_id} "
                            f"teacher_server_prompt_len={len(teacher_server_prompt_ids)} "
                            f"chunk_len={len(chunk)} "
                            f"required_len={teacher_required_len} "
                            f"max_model_len={teacher_max_model_len}",
                            stacklevel=1,
                        )
                        break
                    multi_modal_data = self._current_multi_modal_data(agent_data)
                    teacher_logprob_range = _build_teacher_logprob_range(
                        teacher_server_prompt_len=len(teacher_server_prompt_ids),
                        teacher_sglang_prefix_surplus=teacher_sglang_prefix_surplus,
                        chunk_len=len(chunk),
                    )
                    logprob_start_len = teacher_logprob_range.sglang_logprob_start_len
                    expected_mm_prefix_surplus = teacher_logprob_range.teacher_sglang_prefix_surplus
                    expected_suffix_len = teacher_logprob_range.expected_logprob_rows
                    _trace_async_skd(
                        "loop.teacher_compute_begin",
                        request_id=agent_data.request_id,
                        chunk_idx=next_chunk_idx,
                        teacher_replica_id=teacher_replica_id,
                        teacher_routing_key=teacher_routing_key,
                        seq_len=len(verify_sequence),
                        teacher_server_prompt_len=len(teacher_server_prompt_ids),
                        logprob_start_len=logprob_start_len,
                        expected_suffix_len=expected_suffix_len,
                        expected_mm_prefix_surplus=expected_mm_prefix_surplus,
                        server_logical_start_len=teacher_logprob_range.server_logical_start_len,
                        sglang_logprob_start_len=teacher_logprob_range.sglang_logprob_start_len,
                        expected_logprob_rows=teacher_logprob_range.expected_logprob_rows,
                        teacher_sglang_prefix_surplus=teacher_logprob_range.teacher_sglang_prefix_surplus,
                        image_count=_safe_len(multi_modal_data.get("images")),
                    )
                    if _SKD_CHUNK_TRACE > 0:
                        emit_async_skd_event(
                            "teacher_verify_request",
                            request_id=agent_data.request_id,
                            chunk_idx=next_chunk_idx,
                            chunk_len=len(chunk),
                            seq_len=len(verify_sequence),
                            teacher_prompt_len=len(teacher_prompt_ids),
                            teacher_server_prompt_len=len(teacher_server_prompt_ids),
                            logprob_start_len=logprob_start_len,
                            expected_logprob_rows=expected_suffix_len,
                            expected_mm_prefix_surplus=expected_mm_prefix_surplus,
                            teacher_server_prompt_tail=_token_edges(list(teacher_server_prompt_ids)),
                            chunk_head=list(chunk[:8]),
                            chunk_tail=list(chunk[-8:]),
                            verify_sequence_tail=_token_edges(list(verify_sequence), head=0, tail=16),
                        )
                    teacher_await_t0 = time.monotonic()
                    _trace_async_skd(
                        "loop.teacher_compute_await_begin",
                        request_id=agent_data.request_id,
                        chunk_idx=next_chunk_idx,
                        teacher_replica_id=teacher_replica_id,
                        teacher_routing_key=teacher_routing_key,
                        seq_len=len(verify_sequence),
                        logprob_start_len=logprob_start_len,
                        expected_suffix_len=expected_suffix_len,
                        expected_mm_prefix_surplus=expected_mm_prefix_surplus,
                        image_count=_safe_len(multi_modal_data.get("images")),
                    )
                    try:
                        teacher_ids, teacher_logprobs = (
                            await self.teacher_server_manager.compute_teacher_logprobs_single(
                                request_id=agent_data.request_id,
                                sequence_ids=verify_sequence,
                                logprob_start_len=logprob_start_len,
                                expected_mm_prefix_surplus=expected_mm_prefix_surplus,
                                expected_logprob_rows=teacher_logprob_range.expected_logprob_rows,
                                multi_modal_data=multi_modal_data,
                                routing_key=teacher_routing_key,
                            )
                        )
                    except Exception as exc:
                        _trace_async_skd(
                            "loop.teacher_compute_await_error",
                            request_id=agent_data.request_id,
                            chunk_idx=next_chunk_idx,
                            elapsed_ms=round((time.monotonic() - teacher_await_t0) * 1000, 1),
                            error_type=type(exc).__name__,
                            error=repr(exc),
                        )
                        raise
                    _trace_async_skd(
                        "loop.teacher_compute_await_done",
                        request_id=agent_data.request_id,
                        chunk_idx=next_chunk_idx,
                        elapsed_ms=round((time.monotonic() - teacher_await_t0) * 1000, 1),
                        teacher_rows=int(teacher_ids.shape[0]),
                        teacher_width=int(teacher_ids.shape[1]) if teacher_ids.dim() > 1 else 1,
                    )
                teacher_ms = (time.monotonic() - teacher_t0) * 1000
                skd_metrics["teacher_verify_ms"] += teacher_ms
                _trace_async_skd(
                    "loop.teacher_compute_done",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    elapsed_ms=round(teacher_ms, 1),
                    teacher_rows=int(teacher_ids.shape[0]),
                    teacher_width=int(teacher_ids.shape[1]) if teacher_ids.dim() > 1 else 1,
                    expected_suffix_len=expected_suffix_len,
                    expected_mm_prefix_surplus=expected_mm_prefix_surplus,
                    teacher_sglang_prefix_surplus=teacher_logprob_range.teacher_sglang_prefix_surplus,
                    server_logical_start_len=teacher_logprob_range.server_logical_start_len,
                    sglang_logprob_start_len=teacher_logprob_range.sglang_logprob_start_len,
                    expected_logprob_rows=teacher_logprob_range.expected_logprob_rows,
                )
                self._emit_teacher_chunk_debug_event(
                    event_name="teacher_verify_rows",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    chunk=chunk,
                    new_tokens=[],
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    rejection_pos=None,
                    extra_fields={
                        "teacher_rows": int(teacher_ids.shape[0]),
                        "teacher_width": int(teacher_ids.shape[1]) if teacher_ids.dim() > 1 else 1,
                    },
                )
                # In SGLang delta mode, teacher_ids / teacher_logprobs only cover the
                # chunk suffix rows needed by SKD, aligned so local row k supervises
                # chunk token k.

                # 3. Accept/Reject
                accept_t0 = time.monotonic()
                _trace_async_skd(
                    "loop.accept_reject_begin",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    chunk_len=len(chunk),
                    teacher_rows=int(teacher_ids.shape[0]),
                    verify_top_k=self.skd_verify_top_k,
                )
                chunk_start = len(agent_data.prompt_ids)
                rejection_pos = None
                for k in range(len(chunk)):
                    teacher_idx = k
                    if teacher_idx < 0 or teacher_idx >= teacher_ids.shape[0]:
                        break
                    # Check if student token is in teacher's top-K for verification
                    teacher_topk_at_pos = teacher_ids[teacher_idx, : self.skd_verify_top_k].tolist()
                    if chunk[k] not in teacher_topk_at_pos:
                        rejection_pos = k
                        break
                accept_ms = (time.monotonic() - accept_t0) * 1000
                _trace_async_skd(
                    "loop.accept_reject_done",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    elapsed_ms=round(accept_ms, 1),
                    rejection_pos=rejection_pos,
                    chunk_len=len(chunk),
                )

                # 4. Accumulate tokens + teacher logprobs
                replacement_t0 = time.monotonic()
                if rejection_pos is not None:
                    accepted_tokens = list(chunk[:rejection_pos])
                    teacher_replacement_idx = rejection_pos
                    teacher_replacement = int(teacher_ids[teacher_replacement_idx, 0].item())
                    new_tokens = accepted_tokens + [teacher_replacement]
                    skd_metrics["accept_count"] += len(accepted_tokens)
                    skd_metrics["reject_count"] += 1
                else:
                    new_tokens = list(chunk)
                    skd_metrics["accept_count"] += len(new_tokens)
                replacement_ms = (time.monotonic() - replacement_t0) * 1000
                _trace_async_skd(
                    "loop.replacement_done",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    elapsed_ms=round(replacement_ms, 1),
                    rejection_pos=rejection_pos,
                    new_tokens_len=len(new_tokens),
                    accepted_len=len(new_tokens) - (1 if rejection_pos is not None else 0),
                )
                self._emit_teacher_chunk_debug_event(
                    event_name="teacher_replacement",
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    chunk=chunk,
                    new_tokens=new_tokens,
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    rejection_pos=rejection_pos,
                )
                self._log_teacher_chunk_debug_console(
                    request_id=agent_data.request_id,
                    chunk_idx=next_chunk_idx,
                    chunk=chunk,
                    new_tokens=new_tokens,
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    rejection_pos=rejection_pos,
                )

                skd_metrics["chunk_count"] += 1

                # === Debug: token-level alignment verification ===
                if do_token_debug and skd_metrics["chunk_count"] <= 3:
                    self._log_token_alignment(
                        chunk, new_tokens, rejection_pos,
                        teacher_ids, chunk_start, agent_data.request_id,
                        skd_metrics["chunk_count"],
                    )

                # === Debug: per-chunk timing log ===
                if _SKD_DEBUG >= 1:
                    acc = len(new_tokens) - (1 if rejection_pos is not None else 0)
                    warnings.warn(
                        f"[SKD_DBG] chunk={skd_metrics['chunk_count']:>3} "
                        f"student={student_ms:>7.1f}ms teacher={teacher_ms:>7.1f}ms "
                        f"teacher_prefix_len={len(teacher_prompt_ids)} "
                        f"teacher_server_prefix_len={len(teacher_server_prompt_ids)} "
                        f"teacher_mm_prefix_surplus={expected_mm_prefix_surplus} "
                        f"teacher_server_start={teacher_logprob_range.server_logical_start_len} "
                        f"teacher_sglang_start={teacher_logprob_range.sglang_logprob_start_len} "
                        f"teacher_suffix_len={len(chunk)} "
                        f"teacher_seq_len={len(verify_sequence)} "
                        f"chunk_len={len(chunk)} accepted={acc} rejected={1 if rejection_pos is not None else 0} "
                        f"new_tokens={len(new_tokens)} "
                        f"prompt_len={chunk_start} total_resp={len(agent_data.response_mask) + len(turn_state.tokens) + len(new_tokens)} "
                        f"req={agent_data.request_id}",
                        stacklevel=1,
                    )

                # Keep verified chunk state turn-local until EOS finalization.
                commit_t0 = time.monotonic()
                _trace_async_skd(
                    "loop.chunk_commit_begin",
                    request_id=agent_data.request_id,
                    chunk_idx=skd_metrics["chunk_count"],
                    new_tokens_len=len(new_tokens),
                    response_len_before=len(agent_data.response_mask),
                    prompt_len_before=len(agent_data.prompt_ids),
                    teacher_rows=int(teacher_ids.shape[0]),
                )
                pending_teacher_id_rows = []
                pending_teacher_logprob_rows = []
                for k in range(len(new_tokens)):
                    teacher_idx = k
                    assert 0 <= teacher_idx < teacher_ids.shape[0], (
                        f"[SKD] invalid teacher_idx={teacher_idx} for committed token {k} "
                        f"(chunk_start={chunk_start}, committed={len(new_tokens)}, teacher_len={teacher_ids.shape[0]})"
                    )
                    teacher_id_row = teacher_ids[teacher_idx].tolist()
                    teacher_logprob_row = teacher_logprobs[teacher_idx].tolist()
                    assert len(teacher_id_row) == self.loss_top_k, (
                        f"[SKD] teacher_ids row width {len(teacher_id_row)} != configured topk {self.loss_top_k}"
                    )
                    assert len(teacher_logprob_row) == self.loss_top_k, (
                        "[SKD] teacher_logprobs row width "
                        f"{len(teacher_logprob_row)} != configured topk {self.loss_top_k}"
                    )
                    pending_teacher_id_rows.append(teacher_id_row)
                    pending_teacher_logprob_rows.append(teacher_logprob_row)

                turn_state.tokens.extend(new_tokens)
                turn_state.teacher_ids_rows.extend(pending_teacher_id_rows)
                turn_state.teacher_logprobs_rows.extend(pending_teacher_logprob_rows)
                turn_state.raw_chunk = list(chunk)
                turn_state.verified_chunk = list(new_tokens)
                pending_turn_chunks += 1
                self._set_pending_turn_state(agent_data, turn_state, pending_chunks=pending_turn_chunks)
                commit_ms = (time.monotonic() - commit_t0) * 1000
                _trace_async_skd(
                    "loop.chunk_commit_state_done",
                    request_id=agent_data.request_id,
                    chunk_idx=skd_metrics["chunk_count"],
                    elapsed_ms=round(commit_ms, 1),
                    response_len=len(agent_data.response_mask) + len(turn_state.tokens),
                    prompt_len=len(agent_data.prompt_ids),
                    server_prompt_len=len(student_request_prompt_ids) + len(new_tokens),
                    teacher_prompt_len=len(teacher_prompt_ids) + len(new_tokens),
                    teacher_server_prompt_len=len(teacher_server_prompt_ids) + len(new_tokens),
                    teacher_rows_accumulated=len(teacher_ids_list) + len(turn_state.teacher_ids_rows),
                    teacher_logprobs_accumulated=len(teacher_logprobs_list) + len(turn_state.teacher_logprobs_rows),
                )
                eos_token_id = self.tokenizer.eos_token_id
                eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
                emit_async_skd_event(
                    "chunk_commit",
                    request_id=agent_data.request_id,
                    chunk_idx=skd_metrics["chunk_count"],
                    student_ms=student_ms,
                    teacher_ms=teacher_ms,
                    chunk_len=len(chunk),
                    accepted=len(new_tokens) - (1 if rejection_pos is not None else 0),
                    rejected=1 if rejection_pos is not None else 0,
                    new_tokens=len(new_tokens),
                    response_len=len(agent_data.response_mask) + len(turn_state.tokens),
                    committed_gen_chunks=int(agent_data.extra_fields.get("skd_committed_gen_chunks", 0)),
                    committed_prefix_tokens=int(agent_data.extra_fields.get("skd_committed_prefix_tokens", 0)),
                    raw_chunk=list(chunk) if _SKD_CHUNK_TRACE > 0 else None,
                    verified_chunk=list(new_tokens) if _SKD_CHUNK_TRACE > 0 else None,
                    teacher_rows=int(teacher_ids.shape[0]) if _SKD_CHUNK_TRACE > 0 else None,
                )
                _append_chunk_live_log(
                    tokenizer=self.tokenizer,
                    request_id=agent_data.request_id,
                    chunk_idx=skd_metrics["chunk_count"],
                    student_ms=student_ms,
                    teacher_ms=teacher_ms,
                    chunk=list(chunk),
                    new_tokens=list(new_tokens),
                    accepted=len(new_tokens) - (1 if rejection_pos is not None else 0),
                    rejected=1 if rejection_pos is not None else 0,
                    response_len=len(agent_data.response_mask) + len(turn_state.tokens),
                    eos_in_new_tokens=any(token_id in new_tokens for token_id in eos_ids),
                )
                _trace_async_skd(
                    "loop.chunk_commit",
                    request_id=agent_data.request_id,
                    teacher_replica_id=agent_data.extra_fields.get("teacher_replica_id"),
                    teacher_routing_key=agent_data.extra_fields.get("teacher_routing_key"),
                    response_len=len(agent_data.response_mask) + len(turn_state.tokens),
                    committed_gen_chunks=int(agent_data.extra_fields.get("skd_committed_gen_chunks", 0)),
                    committed_prefix_tokens=int(agent_data.extra_fields.get("skd_committed_prefix_tokens", 0)),
                    termination_reason=termination_reason,
                )

                # 5. Termination checks within chunk loop
                # Note: stop_reason == "completed" covers both EOS and max_tokens,
                # so we check for actual EOS tokens instead.
                if any(t in new_tokens for t in eos_ids):
                    termination_reason = "eos"
                    break
                if skd_metrics["chunk_count"] >= self.max_chunks_per_sample:
                    termination_reason = "max_chunks"
                    break
                if stop_after_skd_chunk:
                    termination_reason = "committed_unit_boundary"
                    agent_data.extra_fields["skd_termination_reason"] = termination_reason
                    if not agent_data.extra_fields.get("max_global_steps"):
                        agent_data.extra_fields.update(chunk_output.extra_fields)
                    if chunk_output.routed_experts is not None:
                        agent_data.routed_experts = chunk_output.routed_experts
                    return AgentState.GENERATING

        # === Post chunk loop: match ToolAgentLoop's post-generation logic ===

        sample_elapsed_ms = (time.monotonic() - sample_start_time) * 1000
        agent_data.extra_fields["skd_termination_reason"] = termination_reason

        # Keep the unfinished turn available for resume/export and delay any
        # committed rollout mutation until the turn has actually ended.
        self._set_pending_turn_state(agent_data, turn_state, pending_chunks=pending_turn_chunks)

        # Update extra_fields (first time only, same pattern as ToolAgentLoop L235-241)
        if chunk_output is not None and not agent_data.extra_fields.get("max_global_steps"):
            agent_data.extra_fields.update(chunk_output.extra_fields)

        # Track routed_experts (same as ToolAgentLoop L250-251, for MoE models)
        if chunk_output is not None and chunk_output.routed_experts is not None:
            agent_data.routed_experts = chunk_output.routed_experts

        forced_cutoff_reasons = {"budget_exhausted", "empty_chunk", "max_chunks", "teacher_context_exhausted"}
        if termination_reason == "eos":
            self._commit_pending_turn_state(
                agent_data,
                turn_state,
                pending_chunks=pending_turn_chunks,
                finalize_assistant_turn=True,
            )
            turn_response_ids = list(agent_data.response_ids)
        else:
            turn_response_ids = list(agent_data.response_ids)

        # Compute accept rate metric
        total = skd_metrics["accept_count"] + skd_metrics["reject_count"]
        if total > 0:
            agent_data.metrics["skd_accept_rate"] = skd_metrics["accept_count"] / total

        # === Per-sample summary log (always emitted via warnings for visibility) ===
        accept_rate = skd_metrics["accept_count"] / total * 100 if total > 0 else 0
        avg_tokens_per_chunk = len(turn_response_ids) / max(skd_metrics["chunk_count"], 1)
        warnings.warn(
            f"[SKD] req={agent_data.request_id} "
            f"done={termination_reason} "
            f"chunks={skd_metrics['chunk_count']}/{self.max_chunks_per_sample} "
            f"resp_len={len(turn_response_ids)} "
            f"accept={skd_metrics['accept_count']} reject={skd_metrics['reject_count']} "
            f"rate={accept_rate:.1f}% "
            f"avg_tok/chunk={avg_tokens_per_chunk:.1f} "
            f"student={skd_metrics['student_gen_ms']:.0f}ms "
            f"teacher={skd_metrics['teacher_verify_ms']:.0f}ms "
            f"total={sample_elapsed_ms:.0f}ms "
            f"prompt={initial_prompt_len} "
            f"teacher_logprobs_accumulated={len(teacher_ids_list) + len(turn_state.teacher_ids_rows)}",
            stacklevel=1,
        )

        # Check termination conditions (same as ToolAgentLoop L253-259)
        if termination_reason in forced_cutoff_reasons:
            return AgentState.TERMINATED
        if termination_reason != "eos":
            return AgentState.GENERATING
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (same as ToolAgentLoop L261-263)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        parse_error = getattr(self.tool_parser, "last_parse_error", None)

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        if parse_error is not None:
            return await self._handle_tool_parse_error(agent_data, parse_error)
        return AgentState.TERMINATED

    def _log_token_alignment(
        self,
        chunk: list[int],
        new_tokens: list[int],
        rejection_pos: int | None,
        teacher_ids: torch.Tensor,
        chunk_start: int,
        request_id: str,
        chunk_num: int,
    ):
        """Log token-level alignment details for verifying offset correctness.

        For each token in new_tokens, shows the local teacher row used for both
        verification and distillation after delta slicing.
        """
        try:
            lines = [f"[SKD_ALIGN] req={request_id} chunk={chunk_num} "
                     f"chunk_start={chunk_start} chunk_len={len(chunk)} "
                     f"rejection_pos={rejection_pos}"]
            # Show first 5 tokens of the chunk for brevity
            show_count = min(len(new_tokens), 5)
            for k in range(show_count):
                student_tok = chunk[k] if k < len(chunk) else -1
                final_tok = new_tokens[k]
                local_idx = k

                if 0 <= local_idx < teacher_ids.shape[0]:
                    local_top5 = teacher_ids[local_idx, :5].tolist()
                else:
                    local_top5 = "OOB"

                in_verify = "✓" if (isinstance(local_top5, list) and student_tok in local_top5) else "✗"
                replaced = " REPLACED" if (rejection_pos is not None and k == rejection_pos) else ""

                # Decode tokens for readability (catch errors for special tokens)
                try:
                    student_text = repr(self.tokenizer.decode([student_tok]))
                    final_text = repr(self.tokenizer.decode([final_tok]))
                except Exception:
                    student_text = f"id={student_tok}"
                    final_text = f"id={final_tok}"

                lines.append(
                    f"  k={k}: student={student_text}({student_tok}) {in_verify} "
                    f"teacher_top5={local_top5} | "
                    f"final={final_text}({final_tok}){replaced}"
                )

            warnings.warn("\n".join(lines), stacklevel=2)
        except Exception as e:
            warnings.warn(f"[SKD_ALIGN] logging error: {e}", stacklevel=2)
