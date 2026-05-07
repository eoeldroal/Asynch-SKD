"""
Unit tests for the actual SKD agent loop implementation.

These tests intentionally avoid Ray, SGLang, vLLM, network servers, and real
tools. They exercise ``SkdAgentLoop`` methods directly with deterministic
fake student/teacher/tool components so the core SKD invariants can be tested
quickly on CPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop, _build_teacher_logprob_range
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.web_skd_agent_loop import WebSkdAgentLoop
from verl.experimental.async_skd.events import async_skd_event_context
from verl.experimental.async_skd.state import SkdPartialState
from verl.workers.rollout.replica import TokenOutput


EOS = 99999
TOOL_CALL_A = 81001
TOOL_CALL_B = 81002
OPEN_TOOL = 91001
CLOSE_TOOL = 91002
LOSS_TOP_K = 4
VERIFY_TOP_K = 3


class FakeTokenizer:
    eos_token_id = EOS

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


class FakeWebPromptTokenizer(FakeTokenizer):
    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "decoded:" + ",".join(str(token_id) for token_id in ids)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        if not text.startswith("decoded:"):
            return []
        suffix = text[len("decoded:") :]
        if not suffix:
            return []
        return [int(part) for part in suffix.split(",")]


class FakeHermesTokenizer:
    eos_token_id = EOS

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        text_parts = []
        for token_id in ids:
            if token_id == OPEN_TOOL:
                text_parts.append("<tool_call>")
            elif token_id == CLOSE_TOOL:
                text_parts.append("</tool_call>")
            elif token_id == EOS:
                text_parts.append("<|im_end|>")
            else:
                text_parts.append(str(token_id))
        return "".join(text_parts)


class FakeStudentServer:
    """Returns preconfigured chunks and records the prompt seen by each call."""

    def __init__(self, chunks: list[list[int]]):
        self.chunks = chunks
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        prompt_text: str | None = None,
    ) -> TokenOutput:
        del video_data
        assert self.call_count < len(self.chunks), (
            f"student chunks exhausted at call {self.call_count}; "
            f"prompt={prompt_ids}, request_id={request_id}"
        )
        chunk = list(self.chunks[self.call_count])
        self.call_log.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "prompt_len": len(prompt_ids),
                "max_tokens": sampling_params.get("max_tokens"),
                "prompt_text": prompt_text,
                "image_data": image_data,
                "chunk": chunk,
            }
        )
        self.call_count += 1
        return TokenOutput(
            token_ids=chunk,
            log_probs=[-0.01] * len(chunk),
            num_preempted=0,
            stop_reason="completed" if EOS in chunk else None,
            extra_fields={
                "global_steps": 7,
                "min_global_steps": 7,
                "max_global_steps": 7,
            },
        )


class FakeTeacherServer:
    """Delta-mode teacher fake matching ``compute_teacher_logprobs_single``."""

    def __init__(self, topk_by_call: list[dict[int, list[int]]] | None = None, *, k: int = LOSS_TOP_K):
        self.topk_by_call = topk_by_call or []
        self.k = k
        self.call_count = 0
        self.call_log: list[dict[str, Any]] = []
        self.released_request_ids: list[str] = []
        self.bound_requests: list[dict[str, str]] = []

    async def compute_teacher_logprobs_single(
        self,
        *,
        request_id: str | None = None,
        sequence_ids: list[int],
        logprob_start_len: int = 0,
        expected_mm_prefix_surplus: int | None = None,
        expected_logprob_rows: int | None = None,
        multi_modal_data: dict[str, Any] | None = None,
        routing_key: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_len = logprob_start_len + 1 if logprob_start_len > 0 else 0
        if expected_logprob_rows is None:
            chunk = list(sequence_ids[prefix_len:])
        else:
            chunk = list(sequence_ids[-int(expected_logprob_rows) :])
        overrides = self.topk_by_call[self.call_count] if self.call_count < len(self.topk_by_call) else {}
        rows: list[list[int]] = []
        logprobs: list[list[float]] = []
        for local_idx, token_id in enumerate(chunk):
            topk = list(overrides.get(local_idx, [token_id]))
            topk = (topk + [0] * self.k)[: self.k]
            rows.append(topk)
            logprobs.append([-(local_idx + 1.0)] * self.k)

        self.call_log.append(
            {
                "request_id": request_id,
                "sequence_ids": list(sequence_ids),
                "logprob_start_len": logprob_start_len,
                "expected_mm_prefix_surplus": expected_mm_prefix_surplus,
                "expected_logprob_rows": expected_logprob_rows,
                "prefix_len": prefix_len,
                "chunk": chunk,
                "rows": rows,
                "multi_modal_data": multi_modal_data,
                "routing_key": routing_key,
            }
        )
        self.call_count += 1
        return torch.tensor(rows, dtype=torch.int32), torch.tensor(logprobs, dtype=torch.float32)

    async def release_sticky_session(self, request_id: str) -> None:
        self.released_request_ids.append(request_id)

    async def bind_sticky_request(self, *, routing_key: str, request_id: str, server_id: str) -> None:
        self.bound_requests.append(
            {
                "routing_key": routing_key,
                "request_id": request_id,
                "server_id": server_id,
            }
        )


class FakeToolParser:
    """Returns a tool call only when the assistant turn contains tool-call tokens."""

    async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
        del tools
        if TOOL_CALL_A in response_ids and TOOL_CALL_B in response_ids:
            return None, [FakeToolCall(name="lookup", arguments='{"query":"weather"}')]
        return None, []


@dataclass
class FakeToolCall:
    name: str
    arguments: str


def make_skd_loop(
    *,
    student_chunks: list[list[int]],
    teacher_topk_by_call: list[dict[int, list[int]]] | None = None,
    chunk_size: int = 8,
    max_chunks: int = 32,
    response_length: int = 128,
) -> SkdAgentLoop:
    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    loop.teacher_server_manager = FakeTeacherServer(teacher_topk_by_call, k=LOSS_TOP_K)
    loop.server_manager = FakeStudentServer(student_chunks)
    loop.tokenizer = FakeTokenizer()
    loop.response_length = response_length
    loop.skd_chunk_size = chunk_size
    loop.skd_verify_top_k = VERIFY_TOP_K
    loop.max_chunks_per_sample = max_chunks
    loop.loss_top_k = LOSS_TOP_K
    loop.teacher_key = "data_source"
    loop.tools = {}
    loop.tool_schemas = []
    loop.tool_parser = FakeToolParser()
    loop.interaction_config_file = None
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.processor = None
    loop.apply_chat_template_kwargs = {}
    return loop


def make_web_skd_loop(
    *,
    student_chunks: list[list[int]],
    teacher_topk_by_call: list[dict[int, list[int]]] | None = None,
    chunk_size: int = 8,
    max_chunks: int = 32,
    response_length: int = 128,
) -> WebSkdAgentLoop:
    loop = WebSkdAgentLoop.__new__(WebSkdAgentLoop)
    loop.teacher_server_manager = FakeTeacherServer(teacher_topk_by_call, k=LOSS_TOP_K)
    loop.server_manager = FakeStudentServer(student_chunks)
    loop.tokenizer = FakeWebPromptTokenizer()
    loop.loop = asyncio.get_running_loop()
    loop.response_length = response_length
    loop.skd_chunk_size = chunk_size
    loop.skd_verify_top_k = VERIFY_TOP_K
    loop.max_chunks_per_sample = max_chunks
    loop.loss_top_k = LOSS_TOP_K
    loop.teacher_key = "data_source"
    loop.tools = {}
    loop.tool_schemas = []
    loop.tool_parser = FakeToolParser()
    loop.interaction_config_file = None
    loop.max_assistant_turns = None
    loop.max_user_turns = None
    loop.processor = None
    loop.apply_chat_template_kwargs = {}
    return loop


def make_agent_data(prompt_ids: list[int] | None = None) -> AgentData:
    agent_data = AgentData(
        messages=[{"role": "user", "content": "question"}],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="req-skd-unit",
        tools_kwargs={},
    )
    agent_data.prompt_ids = list(prompt_ids or [1, 2, 3])
    agent_data.extra_fields["teacher_prompt_ids"] = list(agent_data.prompt_ids)
    return agent_data


def teacher_rows(agent_data: AgentData) -> list[list[int]]:
    return agent_data.extra_fields["teacher_ids_list"]


def teacher_logprobs(agent_data: AgentData) -> list[list[float]]:
    return agent_data.extra_fields["teacher_logprobs_list"]


def assert_skd_alignment(agent_data: AgentData) -> None:
    assert len(agent_data.response_mask) == len(teacher_rows(agent_data))
    assert len(agent_data.response_mask) == len(teacher_logprobs(agent_data))


def assert_masked_teacher_rows(agent_data: AgentData) -> None:
    rows = zip(agent_data.response_mask, teacher_rows(agent_data), teacher_logprobs(agent_data), strict=True)
    for mask, ids, logps in rows:
        if mask == 0:
            assert ids == [0] * LOSS_TOP_K
            assert logps == [0.0] * LOSS_TOP_K
        else:
            assert ids != [0] * LOSS_TOP_K


def assert_committed_tokens_inside_teacher_topk(agent_data: AgentData) -> None:
    response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
    rows = zip(response_ids, agent_data.response_mask, teacher_rows(agent_data), strict=True)
    for token_id, mask, ids in rows:
        if mask == 1:
            assert token_id in ids[:VERIFY_TOP_K], f"token {token_id} not in teacher top-k row {ids}"


@pytest.mark.asyncio
async def test_skd_teacher_verification_forwards_sample_teacher_routing_key():
    loop = make_skd_loop(student_chunks=[[10, EOS]], max_chunks=1)
    agent_data = make_agent_data()
    agent_data.extra_fields["teacher_routing_key"] = "math_teacher"

    await loop._handle_generating_state(agent_data, {}, ignore_termination=True)

    assert loop.teacher_server_manager.call_log[0]["routing_key"] == "math_teacher"


@pytest.mark.asyncio
async def test_skd_pending_state_uses_per_sample_active_tool_schemas():
    loop = make_skd_loop(student_chunks=[[EOS]])
    seen_tool_schemas: list[list[Any]] = []

    async def fake_apply_chat_template(messages, tools=None, images=None, videos=None, **kwargs):
        del messages, images, videos, kwargs
        seen_tool_schemas.append(list(tools or []))
        return [1, 2, 3]

    agent_data = make_agent_data()
    agent_data._active_tool_schemas = [{"function": {"name": "selected_tool"}}]
    loop.tool_schemas = [{"function": {"name": "global_tool"}}]
    loop.apply_chat_template = fake_apply_chat_template

    state = await loop._handle_pending_state(agent_data, {})

    assert state == AgentState.GENERATING
    assert seen_tool_schemas == [[{"function": {"name": "selected_tool"}}]]


def test_skd_finalize_slices_routed_experts_like_tool_agent_loop():
    loop = make_skd_loop(student_chunks=[[EOS]], response_length=4)
    agent_data = make_agent_data(prompt_ids=[1, 2, 3])
    agent_data.response_mask = [1, 1]
    agent_data.prompt_ids += [10, EOS]
    agent_data.routed_experts = torch.arange(20).reshape(10, 1, 2)

    output = loop._finalize_boundary_agent_output(agent_data)

    assert output.routed_experts.shape[0] == 7


@pytest.mark.asyncio
async def test_full_skd_tool_trajectory_e2e(monkeypatch):
    """Exercise multi-chunk SKD, first rejection, tool macro-step, export, and second turn."""

    async def fake_tool_step(self, agent_data):
        del self
        tool_tokens = [900, 901, 902, 903]
        agent_data.messages.append({"role": "tool", "content": "tool result"})
        agent_data.prompt_ids += tool_tokens
        agent_data.response_mask += [0] * len(tool_tokens)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step)

    loop = make_skd_loop(
        student_chunks=[
            [10, 11, 12],
            [20, 777, 22],
            [TOOL_CALL_A, TOOL_CALL_B, EOS],
            [30, 31],
            [888, 41],
            [50, EOS],
        ],
        teacher_topk_by_call=[
            {},
            {1: [21, 210, 211]},
            {},
            {},
            {0: [40, 400, 401]},
            {},
        ],
    )
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)
    assert next_state == AgentState.PROCESSING_TOOLS
    assert not loop._can_export_partial_state(agent_data, next_state)
    with pytest.raises(ValueError, match="Cannot export SKD partial state"):
        loop._export_partial_state(
            agent_data,
            next_state,
            sample_id="sample-pending-tool-result",
            logical_step=10,
            source_type="lookahead",
        )
    assert agent_data.response_ids == [10, 11, 12, 20, 21, TOOL_CALL_A, TOOL_CALL_B, EOS]
    assert 777 not in agent_data.response_ids
    assert 22 not in agent_data.response_ids
    assert_skd_alignment(agent_data)
    assert_committed_tokens_inside_teacher_topk(agent_data)
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 3
    assert agent_data.extra_fields["skd_committed_env_units"] == 0
    assert agent_data.extra_fields["skd_committed_prefix_tokens"] == 8
    assert agent_data.extra_fields["rollout_min_version"] == 7
    assert agent_data.extra_fields["rollout_max_version"] == 7

    next_state = await SkdAgentLoop._handle_processing_tools_state(loop, agent_data)
    assert next_state == AgentState.GENERATING
    assert agent_data.response_mask[-4:] == [0, 0, 0, 0]
    assert agent_data.extra_fields["teacher_prompt_ids"][-4:] == [900, 901, 902, 903]
    assert_skd_alignment(agent_data)
    assert_masked_teacher_rows(agent_data)
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 3
    assert agent_data.extra_fields["skd_committed_env_units"] == 1
    assert agent_data.extra_fields["skd_committed_prefix_tokens"] == 12

    assert loop._can_export_partial_state(agent_data, next_state)
    partial = loop._export_partial_state(
        agent_data,
        next_state,
        sample_id="sample-tool-closed",
        logical_step=10,
        source_type="lookahead",
    )
    assert isinstance(partial, SkdPartialState)
    assert partial.sample_id == "sample-tool-closed"
    assert partial.logical_step == 10
    assert partial.source_type == "lookahead"
    assert partial.agent_state == AgentState.GENERATING.value
    assert partial.prompt_ids == agent_data.prompt_ids
    assert partial.teacher_prompt_ids == agent_data.extra_fields["teacher_prompt_ids"]
    assert partial.response_mask == agent_data.response_mask
    assert partial.extra_fields["teacher_ids_list"] == teacher_rows(agent_data)
    assert partial.extra_fields["teacher_logprobs_list"] == teacher_logprobs(agent_data)
    assert partial.rollout_birth_version == 7
    assert partial.rollout_min_version == 7
    assert partial.rollout_max_version == 7
    assert partial.committed_gen_chunks == 3
    assert partial.committed_env_units == 1
    assert partial.committed_prefix_tokens == 12

    restored_loop = make_skd_loop(
        student_chunks=[
            [30, 31],
            [888, 41],
            [50, EOS],
        ],
        teacher_topk_by_call=[
            {},
            {0: [40, 400, 401]},
            {},
        ],
    )
    restored_agent_data, restored_state = restored_loop._restore_partial_state(partial)
    assert restored_state == AgentState.GENERATING
    assert restored_agent_data is not agent_data
    assert restored_agent_data.prompt_ids == partial.prompt_ids
    assert restored_agent_data.extra_fields["teacher_prompt_ids"] == partial.teacher_prompt_ids
    assert restored_agent_data.response_mask == partial.response_mask
    assert restored_agent_data.extra_fields["teacher_ids_list"] == partial.extra_fields["teacher_ids_list"]
    assert restored_agent_data.extra_fields["teacher_logprobs_list"] == partial.extra_fields["teacher_logprobs_list"]
    assert restored_agent_data.extra_fields["skd_committed_gen_chunks"] == 3
    assert restored_agent_data.extra_fields["skd_committed_env_units"] == 1
    assert restored_agent_data.extra_fields["skd_committed_prefix_tokens"] == 12
    assert_skd_alignment(restored_agent_data)

    next_state = await SkdAgentLoop._handle_generating_state(restored_loop, restored_agent_data, {}, False)
    assert next_state == AgentState.TERMINATED

    expected_response = [
        10,
        11,
        12,
        20,
        21,
        TOOL_CALL_A,
        TOOL_CALL_B,
        EOS,
        900,
        901,
        902,
        903,
        30,
        31,
        40,
        50,
        EOS,
    ]
    expected_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    assert restored_agent_data.prompt_ids == [1, 2, 3] + expected_response
    assert restored_agent_data.response_mask == expected_mask
    assert restored_agent_data.extra_fields["teacher_prompt_ids"] == [1, 2, 3] + expected_response
    assert 888 not in expected_response
    assert 41 not in expected_response
    assert restored_agent_data.assistant_turns == 2
    assert restored_agent_data.user_turns == 1
    assert loop.server_manager.call_count == 3
    assert loop.teacher_server_manager.call_count == 3
    assert restored_loop.server_manager.call_count == 3
    assert restored_loop.teacher_server_manager.call_count == 3
    assert_skd_alignment(restored_agent_data)
    assert_masked_teacher_rows(restored_agent_data)
    assert_committed_tokens_inside_teacher_topk(restored_agent_data)
    assert restored_agent_data.extra_fields["skd_termination_reason"] == "eos"
    assert restored_agent_data.extra_fields["skd_committed_gen_chunks"] == 6
    assert restored_agent_data.extra_fields["skd_committed_env_units"] == 1
    assert restored_agent_data.extra_fields["skd_committed_prefix_tokens"] == len(expected_response)
    assert restored_agent_data.extra_fields["rollout_birth_version"] == 7
    assert restored_agent_data.extra_fields["rollout_min_version"] == 7
    assert restored_agent_data.extra_fields["rollout_max_version"] == 7


@pytest.mark.asyncio
async def test_rejection_at_first_token_discards_suffix():
    loop = make_skd_loop(
        student_chunks=[[777, 20, 30]],
        teacher_topk_by_call=[{0: [100, 101, 102]}],
        max_chunks=1,
    )
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    assert not loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.response_ids == [100]
    assert agent_data.prompt_ids == [1, 2, 3, 100]
    assert agent_data.response_mask == [1]
    assert teacher_rows(agent_data) == [[100, 101, 102, 0]]
    assert 777 not in agent_data.prompt_ids
    assert 20 not in agent_data.prompt_ids
    assert 30 not in agent_data.prompt_ids
    assert agent_data.metrics["skd"]["reject_count"] == 1
    assert agent_data.metrics["skd"]["accept_count"] == 0
    assert_skd_alignment(agent_data)
    assert_committed_tokens_inside_teacher_topk(agent_data)
    assert agent_data.extra_fields["skd_termination_reason"] == "max_chunks"
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 1
    assert agent_data.extra_fields["skd_committed_env_units"] == 0
    assert agent_data.extra_fields["skd_committed_prefix_tokens"] == 1
    assert agent_data.extra_fields["rollout_min_version"] == 7
    assert agent_data.extra_fields["rollout_max_version"] == 7


@pytest.mark.asyncio
async def test_skd_chunk_commit_emits_realtime_progress_event(tmp_path, monkeypatch):
    event_path = tmp_path / "async_skd_events.jsonl"
    monkeypatch.setenv("VERL_ASYNC_SKD_EVENT_LOG", str(event_path))
    loop = make_skd_loop(
        student_chunks=[[777, 20, 30]],
        teacher_topk_by_call=[{0: [100, 101, 102]}],
        max_chunks=1,
    )
    agent_data = make_agent_data([1, 2, 3])

    with async_skd_event_context(
        sample_id="sample-reject",
        scheduler_worker_idx=0,
        source_type="base_current",
        barrier_role="current",
    ):
        await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    [chunk_event] = [event for event in events if event["event"] == "chunk_commit"]
    assert chunk_event["sample_id"] == "sample-reject"
    assert chunk_event["chunk_idx"] == 1
    assert chunk_event["accepted"] == 0
    assert chunk_event["rejected"] == 1
    assert chunk_event["new_tokens"] == 1
    assert chunk_event["response_len"] == 1


@pytest.mark.asyncio
async def test_skd_generation_can_pause_at_committed_chunk_boundary_and_resume():
    loop = make_skd_loop(
        student_chunks=[
            [10, 11],
            [20, EOS],
        ],
        teacher_topk_by_call=[
            {},
            {},
        ],
    )
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [1, 2, 3]
    assert agent_data.response_ids == [10, 11]
    assert agent_data.response_mask == []
    assert agent_data.assistant_turns == 0
    assert agent_data.extra_fields["skd_termination_reason"] == "committed_unit_boundary"
    assert agent_data.extra_fields["skd_pending_turn_response_ids"] == [10, 11]
    assert agent_data.extra_fields["skd_pending_turn_state"] == {
        "tokens": [10, 11],
        "teacher_ids_rows": [[10, 0, 0, 0], [11, 0, 0, 0]],
        "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K],
        "raw_chunk": [10, 11],
        "verified_chunk": [10, 11],
    }
    assert_skd_alignment(agent_data)

    partial = loop._export_partial_state(
        agent_data,
        next_state,
        sample_id="sample-paused-after-one-chunk",
        logical_step=11,
        source_type="lookahead",
    )

    restored_loop = make_skd_loop(
        student_chunks=[
            [20, EOS],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    restored_agent_data, restored_state = restored_loop._restore_partial_state(partial)
    assert restored_state == AgentState.GENERATING
    assert restored_agent_data.response_ids == [10, 11]
    assert restored_agent_data.extra_fields["skd_pending_turn_response_ids"] == [10, 11]

    next_state = await SkdAgentLoop._handle_generating_state(restored_loop, restored_agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    assert restored_agent_data.prompt_ids == [1, 2, 3, 10, 11, 20, EOS]
    assert restored_agent_data.response_ids == [10, 11, 20, EOS]
    assert restored_agent_data.response_mask == [1, 1, 1, 1]
    assert restored_agent_data.assistant_turns == 1
    assert "skd_pending_turn_response_ids" not in restored_agent_data.extra_fields
    assert restored_agent_data.extra_fields["skd_termination_reason"] == "eos"
    assert restored_agent_data.extra_fields["skd_committed_gen_chunks"] == 2
    assert restored_agent_data.extra_fields["skd_committed_prefix_tokens"] == 4
    assert_skd_alignment(restored_agent_data)


@pytest.mark.asyncio
async def test_skd_chunks_do_not_commit_prompt_state_before_eos():
    loop = make_skd_loop(
        student_chunks=[
            [10, 11],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    agent_data = make_agent_data([1, 2, 3])
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_ids_list"] = []
    agent_data.extra_fields["teacher_logprobs_list"] = []
    agent_data.extra_fields["skd_committed_gen_chunks"] = 0
    agent_data.extra_fields["skd_committed_env_units"] = 0
    agent_data.extra_fields["skd_committed_prefix_tokens"] = 0

    before_commit_state = {
        "prompt_ids": list(agent_data.prompt_ids),
        "server_prompt_ids": list(agent_data.extra_fields["server_prompt_ids"]),
        "teacher_prompt_ids": list(agent_data.extra_fields["teacher_prompt_ids"]),
        "teacher_server_prompt_ids": list(agent_data.extra_fields["teacher_server_prompt_ids"]),
        "response_mask": list(agent_data.response_mask),
        "teacher_ids_list": list(agent_data.extra_fields["teacher_ids_list"]),
        "teacher_logprobs_list": list(agent_data.extra_fields["teacher_logprobs_list"]),
        "skd_committed_gen_chunks": agent_data.extra_fields["skd_committed_gen_chunks"],
        "skd_committed_env_units": agent_data.extra_fields["skd_committed_env_units"],
        "skd_committed_prefix_tokens": agent_data.extra_fields["skd_committed_prefix_tokens"],
    }

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert agent_data.response_ids == [10, 11]
    assert agent_data.extra_fields["skd_pending_turn_response_ids"] == [10, 11]
    assert agent_data.extra_fields["skd_pending_turn_state"] == {
        "tokens": [10, 11],
        "teacher_ids_rows": [[10, 0, 0, 0], [11, 0, 0, 0]],
        "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K],
        "raw_chunk": [10, 11],
        "verified_chunk": [10, 11],
    }
    after_commit_state = {
        "prompt_ids": list(agent_data.prompt_ids),
        "server_prompt_ids": list(agent_data.extra_fields["server_prompt_ids"]),
        "teacher_prompt_ids": list(agent_data.extra_fields["teacher_prompt_ids"]),
        "teacher_server_prompt_ids": list(agent_data.extra_fields["teacher_server_prompt_ids"]),
        "response_mask": list(agent_data.response_mask),
        "teacher_ids_list": list(agent_data.extra_fields["teacher_ids_list"]),
        "teacher_logprobs_list": list(agent_data.extra_fields["teacher_logprobs_list"]),
        "skd_committed_gen_chunks": agent_data.extra_fields["skd_committed_gen_chunks"],
        "skd_committed_env_units": agent_data.extra_fields["skd_committed_env_units"],
        "skd_committed_prefix_tokens": agent_data.extra_fields["skd_committed_prefix_tokens"],
    }
    assert after_commit_state == before_commit_state


@pytest.mark.asyncio
async def test_skd_teacher_request_counts_teacher_only_text_as_server_prefix_not_surplus():
    loop = make_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    teacher_prefix = [1, 2, 3, 101, 102, 103]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = list(teacher_prefix)
    agent_data.extra_fields["teacher_prompt_ids"] = list(teacher_prefix)
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 0

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == teacher_prefix + [10, 11]
    assert teacher_call["logprob_start_len"] == len(teacher_prefix) - 1
    assert teacher_call["expected_mm_prefix_surplus"] == 0
    assert teacher_call["expected_logprob_rows"] == 2
    assert teacher_call["chunk"] == [10, 11]
    assert teacher_rows(agent_data) == [[10, 0, 0, 0], [11, 0, 0, 0]]


@pytest.mark.asyncio
async def test_skd_teacher_request_uses_tracked_multimodal_surplus_for_sglang_start():
    loop = make_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3] + [900] * 959
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 959

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == [1, 2, 3, 10, 11]
    assert teacher_call["logprob_start_len"] == 961
    assert teacher_call["expected_mm_prefix_surplus"] == 959
    assert teacher_call["expected_logprob_rows"] == 2
    assert teacher_call["chunk"] == [10, 11]
    assert teacher_rows(agent_data) == [[10, 0, 0, 0], [11, 0, 0, 0]]


@pytest.mark.asyncio
async def test_skd_teacher_request_prefers_tracked_surplus_over_prompt_length_gap():
    loop = make_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3] + [900] * 100
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 7

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == [1, 2, 3, 10, 11]
    assert teacher_call["logprob_start_len"] == 9
    assert teacher_call["expected_mm_prefix_surplus"] == 7
    assert teacher_call["expected_logprob_rows"] == 2
    assert teacher_call["chunk"] == [10, 11]


@pytest.mark.asyncio
async def test_skd_text_only_teacher_gap_is_not_treated_as_sglang_surplus():
    loop = make_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3] + [900] * 17

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == [1, 2, 3, 10, 11]
    assert teacher_call["logprob_start_len"] == 2
    assert teacher_call["expected_mm_prefix_surplus"] == 0
    assert teacher_call["expected_logprob_rows"] == 2


@pytest.mark.asyncio
async def test_skd_multimodal_teacher_requires_explicit_sglang_surplus():
    loop = make_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3] + [900] * 17

    with pytest.raises(ValueError, match="teacher_sglang_prefix_surplus is required"):
        await loop._handle_generating_state(
            agent_data,
            {"max_tokens": 2},
            ignore_termination=False,
            stop_after_skd_chunk=True,
        )


@pytest.mark.asyncio
async def test_web_skd_image_generation_keeps_input_ids_request_view():
    loop = make_web_skd_loop(student_chunks=[[10, EOS]], chunk_size=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 0

    async def _skip_reward(*args, **kwargs):
        del args, kwargs

    async def _recompute_student(agent_data_arg, messages):
        del agent_data_arg, messages
        return [1, 2, 3]

    async def _recompute_teacher(agent_data_arg, teacher_messages):
        del agent_data_arg, teacher_messages
        return [1, 2, 3]

    loop._recompute_server_prompt_ids = _recompute_student
    loop._recompute_teacher_server_prompt_ids = _recompute_teacher
    loop._finalize_with_web_osgym_reward = _skip_reward

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 8},
        ignore_termination=False,
        stop_after_skd_chunk=False,
    )

    assert state == AgentState.TERMINATED
    student_call = loop.server_manager.call_log[0]
    assert student_call["prompt_ids"] == [1, 2, 3]
    assert student_call["prompt_text"] is None
    assert student_call["image_data"] == ["image-1"]


@pytest.mark.asyncio
async def test_web_skd_teacher_verify_rebuilds_request_views_from_current_messages():
    loop = make_web_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields.update(
        {
            "teacher_prompt_ids": [1, 2, 3, 900, 901],
            "server_prompt_ids": [999],
            "teacher_server_prompt_ids": [888],
            "teacher_sglang_prefix_surplus": 0,
            "web_osgym_teacher_messages": [{"role": "user", "content": "task"}, {"role": "tool", "content": [{"type": "image"}]}],
        }
    )

    async def _recompute_student(agent_data_arg, messages):
        del agent_data_arg, messages
        return [1, 2, 3]

    async def _recompute_teacher(agent_data_arg, teacher_messages):
        del agent_data_arg, teacher_messages
        return [1, 2, 3]

    loop._recompute_server_prompt_ids = _recompute_student
    loop._recompute_teacher_server_prompt_ids = _recompute_teacher

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    student_call = loop.server_manager.call_log[0]
    assert student_call["prompt_ids"] == [1, 2, 3]
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == [1, 2, 3, 10, 11]
    assert teacher_call["expected_mm_prefix_surplus"] == 2
    assert teacher_call["expected_logprob_rows"] == 2
    assert teacher_rows(agent_data) == [[10, 0, 0, 0], [11, 0, 0, 0]]


@pytest.mark.asyncio
async def test_web_skd_teacher_verification_span_stays_stable_across_image_tool_boundary():
    loop = make_web_skd_loop(student_chunks=[[10, 11]], chunk_size=2, response_length=8)
    agent_data = make_agent_data([1, 2, 3, 61, 62, 63, 64])
    agent_data.image_data = ["image-1"]
    original_teacher_server_prompt_ids = [1, 2, 3, 91, 92]
    agent_data.extra_fields.update(
        {
            "server_prompt_ids": [1, 2, 3, 81, 82],
            "teacher_prompt_ids": [1, 2, 3, 71, 72, 73, 74],
            "teacher_server_prompt_ids": list(original_teacher_server_prompt_ids),
            "teacher_sglang_prefix_surplus": 2,
            "web_osgym_teacher_messages": [{"role": "user", "content": "question"}, {"role": "tool", "content": [{"type": "image"}]}],
        }
    )

    async def _recompute_student(agent_data_arg, messages):
        del agent_data_arg, messages
        return [1, 2, 3, 81, 82]

    async def _recompute_teacher(agent_data_arg, teacher_messages):
        del agent_data_arg, teacher_messages
        return list(original_teacher_server_prompt_ids)

    loop._recompute_server_prompt_ids = _recompute_student
    loop._recompute_teacher_server_prompt_ids = _recompute_teacher

    state = await loop._handle_generating_state(
        agent_data,
        {"max_tokens": 2},
        ignore_termination=False,
        stop_after_skd_chunk=True,
    )

    assert state == AgentState.GENERATING
    student_call = loop.server_manager.call_log[0]
    assert student_call["prompt_ids"] == [1, 2, 3, 81, 82]
    assert student_call["image_data"] == ["image-1"]
    teacher_call = loop.teacher_server_manager.call_log[0]
    assert teacher_call["sequence_ids"] == original_teacher_server_prompt_ids + [10, 11]
    assert teacher_call["logprob_start_len"] == 6
    assert teacher_call["expected_mm_prefix_surplus"] == 2
    assert teacher_call["expected_logprob_rows"] == 2
    assert teacher_call["chunk"] == [10, 11]
    assert agent_data.extra_fields["teacher_server_prompt_ids"] == original_teacher_server_prompt_ids + [10, 11]
    assert agent_data.extra_fields["teacher_sglang_prefix_surplus"] == 2
    assert teacher_rows(agent_data) == [[10, 0, 0, 0], [11, 0, 0, 0]]


def test_skd_text_tool_delta_updates_student_and_teacher_server_streams():
    loop = make_skd_loop(student_chunks=[])
    agent_data = make_agent_data([1, 2, 3, 70, 71])
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]

    loop._append_student_prompt_delta_to_teacher_stream(agent_data, prev_prompt_len=3)

    assert agent_data.extra_fields["server_prompt_ids"] == [1, 2, 3, 70, 71]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [1, 2, 3, 70, 71]
    assert agent_data.extra_fields["teacher_server_prompt_ids"] == [1, 2, 3, 70, 71]


def test_build_teacher_logprob_range_multimodal_scalar_shifts_start():
    result = _build_teacher_logprob_range(
        teacher_server_prompt_len=128,
        teacher_sglang_prefix_surplus=1918,
        chunk_len=64,
    )

    assert result.server_logical_start_len == 127
    assert result.sglang_logprob_start_len == 2045
    assert result.expected_logprob_rows == 64
    assert result.teacher_sglang_prefix_surplus == 1918


def test_build_teacher_logprob_range_rejects_negative_surplus():
    with pytest.raises(ValueError, match="teacher_sglang_prefix_surplus must be non-negative"):
        _build_teacher_logprob_range(
            teacher_server_prompt_len=128,
            teacher_sglang_prefix_surplus=-1,
            chunk_len=64,
        )


@pytest.mark.asyncio
async def test_skd_export_allows_open_tool_call_prefix_without_eos():
    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert_skd_alignment(agent_data)


@pytest.mark.asyncio
async def test_skd_export_allows_closed_tool_call_without_eos_as_generation_prefix():
    class _ParserMustNotRun:
        async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
            raise AssertionError("tool parser must not run before EOS")

    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11, CLOSE_TOOL],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    loop.tool_parser = _ParserMustNotRun()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(
        loop,
        agent_data,
        {},
        False,
        stop_after_skd_chunk=True,
    )

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    partial = loop._export_partial_state(
        agent_data,
        next_state,
        sample_id="sample-closed-tool-no-eos",
        logical_step=10,
        source_type="lookahead",
    )
    assert partial.agent_state == AgentState.GENERATING.value
    assert partial.response_ids == [OPEN_TOOL, 11, CLOSE_TOOL]
    assert partial.response_mask == []
    assert partial.extra_fields["skd_pending_turn_state"] == {
        "tokens": [OPEN_TOOL, 11, CLOSE_TOOL],
        "teacher_ids_rows": [[OPEN_TOOL, 0, 0, 0], [11, 0, 0, 0], [CLOSE_TOOL, 0, 0, 0]],
        "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K, [-3.0] * LOSS_TOP_K],
        "raw_chunk": [OPEN_TOOL, 11, CLOSE_TOOL],
        "verified_chunk": [OPEN_TOOL, 11, CLOSE_TOOL],
    }
    assert_skd_alignment(agent_data)


@pytest.mark.asyncio
async def test_skd_closed_tool_call_without_eos_does_not_invoke_tool_parser():
    class _ParserMustNotRun:
        async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
            raise AssertionError(
                f"tool parser must not run before EOS: response_ids={response_ids!r}, tools={tools!r}"
            )

    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11, CLOSE_TOOL],
        ],
        teacher_topk_by_call=[
            {},
        ],
        max_chunks=1,
    )
    loop.tokenizer = FakeHermesTokenizer()
    loop.tool_parser = _ParserMustNotRun()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.GENERATING
    assert agent_data.response_ids == [OPEN_TOOL, 11, CLOSE_TOOL]


@pytest.mark.asyncio
async def test_skd_max_chunks_cutoff_flushes_turn_buffer_without_tool_processing():
    class _ParserMustNotRun:
        async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
            raise AssertionError(
                f"tool parser must not run at max-chunks cutoff: response_ids={response_ids!r}, tools={tools!r}"
            )

    loop = make_skd_loop(
        student_chunks=[
            [TOOL_CALL_A, TOOL_CALL_B, 33],
        ],
        teacher_topk_by_call=[
            {},
        ],
        max_chunks=1,
    )
    loop.tool_parser = _ParserMustNotRun()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    assert agent_data.extra_fields["skd_termination_reason"] == "max_chunks"
    assert agent_data.response_ids == [TOOL_CALL_A, TOOL_CALL_B, 33]
    assert "skd_pending_turn_response_ids" not in agent_data.extra_fields


@pytest.mark.asyncio
async def test_skd_boundary_driver_closes_tool_macro_step_before_export(monkeypatch):
    async def fake_tool_step(self, agent_data):
        del self
        tool_tokens = [900, 901]
        agent_data.messages.append({"role": "tool", "content": "tool result"})
        agent_data.prompt_ids += tool_tokens
        agent_data.response_mask += [0] * len(tool_tokens)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step)

    loop = make_skd_loop(
        student_chunks=[
            [TOOL_CALL_A, TOOL_CALL_B, EOS],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    agent_data = make_agent_data([1, 2, 3])

    next_state = await loop._run_until_exportable_boundary(agent_data, AgentState.GENERATING, {})

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [1, 2, 3, TOOL_CALL_A, TOOL_CALL_B, EOS, 900, 901]
    assert agent_data.response_mask == [1, 1, 1, 0, 0]
    assert agent_data.response_ids == [TOOL_CALL_A, TOOL_CALL_B, EOS]
    assert agent_data.assistant_turns == 1
    assert agent_data.user_turns == 1
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 1
    assert agent_data.extra_fields["skd_committed_env_units"] == 1
    assert_skd_alignment(agent_data)
    assert_masked_teacher_rows(agent_data)


@pytest.mark.asyncio
async def test_skd_boundary_driver_closes_eos_tool_call_before_export(monkeypatch):
    async def fake_tool_step(self, agent_data):
        del self
        tool_tokens = [900, 901]
        agent_data.messages.append({"role": "tool", "content": "tool result"})
        agent_data.prompt_ids += tool_tokens
        agent_data.response_mask += [0] * len(tool_tokens)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    class _Parser:
        async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]):
            del response_ids, tools
            return None, [FakeToolCall(name="lookup", arguments='{"query":"weather"}')]

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step)

    loop = make_skd_loop(
        student_chunks=[
            [OPEN_TOOL, 11, CLOSE_TOOL, EOS],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )
    loop.tokenizer = FakeHermesTokenizer()
    loop.tool_parser = _Parser()
    agent_data = make_agent_data([1, 2, 3])

    next_state = await loop._run_until_exportable_boundary(agent_data, AgentState.GENERATING, {})

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [1, 2, 3, OPEN_TOOL, 11, CLOSE_TOOL, EOS, 900, 901]
    assert agent_data.response_mask == [1, 1, 1, 1, 0, 0]
    assert agent_data.extra_fields["skd_committed_gen_chunks"] == 1
    assert agent_data.extra_fields["skd_committed_env_units"] == 1
    assert_skd_alignment(agent_data)
    assert_masked_teacher_rows(agent_data)


@pytest.mark.asyncio
async def test_skd_run_until_exportable_boundary_fresh_returns_partial():
    async def fake_process_vision_info(messages):
        del messages
        return {}

    async def fake_apply_chat_template(messages, tools=None, images=None, videos=None, **kwargs):
        del messages, tools, images, videos, kwargs
        return [1, 2, 3]

    loop = make_skd_loop(
        student_chunks=[
            [10, 11],
            [20, EOS],
        ],
        teacher_topk_by_call=[
            {},
            {},
        ],
    )
    loop.tool_schemas = []
    loop.process_vision_info = fake_process_vision_info
    loop.apply_chat_template = fake_apply_chat_template

    result = await loop.run_until_exportable_boundary(
        {},
        sample_id="fresh-boundary",
        logical_step=12,
        source_type="lookahead",
        raw_prompt=[{"role": "user", "content": "question"}],
        tools_kwargs={"session": "abc"},
    )

    assert isinstance(result, SkdPartialState)
    assert result.sample_id == "fresh-boundary"
    assert result.logical_step == 12
    assert result.source_type == "lookahead"
    assert result.agent_state == AgentState.GENERATING.value
    assert result.prompt_ids == [1, 2, 3]
    assert result.response_ids == [10, 11]
    assert result.response_mask == []
    assert result.assistant_turns == 0
    assert result.tools_kwargs == {"session": "abc"}
    assert result.extra_fields["skd_pending_turn_response_ids"] == [10, 11]
    assert result.extra_fields["skd_pending_turn_state"] == {
        "tokens": [10, 11],
        "teacher_ids_rows": [[10, 0, 0, 0], [11, 0, 0, 0]],
        "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K],
        "raw_chunk": [10, 11],
        "verified_chunk": [10, 11],
    }
    assert result.extra_fields["raw_prompt"] == [{"role": "user", "content": "question"}]
    assert result.extra_fields["teacher_ids_list"] == []
    assert result.extra_fields["teacher_logprobs_list"] == []
    assert loop.teacher_server_manager.released_request_ids == []


@pytest.mark.asyncio
async def test_skd_run_until_exportable_boundary_resume_partial_keeps_teacher_replica_id():
    partial = SkdPartialState(
        sample_id="resume-boundary",
        logical_step=13,
        source_type="lookahead",
        agent_state=AgentState.GENERATING.value,
        request_id="req-resume-boundary",
        tools_kwargs={},
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=[1, 2, 3],
        teacher_prompt_ids=[1, 2, 3],
        response_ids=[10],
        response_mask=[],
        response_logprobs=[],
        assistant_turns=0,
        user_turns=0,
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=7,
        committed_gen_chunks=0,
        committed_env_units=0,
        committed_prefix_tokens=0,
        metrics={},
        extra_fields={
            "teacher_prompt_ids": [1, 2, 3],
            "teacher_ids_list": [],
            "teacher_logprobs_list": [],
            "skd_pending_turn_response_ids": [10],
            "skd_pending_turn_state": {
                "tokens": [10],
                "teacher_ids_rows": [[10, 0, 0, 0]],
                "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K],
                "raw_chunk": [10],
                "verified_chunk": [10],
            },
            "skd_pending_turn_chunks": 1,
            "skd_committed_gen_chunks": 0,
            "skd_committed_env_units": 0,
            "skd_committed_prefix_tokens": 0,
            "rollout_birth_version": 7,
            "rollout_min_version": 7,
            "rollout_max_version": 7,
            "teacher_replica_id": "teacher-replica-1",
            "raw_prompt": [{"role": "user", "content": "question"}],
        },
    )

    loop = make_skd_loop(
        student_chunks=[
            [20],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )

    result = await loop.run_until_exportable_boundary(
        {},
        sample_id="resume-boundary",
        logical_step=13,
        source_type="lookahead",
        partial_state=partial,
    )

    assert isinstance(result, SkdPartialState)
    assert result.extra_fields["teacher_replica_id"] == "teacher-replica-1"
    assert result.prompt_ids == [1, 2, 3]
    assert result.response_ids == [10, 20]
    assert result.response_mask == []
    assert result.extra_fields["teacher_ids_list"] == []
    assert result.extra_fields["teacher_logprobs_list"] == []
    assert result.extra_fields["skd_pending_turn_state"] == {
        "tokens": [10, 20],
        "teacher_ids_rows": [[10, 0, 0, 0], [20, 0, 0, 0]],
        "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-1.0] * LOSS_TOP_K],
        "raw_chunk": [20],
        "verified_chunk": [20],
    }
    assert loop.teacher_server_manager.released_request_ids == []


@pytest.mark.asyncio
async def test_skd_teacher_verification_binds_sticky_request_to_pinned_replica():
    loop = make_skd_loop(
        student_chunks=[[10, EOS]],
        teacher_topk_by_call=[{}, {}],
    )
    agent_data = make_agent_data()
    agent_data.extra_fields["teacher_routing_key"] = "default"
    agent_data.extra_fields["teacher_replica_id"] = "teacher-replica-7"

    next_state = await loop._handle_generating_state(
        agent_data,
        {
            "max_tokens": 8,
        },
    )

    assert next_state == AgentState.TERMINATED
    assert loop.teacher_server_manager.bound_requests
    assert loop.teacher_server_manager.bound_requests[0] == {
        "routing_key": "default",
        "request_id": agent_data.request_id,
        "server_id": "teacher-replica-7",
    }


@pytest.mark.asyncio
async def test_skd_teacher_verification_binds_sticky_request_without_routing_key_for_single_teacher():
    loop = make_skd_loop(
        student_chunks=[[10, EOS]],
        teacher_topk_by_call=[{}, {}],
    )
    agent_data = make_agent_data()
    agent_data.extra_fields["teacher_replica_id"] = "teacher-server-0"

    next_state = await loop._handle_generating_state(
        agent_data,
        {
            "max_tokens": 8,
        },
    )

    assert next_state == AgentState.TERMINATED
    assert loop.teacher_server_manager.bound_requests
    assert loop.teacher_server_manager.bound_requests[0] == {
        "routing_key": None,
        "request_id": agent_data.request_id,
        "server_id": "teacher-server-0",
    }


@pytest.mark.asyncio
async def test_skd_run_until_exportable_boundary_fresh_preserves_teacher_assignment_from_kwargs():
    loop = make_skd_loop(
        student_chunks=[[20]],
        teacher_topk_by_call=[{}],
    )

    async def fake_apply_chat_template(messages, tools=None, images=None, videos=None, **kwargs):
        del messages, tools, images, videos, kwargs
        return [1, 2, 3]

    loop.apply_chat_template = fake_apply_chat_template

    result = await loop.run_until_exportable_boundary(
        {},
        sample_id="fresh-boundary-teacher-pin",
        logical_step=12,
        source_type="lookahead",
        raw_prompt=[{"role": "user", "content": "question"}],
        tools_kwargs={"session": "abc"},
        data_source="default",
        teacher_replica_id="teacher-server-1",
    )

    assert isinstance(result, SkdPartialState)
    assert result.extra_fields["teacher_replica_id"] == "teacher-server-1"
    assert result.extra_fields["teacher_routing_key"] == "default"


@pytest.mark.asyncio
async def test_skd_run_until_exportable_boundary_resume_returns_completed_output():
    partial = SkdPartialState(
        sample_id="resume-boundary",
        logical_step=13,
        source_type="lookahead",
        agent_state=AgentState.GENERATING.value,
        request_id="req-resume-boundary",
        tools_kwargs={},
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=[1, 2, 3],
        teacher_prompt_ids=[1, 2, 3],
        response_ids=[10, 11],
        response_mask=[],
        response_logprobs=[],
        assistant_turns=0,
        user_turns=0,
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=7,
        committed_gen_chunks=0,
        committed_env_units=0,
        committed_prefix_tokens=0,
        metrics={},
        extra_fields={
            "teacher_prompt_ids": [1, 2, 3],
            "teacher_ids_list": [],
            "teacher_logprobs_list": [],
            "skd_pending_turn_response_ids": [10, 11],
            "skd_pending_turn_state": {
                "tokens": [10, 11],
                "teacher_ids_rows": [[10, 0, 0, 0], [11, 0, 0, 0]],
                "teacher_logprobs_rows": [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K],
                "raw_chunk": [10, 11],
                "verified_chunk": [10, 11],
            },
            "skd_pending_turn_chunks": 1,
            "skd_committed_gen_chunks": 0,
            "skd_committed_env_units": 0,
            "skd_committed_prefix_tokens": 0,
            "rollout_birth_version": 7,
            "rollout_min_version": 7,
            "rollout_max_version": 7,
            "raw_prompt": [{"role": "user", "content": "question"}],
        },
    )

    loop = make_skd_loop(
        student_chunks=[
            [20, EOS],
        ],
        teacher_topk_by_call=[
            {},
        ],
    )

    result = await loop.run_until_exportable_boundary(
        {},
        sample_id="resume-boundary",
        logical_step=13,
        source_type="lookahead",
        partial_state=partial,
    )

    assert isinstance(result, AgentLoopOutput)
    assert result.prompt_ids == [1, 2, 3]
    assert result.response_ids == [10, 11, 20, EOS]
    assert result.response_mask == [1, 1, 1, 1]
    assert result.num_turns == 2
    assert result.extra_fields["skd_termination_reason"] == "eos"
    assert "skd_pending_turn_response_ids" not in result.extra_fields
    assert result.extra_fields["teacher_ids_list"] == [
        [10, 0, 0, 0],
        [11, 0, 0, 0],
        [20, 0, 0, 0],
        [EOS, 0, 0, 0],
    ]
    assert result.extra_fields["parent_request_id"] is None


@pytest.mark.asyncio
async def test_skd_run_from_partial_to_completion_ignores_exportable_intermediate_boundary():
    partial = SkdPartialState(
        sample_id="resume-to-completion",
        logical_step=13,
        source_type="lookahead",
        agent_state=AgentState.GENERATING.value,
        request_id="req-resume-to-completion",
        tools_kwargs={},
        messages=[{"role": "user", "content": "question"}],
        prompt_ids=[1, 2, 3, 10],
        teacher_prompt_ids=[1, 2, 3, 10],
        response_ids=[10],
        response_mask=[1],
        response_logprobs=[],
        assistant_turns=0,
        user_turns=0,
        rollout_birth_version=7,
        rollout_min_version=7,
        rollout_max_version=7,
        committed_gen_chunks=1,
        committed_env_units=0,
        committed_prefix_tokens=1,
        metrics={},
        extra_fields={
            "teacher_prompt_ids": [1, 2, 3, 10],
            "teacher_ids_list": [[10, 0, 0, 0]],
            "teacher_logprobs_list": [[-1.0] * LOSS_TOP_K],
            "skd_pending_turn_response_ids": [10],
            "skd_committed_gen_chunks": 1,
            "skd_committed_env_units": 0,
            "skd_committed_prefix_tokens": 1,
            "rollout_birth_version": 7,
            "rollout_min_version": 7,
            "rollout_max_version": 7,
            "raw_prompt": [{"role": "user", "content": "question"}],
        },
    )

    loop = make_skd_loop(
        student_chunks=[
            [20],
            [30, EOS],
        ],
        teacher_topk_by_call=[
            {},
            {},
        ],
    )

    old_request_id = partial.request_id
    result = await loop.run_from_partial_to_completion({}, partial_state=partial)

    assert isinstance(result, AgentLoopOutput)
    assert result.prompt_ids == [1, 2, 3]
    assert result.response_ids == [10, 20, 30, EOS]
    assert result.response_mask == [1, 1, 1, 1]
    assert loop.server_manager.call_count == 2
    assert result.extra_fields["skd_termination_reason"] == "eos"
    assert "skd_pending_turn_response_ids" not in result.extra_fields
    new_student_request_ids = {entry["request_id"] for entry in loop.server_manager.call_log}
    new_teacher_request_ids = {entry["request_id"] for entry in loop.teacher_server_manager.call_log}
    assert len(new_student_request_ids) == 1
    assert new_student_request_ids == new_teacher_request_ids
    [new_request_id] = list(new_teacher_request_ids)
    assert new_request_id != old_request_id
    assert result.extra_fields["parent_request_id"] == old_request_id
    assert set(loop.teacher_server_manager.released_request_ids) == {old_request_id, new_request_id}
    assert len(loop.teacher_server_manager.released_request_ids) == 2


@pytest.mark.asyncio
async def test_skd_generating_state_handles_budget_exhausted_before_first_chunk():
    loop = make_skd_loop(student_chunks=[], response_length=1)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.prompt_ids = [1, 2, 3, 10]
    agent_data.response_mask = [1]
    agent_data.response_ids = [10]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3, 10]
    agent_data.extra_fields["teacher_ids_list"] = [[10, 11, 12, 13]]
    agent_data.extra_fields["teacher_logprobs_list"] = [[-1.0] * LOSS_TOP_K]

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    assert agent_data.prompt_ids == [1, 2, 3, 10]
    assert agent_data.response_mask == [1]
    assert agent_data.response_ids == []
    assert agent_data.assistant_turns == 0
    assert agent_data.extra_fields["skd_termination_reason"] == "budget_exhausted"
    assert_skd_alignment(agent_data)


@pytest.mark.asyncio
async def test_skd_teacher_verification_receives_current_tool_images():
    image = object()
    loop = make_skd_loop(student_chunks=[[10, EOS]])
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = [image]
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 0

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    assert loop.teacher_server_manager.call_log[0]["multi_modal_data"] == {"images": [image]}


@pytest.mark.asyncio
async def test_web_skd_image_generation_appends_suffix_to_server_prompt_ids_without_prompt_text():
    loop = make_web_skd_loop(student_chunks=[[10, EOS]], chunk_size=8)
    agent_data = make_agent_data([1, 2, 3])
    agent_data.image_data = ["image-1"]
    agent_data.extra_fields["server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_server_prompt_ids"] = [1, 2, 3]
    agent_data.extra_fields["teacher_sglang_prefix_surplus"] = 0

    next_state = await SkdAgentLoop._handle_generating_state(loop, agent_data, {}, False)

    assert next_state == AgentState.TERMINATED
    student_call = loop.server_manager.call_log[0]
    assert student_call["prompt_ids"] == [1, 2, 3]
    assert student_call["prompt_text"] is None
    assert student_call["image_data"] == ["image-1"]
    assert agent_data.extra_fields["server_prompt_ids"] == [1, 2, 3, 10, EOS]
    assert agent_data.prompt_ids == [1, 2, 3, 10, EOS]
    assert agent_data.extra_fields["teacher_server_prompt_ids"] == [1, 2, 3, 10, EOS]
    assert "student_generate_prompt_text_used" not in agent_data.extra_fields
    assert "student_generate_roundtrip_match" not in agent_data.extra_fields
    assert_skd_alignment(agent_data)


@pytest.mark.asyncio
async def test_tool_macro_step_appends_dummy_teacher_rows(monkeypatch):
    async def fake_tool_step(self, agent_data):
        del self
        agent_data.prompt_ids += [71, 72, 73]
        agent_data.response_mask += [0, 0, 0]
        agent_data.user_turns += 1
        return AgentState.GENERATING

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step)

    loop = make_skd_loop(student_chunks=[])
    agent_data = make_agent_data([41])
    agent_data.response_mask = [1, 1]
    agent_data.prompt_ids = [41, 11, 12]
    agent_data.extra_fields["teacher_prompt_ids"] = [91, 11, 12]
    agent_data.extra_fields["teacher_ids_list"] = [[11, 110, 111, 0], [12, 120, 121, 0]]
    agent_data.extra_fields["teacher_logprobs_list"] = [[-1.0] * LOSS_TOP_K, [-2.0] * LOSS_TOP_K]

    next_state = await SkdAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert next_state == AgentState.GENERATING
    assert loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [41, 11, 12, 71, 72, 73]
    assert agent_data.response_mask == [1, 1, 0, 0, 0]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [91, 11, 12, 71, 72, 73]
    assert teacher_rows(agent_data)[-3:] == [[0] * LOSS_TOP_K] * 3
    assert teacher_logprobs(agent_data)[-3:] == [[0.0] * LOSS_TOP_K] * 3
    assert_skd_alignment(agent_data)
    assert_masked_teacher_rows(agent_data)
    assert agent_data.extra_fields.get("skd_committed_gen_chunks", 0) == 0
    assert agent_data.extra_fields["skd_committed_env_units"] == 1
    assert agent_data.extra_fields["skd_committed_prefix_tokens"] == 3


@pytest.mark.asyncio
async def test_budget_boundary_does_not_append_partial_tool_result(monkeypatch):
    async def fake_tool_step_budget_hit(self, agent_data):
        del self, agent_data
        return AgentState.TERMINATED

    monkeypatch.setattr(ToolAgentLoop, "_handle_processing_tools_state", fake_tool_step_budget_hit)

    loop = make_skd_loop(student_chunks=[])
    agent_data = make_agent_data([41])
    agent_data.prompt_ids = [41, 11]
    agent_data.response_mask = [1]
    agent_data.extra_fields["teacher_prompt_ids"] = [91, 11]
    agent_data.extra_fields["teacher_ids_list"] = [[11, 110, 111, 0]]
    agent_data.extra_fields["teacher_logprobs_list"] = [[-1.0] * LOSS_TOP_K]

    next_state = await SkdAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert next_state == AgentState.TERMINATED
    assert not loop._can_export_partial_state(agent_data, next_state)
    assert agent_data.prompt_ids == [41, 11]
    assert agent_data.response_mask == [1]
    assert agent_data.extra_fields["teacher_prompt_ids"] == [91, 11]
    assert_skd_alignment(agent_data)
    assert agent_data.extra_fields.get("skd_committed_env_units", 0) == 0
    assert agent_data.extra_fields.get("skd_committed_prefix_tokens", 0) == 0


@pytest.mark.asyncio
async def test_teacher_reconstruction_preserves_mixed_actual_and_dummy_rows():
    class DummyWorker:
        stream_teacher_with_rollout = False

    output = AgentLoopOutput(
        prompt_ids=[101, 102, 103, 104],
        response_ids=[10, 11, 900, 901, 30, 40, EOS],
        response_mask=[1, 1, 0, 0, 1, 1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": [
                [10, 110, 111, 0],
                [11, 120, 121, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [30, 130, 131, 0],
                [40, 140, 141, 0],
                [EOS, 150, 151, 0],
            ],
            "teacher_logprobs_list": [
                [-1.0] * LOSS_TOP_K,
                [-2.0] * LOSS_TOP_K,
                [0.0] * LOSS_TOP_K,
                [0.0] * LOSS_TOP_K,
                [-3.0] * LOSS_TOP_K,
                [-4.0] * LOSS_TOP_K,
                [-5.0] * LOSS_TOP_K,
            ],
        },
    )

    await AgentLoopWorker._compute_teacher_logprobs(
        DummyWorker(),
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
    )

    prompt_len = len(output.prompt_ids)
    response_len = len(output.response_ids)
    ids_slice = output.extra_fields["teacher_ids"][prompt_len - 1 : prompt_len + response_len - 1]
    logps_slice = output.extra_fields["teacher_logprobs"][prompt_len - 1 : prompt_len + response_len - 1]

    assert ids_slice.tolist() == [
        [10, 110, 111, 0],
        [11, 120, 121, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [30, 130, 131, 0],
        [40, 140, 141, 0],
        [EOS, 150, 151, 0],
    ]
    assert logps_slice.tolist() == [
        [-1.0] * LOSS_TOP_K,
        [-2.0] * LOSS_TOP_K,
        [0.0] * LOSS_TOP_K,
        [0.0] * LOSS_TOP_K,
        [-3.0] * LOSS_TOP_K,
        [-4.0] * LOSS_TOP_K,
        [-5.0] * LOSS_TOP_K,
    ]
