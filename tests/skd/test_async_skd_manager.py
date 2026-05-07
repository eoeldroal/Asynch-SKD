"""Unit tests for AsyncSkdAgentLoopManager sample-level scheduling."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.manager import AsyncSkdAgentLoopManager
from verl.experimental.async_skd.state import SkdPartialState
from verl.protocol import DataProto


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeWorker:
    def __init__(self, *, name: str, delays: dict[int, float], calls: list[tuple[str, int]]):
        self._name = name
        self._delays = delays
        self._calls = calls
        self.generate_sequence_single = _RemoteMethod(self._generate_sequence_single)

    async def _generate_sequence_single(self, sample: DataProto) -> DataProto:
        input_pos = int(sample.non_tensor_batch["input_pos"][0])
        self._calls.append((self._name, input_pos))
        await asyncio.sleep(self._delays.get(input_pos, 0.0))
        return _make_output(input_pos)


class _FakeTeacherModelManager:
    def __init__(self, server_addresses: dict[str, list[str]]):
        self.server_addresses = server_addresses


def _make_prompts(batch_size: int) -> DataProto:
    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.arange(batch_size, dtype=torch.long).unsqueeze(-1)},
        non_tensors={
            "input_pos": np.array(list(range(batch_size)), dtype=object),
            "preferred_worker": np.array([f"sample-{i}" for i in range(batch_size)], dtype=object),
        },
        meta_info={"global_steps": 3, "validate": False},
    )


def _make_output(input_pos: int) -> DataProto:
    prompt_len = 2
    response_len = 3
    seq_len = prompt_len + response_len
    prompts = torch.tensor([[100 + input_pos, 200 + input_pos]], dtype=torch.long)
    responses = torch.tensor([[input_pos, input_pos + 10, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={
            "input_pos": np.array([input_pos], dtype=object),
            "payload": np.array([f"out-{input_pos}"], dtype=object),
        },
        meta_info={
            "metrics": [
                {
                    "generate_sequences": float(input_pos + 1),
                    "tool_calls": float(input_pos % 2),
                    "num_preempted": -1,
                }
            ]
        },
    )




def _make_output_with_prompt_width(input_pos: int, prompt_len: int, *, with_routed_experts: bool = False) -> DataProto:
    response_len = 3
    seq_len = prompt_len + response_len
    prompts = torch.arange(100 + input_pos, 100 + input_pos + prompt_len, dtype=torch.long).unsqueeze(0)
    responses = torch.tensor([[input_pos, input_pos + 10, 0]], dtype=torch.long)
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if with_routed_experts:
        tensors["routed_experts"] = torch.full((1, seq_len, 1, 1), input_pos + 1, dtype=torch.long)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "input_pos": np.array([input_pos], dtype=object),
            "payload": np.array([f"out-{input_pos}"], dtype=object),
        },
        meta_info={
            "metrics": [
                {
                    "generate_sequences": float(input_pos + 1),
                    "tool_calls": float(input_pos % 2),
                    "num_preempted": -1,
                }
            ]
        },
    )

def _make_manager(*, mode: str = "sample_async", rollout_n: int = 1, delays: dict[int, float] | None = None):
    calls: list[tuple[str, int]] = []
    manager = AsyncSkdAgentLoopManager.__new__(AsyncSkdAgentLoopManager)

    manager.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": rollout_n,
                    "agent": {"async_skd_mode": mode},
                }
            }
        }
    )
    manager.rollout_config = OmegaConf.create({"n": rollout_n})
    manager.set_async_skd_pad_token_id(0)
    manager.stream_teacher_with_rollout = False
    manager.agent_loop_workers = [
        _FakeWorker(name="worker-0", delays=delays or {}, calls=calls),
        _FakeWorker(name="worker-1", delays=delays or {}, calls=calls),
    ]
    return manager, calls


@pytest.mark.asyncio
async def test_sample_async_manager_preserves_input_order_under_out_of_order_completion():
    manager, calls = _make_manager(delays={0: 0.03, 1: 0.0, 2: 0.02, 3: 0.0})
    output = await manager.generate_sequences(_make_prompts(4))

    assert output.non_tensor_batch["input_pos"].tolist() == [0, 1, 2, 3]
    assert output.non_tensor_batch["payload"].tolist() == ["out-0", "out-1", "out-2", "out-3"]
    assert output.batch["responses"][:, 0].tolist() == [0, 1, 2, 3]
    assert "timing" in output.meta_info
    assert output.meta_info["timing"]["agent_loop/generate_sequences/max"] == 4.0

    # All base samples are submitted up front.  This preserves the concurrency
    # of the existing per-worker generate_sequences(chunk) path while exposing
    # per-sample completion events to the manager.
    assert calls == [("worker-0", 0), ("worker-0", 1), ("worker-1", 2), ("worker-1", 3)]


@pytest.mark.asyncio
async def test_sample_async_manager_rejects_rollout_n_greater_than_one():
    manager, _ = _make_manager(rollout_n=2)

    with pytest.raises(ValueError, match="rollout.n == 1"):
        await manager.generate_sequences(_make_prompts(1))


def test_async_skd_manager_mode_defaults_to_sync():
    manager, _ = _make_manager(mode="sample_async")
    assert manager._async_skd_mode() == "sample_async"

    manager.config = OmegaConf.create({})
    assert manager._async_skd_mode() == "sync"


def test_teacher_server_id_map_refreshes_when_initial_cache_is_empty():
    manager, _ = _make_manager()
    manager._teacher_server_ids_by_routing_key = {}
    manager.teacher_model_manager = _FakeTeacherModelManager(
        {
            "default": ["teacher-0", "teacher-1"],
        }
    )

    server_id_map = manager._teacher_server_id_map()

    assert server_id_map == {"default": ["teacher-0", "teacher-1"]}
    assert manager._teacher_server_ids_by_routing_key == {"default": ["teacher-0", "teacher-1"]}


def test_resolve_teacher_routing_key_recovers_from_empty_cache_via_lazy_refresh():
    manager, _ = _make_manager()
    manager._teacher_server_ids_by_routing_key = {}
    manager.teacher_model_manager = _FakeTeacherModelManager(
        {
            "default": ["teacher-0", "teacher-1"],
        }
    )

    resolved = manager._resolve_teacher_routing_key("nvidia/Nemotron-Cascade-RL-Math")

    assert resolved == "default"


def test_teacher_server_id_map_loads_from_parent_teacher_model_manager():
    manager, _ = _make_manager()
    manager._teacher_server_ids_by_routing_key = {}
    manager.teacher_model_manager = _FakeTeacherModelManager(
        {
            "default": [
                "http://teacher-0",
                "http://teacher-1",
            ]
        }
    )

    server_id_map = manager._teacher_server_id_map()

    assert server_id_map == {"default": ["http://teacher-0", "http://teacher-1"]}
    assert manager._resolve_teacher_routing_key("nvidia/Nemotron-Cascade-RL-Math") == "default"


def test_parent_teacher_model_manager_is_used_when_worker_teacher_client_is_absent():
    manager, _ = _make_manager()
    manager._teacher_server_ids_by_routing_key = {}
    manager.teacher_model_manager = _FakeTeacherModelManager(
        {
            "default": [
                "http://teacher-0",
                "http://teacher-1",
            ]
        }
    )

    assert manager._teacher_replica_ids_for_planning(routing_key="nvidia/Nemotron-Cascade-RL-Math") == [
        "http://teacher-0",
        "http://teacher-1",
    ]


def test_teacher_sticky_carryover_can_be_disabled_via_config():
    manager, _ = _make_manager()
    manager.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "agent": {
                        "async_skd_mode": "lookahead",
                        "async_skd_teacher_sticky_carryover": False,
                    },
                }
            }
        }
    )
    manager._teacher_server_ids_by_routing_key = {"default": ["teacher-0", "teacher-1"]}
    manager._teacher_replica_pin_by_sample_id = {"carry-1": "teacher-1"}
    manager._teacher_routing_key_by_sample_id = {"carry-1": "default"}

    partial = SkdPartialState(
        sample_id="carry-1",
        logical_step=4,
        source_type="resumed_current",
        agent_state="generating",
        request_id="req-carry-1",
        extra_fields={"teacher_replica_id": "teacher-1", "teacher_routing_key": "default"},
    )

    assignments = manager._plan_teacher_replica_assignments(
        carryover_sample_ids=["carry-1"],
        fresh_sample_ids=[],
        carryover_partials=[partial],
        fresh_payloads_by_sample_id={},
    )

    assert assignments == {"carry-1": "teacher-0"}
    assert manager._teacher_replica_last_plan_stats == {
        "async_skd/teacher_pinned_carryover_count": 0,
        "async_skd/teacher_fallback_carryover_count": 0,
    }


def test_lookahead_prefetch_limit_allows_two_step_horizon():
    manager, _ = _make_manager()
    manager.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "agent": {
                        "async_skd_mode": "lookahead",
                        "async_skd_prefetch_limit": 80,
                    },
                }
            }
        }
    )

    assert manager._lookahead_prefetch_limit(64) == 80


def test_async_skd_manager_finalize_outputs_aligns_prompt_width_before_concat():
    manager, _ = _make_manager()
    manager.set_async_skd_pad_token_id(0)

    outputs = [
        _make_output_with_prompt_width(0, 2),
        _make_output_with_prompt_width(1, 5),
    ]

    finalized = manager._finalize_outputs(outputs)

    assert finalized.batch["prompts"].shape == (2, 5)
    assert finalized.batch["input_ids"].shape == (2, 8)
    assert finalized.batch["attention_mask"].shape == (2, 8)
    assert finalized.batch["position_ids"].shape == (2, 8)
    assert finalized.non_tensor_batch["input_pos"].tolist() == [0, 1]
    assert finalized.batch["prompts"][0].tolist() == [0, 0, 0, 100, 101]
    assert finalized.batch["prompts"][1].tolist() == [101, 102, 103, 104, 105]


def test_async_skd_manager_finalize_outputs_uses_injected_pad_token():
    manager, _ = _make_manager()
    manager.set_async_skd_pad_token_id(7)

    outputs = [
        _make_output_with_prompt_width(0, 2),
        _make_output_with_prompt_width(1, 5),
    ]

    finalized = manager._finalize_outputs(outputs)

    assert finalized.batch["prompts"][0].tolist() == [7, 7, 7, 100, 101]


def test_async_skd_manager_finalize_outputs_rejects_missing_explicit_pad_token_id():
    manager, _ = _make_manager()
    manager._async_skd_pad_token_id_value = None

    with pytest.raises(ValueError, match="pad_token_id"):
        manager._finalize_outputs([
            _make_output_with_prompt_width(0, 2),
            _make_output_with_prompt_width(1, 5),
        ])


def test_async_skd_manager_rejects_none_pad_token_in_setter():
    manager, _ = _make_manager()

    with pytest.raises(ValueError, match="explicit tokenizer pad_token_id"):
        manager.set_async_skd_pad_token_id(None)


def test_async_skd_manager_finalize_outputs_aligns_routed_experts_with_prompt_width():
    manager, _ = _make_manager()
    manager.set_async_skd_pad_token_id(0)

    outputs = [
        _make_output_with_prompt_width(0, 2, with_routed_experts=True),
        _make_output_with_prompt_width(1, 5, with_routed_experts=True),
    ]

    finalized = manager._finalize_outputs(outputs)

    assert finalized.batch["routed_experts"].shape == (2, 8, 1, 1)
    assert finalized.batch["routed_experts"][0, :3].eq(0).all()
    assert finalized.batch["routed_experts"][0, 3:].eq(1).all()
    assert finalized.batch["routed_experts"][1].eq(2).all()


def test_async_skd_manager_finalize_outputs_rejects_missing_prompts():
    manager, _ = _make_manager()
    manager.set_async_skd_pad_token_id(0)
    bad_output = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
        },
        non_tensors={"input_pos": np.array([0], dtype=object)},
        meta_info={"metrics": [{"generate_sequences": 1.0, "tool_calls": 0.0, "num_preempted": -1}]},
    )

    with pytest.raises(ValueError, match="requires 'prompts'"):
        manager._finalize_outputs([bad_output])
