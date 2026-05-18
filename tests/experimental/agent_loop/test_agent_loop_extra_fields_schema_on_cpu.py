# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import asyncio
import warnings
from functools import partial
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.reward import extract_reward
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.reward_manager.naive import NaiveRewardManager
from verl.workers.rollout.replica import TokenOutput


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data
        # Return a short, deterministic "generation" for testing.
        return TokenOutput(token_ids=prompt_ids[-1:] + [11, 12, 13], log_probs=[0.0, 0.0, 0.0, 0.0])

    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        # Minimal tokenization: return a small prompt.
        return [101, 102]

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [0] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [0] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _to_internal(
    *,
    output_prompt_ids: list[int],
    output_response_ids: list[int],
    output_response_mask: list[int],
    metrics: AgentLoopMetrics,
    extra_fields: dict[str, Any],
    reward_score: float | None = None,
    num_turns: int,
    prompt_len: int,
    response_len: int,
) -> _InternalAgentLoopOutput:
    prompt_ids = _pad_1d(output_prompt_ids, length=prompt_len, pad_id=0)
    response_ids = _pad_1d(output_response_ids, length=response_len, pad_id=0)
    response_mask = _pad_1d(output_response_mask, length=response_len, pad_id=0)

    seq_len = prompt_len + response_len
    attention_mask = _pad_1d([1] * len(output_prompt_ids), length=prompt_len, pad_id=0) + _pad_1d(
        [1] * len(output_response_ids),
        length=response_len,
        pad_id=0,
    )
    input_ids = prompt_ids + response_ids
    position_ids = list(range(seq_len))

    def t(x: list[int]) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    return _InternalAgentLoopOutput(
        prompt_ids=t(prompt_ids),
        response_ids=t(response_ids),
        response_mask=t(response_mask),
        attention_mask=t(attention_mask),
        input_ids=t(input_ids),
        position_ids=t(position_ids),
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=reward_score,
        num_turns=num_turns,
        metrics=metrics,
        extra_fields=extra_fields,
    )


class _FakeRewardRemoteMethod:
    def __init__(self, result: dict[str, Any]):
        self.result = result
        self.calls: list[Any] = []

    async def remote(self, data):
        self.calls.append(data)
        return self.result


class _FakeRewardWorkerHandle:
    def __init__(self, result: dict[str, Any]):
        self.compute_score = _FakeRewardRemoteMethod(result)


@pytest.mark.asyncio
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu():
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                "model": {},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    processor = None

    trainer_config = DictConfigWrap(config)
    data_config = DictConfigWrap(config.data)

    single_turn = SingleTurnAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        data_config=data_config,
    )

    raw_prompt = [{"role": "user", "content": "hi"}]
    sampling_params: dict[str, Any] = {}

    out = await single_turn.run(sampling_params=sampling_params, raw_prompt=raw_prompt)

    # Agent loop outputs should always contain these fields with consistent types.
    assert out.extra_fields["turn_scores"] == []
    assert out.extra_fields["tool_rewards"] == []

    internal_a = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields=out.extra_fields,
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    internal_b = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields={**out.extra_fields, "tool_parse_error_retry_count": 2},
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    # Mimic two "worker chunks" and concatenate as in training.
    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    merged = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a, internal_b],
        input_non_tensor_batch={
            "index": np.array([0, 1], dtype=object),
            "agent_name": np.array(["single_turn_agent", "single_turn_agent"], dtype=object),
        },
    )

    # Stable schema: present regardless of which loop produced a sample.
    stable_keys = (
        "turn_scores",
        "tool_rewards",
        "min_global_steps",
        "max_global_steps",
        "extras",
        "tool_parse_error_retry_count",
    )
    for key in stable_keys:
        assert key in merged.non_tensor_batch, f"missing key in merged batch: {key}"
        assert merged.non_tensor_batch[key].shape == (2,), (
            f"invalid shape for {key}: {merged.non_tensor_batch[key].shape}"
        )

    # And the list-typed fields are actually lists (not missing / scalar).
    assert merged.non_tensor_batch["turn_scores"][0] == []
    assert merged.non_tensor_batch["tool_rewards"][0] == []
    assert merged.non_tensor_batch["tool_parse_error_retry_count"][0] == 0
    assert merged.non_tensor_batch["tool_parse_error_retry_count"][1] == 2


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        distillation_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.reward_loop_worker_handles = None

    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        internal = await AgentLoopWorker._agent_loop_postprocess(
            _DummyWorker(),
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    expected = torch.tensor(routed_experts.copy()).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


def test_web_osgym_training_reward_is_computed_by_reward_manager_from_extra_info():
    metrics = AgentLoopMetrics()
    internal = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_attempted_tool_calls": 2,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 2,
                "web_osgym_termination_reason": "model_done",
            }
        },
        num_turns=4,
        prompt_len=2,
        response_len=2,
    )

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    batch = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal],
        input_non_tensor_batch={
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["web_tool_agent"], dtype=object),
            "data_source": np.array(["webgym_rl"], dtype=object),
            "reward_model": np.array([{"ground_truth": "env_reward"}], dtype=object),
            "extra_info": np.array([{"task_id": "demo-task"}], dtype=object),
        },
    )

    assert "rm_scores" not in batch.batch
    merged_extra_info = batch.non_tensor_batch["extra_info"][0]
    assert merged_extra_info["task_id"] == "demo-task"
    assert merged_extra_info["web_osgym_env_reward_score"] == 1.0
    assert merged_extra_info["web_osgym_attempted_tool_calls"] == 2
    assert merged_extra_info["web_osgym_first_valid_tool_call_index"] == 1
    assert merged_extra_info["web_osgym_valid_tool_calls"] == 2

    reward_manager = NaiveRewardManager(
        tokenizer=_FakeTokenizer(),
        num_examine=0,
        compute_score=partial(
            compute_score_webgym_rl,
            format_reward_alpha=0.03,
            format_reward_tau=2.0,
        ),
    )
    reward_result = reward_manager(batch, return_dict=True)

    expected_score = 1.0 + 0.03
    reward_tensor = reward_result["reward_tensor"]
    assert reward_tensor.shape == batch.batch["responses"].shape
    assert reward_tensor[0, 1].item() == pytest.approx(expected_score)
    assert reward_result["reward_extra_info"]["web_osgym_env_reward_score"] == [1.0]
    assert reward_result["reward_extra_info"]["web_osgym_format_reward"] == [1.0]
    assert reward_result["reward_extra_info"]["web_osgym_attempted_tool_calls"] == [2]
    assert reward_result["reward_extra_info"]["web_osgym_first_valid_tool_call_index"] == [1]
    assert reward_result["reward_extra_info"]["web_osgym_valid_tool_calls"] == [2]


@pytest.mark.asyncio
async def test_async_reward_loop_preserves_request_id_when_merging_reward_extra_info():
    reward_handle = _FakeRewardWorkerHandle(
        {
            "reward_score": 1.012,
            "reward_extra_info": {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_attempted_tool_calls": 2,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 2,
            },
        }
    )

    dummy_worker = type(
        "_DummyWorker",
        (),
        {
            "reward_loop_worker_handles": [reward_handle],
            "loop": asyncio.get_running_loop(),
        },
    )()

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={"reward_extra_info": {"request_id": "req-1"}},
    )

    await AgentLoopWorker._compute_score(
        dummy_worker,
        output,
        prompts=torch.tensor([[101, 102]], dtype=torch.long),
        responses=torch.tensor([[11, 12]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        input_ids=torch.tensor([[101, 102, 11, 12]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        kwargs={"data_source": "webgym_rl", "reward_model": {"ground_truth": "env_reward"}},
    )

    assert output.reward_score == pytest.approx(1.012)
    assert output.extra_fields["reward_extra_info"] == {
        "request_id": "req-1",
        "web_osgym_env_reward_score": 1.0,
        "web_osgym_format_reward": 0.4,
        "web_osgym_attempted_tool_calls": 2,
        "web_osgym_first_valid_tool_call_index": 1,
        "web_osgym_valid_tool_calls": 2,
    }


def test_postprocess_merges_web_osgym_trajectory_dir_from_reward_extra_info_into_extra_info():
    metrics = AgentLoopMetrics()
    internal = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "request_id": "req-1",
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_trajectory_dir": "/tmp/trajectory/req-1",
            }
        },
        reward_score=None,
        num_turns=2,
        prompt_len=2,
        response_len=2,
    )

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    batch = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal],
        input_non_tensor_batch={
            "uid": np.array(["uid-1"], dtype=object),
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["web_tool_agent"], dtype=object),
            "data_source": np.array(["webgym_rl"], dtype=object),
            "reward_model": np.array([{"ground_truth": "env_reward"}], dtype=object),
            "extra_info": np.array([{"task_id": "demo-a"}], dtype=object),
        },
    )

    merged_extra_info = batch.non_tensor_batch["extra_info"].tolist()[0]
    assert merged_extra_info["task_id"] == "demo-a"
    assert merged_extra_info["web_osgym_trajectory_dir"] == "/tmp/trajectory/req-1"


def test_web_osgym_async_reward_path_drives_training_reward_breakdown_and_advantage():
    metrics = AgentLoopMetrics()
    shaped_score_a = 1.0 + 0.03
    shaped_score_b = 0.0

    internal_a = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "request_id": "req-1",
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 1.0,
                "web_osgym_attempted_tool_calls": 2,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 2,
            },
            "web_osgym_log_global_step": 7,
        },
        reward_score=shaped_score_a,
        num_turns=4,
        prompt_len=2,
        response_len=2,
    )
    internal_b = _to_internal(
        output_prompt_ids=[201, 202],
        output_response_ids=[21, 22],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "request_id": "req-2",
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.0,
                "web_osgym_attempted_tool_calls": 0,
                "web_osgym_first_valid_tool_call_index": 0,
                "web_osgym_valid_tool_calls": 0,
            },
            "web_osgym_log_global_step": 7,
        },
        reward_score=shaped_score_b,
        num_turns=2,
        prompt_len=2,
        response_len=2,
    )

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    batch = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a, internal_b],
        input_non_tensor_batch={
            "uid": np.array(["uid-1", "uid-1"], dtype=object),
            "index": np.array([0, 0], dtype=object),
            "agent_name": np.array(["web_tool_agent", "web_tool_agent"], dtype=object),
            "data_source": np.array(["webgym_rl", "webgym_rl"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": "env_reward"}, {"ground_truth": "env_reward"}], dtype=object
            ),
            "extra_info": np.array([{"task_id": "demo-a"}, {"task_id": "demo-b"}], dtype=object),
        },
    )

    reward_tensor, reward_extra_infos_dict = extract_reward(batch)
    assert reward_tensor.sum(dim=-1).tolist() == pytest.approx([shaped_score_a, shaped_score_b])
    assert reward_extra_infos_dict["request_id"].tolist() == ["req-1", "req-2"]

    breakdown = SeparateRayPPOTrainer._compute_reward_breakdown_metrics(reward_tensor, reward_extra_infos_dict)
    assert breakdown["score/sum"] == pytest.approx((shaped_score_a + shaped_score_b) / 2.0)
    assert breakdown["score/env"] == pytest.approx(0.5)
    assert breakdown["score/format"] == pytest.approx(0.5)

    batch.batch["token_level_scores"] = reward_tensor
    batch.batch["token_level_rewards"] = reward_tensor
    batch = compute_advantage(
        batch,
        adv_estimator=AdvantageEstimator.GRPO,
        num_repeat=2,
        norm_adv_by_std_in_grpo=True,
        config=None,
    )
    assert batch.batch["advantages"][0, -1].item() > 0.0
    assert batch.batch["advantages"][1, -1].item() < 0.0
