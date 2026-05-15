from __future__ import annotations

import asyncio
import json
import math
from functools import partial

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker, _InternalAgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager as ExperimentalNaiveRewardManager
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator


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
    extra_fields: dict,
    reward_score: float,
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


class _ResourcePoolStub:
    def get_n_gpus(self) -> int:
        return 1


class _TokenizerStub:
    def batch_decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return [f"decoded-{idx}" for idx in range(len(ids))]

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "decoded"


class _RewardToolStub:
    def __init__(self):
        self._instance_dict = {
            "instance-1": {
                "task_id": "task-1",
                "request_id": 101,
                "include_a11y": False,
                "reward": None,
            }
        }

    async def calc_reward(self, instance_id, **kwargs):
        del instance_id, kwargs
        return 0.0


class _RewardRemoteMethod:
    def __init__(self, reward_manager):
        self.reward_manager = reward_manager

    async def remote(self, data):
        return await self.reward_manager.run_single(data)


class _RewardWorkerHandle:
    def __init__(self, reward_manager):
        self.compute_score = _RewardRemoteMethod(reward_manager)


def test_separate_trainer_web_osgym_reward_path_drives_metrics_and_advantage(tmp_path):
    shaped_score_a = 1.0 + 0.1
    shaped_score_b = 0.0 + 0.1

    metrics = AgentLoopMetrics()
    internal_a = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "request_id": "req-1",
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_raw_format_reward": 1.0,
                "web_osgym_non_grounding_adjacency_ratio": 0.2,
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
                "web_osgym_format_reward": 0.4,
                "web_osgym_raw_format_reward": 1.0,
                "web_osgym_non_grounding_adjacency_ratio": 0.6,
                "web_osgym_attempted_tool_calls": 1,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 1,
            },
            "web_osgym_log_global_step": 7,
        },
        reward_score=shaped_score_b,
        num_turns=3,
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
            "request_id": np.array(["req-1", "req-2"], dtype=object),
        },
    )

    summary_root = tmp_path / "rollout_data"
    step_dir = summary_root / "step_7"
    for request_id, session_id in (("req-1", 101), ("req-2", 202)):
        session_dir = step_dir / f"task___index___{session_id}"
        session_dir.mkdir(parents=True)
        (session_dir / "summary.json").write_text(
            json.dumps(
                {
                    "task_id": "task",
                    "sample_uid": "index",
                    "global_step": 7,
                    "session_id": session_id,
                    "request_id": request_id,
                    "reward_score": 0.0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.metrics = {}
    trainer.timing_raw = {"step": 1.0}
    trainer.resource_pool_manager = _ResourcePoolStub()
    trainer.tokenizer = _TokenizerStub()
    trainer.config = OmegaConf.create(
        {
            "trainer": {"rollout_data_dir": str(summary_root)},
            "algorithm": {
                "use_kl_in_reward": False,
                "adv_estimator": AdvantageEstimator.GRPO,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )
    trainer._dump_generations = lambda **kwargs: None

    batch = trainer._fit_compute_reward(batch)
    assert trainer.reward_tensor.sum(dim=-1).tolist() == pytest.approx([shaped_score_a, shaped_score_b])

    batch = trainer._fit_compute_advantage(batch)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    assert batch.batch["token_level_scores"].sum(dim=-1).tolist() == pytest.approx([shaped_score_a, shaped_score_b])
    assert batch.batch["token_level_rewards"].sum(dim=-1).tolist() == pytest.approx([shaped_score_a, shaped_score_b])
    assert batch.batch["advantages"][0, -1].item() > 0.0
    assert batch.batch["advantages"][1, -1].item() < 0.0

    trainer._fit_collect_metrics(batch)
    assert trainer.metrics["score/sum"] == pytest.approx((shaped_score_a + shaped_score_b) / 2.0)
    assert trainer.metrics["score/env"] == pytest.approx(0.5)
    assert trainer.metrics["score/format"] == pytest.approx(0.6)
    assert trainer.metrics["score/format_raw"] == pytest.approx(1.0)
    assert trainer.metrics["score/non_grounding_adjacency_ratio"] == pytest.approx(0.4)
    assert trainer.metrics["critic/score/mean"] == pytest.approx((shaped_score_a + shaped_score_b) / 2.0)

    trainer._fit_dump_data(batch)
    summary_a = json.loads((step_dir / "task___index___101" / "summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((step_dir / "task___index___202" / "summary.json").read_text(encoding="utf-8"))
    assert summary_a["reward"]["sum"] == pytest.approx(shaped_score_a)
    assert summary_a["reward"]["env"] == pytest.approx(1.0)
    assert summary_a["reward"]["format"] == pytest.approx(0.8)
    assert summary_a["reward"]["raw_format"] == pytest.approx(1.0)
    assert summary_a["reward"]["non_grounding_adjacency_ratio"] == pytest.approx(0.2)
    assert summary_b["reward"]["sum"] == pytest.approx(shaped_score_b)
    assert summary_b["reward"]["env"] == pytest.approx(0.0)
    assert summary_b["reward"]["format"] == pytest.approx(0.4)
    assert summary_b["reward"]["raw_format"] == pytest.approx(1.0)
    assert summary_b["reward"]["non_grounding_adjacency_ratio"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_fast_tool_reward_loop_path_uses_attempted_and_valid_counts():
    reward_manager = ExperimentalNaiveRewardManager(
        config=OmegaConf.create({}),
        tokenizer=_TokenizerStub(),
        compute_score=partial(
            compute_score_webgym_rl,
            format_reward_alpha=0.1,
            format_reward_tau=2.0,
        ),
    )
    reward_handle = _RewardWorkerHandle(reward_manager)

    loop = WebOsGymLoopMixin()
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-1",
        tools_kwargs={},
    )
    agent_data._active_tools = {"computer": _RewardToolStub()}
    agent_data.extra_fields.update(
        {
            "web_osgym_instance_id": "instance-1",
            "web_osgym_task_id": "task-1",
            "web_osgym_session_id": 101,
            "web_osgym_include_a11y": False,
            "web_osgym_trajectory_counts": {
                "attempted_tool_call_count": 6,
                "valid_tool_call_count": 1,
                "first_valid_tool_call_index": 4,
            },
        }
    )
    await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="tool_response_budget_exhausted")

    agent_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=7,
        metrics=AgentLoopMetrics(),
        extra_fields=agent_data.extra_fields,
    )

    dummy_worker = type(
        "_DummyRewardWorker",
        (),
        {
            "reward_loop_worker_handles": [reward_handle],
            "loop": asyncio.get_running_loop(),
            "distillation_enabled": False,
        },
    )()

    await AgentLoopWorker._compute_score(
        dummy_worker,
        agent_output,
        prompts=torch.tensor([[101, 102]], dtype=torch.long),
        responses=torch.tensor([[11, 12]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        input_ids=torch.tensor([[101, 102, 11, 12]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        kwargs={"data_source": "webgym_rl", "reward_model": {"ground_truth": "env_reward"}},
    )

    expected_format = (1.0 / 6.0) * math.exp(-1.5) - 0.15
    expected_score = 0.1 * expected_format
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_format_reward"] == pytest.approx(expected_format)
    assert agent_output.reward_score == pytest.approx(expected_score)
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_first_valid_tool_call_index"] == 4

    internal = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=agent_output.metrics,
        extra_fields=agent_output.extra_fields,
        reward_score=agent_output.reward_score,
        num_turns=agent_output.num_turns,
        prompt_len=2,
        response_len=2,
    )
    batch = AgentLoopWorker._postprocess(
        type("_DummyPostprocessWorker", (), {"reward_loop_worker_handles": None, "distillation_enabled": False})(),
        inputs=[internal],
        input_non_tensor_batch={
            "uid": np.array(["uid-1"], dtype=object),
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["web_tool_agent"], dtype=object),
            "data_source": np.array(["webgym_rl"], dtype=object),
            "reward_model": np.array([{"ground_truth": "env_reward"}], dtype=object),
            "extra_info": np.array([{"task_id": "demo"}], dtype=object),
        },
    )

    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.metrics = {}
    trainer.timing_raw = {"step": 1.0}
    trainer.resource_pool_manager = _ResourcePoolStub()
    trainer.tokenizer = _TokenizerStub()
    trainer.config = OmegaConf.create(
        {
            "trainer": {"rollout_data_dir": None},
            "algorithm": {
                "use_kl_in_reward": False,
                "adv_estimator": AdvantageEstimator.GRPO,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 1}},
        }
    )

    batch = trainer._fit_compute_reward(batch)
    assert trainer.reward_tensor.sum(dim=-1).item() == pytest.approx(expected_score)
    batch = trainer._fit_compute_advantage(batch)
    assert batch.batch["token_level_rewards"].sum(dim=-1).item() == pytest.approx(expected_score)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    trainer._fit_collect_metrics(batch)
    assert trainer.metrics["score/format"] == pytest.approx(expected_format)


@pytest.mark.asyncio
async def test_fast_tool_reward_loop_path_can_gate_format_reward_on_env_score():
    reward_manager = ExperimentalNaiveRewardManager(
        config=OmegaConf.create({}),
        tokenizer=_TokenizerStub(),
        compute_score=partial(
            compute_score_webgym_rl,
            format_reward_alpha=0.5,
            format_reward_tau=2.0,
            format_reward_gate_by_env_score=True,
        ),
    )
    reward_handle = _RewardWorkerHandle(reward_manager)

    loop = WebOsGymLoopMixin()
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-gated",
        tools_kwargs={},
    )
    agent_data._active_tools = {"computer": _RewardToolStub()}
    agent_data.extra_fields.update(
        {
            "web_osgym_instance_id": "instance-1",
            "web_osgym_task_id": "task-1",
            "web_osgym_session_id": 201,
            "web_osgym_include_a11y": False,
            "web_osgym_trajectory_counts": {
                "attempted_tool_call_count": 8,
                "valid_tool_call_count": 8,
                "first_valid_tool_call_index": 1,
            },
        }
    )
    await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="tool_response_budget_exhausted")

    agent_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=9,
        metrics=AgentLoopMetrics(),
        extra_fields=agent_data.extra_fields,
    )

    dummy_worker = type(
        "_DummyRewardWorker",
        (),
        {
            "reward_loop_worker_handles": [reward_handle],
            "loop": asyncio.get_running_loop(),
            "distillation_enabled": False,
        },
    )()

    await AgentLoopWorker._compute_score(
        dummy_worker,
        agent_output,
        prompts=torch.tensor([[101, 102]], dtype=torch.long),
        responses=torch.tensor([[11, 12]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        input_ids=torch.tensor([[101, 102, 11, 12]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        kwargs={"data_source": "webgym_rl", "reward_model": {"ground_truth": "env_reward"}},
    )

    assert agent_output.reward_score == pytest.approx(0.0)
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_env_reward_score"] == pytest.approx(0.0)
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_format_reward"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_fast_tool_reward_loop_path_decays_positive_format_reward_under_non_grounding_repetition():
    reward_manager = ExperimentalNaiveRewardManager(
        config=OmegaConf.create({}),
        tokenizer=_TokenizerStub(),
        compute_score=partial(
            compute_score_webgym_rl,
            format_reward_alpha=0.1,
            format_reward_tau=2.0,
        ),
    )
    reward_handle = _RewardWorkerHandle(reward_manager)

    loop = WebOsGymLoopMixin()
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-repeat-decay",
        tools_kwargs={},
    )
    agent_data._active_tools = {"computer": _RewardToolStub()}
    agent_data.extra_fields.update(
        {
            "web_osgym_instance_id": "instance-1",
            "web_osgym_task_id": "task-1",
            "web_osgym_session_id": 201,
            "web_osgym_include_a11y": False,
            "web_osgym_trajectory_counts": {
                "attempted_tool_call_count": 14,
                "valid_tool_call_count": 14,
                "first_valid_tool_call_index": 1,
                "executed_action_count": 14,
                "non_grounding_adjacent_pair_count": 10,
            },
        }
    )
    await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")

    agent_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=15,
        metrics=AgentLoopMetrics(),
        extra_fields=agent_data.extra_fields,
    )

    dummy_worker = type(
        "_DummyRewardWorker",
        (),
        {
            "reward_loop_worker_handles": [reward_handle],
            "loop": asyncio.get_running_loop(),
            "distillation_enabled": False,
        },
    )()

    await AgentLoopWorker._compute_score(
        dummy_worker,
        agent_output,
        prompts=torch.tensor([[101, 102]], dtype=torch.long),
        responses=torch.tensor([[11, 12]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        input_ids=torch.tensor([[101, 102, 11, 12]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        kwargs={"data_source": "webgym_rl", "reward_model": {"ground_truth": "env_reward"}},
    )

    expected_ratio = 10.0 / 13.0
    expected_effective_format = 1.0 * (1.0 - expected_ratio)
    assert agent_output.reward_score == pytest.approx(0.1 * expected_effective_format)
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_raw_format_reward"] == pytest.approx(1.0)
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_non_grounding_adjacency_ratio"] == pytest.approx(
        expected_ratio
    )
    assert agent_output.extra_fields["reward_extra_info"]["web_osgym_format_reward"] == pytest.approx(
        expected_effective_format
    )


def test_separate_trainer_gated_env_zero_group_produces_zero_grpo_advantage():
    metrics = AgentLoopMetrics()

    internal_a = _to_internal(
        output_prompt_ids=[101, 102],
        output_response_ids=[11, 12],
        output_response_mask=[1, 1],
        metrics=metrics,
        extra_fields={
            "reward_extra_info": {
                "request_id": "req-1",
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.0,
                "web_osgym_attempted_tool_calls": 8,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 8,
            },
            "web_osgym_log_global_step": 11,
        },
        reward_score=0.0,
        num_turns=9,
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
                "web_osgym_attempted_tool_calls": 8,
                "web_osgym_first_valid_tool_call_index": 1,
                "web_osgym_valid_tool_calls": 8,
            },
            "web_osgym_log_global_step": 11,
        },
        reward_score=0.0,
        num_turns=9,
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
            "request_id": np.array(["req-1", "req-2"], dtype=object),
        },
    )

    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.metrics = {}
    trainer.timing_raw = {"step": 1.0}
    trainer.resource_pool_manager = _ResourcePoolStub()
    trainer.tokenizer = _TokenizerStub()
    trainer.config = OmegaConf.create(
        {
            "trainer": {"rollout_data_dir": None},
            "algorithm": {
                "use_kl_in_reward": False,
                "adv_estimator": AdvantageEstimator.GRPO,
                "gamma": 1.0,
                "lam": 1.0,
                "norm_adv_by_std_in_grpo": True,
            },
            "actor_rollout_ref": {"rollout": {"n": 2}},
        }
    )

    batch = trainer._fit_compute_reward(batch)
    batch = trainer._fit_compute_advantage(batch)
    assert batch.batch["token_level_rewards"].sum(dim=-1).tolist() == pytest.approx([0.0, 0.0])
    assert batch.batch["advantages"].sum(dim=-1).tolist() == pytest.approx([0.0, 0.0])
