from __future__ import annotations

import asyncio
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import openai
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from tensordict import TensorDict

from WebOSWorld.webgym_rl import reward_fn_webgym_rl as webgym_reward_fn_module
from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker, _InternalAgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.experimental.reward_loop import reward_loop as reward_loop_module
from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager as ExperimentalNaiveRewardManager
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.ppo.core_algos import AdvantageEstimator


@pytest.fixture(autouse=True)
def _block_live_openai_client(monkeypatch):
    def _raise_unexpected_openai(*args, **kwargs):
        raise AssertionError("Unexpected live OpenAI client construction in test")

    monkeypatch.setattr(openai, "OpenAI", _raise_unexpected_openai)
    monkeypatch.setattr(openai, "AsyncOpenAI", _raise_unexpected_openai)
    monkeypatch.setattr(webgym_reward_fn_module, "_OPENAI_CLIENT", None)
    monkeypatch.setattr(webgym_reward_fn_module, "_ASYNC_OPENAI_CLIENT", None)


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


class _RewardBatchRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, data):
        return self._fn(data)


class _RewardBatchWorkerHandle:
    def __init__(self, fn):
        self.compute_score_batch = _RewardBatchRemoteMethod(fn)


class _LocalAsyncRemoteMethod:
    def __init__(self, worker, method_name: str):
        self._worker = worker
        self._method_name = method_name

    def remote(self, data):
        async def _run():
            loop = asyncio.get_running_loop()
            self._worker.loop = loop
            if hasattr(self._worker, "reward_manager"):
                self._worker.reward_manager.loop = loop
            method = getattr(self._worker, self._method_name)
            return await method(data)

        return asyncio.run(_run())


class _LocalRewardLoopWorkerHandle:
    def __init__(self, worker):
        self.compute_score_batch = _LocalAsyncRemoteMethod(worker, "compute_score_batch")


class _LocalRewardLoopRemoteBuilder:
    def __init__(self, worker_cls, kwargs):
        self._worker_cls = worker_cls
        self.kwargs = kwargs

    def remote(self, config, reward_router_address):
        worker = self._worker_cls(config, reward_router_address)
        return _LocalRewardLoopWorkerHandle(worker)


class _LocalRewardLoopRemoteClass:
    def __init__(self, worker_cls):
        self._worker_cls = worker_cls

    def options(self, **kwargs):
        return _LocalRewardLoopRemoteBuilder(self._worker_cls, kwargs)


class _BatchRewardLoopManagerStub:
    reward_loop_worker_handles = None

    def __init__(
        self,
        scores: list[float],
        env_scores: list[float],
        *,
        judge_used: list[bool] | None = None,
        judge_scores: list[float] | None = None,
    ):
        self.scores = scores
        self.env_scores = env_scores
        self.judge_used = judge_used or [False] * len(scores)
        self.judge_scores = judge_scores or [0.0] * len(scores)

    def compute_rm_score(self, batch: DataProto) -> DataProto:
        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        valid_response_length = batch.batch["attention_mask"][:, batch.batch["prompts"].size(1) :].sum(dim=1)
        rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = torch.tensor(
            self.scores, dtype=torch.float32
        )
        existing_env_scores = batch.non_tensor_batch["web_osgym_env_reward_score"]
        return DataProto(
            batch=TensorDict({"rm_scores": rm_scores}, batch_size=len(batch)),
            non_tensor_batch={
                "web_osgym_env_reward_score": existing_env_scores,
                "web_osgym_llm_judge_used": np.array(self.judge_used, dtype=object),
                "web_osgym_llm_judge_score": np.array(self.judge_scores, dtype=object),
            },
            meta_info={
                "reward_extra_keys": [
                    "web_osgym_env_reward_score",
                    "web_osgym_llm_judge_used",
                    "web_osgym_llm_judge_score",
                ]
            },
        )


def _make_reward_loop_batch(
    *,
    uid_to_env_scores: dict[str, list[float]],
    validate: bool = False,
    include_judge_standard: bool = False,
) -> DataProto:
    metrics = AgentLoopMetrics()
    internals = []
    uids: list[str] = []
    indexes: list[int] = []
    data_sources: list[str] = []
    reward_models: list[dict] = []
    extra_infos: list[dict] = []

    sample_index = 0
    for uid, env_scores in uid_to_env_scores.items():
        for env_score in env_scores:
            internals.append(
                _to_internal(
                    output_prompt_ids=[101, 102],
                    output_response_ids=[11, 12],
                    output_response_mask=[1, 1],
                    metrics=metrics,
                    extra_fields={
                        "reward_extra_info": {
                            "web_osgym_env_reward_score": float(env_score),
                            "web_osgym_trajectory_dir": f"/tmp/{uid}/{sample_index}",
                        }
                    },
                    reward_score=None,
                    num_turns=2,
                    prompt_len=2,
                    response_len=2,
                )
            )
            uids.append(uid)
            indexes.append(sample_index)
            data_sources.append("webgym_rl")
            reward_models.append({"ground_truth": "env_reward"})
            extra_info = {"task_name": f"task-{uid}"}
            if include_judge_standard:
                extra_info["judge_standard"] = [
                    {"id": "task_progress", "text": f"The task for {uid} shows progress."}
                ]
            extra_infos.append(extra_info)
            sample_index += 1

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    batch = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=internals,
        input_non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "index": np.array(indexes, dtype=object),
            "agent_name": np.array(["web_tool_agent"] * len(internals), dtype=object),
            "data_source": np.array(data_sources, dtype=object),
            "reward_model": np.array(reward_models, dtype=object),
            "extra_info": np.array(extra_infos, dtype=object),
        },
    )
    batch.meta_info["validate"] = validate
    return batch


def _write_runtime_trajectory_dir(root: Path, sample_name: str, *, turn_count: int = 3) -> Path:
    trajectory_dir = root / sample_name
    image_dir = trajectory_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = [{"event_type": "initial_observation", "image_paths": ["images/init.png"]}]
    Image.new("RGB", (2, 2), (255, 255, 255)).save(image_dir / "init.png")

    for turn_idx in range(1, turn_count + 1):
        image_name = f"images/turn_{turn_idx}.png"
        Image.new("RGB", (2, 2), (255, 255, 255)).save(image_dir / f"turn_{turn_idx}.png")
        events.append(
            {
                "event_type": "assistant_turn",
                "assistant_turn": turn_idx,
                "image_paths": [image_name],
                "actions": [
                    {
                        "action_type": "click",
                        "target": f"button-{turn_idx}",
                    }
                ],
                "result": {
                    "action_count": 1,
                    "invalid_action": False,
                },
            }
        )

    (trajectory_dir / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return trajectory_dir


def _make_runtime_reward_batch(
    tmp_path: Path,
    *,
    uid_to_env_scores: dict[str, list[float]],
    validate: bool = False,
) -> DataProto:
    metrics = AgentLoopMetrics()
    internals = []
    uids: list[str] = []
    indexes: list[int] = []
    data_sources: list[str] = []
    reward_models: list[dict] = []
    extra_infos: list[dict] = []

    sample_index = 0
    for uid, env_scores in uid_to_env_scores.items():
        for env_score in env_scores:
            trajectory_dir = _write_runtime_trajectory_dir(tmp_path, f"{uid}-{sample_index}", turn_count=3)
            internals.append(
                _to_internal(
                    output_prompt_ids=[101, 102],
                    output_response_ids=[11, 12],
                    output_response_mask=[1, 1],
                    metrics=metrics,
                    extra_fields={
                        "reward_extra_info": {
                            "request_id": f"req-{sample_index}",
                            "web_osgym_env_reward_score": float(env_score),
                            "web_osgym_trajectory_dir": str(trajectory_dir),
                            "web_osgym_attempted_tool_calls": 2,
                            "web_osgym_valid_tool_calls": 2,
                            "web_osgym_first_valid_tool_call_index": 1,
                        }
                    },
                    reward_score=None,
                    num_turns=3,
                    prompt_len=2,
                    response_len=2,
                )
            )
            uids.append(uid)
            indexes.append(sample_index)
            data_sources.append("webgym_rl")
            reward_models.append({"ground_truth": "env_reward"})
            extra_infos.append(
                {
                    "task_name": f"task-{uid}",
                    "judge_standard": [
                        {"id": "progress", "text": f"The task for {uid} shows progress."},
                        {"id": "final_state", "text": f"The task for {uid} reaches the requested final state."},
                    ],
                }
            )
            sample_index += 1

    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    batch = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=internals,
        input_non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            "index": np.array(indexes, dtype=object),
            "agent_name": np.array(["web_tool_agent"] * len(internals), dtype=object),
            "data_source": np.array(data_sources, dtype=object),
            "reward_model": np.array(reward_models, dtype=object),
            "extra_info": np.array(extra_infos, dtype=object),
        },
    )
    batch.meta_info["validate"] = validate
    return batch


class _FakeJudgeResponse:
    def __init__(self, output_text: str):
        self.output_text = output_text


class _FakeJudgeResponsesAPI:
    def __init__(self, requests: list[dict], output_text: str):
        self._requests = requests
        self._output_text = output_text

    def create(self, **request):
        self._requests.append(request)
        return _FakeJudgeResponse(self._output_text)


class _FakeJudgeClientWithOptions:
    def __init__(self, requests: list[dict], timeouts: list[float], timeout: float, output_text: str):
        self.responses = _FakeJudgeResponsesAPI(requests, output_text)
        self._timeouts = timeouts
        self._timeouts.append(timeout)


class _FakeJudgeClient:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        return _FakeJudgeClientWithOptions(self.requests, self.timeouts, timeout, self.output_text)


class _FakeAsyncJudgeResponsesAPI:
    def __init__(self, requests: list[dict], output_text: str):
        self._requests = requests
        self._output_text = output_text

    async def create(self, **request):
        self._requests.append(request)
        return _FakeJudgeResponse(self._output_text)


class _FakeAsyncJudgeClientWithOptions:
    def __init__(self, requests: list[dict], timeouts: list[float], timeout: float, output_text: str):
        self.responses = _FakeAsyncJudgeResponsesAPI(requests, output_text)
        self._timeouts = timeouts
        self._timeouts.append(timeout)


class _FakeAsyncJudgeClient:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        return _FakeAsyncJudgeClientWithOptions(self.requests, self.timeouts, timeout, self.output_text)


def _build_runtime_reward_loop_config(*, num_workers: int, rollout_n: int, val_n: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "reward": {
                "num_workers": num_workers,
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {
                    "source": "register",
                    "name": "naive",
                    "module": {"path": None, "name": "custom_reward_manager"},
                },
                "custom_reward_function": {
                    "path": "/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py",
                    "name": "compute_score_webgym_rl",
                    "reward_kwargs": {
                        "format_reward_alpha": 0.0,
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                        "llm_judge_model": "gpt-5.4-mini",
                        "llm_judge_reasoning_effort": "medium",
                        "llm_judge_image_detail": "auto",
                        "llm_judge_timeout_seconds": 17,
                        "llm_judge_max_concurrency": 2,
                    },
                },
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "dummy-model",
                    "tokenizer_path": None,
                },
                "rollout": {
                    "n": rollout_n,
                    "val_kwargs": {"n": val_n},
                },
            },
        }
    )


def _build_fake_judge_output(rollout_n: int) -> str:
    labels = webgym_reward_fn_module.build_compare_labels(rollout_n)
    ranks = [1, 1] + list(range(2, rollout_n))
    payload = {f"{label}_rank": rank for label, rank in zip(labels, ranks, strict=True)}
    payload["reason"] = f"{rollout_n}-way ranking"
    return json.dumps(payload)


def _build_runtime_reward_loop_manager(
    monkeypatch,
    config: OmegaConf,
) -> tuple[RewardLoopManager, _FakeJudgeClient, _FakeAsyncJudgeClient]:
    output_text = _build_fake_judge_output(int(config.actor_rollout_ref.rollout.n))
    fake_client = _FakeJudgeClient(output_text)
    fake_async_client = _FakeAsyncJudgeClient(output_text)

    monkeypatch.setattr(reward_loop_module.ray, "nodes", lambda: [{"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 4}}])
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)
    monkeypatch.setattr(reward_loop_module.ray, "remote", lambda cls: _LocalRewardLoopRemoteClass(cls))
    monkeypatch.setattr(reward_loop_module, "copy_to_local", lambda path: path)
    monkeypatch.setattr(reward_loop_module, "hf_tokenizer", lambda *args, **kwargs: _TokenizerStub())
    monkeypatch.setattr(openai, "OpenAI", lambda api_key=None: fake_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda api_key=None: fake_async_client)
    monkeypatch.setattr(webgym_reward_fn_module, "_OPENAI_CLIENT", None)
    monkeypatch.setattr(webgym_reward_fn_module, "_ASYNC_OPENAI_CLIENT", None)

    manager = RewardLoopManager(config)
    return manager, fake_client, fake_async_client


def test_webgym_reward_extra_infos_roundtrip_through_real_ray_transport():
    import ray

    packed_non_tensor_batch, reward_extra_keys = webgym_reward_fn_module.pack_webgym_reward_extra_infos(
        [
            {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_llm_judge_used": False,
                "web_osgym_llm_judge_reason": "first",
            },
            {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.5,
                "web_osgym_llm_judge_rank": 2,
                "web_osgym_llm_judge_reason": "second",
            },
        ]
    )
    batch = DataProto(
        batch=TensorDict({"rm_scores": torch.zeros((2, 1), dtype=torch.float32)}, batch_size=[2]),
        non_tensor_batch=packed_non_tensor_batch,
        meta_info={"reward_extra_keys": reward_extra_keys},
    )

    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False, ignore_reinit_error=False)
    try:

        @ray.remote
        def _roundtrip(data: DataProto):
            from WebOSWorld.webgym_rl import reward_fn_webgym_rl as reward_module

            return [item.model_dump() for item in reward_module.extract_webgym_reward_extra_infos_from_batch(data)]

        restored = ray.get(_roundtrip.remote(batch))
    finally:
        ray.shutdown()

    expected = [
        webgym_reward_fn_module.validate_webgym_reward_extra_info(
            {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_llm_judge_used": False,
                "web_osgym_llm_judge_reason": "first",
            }
        ).model_dump(),
        webgym_reward_fn_module.validate_webgym_reward_extra_info(
            {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.5,
                "web_osgym_llm_judge_rank": 2,
                "web_osgym_llm_judge_reason": "second",
            }
        ).model_dump(),
    ]

    assert restored == expected


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
                "web_osgym_llm_judge_used": False,
                "web_osgym_llm_judge_score": 0.0,
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
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.75,
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
    captured_dump: dict[str, object] = {}

    def _capture_dump(**kwargs):
        captured_dump.update(kwargs)

    trainer._dump_generations = _capture_dump

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
    assert trainer.metrics["score/zero group"] == pytest.approx(0.0)
    assert trainer.metrics["score/llm judge"] == pytest.approx(0.75)
    assert trainer.metrics["score/format"] == pytest.approx(0.6)
    assert trainer.metrics["score/format_raw"] == pytest.approx(1.0)
    assert trainer.metrics["score/non_grounding_adjacency_ratio"] == pytest.approx(0.4)
    assert trainer.metrics["critic/score/mean"] == pytest.approx((shaped_score_a + shaped_score_b) / 2.0)

    batch.batch["token_level_scores"] = torch.zeros_like(batch.batch["token_level_scores"])

    trainer._fit_dump_data(batch)
    assert captured_dump["scores"] == pytest.approx([shaped_score_a, shaped_score_b])
    assert list(captured_dump["reward_extra_infos_dict"]["request_id"]) == ["req-1", "req-2"]
    assert list(captured_dump["reward_extra_infos_dict"]["uid"]) == ["uid-1", "uid-1"]
    assert list(captured_dump["reward_extra_infos_dict"]["index"]) == [0, 0]
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


def test_reward_loop_manager_disables_agent_reward_loop_when_group_level_judge_is_enabled():
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                        "reward_kwargs": {
                            "llm_judge_enable": True,
                            "llm_judge_only_zerogroup": True,
                    }
                },
            }
        }
    )
    manager.reward_loop_workers = [object()]

    assert manager.reward_loop_worker_handles is None


def test_reward_loop_manager_does_not_assign_fixed_actor_names(monkeypatch):
    option_calls: list[dict] = []

    class _RemoteBuilder:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def remote(self, config, reward_router_address):
            del config, reward_router_address
            option_calls.append(self.kwargs)
            return object()

    class _RewardLoopWorkersClassStub:
        def options(self, **kwargs):
            return _RemoteBuilder(kwargs)

    monkeypatch.setattr(
        reward_loop_module.ray,
        "nodes",
        lambda: [{"NodeID": "a" * 56, "Alive": True, "Resources": {"CPU": 4}}],
    )

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "num_workers": 2,
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {"reward_kwargs": {}},
            }
        }
    )
    manager.reward_router_address = None
    manager.reward_loop_workers_class = _RewardLoopWorkersClassStub()

    manager._init_reward_loop_workers()

    assert len(manager.reward_loop_workers) == 2
    assert all("name" not in kwargs for kwargs in option_calls)


def test_reward_loop_manager_only_triggers_judge_for_all_zero_uid_groups(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    call_flags: list[list[bool]] = []

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        batch_flags: list[bool] = []
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            group_active = bool(extra_info.get("web_osgym_llm_judge_group_active"))
            batch_flags.append(group_active)
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": False,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        call_flags.append(batch_flags)
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                    "reward_manager": {"name": "naive"},
                    "custom_reward_function": {
                        "reward_kwargs": {
                            "llm_judge_enable": True,
                            "llm_judge_only_zerogroup": True,
                        }
                    },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch)]
    manager.zero_group_compare_fn = lambda extra_infos, **kwargs: [
        {
            "reward_score": score,
            "reward_extra_info": {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": score,
                "web_osgym_llm_judge_rank": rank,
            },
        }
        for score, rank in [(1.0, 1), (1.0, 1), (0.5, 2), (0.0, 3)]
    ]

    batch = _make_reward_loop_batch(
        uid_to_env_scores={
            "uid-zero": [0.0, 0.0, 0.0, 0.0],
            "uid-mixed": [1.0, 0.0, 0.0, 0.0],
        },
        include_judge_standard=True,
    )
    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx([1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert call_flags == [[False] * 8]


def test_reward_loop_manager_skips_group_level_judge_when_training_group_is_incomplete(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    call_flags: list[list[bool]] = []

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        batch_flags: list[bool] = []
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            group_active = bool(extra_info.get("web_osgym_llm_judge_group_active"))
            batch_flags.append(group_active)
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": group_active,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        call_flags.append(batch_flags)
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                    "reward_manager": {"name": "naive"},
                    "custom_reward_function": {
                        "reward_kwargs": {
                            "llm_judge_enable": True,
                            "llm_judge_only_zerogroup": True,
                        }
                    },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch)]
    compare_input_validator = reward_loop_module.load_extern_object(
        module_path="/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py",
        object_name="validate_zero_group_compare_extra_infos",
    )

    def _compare_with_validation(extra_infos, **kwargs):
        del kwargs
        compare_input_validator(extra_infos)
        return [
            {
                "reward_score": 1.0,
                "reward_extra_info": {
                    "web_osgym_env_reward_score": 0.0,
                    "web_osgym_llm_judge_used": True,
                    "web_osgym_llm_judge_score": 1.0,
                    "web_osgym_llm_judge_rank": 1,
                },
            }
            for _ in extra_infos
        ]

    manager.zero_group_compare_fn = _compare_with_validation

    batch = _make_reward_loop_batch(
        uid_to_env_scores={"uid-short": [0.0, 0.0, 0.0]},
        validate=False,
        include_judge_standard=True,
    )
    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx([0.0, 0.0, 0.0])
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [False, False, False]
    assert call_flags == [[False, False, False]]


def test_reward_loop_manager_uses_val_n_for_validation_group_level_judge(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    call_flags: list[list[bool]] = []

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        batch_flags: list[bool] = []
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            group_active = bool(extra_info.get("web_osgym_llm_judge_group_active"))
            batch_flags.append(group_active)
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": False,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        call_flags.append(batch_flags)
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                    }
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch)]
    manager.zero_group_compare_fn = lambda extra_infos, **kwargs: [
        {
            "reward_score": score,
            "reward_extra_info": {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": score,
                "web_osgym_llm_judge_rank": rank,
            },
        }
        for score, rank in [(1.0, 1), (0.5, 2), (0.0, 3), (0.0, 3)]
    ]

    batch = _make_reward_loop_batch(
        uid_to_env_scores={"uid-val": [0.0, 0.0, 0.0, 0.0]},
        validate=True,
        include_judge_standard=True,
    )
    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [True, True, True, True]
    assert call_flags == [[False, False, False, False]]


def test_reward_loop_manager_handles_uneven_selected_judge_batch_with_multiple_workers(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": False,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                    }
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch) for _ in range(8)]
    manager.zero_group_compare_fn = lambda extra_infos, **kwargs: [
        {
            "reward_score": score,
            "reward_extra_info": {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": score,
                "web_osgym_llm_judge_rank": rank,
            },
        }
        for score, rank in [(1.0, 1), (1.0, 1), (0.5, 2), (0.0, 3)]
    ]

    batch = _make_reward_loop_batch(
        uid_to_env_scores={
            "uid-zero-a": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-b": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-c": [0.0, 0.0, 0.0, 0.0],
            "uid-mixed": [1.0, 0.0, 0.0, 0.0],
        },
        include_judge_standard=True,
    )
    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx(
        [1.0, 1.0, 0.5, 0.0] * 3 + [1.0, 0.0, 0.0, 0.0]
    )
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_reward_loop_manager_async_judge_respects_max_concurrency(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": False,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                        "llm_judge_max_concurrency": 2,
                    }
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch) for _ in range(3)]
    manager.zero_group_compare_fn = None

    active = 0
    max_active = 0

    async def _fake_compare_async(extra_infos, **kwargs):
        del kwargs
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return [
            {
                "reward_score": score,
                "reward_extra_info": {
                    "web_osgym_env_reward_score": 0.0,
                    "web_osgym_llm_judge_used": True,
                    "web_osgym_llm_judge_score": score,
                    "web_osgym_llm_judge_rank": rank,
                },
            }
            for score, rank in [(1.0, 1), (1.0, 1), (0.5, 2), (0.0, 3)]
        ]

    manager.zero_group_compare_async_fn = _fake_compare_async

    batch = _make_reward_loop_batch(
        uid_to_env_scores={
            "uid-zero-a": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-b": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-c": [0.0, 0.0, 0.0, 0.0],
        },
        include_judge_standard=True,
    )
    result = manager.compute_rm_score(batch)

    assert max_active == 2
    assert result.batch["rm_scores"].sum(dim=-1).tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0] * 3)
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [True] * 12


def test_load_zero_group_compare_async_fn_fails_closed_on_import_error(monkeypatch):
    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": "/tmp/missing_reward_fn.py",
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                    },
                }
            }
        }
    )

    def _raise_load_error(*args, **kwargs):
        raise ImportError("missing async compare fn")

    monkeypatch.setattr(reward_loop_module, "load_extern_object", _raise_load_error)

    with pytest.raises(RuntimeError, match="Failed to load compare_zero_group_webgym_rl_async"):
        reward_loop_module._load_zero_group_compare_async_fn(config)


def test_runtime_reward_path_uses_real_manager_worker_and_custom_reward_fn(monkeypatch, tmp_path):
    config = _build_runtime_reward_loop_config(num_workers=4, rollout_n=4, val_n=4)
    manager, fake_client, fake_async_client = _build_runtime_reward_loop_manager(monkeypatch, config)

    batch = _make_runtime_reward_batch(
        tmp_path,
        uid_to_env_scores={
            "uid-zero": [0.0, 0.0, 0.0, 0.0],
            "uid-mixed": [1.0, 0.0, 0.0, 0.0],
        },
    )

    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.reward_loop_manager = manager
    trainer.timing_raw = {"step": 1.0}

    batch = trainer._fit_compute_reward(batch)

    assert trainer.reward_tensor.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert trainer.reward_extra_infos_dict["web_osgym_llm_judge_used"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert len(fake_client.requests) == 0
    assert len(fake_async_client.requests) == 1
    assert fake_async_client.timeouts == [17.0]
    assert all(request["model"] == "gpt-5.4-mini" for request in fake_async_client.requests)


def test_runtime_reward_path_handles_uneven_selected_batch_with_real_manager(monkeypatch, tmp_path):
    config = _build_runtime_reward_loop_config(num_workers=8, rollout_n=4, val_n=4)
    manager, fake_client, fake_async_client = _build_runtime_reward_loop_manager(monkeypatch, config)

    batch = _make_runtime_reward_batch(
        tmp_path,
        uid_to_env_scores={
            "uid-zero-a": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-b": [0.0, 0.0, 0.0, 0.0],
            "uid-zero-c": [0.0, 0.0, 0.0, 0.0],
            "uid-mixed": [1.0, 0.0, 0.0, 0.0],
        },
    )

    result = manager.compute_rm_score(batch)

    assert result.batch["rm_scores"].sum(dim=-1).tolist() == pytest.approx(
        [1.0, 1.0, 0.5, 0.0] * 3 + [1.0, 0.0, 0.0, 0.0]
    )
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert len(fake_client.requests) == 0
    assert len(fake_async_client.requests) == 3
    assert fake_async_client.timeouts == [17.0] * 3


def test_runtime_reward_path_supports_six_way_real_manager(monkeypatch, tmp_path):
    config = _build_runtime_reward_loop_config(num_workers=6, rollout_n=6, val_n=6)
    manager, fake_client, fake_async_client = _build_runtime_reward_loop_manager(monkeypatch, config)

    batch = _make_runtime_reward_batch(
        tmp_path,
        uid_to_env_scores={
            "uid-zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "uid-mixed": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    result = manager.compute_rm_score(batch)

    assert result.batch["rm_scores"].sum(dim=-1).tolist() == pytest.approx(
        [1.0, 1.0, 0.75, 0.5, 0.25, 0.0] + [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert len(fake_client.requests) == 0
    assert len(fake_async_client.requests) == 1
    assert fake_async_client.timeouts == [17.0]


def test_reward_loop_manager_skips_group_level_judge_without_judge_standard(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    call_flags: list[list[bool]] = []

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        batch_flags: list[bool] = []
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            group_active = bool(extra_info.get("web_osgym_llm_judge_group_active"))
            batch_flags.append(group_active)
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": group_active,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        call_flags.append(batch_flags)
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                    }
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch)]
    compare_input_validator = reward_loop_module.load_extern_object(
        module_path="/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py",
        object_name="validate_zero_group_compare_extra_infos",
    )

    def _compare_with_validation(extra_infos, **kwargs):
        del kwargs
        compare_input_validator(extra_infos)
        return [
            {
                "reward_score": 1.0,
                "reward_extra_info": {
                    "web_osgym_env_reward_score": 0.0,
                    "web_osgym_llm_judge_used": True,
                    "web_osgym_llm_judge_score": 1.0,
                    "web_osgym_llm_judge_rank": 1,
                },
            }
            for _ in extra_infos
        ]

    manager.zero_group_compare_fn = _compare_with_validation

    batch = _make_reward_loop_batch(uid_to_env_scores={"uid-zero": [0.0, 0.0, 0.0, 0.0]}, include_judge_standard=False)
    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [False, False, False, False]
    assert call_flags == [[False, False, False, False]]


def test_reward_loop_manager_skips_group_level_judge_with_invalid_compare_input(monkeypatch):
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda value: value)

    call_flags: list[list[bool]] = []

    def _fake_compute_score_batch(data: DataProto) -> list[dict]:
        batch_flags: list[bool] = []
        outputs: list[dict] = []
        for item in data:
            extra_info = item.non_tensor_batch.get("extra_info", {})
            env_reward = float(extra_info.get("web_osgym_env_reward_score", 0.0))
            group_active = bool(extra_info.get("web_osgym_llm_judge_group_active"))
            batch_flags.append(group_active)
            outputs.append(
                {
                    "reward_score": env_reward,
                    "reward_extra_info": {
                        "web_osgym_env_reward_score": env_reward,
                        "web_osgym_llm_judge_used": group_active,
                        "web_osgym_llm_judge_score": 0.0,
                    },
                }
            )
        call_flags.append(batch_flags)
        return outputs

    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.config = OmegaConf.create(
        {
            "reward": {
                "reward_model": {"enable": False, "enable_resource_pool": False},
                "reward_manager": {"name": "naive"},
                "custom_reward_function": {
                    "path": "/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py",
                    "reward_kwargs": {
                        "llm_judge_enable": True,
                        "llm_judge_only_zerogroup": True,
                    },
                },
            },
            "actor_rollout_ref": {"rollout": {"n": 4, "val_kwargs": {"n": 4}}},
        }
    )
    manager.reward_model_manager = None
    manager.reward_loop_workers = [_RewardBatchWorkerHandle(_fake_compute_score_batch)]
    compare_input_validator = reward_loop_module.load_extern_object(
        module_path="/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py",
        object_name="validate_zero_group_compare_extra_infos",
    )

    def _compare_with_validation(extra_infos, **kwargs):
        del kwargs
        compare_input_validator(extra_infos)
        return [
            {
                "reward_score": 1.0,
                "reward_extra_info": {
                    "web_osgym_env_reward_score": 0.0,
                    "web_osgym_llm_judge_used": True,
                    "web_osgym_llm_judge_score": 1.0,
                    "web_osgym_llm_judge_rank": 1,
                },
            }
            for _ in extra_infos
        ]

    manager.zero_group_compare_fn = _compare_with_validation
    batch = _make_reward_loop_batch(uid_to_env_scores={"uid-zero": [0.0, 0.0, 0.0, 0.0]}, include_judge_standard=True)
    extra_infos = batch.non_tensor_batch["extra_info"].copy()
    extra_infos[1] = dict(extra_infos[1])
    extra_infos[1]["web_osgym_trajectory_dir"] = ""
    batch.non_tensor_batch["extra_info"] = extra_infos

    result = manager.compute_rm_score(batch)

    reward_sums = result.batch["rm_scores"].sum(dim=-1).tolist()
    assert reward_sums == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [False, False, False, False]
    assert call_flags == [[False, False, False, False]]


def test_separate_trainer_computes_batch_reward_for_custom_reward_when_streaming_reward_is_disabled():
    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.reward_loop_manager = _BatchRewardLoopManagerStub(scores=[0.25, 0.75], env_scores=[0.0, 1.0])
    trainer.timing_raw = {"step": 1.0}

    batch = _make_reward_loop_batch(uid_to_env_scores={"uid-a": [0.0], "uid-b": [1.0]}, validate=False)

    batch = trainer._fit_compute_reward(batch)

    assert trainer.reward_tensor.sum(dim=-1).tolist() == pytest.approx([0.25, 0.75])
    assert trainer.reward_extra_infos_dict["web_osgym_env_reward_score"].tolist() == [0.0, 1.0]


def test_fully_async_trainer_initializes_reward_loop_even_without_trainer_validate(monkeypatch):
    trainer_cls = FullyAsyncTrainer.__ray_actor_class__
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.config = OmegaConf.create({"async_training": {"use_trainer_do_validate": False}})

    called: dict[str, bool] = {"value": False}

    def _fake_init_reward_loop(self):
        called["value"] = True
        self.reward_loop_manager = type("_RewardLoopManagerStub", (), {"reward_loop_worker_handles": None})()

    monkeypatch.setattr(SeparateRayPPOTrainer, "_init_reward_loop", _fake_init_reward_loop)

    trainer._init_reward_loop()

    assert called["value"] is True
    assert hasattr(trainer, "reward_loop_manager")


def test_separate_trainer_skips_llm_judge_metric_when_no_judge_sample_exists():
    trainer = SeparateRayPPOTrainer.__new__(SeparateRayPPOTrainer)
    trainer.use_rm = False
    trainer.use_critic = False
    trainer.reward_loop_manager = _BatchRewardLoopManagerStub(scores=[0.25, 0.75], env_scores=[0.0, 1.0])
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

    batch = _make_reward_loop_batch(uid_to_env_scores={"uid-a": [0.0], "uid-b": [1.0]}, validate=False)
    batch = trainer._fit_compute_reward(batch)
    batch = trainer._fit_compute_advantage(batch)
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
    trainer._fit_collect_metrics(batch)

    assert trainer.metrics["score/zero group"] == pytest.approx(0.5)
    assert "score/llm judge" not in trainer.metrics
