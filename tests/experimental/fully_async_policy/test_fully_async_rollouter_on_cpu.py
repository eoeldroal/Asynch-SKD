from types import SimpleNamespace

import httpx
import numpy as np
from omegaconf import OmegaConf
import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.detach_utils import prepare_single_generation_data
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.protocol import DataProto
from verl.utils.rollout_trace import RolloutTraceConfig

from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _FakeTokenizer

_RollouterImpl = FullyAsyncRollouter.__ray_actor_class__


class _FakeManager:
    def __init__(self, exc=None, ret=None):
        self.exc = exc
        self.ret = ret

    async def generate_sequences_single(self, full_batch):
        del full_batch
        if self.exc is not None:
            raise self.exc
        return self.ret


class _FakeMessageQueueClient:
    def __init__(self):
        self.samples = []

    async def put_sample(self, sample):
        self.samples.append(sample)
        return True


class _FakeWrappedTimeout(Exception):
    def as_instanceof_cause(self):
        return httpx.ReadTimeout("wrapped timeout")


def _minimal_output() -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[201, 202],
        response_mask=[1, 1],
        response_logprobs=[-0.1, -0.2],
        multi_modal_data=None,
        reward_score=0.0,
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


class _TimeoutWorker(AgentLoopWorker):
    def __init__(self):
        self.rollout_config = OmegaConf.create(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "calculate_log_probs": True,
                "prompt_length": 16,
                "response_length": 64,
                "val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                "agent": {"default_agent_loop": "web_tool_agent"},
            }
        )
        self.tokenizer = _FakeTokenizer()
        self.raw_calls = []

    async def _run_raw_agent_loop(
        self,
        sampling_params: dict,
        trajectory: dict,
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> AgentLoopOutput:
        self.raw_calls.append(
            {
                "sampling_params": sampling_params,
                "trajectory": dict(trajectory),
                "agent_name": agent_name,
                "trace": trace,
                "kwargs": dict(kwargs),
            }
        )
        if trajectory["rollout_n"] == 2:
            raise httpx.ReadTimeout("replica 2 timed out")
        return _minimal_output()


class _ManagerViaWorker:
    def __init__(self, worker):
        self.worker = worker

    async def generate_sequences_single(self, full_batch):
        return await self.worker.generate_sequences(full_batch)


@pytest.mark.asyncio
async def test_process_single_sample_streaming_drops_timeout_group_after_real_repeat_and_gather_path(capsys):
    RolloutTraceConfig.reset()
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 6,
                    "multi_turn": {"enable": True},
                }
            }
        }
    )
    full_batch = prepare_single_generation_data(
        {
            "raw_prompt": np.array([[{"role": "user", "content": "task"}]], dtype=object),
            "agent_name": np.array(["web_tool_agent"], dtype=object),
            "uid": np.array(["sample-a"], dtype=object),
            "index": np.array([7], dtype=object),
            "task_id": np.array(["task-123"], dtype=object),
        },
        config,
    )
    full_batch.meta_info["global_steps"] = 12
    full_batch.meta_info["validate"] = False

    worker = _TimeoutWorker()
    rollouter = object.__new__(_RollouterImpl)
    rollouter.async_rollout_manager = _ManagerViaWorker(worker)
    rollouter.message_queue_client = _FakeMessageQueueClient()
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0
    rollouter.timeout_dropped_samples = 0
    rollouter.processed_sample_count = 0
    sample = RolloutSample(
        full_batch=full_batch,
        sample_id="sample_0",
        epoch=0,
        rollout_status={},
    )

    await rollouter._process_single_sample_streaming(sample)

    captured = capsys.readouterr()
    assert "Dropped timeout sample group" in captured.out
    assert "sample_0" in captured.out
    assert "task-123" in captured.out
    assert "index=7" in captured.out
    assert len(worker.raw_calls) == 6
    assert sorted(call["trajectory"]["rollout_n"] for call in worker.raw_calls) == [0, 1, 2, 3, 4, 5]
    assert rollouter.timeout_dropped_samples == 1
    assert rollouter.processed_sample_count == 1
    assert rollouter.total_generated_samples == 0
    assert rollouter.message_queue_client.samples == []


@pytest.mark.asyncio
async def test_process_single_sample_streaming_reraises_non_timeout_failure():
    rollouter = object.__new__(_RollouterImpl)
    rollouter.async_rollout_manager = _FakeManager(exc=RuntimeError("boom"))
    rollouter.message_queue_client = _FakeMessageQueueClient()
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0
    rollouter.timeout_dropped_samples = 0
    rollouter.processed_sample_count = 0
    sample = RolloutSample(full_batch=SimpleNamespace(non_tensor_batch={}), sample_id="sample_1", epoch=0, rollout_status={})

    with pytest.raises(RuntimeError, match="boom"):
        await rollouter._process_single_sample_streaming(sample)

    assert rollouter.timeout_dropped_samples == 0
    assert rollouter.processed_sample_count == 0
    assert rollouter.message_queue_client.samples == []


@pytest.mark.asyncio
async def test_process_single_sample_streaming_drops_wrapped_timeout_failure():
    rollouter = object.__new__(_RollouterImpl)
    rollouter.async_rollout_manager = _FakeManager(exc=_FakeWrappedTimeout())
    rollouter.message_queue_client = _FakeMessageQueueClient()
    rollouter.total_generated_samples = 0
    rollouter.dropped_stale_samples = 0
    rollouter.timeout_dropped_samples = 0
    rollouter.processed_sample_count = 0
    sample = RolloutSample(
        full_batch=SimpleNamespace(non_tensor_batch={"task_id": ["task-xyz"], "index": [3]}),
        sample_id="sample_wrapped",
        epoch=0,
        rollout_status={},
    )

    await rollouter._process_single_sample_streaming(sample)

    assert rollouter.timeout_dropped_samples == 1
    assert rollouter.processed_sample_count == 1
    assert rollouter.message_queue_client.samples == []
