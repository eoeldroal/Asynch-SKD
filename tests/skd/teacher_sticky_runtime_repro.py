"""Runtime-style teacher sticky carryover repro.

This is intentionally heavier than the CPU-only unit tests. It exercises:

DataProto
  -> AsyncSkdAgentLoopWorker.generate_skd_until_boundary(...)
  -> real SkdAgentLoop
  -> real AsyncTeacherLLMServerManager
  -> real GlobalRequestLoadBalancer (Ray actor)
  -> Ray teacher server actors
  -> partial export
  -> AsyncSkdAgentLoopWorker.generate_skd_from_partial_to_completion(...)

The goal is to verify that teacher server assignment survives fresh->partial
export and partial->resume, even when the resumed trajectory receives a new
outer request_id.
"""

from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from typing import Any

import numpy as np
import ray

import verl.experimental.agent_loop.skd_agent_loop as skd_agent_loop_module
import verl.experimental.async_skd.worker as async_skd_worker_module
import verl.experimental.teacher_loop.teacher_manager as teacher_manager_module
from verl.experimental.agent_loop.agent_loop import GlobalRequestLoadBalancer
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.protocol import DataProto
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.config.rollout import RolloutConfig
from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _FakeTokenizer
from tests.skd.test_skd_logic import make_skd_loop


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def make_single_batch_with_teacher_assignment() -> DataProto:
    raw_prompt = [{"role": "user", "content": "hi"}]
    return DataProto.from_dict(
        non_tensors={
            "raw_prompt": _object_array([raw_prompt]),
            "index": np.array([7], dtype=object),
            "agent_name": np.array(["skd_agent"], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
            "data_source": np.array(["default"], dtype=object),
            "teacher_replica_id": np.array(["teacher-server-1"], dtype=object),
        },
        meta_info={"global_steps": 12, "validate": False},
    )


@ray.remote
class _TeacherServerActor:
    def __init__(self, server_id: str):
        self.server_id = server_id
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        **kwargs: Any,
    ) -> Any:
        del image_data, video_data, kwargs
        start = int(sampling_params.get("prompt_logprobs_start_len", 0))
        topk = int(sampling_params.get("prompt_logprobs", 0))
        if start > 0:
            suffix = list(prompt_ids[start + 1 :])
        else:
            suffix = list(prompt_ids)
        rows = [[token_id] + [0] * max(topk - 1, 0) for token_id in suffix]
        logprobs = [[-1.0] * max(topk, 1) for _ in suffix]
        self.calls.append(
            {
                "server_id": self.server_id,
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
                "suffix": suffix,
            }
        )
        return SimpleNamespace(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
            stop_reason="completed",
            extra_fields={
                "prompt_ids": rows,
                "prompt_logprobs": logprobs,
                "server_id": self.server_id,
            },
        )

    def get_calls(self) -> list[dict[str, Any]]:
        return list(self.calls)


class _RuntimeBoundaryWorker(AsyncSkdAgentLoopWorker):
    reward_loop_worker_handles = None
    distillation_enabled = False
    stream_teacher_with_rollout = False
    processor = None

    def __init__(self, teacher_server_manager: AsyncTeacherLLMServerManager):
        self.rollout_config = SimpleNamespace(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            calculate_log_probs=False,
            prompt_length=4,
            response_length=16,
            val_kwargs=SimpleNamespace(temperature=0.0, top_p=1.0, top_k=-1),
            agent=SimpleNamespace(default_agent_loop="skd_agent"),
        )
        self.tokenizer = _FakeTokenizer()
        self.tokenizer.pad_token_id = 0
        loop = make_skd_loop(
            student_chunks=[[10], [11, 99999]],
            teacher_topk_by_call=[],
            response_length=16,
        )
        loop.teacher_server_manager = teacher_server_manager

        async def fake_apply_chat_template(messages, tools=None, images=None, videos=None, **kwargs):
            del messages, tools, images, videos, kwargs
            return [1, 2, 3]

        loop.apply_chat_template = fake_apply_chat_template
        self.loop: SkdAgentLoop = loop

    def _get_or_create_agent_loop(self, agent_name: str):
        assert agent_name == "skd_agent"
        return self.loop


def _distillation_config() -> SimpleNamespace:
    inference = RolloutConfig(
        name="sglang",
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    teacher = DistillationTeacherModelConfig(
        key="default",
        model_path="/tmp/fake-teacher",
        inference=inference,
        num_replicas=2,
    )
    distillation = DistillationConfig(
        teacher_key="data_source",
        teacher_models={"default": teacher},
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk",
            topk=4,
            use_policy_gradient=False,
        ),
    )
    return SimpleNamespace(distillation=distillation)


async def run_repro() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, local_mode=False)

    server0 = _TeacherServerActor.remote("teacher-server-0")
    server1 = _TeacherServerActor.remote("teacher-server-1")
    load_balancer = GlobalRequestLoadBalancer.remote(
        server_actor_ids=["teacher-server-0", "teacher-server-1"]
    )
    teacher_manager = AsyncTeacherLLMServerManager(
        config=_distillation_config(),
        servers={"default": [("teacher-server-0", server0), ("teacher-server-1", server1)]},
        load_balancer_handle={"default": load_balancer},
    )

    worker = _RuntimeBoundaryWorker(teacher_manager)
    batch = make_single_batch_with_teacher_assignment()

    skd_agent_loop_module._ASYNC_SKD_TRACE = 1
    async_skd_worker_module._ASYNC_SKD_TRACE = 1
    teacher_manager_module._ASYNC_SKD_TRACE = 1

    stdout = StringIO()
    with redirect_stdout(stdout):
        partial_sample = await worker.generate_skd_until_boundary(
            batch,
            sample_id="runtime-fresh-sample",
            logical_step=12,
            source_type="lookahead",
        )
        partial = partial_sample.require_partial()
        assert partial.extra_fields["teacher_replica_id"] == "teacher-server-1"
        assert partial.extra_fields["teacher_routing_key"] == "default"

        server0_calls = ray.get(server0.get_calls.remote())
        server1_calls = ray.get(server1.get_calls.remote())
        assert len(server0_calls) == 0, server0_calls
        assert len(server1_calls) == 1, server1_calls

        completed = await worker.generate_skd_from_partial_to_completion(partial)
        assert completed.kind == "completed"

        server0_calls = ray.get(server0.get_calls.remote())
        server1_calls = ray.get(server1.get_calls.remote())
        assert len(server0_calls) == 0, server0_calls
        assert len(server1_calls) == 2, server1_calls

        first_prompt = server1_calls[0]["prompt_ids"]
        second_prompt = server1_calls[1]["prompt_ids"]
        assert first_prompt == [1, 2, 3, 10], first_prompt
        assert second_prompt == [1, 2, 3, 10, 11, 99999], second_prompt

    trace_output = stdout.getvalue()
    required_stages = [
        "stage=worker.generate_until_boundary.entry",
        "stage=loop.init_boundary_agent_data",
        "stage=loop.teacher_bind_attempt",
        "stage=teacher.bind_sticky_request",
        "stage=loop.export_partial_state",
        "stage=worker.generate_until_boundary.partial",
        "stage=worker.generate_from_partial.entry",
        "stage=loop.restore_partial_state",
        "stage=loop.resume_request_rebind",
    ]
    for stage in required_stages:
        assert stage in trace_output, trace_output
    assert "teacher_replica_id='teacher-server-1'" in trace_output, trace_output
    assert "teacher_routing_key='default'" in trace_output, trace_output

    print("runtime teacher sticky repro passed")


def main() -> None:
    try:
        asyncio.run(run_repro())
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
