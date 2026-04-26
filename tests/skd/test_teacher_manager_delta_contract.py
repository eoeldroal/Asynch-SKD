from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from verl.experimental.teacher_loop import teacher_manager
from verl.workers.rollout.replica import TokenOutput


class FakeRemoteMethod:
    def __init__(self, result: Any = None):
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def remote(self, **kwargs: Any) -> "_AwaitableValue":
        self.calls.append(dict(kwargs))
        return _AwaitableValue(self.result)


class _AwaitableValue:
    def __init__(self, value: Any):
        self.value = value

    def __await__(self):
        if False:
            yield None
        return self.value


class FakeLoadBalancer:
    def __init__(self, server_id: str = "server-0"):
        self.acquire_server = FakeRemoteMethod(server_id)
        self.release_server = FakeRemoteMethod(None)


class FakeRayServer:
    def __init__(self, output: TokenOutput):
        self.output = output
        self.generate = FakeRemoteMethod(output)


def _fake_output(rows: list[list[int]], logprobs: list[list[float]]) -> TokenOutput:
    return TokenOutput(
        token_ids=[],
        log_probs=[],
        num_preempted=0,
        stop_reason="completed",
        extra_fields={"prompt_ids": rows, "prompt_logprobs": logprobs},
    )


def _distillation_config(inference_name: str = "sglang") -> SimpleNamespace:
    return SimpleNamespace(
        teacher_key="data_source",
        teacher_models={
            "default": SimpleNamespace(
                inference=SimpleNamespace(name=inference_name, temperature=1.0),
            )
        },
        distillation_loss=SimpleNamespace(
            topk=2,
            loss_settings=SimpleNamespace(use_topk=True),
        ),
    )


def _make_manager(monkeypatch: pytest.MonkeyPatch, output: TokenOutput, inference_name: str = "sglang"):
    monkeypatch.setattr(teacher_manager, "omega_conf_to_dataclass", lambda distillation: distillation)

    config = SimpleNamespace(distillation=_distillation_config(inference_name))
    server = FakeRayServer(output)
    load_balancer = FakeLoadBalancer()
    manager = teacher_manager.AsyncTeacherLLMServerManager(
        config=config,
        servers={"default": [("server-0", server)]},
        load_balancer_handle={"default": load_balancer},
    )
    return manager, server, load_balancer


def test_compute_teacher_logprobs_single_accepts_request_id_and_delta_start_len_for_sglang(monkeypatch):
    expected_ids = [[41, 42], [51, 52]]
    expected_logprobs = [[-0.1, -0.2], [-0.3, -0.4]]
    manager, fake_server, fake_load_balancer = _make_manager(monkeypatch, _fake_output(expected_ids, expected_logprobs))

    teacher_ids, teacher_logprobs = asyncio.run(
        manager.compute_teacher_logprobs_single(
            sequence_ids=[1, 2, 3, 4, 5],
            request_id="req-delta",
            logprob_start_len=2,
            multi_modal_data={"images": ["image"], "videos": ["video"]},
        )
    )

    assert teacher_ids.shape[0] == 2
    assert teacher_logprobs.shape[0] == 2
    assert torch.equal(teacher_ids, torch.tensor(expected_ids, dtype=torch.int32))
    assert torch.equal(teacher_logprobs, torch.tensor(expected_logprobs))
    assert fake_load_balancer.acquire_server.calls == [{"request_id": "req-delta"}]
    assert fake_load_balancer.release_server.calls == [{"server_id": "server-0"}]
    assert len(fake_server.generate.calls) == 1
    fake_generate_call = fake_server.generate.calls[0]
    assert isinstance(fake_generate_call.pop("request_id"), str)
    assert fake_generate_call == {
        "prompt_ids": [1, 2, 3, 4, 5],
        "sampling_params": {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 2,
            "prompt_logprobs_start_len": 2,
        },
        "image_data": ["image"],
        "video_data": ["video"],
    }


def test_compute_teacher_logprobs_single_rejects_delta_mode_for_non_sglang_teacher(monkeypatch):
    manager, _, _ = _make_manager(monkeypatch, _fake_output([[1], [2]], [[-0.1], [-0.2]]), inference_name="vllm")

    with pytest.raises(ValueError, match="requires SGLang teacher inference"):
        asyncio.run(
            manager.compute_teacher_logprobs_single(
                sequence_ids=[1, 2, 3, 4, 5],
                request_id="req-delta",
                logprob_start_len=2,
            )
        )


def test_compute_teacher_logprobs_single_rejects_wrong_delta_suffix_length(monkeypatch):
    manager, _, _ = _make_manager(monkeypatch, _fake_output([[41, 42]], [[-0.1, -0.2]]))

    with pytest.raises(ValueError, match="Unexpected teacher logprob length"):
        asyncio.run(
            manager.compute_teacher_logprobs_single(
                sequence_ids=[1, 2, 3, 4, 5],
                request_id="req-delta",
                logprob_start_len=2,
            )
        )
