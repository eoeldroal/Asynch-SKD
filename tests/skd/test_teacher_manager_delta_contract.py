from __future__ import annotations

import asyncio
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest
import torch


class _PlaceholderConfig:
    pass


class _FakeAsyncLLMServerManager:
    def __init__(self, config: Any, servers: list[tuple[str, Any]], load_balancer_handle: Any):
        self.config = config
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle = dict(servers)
        teacher_models = getattr(getattr(config, "distillation", None), "teacher_models", {})
        first_teacher = next(iter(teacher_models.values()), None)
        inference = getattr(first_teacher, "inference", None)
        self._temperature = getattr(inference, "temperature", None)

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
    ) -> Any:
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        try:
            if self._temperature is not None:
                sampling_params = dict(sampling_params)
                sampling_params.setdefault("temperature", self._temperature)
            handle = self._server_id_to_handle[server_id]
            return await handle.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )
        finally:
            self._load_balancer.release_server.remote(server_id=server_id)


def _install_import_stubs() -> None:
    import types

    ray_module = types.ModuleType("ray")
    ray_module.remote = lambda obj=None, **kwargs: obj if obj is not None else (lambda cls: cls)
    ray_module.get = lambda value: value
    ray_module.actor = types.SimpleNamespace(ActorHandle=object)
    sys.modules.setdefault("ray", ray_module)

    verl_module = sys.modules.setdefault("verl", types.ModuleType("verl"))
    verl_module.__path__ = []
    experimental_module = sys.modules.setdefault("verl.experimental", types.ModuleType("verl.experimental"))
    experimental_module.__path__ = []
    agent_loop_module = types.ModuleType("verl.experimental.agent_loop")
    agent_loop_module.AsyncLLMServerManager = _FakeAsyncLLMServerManager
    sys.modules["verl.experimental.agent_loop"] = agent_loop_module

    utils_module = sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
    utils_module.__path__ = []
    config_module = types.ModuleType("verl.utils.config")
    config_module.omega_conf_to_dataclass = lambda value: value
    sys.modules["verl.utils.config"] = config_module

    workers_module = sys.modules.setdefault("verl.workers", types.ModuleType("verl.workers"))
    workers_module.__path__ = []
    config_types_module = types.ModuleType("verl.workers.config")
    config_types_module.DistillationConfig = _PlaceholderConfig
    config_types_module.DistillationLossConfig = _PlaceholderConfig
    config_types_module.DistillationTeacherModelConfig = _PlaceholderConfig
    sys.modules["verl.workers.config"] = config_types_module


def _load_teacher_manager():
    _install_import_stubs()
    module_path = Path(__file__).resolve().parents[2] / "verl" / "experimental" / "teacher_loop" / "teacher_manager.py"
    spec = importlib.util.spec_from_file_location("teacher_manager_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


teacher_manager = _load_teacher_manager()


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
        self.bind_request_to_server = FakeRemoteMethod(None)
        self.release_request_binding = FakeRemoteMethod(None)


class FakeRayServer:
    def __init__(self, output: Any):
        self.output = output
        self.generate = FakeRemoteMethod(output)


def _fake_output(rows: list[list[int]], logprobs: list[list[float]]) -> Any:
    return SimpleNamespace(
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


def test_bind_sticky_request_targets_the_resolved_teacher_load_balancer(monkeypatch):
    manager, _, load_balancer = _make_manager(monkeypatch, _fake_output([[1], [2]], [[-0.1], [-0.2]]))

    asyncio.run(
        manager.bind_sticky_request(
            routing_key="default",
            request_id="carry-1",
            server_id="server-0",
        )
    )

    assert load_balancer.bind_request_to_server.calls == [
        {"request_id": "carry-1", "server_id": "server-0"}
    ]


def test_release_sticky_session_clears_the_bound_request(monkeypatch):
    manager, _, load_balancer = _make_manager(monkeypatch, _fake_output([[1], [2]], [[-0.1], [-0.2]]))

    asyncio.run(
        manager.bind_sticky_request(
            routing_key="default",
            request_id="carry-1",
            server_id="server-0",
        )
    )
    asyncio.run(manager.release_sticky_session("carry-1", routing_key="default"))

    assert load_balancer.release_request_binding.calls == [{"request_id": "carry-1"}]
