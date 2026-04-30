from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

if "sglang" not in sys.modules:
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    entrypoints_module = types.ModuleType("sglang.srt.entrypoints")
    engine_module = types.ModuleType("sglang.srt.entrypoints.engine")
    http_server_module = types.ModuleType("sglang.srt.entrypoints.http_server")
    managers_module = types.ModuleType("sglang.srt.managers")
    io_struct_module = types.ModuleType("sglang.srt.managers.io_struct")
    tokenizer_manager_module = types.ModuleType("sglang.srt.managers.tokenizer_manager")

    class _StubGenerateReqInput:
        def __init__(self, **kwargs):
            self.rid = kwargs.get("rid")
            self.input_ids = kwargs.get("input_ids")
            self.text = kwargs.get("text")
            self.sampling_params = kwargs.get("sampling_params")
            self.return_logprob = kwargs.get("return_logprob")
            self.image_data = kwargs.get("image_data")
            self.logprob_start_len = kwargs.get("logprob_start_len")
            self.top_logprobs_num = kwargs.get("top_logprobs_num")
            self.lora_path = None

    class _StubReqInput:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _StubServerStatus:
        pass

    http_server_module.ServerArgs = object
    http_server_module._GlobalState = object
    http_server_module.app = object()
    http_server_module.set_global_state = lambda *args, **kwargs: None
    io_struct_module.ContinueGenerationReqInput = _StubReqInput
    io_struct_module.GenerateReqInput = _StubGenerateReqInput
    io_struct_module.PauseGenerationReqInput = _StubReqInput
    io_struct_module.ReleaseMemoryOccupationReqInput = _StubReqInput
    io_struct_module.ResumeMemoryOccupationReqInput = _StubReqInput
    tokenizer_manager_module.ServerStatus = _StubServerStatus

    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = srt_module
    sys.modules["sglang.srt.entrypoints"] = entrypoints_module
    sys.modules["sglang.srt.entrypoints.engine"] = engine_module
    sys.modules["sglang.srt.entrypoints.http_server"] = http_server_module
    sys.modules["sglang.srt.managers"] = managers_module
    sys.modules["sglang.srt.managers.io_struct"] = io_struct_module
    sys.modules["sglang.srt.managers.tokenizer_manager"] = tokenizer_manager_module

if "verl.workers.rollout.sglang_rollout.sglang_rollout" not in sys.modules:
    sglang_rollout_module = types.ModuleType("verl.workers.rollout.sglang_rollout.sglang_rollout")
    sglang_rollout_module._set_envs_and_config = lambda *args, **kwargs: None
    sys.modules["verl.workers.rollout.sglang_rollout.sglang_rollout"] = sglang_rollout_module

if "verl.workers.rollout.sglang_rollout.utils" not in sys.modules:
    sglang_utils_module = types.ModuleType("verl.workers.rollout.sglang_rollout.utils")
    sglang_utils_module.SGLANG_LORA_NAME = "stub-lora"
    sys.modules["verl.workers.rollout.sglang_rollout.utils"] = sglang_utils_module

if "verl.workers.rollout.utils" not in sys.modules:
    rollout_utils_module = types.ModuleType("verl.workers.rollout.utils")
    rollout_utils_module.get_max_position_embeddings = lambda *args, **kwargs: 4096
    rollout_utils_module.run_uvicorn = lambda *args, **kwargs: None
    sys.modules["verl.workers.rollout.utils"] = rollout_utils_module

from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer


class _FakeTokenizerManager:
    def __init__(self):
        self.requests = []

    def generate_request(self, generate_request, _):
        async def _iterator():
            self.requests.append(generate_request)
            yield {
                "output_ids": [42],
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "output_token_logprobs": [],
                },
            }

        return _iterator()


def _build_server() -> SGLangHttpServer:
    server = SGLangHttpServer.__new__(SGLangHttpServer)
    server.config = SimpleNamespace(
        max_model_len=128,
        response_length=64,
        prompt_length=64,
        enable_rollout_routing_replay=False,
    )
    server.model_config = SimpleNamespace(lora_rank=0)
    server.tokenizer_manager = _FakeTokenizerManager()
    server.global_steps = 0
    return server


@pytest.mark.asyncio
async def test_sglang_generate_uses_text_request_when_prompt_text_is_provided():
    server = _build_server()

    output = await server.generate(
        prompt_ids=[1, 2, 3],
        prompt_text="<|im_start|>user\n<image><|im_end|>\n<|im_start|>assistant\n",
        sampling_params={"max_tokens": 1},
        request_id="req-native-text",
        image_data=["image-1"],
    )

    request = server.tokenizer_manager.requests[0]
    assert output.token_ids == [42]
    assert request.text == "<|im_start|>user\n<image><|im_end|>\n<|im_start|>assistant\n"
    assert request.input_ids is None
    assert request.image_data == ["image-1"]
    assert request.sampling_params["max_new_tokens"] == 1


@pytest.mark.asyncio
async def test_sglang_generate_keeps_input_ids_request_without_prompt_text():
    server = _build_server()

    output = await server.generate(
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 1},
        request_id="req-input-ids",
        image_data=["image-1"],
    )

    request = server.tokenizer_manager.requests[0]
    assert output.token_ids == [42]
    assert request.text is None
    assert request.input_ids == [1, 2, 3]
    assert request.image_data == ["image-1"]


@pytest.mark.asyncio
async def test_sglang_generate_rejects_prompt_text_for_prompt_logprobs():
    server = _build_server()

    with pytest.raises(ValueError, match="prompt_text is not supported for prompt_logprobs"):
        await server.generate(
            prompt_ids=[1, 2, 3, 4],
            prompt_text="<|im_start|>user\n<image><|im_end|>\n<|im_start|>assistant\n",
            sampling_params={"max_tokens": 0, "prompt_logprobs": 2, "prompt_logprobs_start_len": 2},
            request_id="req-text-logprobs",
            image_data=["image-1"],
        )
