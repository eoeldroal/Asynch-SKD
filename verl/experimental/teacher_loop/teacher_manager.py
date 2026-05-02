# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import os
import time
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)

_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 1


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, list[tuple[str, ray.actor.ActorHandle]]],
        load_balancer_handle: dict[str, ray.actor.ActorHandle],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(servers.keys()) != expected:
            raise ValueError(f"server keys {sorted(servers)} do not match teacher routing keys {sorted(expected)}.")
        if set(load_balancer_handle.keys()) != expected:
            raise ValueError(
                f"load_balancer_handle keys {sorted(load_balancer_handle)} do not match "
                f"teacher routing keys {sorted(expected)}."
            )

        self.server_managers: dict[str, AsyncLLMServerManager] = {
            key: AsyncLLMServerManager(
                config=config,
                servers=servers[key],
                load_balancer_handle=load_balancer_handle[key],
            )
            for key in self.teacher_model_configs
        }

    def server_ids_for_routing_key(self, routing_key: Optional[str] = None) -> list[str]:
        teacher_key = self._resolve_teacher_key(routing_key)
        return list(self.server_managers[teacher_key]._server_id_to_handle.keys())

    def server_ids_by_routing_key(self) -> dict[str, list[str]]:
        return {
            teacher_key: list(server_manager._server_id_to_handle.keys())
            for teacher_key, server_manager in self.server_managers.items()
        }

    def max_model_len_for_routing_key(self, routing_key: Optional[str] = None) -> Optional[int]:
        teacher_key = self._resolve_teacher_key(routing_key)
        return self.teacher_model_configs[teacher_key].inference.max_model_len

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def bind_sticky_request(self, *, routing_key: Optional[str] = None, request_id: str, server_id: str) -> None:
        teacher_key = self._resolve_teacher_key(routing_key)
        _trace_async_skd(
            "teacher.bind_sticky_request",
            teacher_key=teacher_key,
            routing_key=routing_key,
            request_id=request_id,
            server_id=server_id,
        )
        await self.server_managers[teacher_key]._load_balancer.bind_request_to_server.remote(
            request_id=request_id,
            server_id=server_id,
        )

    async def release_sticky_session(self, request_id: str, routing_key: Optional[str] = None) -> None:
        teacher_key = self._resolve_teacher_key(routing_key)
        await self.server_managers[teacher_key]._load_balancer.release_request_binding.remote(
            request_id=request_id,
        )

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        request_id: Optional[str] = None,
        logprob_start_len: int = 0,
        expected_mm_prefix_surplus: Optional[int] = None,
        expected_logprob_rows: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        teacher_backend = teacher_model_config.inference.name
        if logprob_start_len > 0 and teacher_backend != "sglang":
            raise ValueError(
                f"SKD delta verification requires SGLang teacher inference for teacher key {teacher_key!r}, "
                f"but backend is {teacher_backend!r}."
            )

        sampling_params = _get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config)
        if logprob_start_len > 0:
            sampling_params["prompt_logprobs_start_len"] = logprob_start_len
        if expected_mm_prefix_surplus is not None:
            sampling_params["expected_mm_prefix_surplus"] = int(expected_mm_prefix_surplus)
        if expected_logprob_rows is not None:
            sampling_params["prompt_logprobs_expected_len"] = int(expected_logprob_rows)

        server_manager = self.server_managers[teacher_key]
        effective_request_id = request_id or uuid4().hex
        if expected_logprob_rows is not None:
            expected_len = int(expected_logprob_rows)
        elif logprob_start_len == 0:
            expected_len = len(sequence_ids)
        else:
            expected_len = len(sequence_ids) - logprob_start_len - 1
        _trace_async_skd(
            "teacher.compute_logprobs_single",
            teacher_key=teacher_key,
            routing_key=routing_key,
            request_id=effective_request_id,
            seq_len=len(sequence_ids),
            logprob_start_len=logprob_start_len,
            expected_len=expected_len,
            expected_mm_prefix_surplus=expected_mm_prefix_surplus,
            expected_logprob_rows=expected_logprob_rows,
            image_count=_safe_len(multi_modal_data.get("images")),
            video_count=_safe_len(multi_modal_data.get("videos")),
            prompt_logprobs=sampling_params.get("prompt_logprobs"),
        )
        generate_t0 = time.monotonic()
        teacher_output = await server_manager.generate(
            request_id=effective_request_id,
            prompt_ids=sequence_ids,
            sampling_params=sampling_params,
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        generate_ms = (time.monotonic() - generate_t0) * 1000
        prompt_ids_rows = teacher_output.extra_fields.get("prompt_ids", [])
        prompt_logprobs_rows = teacher_output.extra_fields.get("prompt_logprobs", [])
        first_prompt_ids = prompt_ids_rows[0] if prompt_ids_rows else []
        _trace_async_skd(
            "teacher.generate_done",
            teacher_key=teacher_key,
            routing_key=routing_key,
            request_id=effective_request_id,
            elapsed_ms=round(generate_ms, 1),
            returned_id_rows=_safe_len(prompt_ids_rows),
            returned_logprob_rows=_safe_len(prompt_logprobs_rows),
            returned_width=_safe_len(first_prompt_ids),
            expected_len=expected_len,
            expected_logprob_rows=expected_logprob_rows,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        tensor_t0 = time.monotonic()
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        tensor_ms = (time.monotonic() - tensor_t0) * 1000
        if expected_len < 0:
            raise ValueError(
                f"Invalid teacher logprob_start_len={logprob_start_len} for seq_len={len(sequence_ids)}."
            )
        if teacher_ids.shape[0] != expected_len or teacher_logprobs.shape[0] != expected_len:
            raise ValueError(
                f"Unexpected teacher logprob length for teacher key {teacher_key!r}: "
                f"ids={teacher_ids.shape[0]}, logprobs={teacher_logprobs.shape[0]}, "
                f"expected={expected_len}, seq_len={len(sequence_ids)}, start={logprob_start_len}."
            )
        _trace_async_skd(
            "teacher.tensorize_done",
            teacher_key=teacher_key,
            routing_key=routing_key,
            request_id=effective_request_id,
            elapsed_ms=round(tensor_ms, 1),
            ids_shape=tuple(teacher_ids.shape),
            logprobs_shape=tuple(teacher_logprobs.shape),
        )
        return teacher_ids, teacher_logprobs
