"""Async SKD manager should not manage teacher lifecycle per rollout."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.async_skd.manager import AsyncSkdAgentLoopManager
from verl.protocol import DataProto


class _FailingTeacherManager:
    async def wake_up(self) -> None:
        raise AssertionError("teacher wake_up should not be called per rollout")

    async def sleep(self) -> None:
        raise AssertionError("teacher sleep should not be called per rollout")


class _NoTeacherLifecycleManager(AsyncSkdAgentLoopManager):
    async def _generate_sequences_sample_async(self, prompts: DataProto) -> list[DataProto]:
        return [prompts]

    def _finalize_outputs(self, outputs: list[DataProto]) -> DataProto:
        return outputs[0]


def _make_manager() -> _NoTeacherLifecycleManager:
    manager = _NoTeacherLifecycleManager.__new__(_NoTeacherLifecycleManager)
    manager.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "agent": {"async_skd_mode": "sample_async"},
                }
            }
        }
    )
    manager.rollout_config = OmegaConf.create({"n": 1})
    manager.stream_teacher_with_rollout = True
    manager.teacher_model_manager = _FailingTeacherManager()
    return manager


def _make_prompts() -> DataProto:
    return DataProto.from_dict(
        tensors={"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)},
        non_tensors={"index": np.array([0], dtype=object)},
    )


@pytest.mark.asyncio
async def test_generate_sequences_does_not_wake_or_sleep_teacher_per_rollout():
    manager = _make_manager()
    prompts = _make_prompts()

    output = await manager.generate_sequences(prompts)

    assert output is prompts
