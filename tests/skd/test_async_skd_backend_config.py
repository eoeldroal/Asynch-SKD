"""Async SKD backend configuration guard tests."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    ASYNC_SKD_MANAGER_CLASS,
    _validate_async_skd_backend_config,
)


def _make_config(
    *,
    rollout_name: str = "sglang",
    teacher_name: str = "sglang",
    loss_mode: str = "forward_kl_topk",
    default_agent_loop: str = "skd_agent",
    manager_class: str | None = ASYNC_SKD_MANAGER_CLASS,
    async_skd_mode: str = "sample_async",
    distillation_enabled: bool = True,
):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": rollout_name,
                    "prompt_length": 4,
                    "response_length": 4,
                    "agent": {
                        "default_agent_loop": default_agent_loop,
                        "agent_loop_manager_class": manager_class,
                        "async_skd_mode": async_skd_mode,
                    },
                }
            },
            "distillation": {
                "_target_": "verl.workers.config.DistillationConfig",
                "enabled": distillation_enabled,
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "teacher_models": {
                    "teacher_model": {
                        "_target_": "verl.workers.config.DistillationTeacherModelConfig",
                        "model_path": "Qwen/Qwen3-8B",
                        "inference": {
                            "_target_": "verl.workers.config.RolloutConfig",
                            "name": teacher_name,
                            "prompt_length": 4,
                            "response_length": 4,
                            "tensor_model_parallel_size": 1,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": 9,
                            "max_num_batched_tokens": 9,
                        },
                    }
                },
                "distillation_loss": {
                    "_target_": "verl.workers.config.DistillationLossConfig",
                    "loss_mode": loss_mode,
                    "topk": 4,
                    "use_policy_gradient": False,
                },
            },
        }
    )


def test_async_skd_backend_guard_accepts_sglang_student_and_teacher():
    _validate_async_skd_backend_config(_make_config())


def test_async_skd_backend_guard_rejects_non_sglang_student_rollout():
    with pytest.raises(ValueError, match="actor_rollout_ref\\.rollout\\.name='vllm'"):
        _validate_async_skd_backend_config(_make_config(rollout_name="vllm"))


def test_async_skd_backend_guard_rejects_non_sglang_teacher():
    with pytest.raises(ValueError, match="teacher 'default'.*inference\\.name='vllm'"):
        _validate_async_skd_backend_config(_make_config(teacher_name="vllm"))


def test_async_skd_backend_guard_ignores_sync_non_skd_rollout():
    _validate_async_skd_backend_config(
        _make_config(
            rollout_name="vllm",
            teacher_name="vllm",
            default_agent_loop="single_turn_agent",
            manager_class=None,
            async_skd_mode="sync",
            distillation_enabled=False,
        )
    )


def test_async_skd_backend_guard_requires_distillation_enabled():
    with pytest.raises(ValueError, match="distillation\\.enabled=True"):
        _validate_async_skd_backend_config(_make_config(distillation_enabled=False))


def test_async_skd_backend_guard_requires_topk_distillation_loss():
    with pytest.raises(ValueError, match="requires a top-k distillation loss"):
        _validate_async_skd_backend_config(_make_config(loss_mode="k3"))
