import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    resolve_max_concurrent_rollout_samples_per_gpu,
)


def _config(rollout_n=8, max_concurrent_samples_per_gpu=16):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "n": rollout_n,
                    "agent": {
                        "max_concurrent_samples_per_gpu": max_concurrent_samples_per_gpu,
                    },
                },
            },
        }
    )


def test_max_concurrent_samples_per_gpu_is_trajectory_budget_in_fully_async():
    config = _config(rollout_n=8, max_concurrent_samples_per_gpu=16)

    assert resolve_max_concurrent_rollout_samples_per_gpu(config) == 2


def test_max_concurrent_samples_per_gpu_must_be_divisible_by_rollout_n():
    config = _config(rollout_n=8, max_concurrent_samples_per_gpu=10)

    with pytest.raises(ValueError, match="must be divisible"):
        resolve_max_concurrent_rollout_samples_per_gpu(config)


def test_default_keeps_internal_rollout_sample_budget():
    config = _config(rollout_n=8, max_concurrent_samples_per_gpu=None)

    assert resolve_max_concurrent_rollout_samples_per_gpu(config) == 16
