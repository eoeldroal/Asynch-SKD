from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_trainer(config_dict: dict, *, async_skd_lookahead: bool = False) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(config_dict)
    trainer._uses_async_skd_lookahead_training = lambda: async_skd_lookahead
    return trainer


def _make_batch(*, force_single_mini_batch: bool = False) -> SimpleNamespace:
    meta_info = {}
    if force_single_mini_batch:
        meta_info["force_single_actor_mini_batch"] = True
    return SimpleNamespace(meta_info=meta_info)


def test_actor_batch_controls_use_fixed_mini_batch_by_default():
    trainer = _make_trainer(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 16,
                    "use_single_actor_mini_batch": False,
                },
                "rollout": {
                    "n": 8,
                },
            },
        }
    )

    controls = trainer._build_actor_batch_controls(
        _make_batch(),
        {
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.long,
            )
        },
    )

    assert controls == {
        "global_batch_size": 128,
        "mini_batch_size": 128,
    }


def test_actor_batch_controls_use_single_mini_batch_when_explicitly_enabled():
    trainer = _make_trainer(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 16,
                    "use_single_actor_mini_batch": True,
                },
                "rollout": {
                    "n": 8,
                },
            },
        }
    )

    controls = trainer._build_actor_batch_controls(
        _make_batch(),
        {
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.long,
            )
        },
    )

    assert controls == {
        "global_batch_size": 2,
        "num_mini_batch": 1,
    }


def test_actor_batch_controls_keep_async_skd_single_mini_batch_path():
    trainer = _make_trainer(
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 16,
                    "use_single_actor_mini_batch": False,
                },
                "rollout": {
                    "n": 8,
                },
            },
        },
        async_skd_lookahead=True,
    )

    controls = trainer._build_actor_batch_controls(
        _make_batch(),
        {
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0],
                ],
                dtype=torch.long,
            )
        },
    )

    assert controls == {
        "global_batch_size": 2,
        "num_mini_batch": 1,
    }
