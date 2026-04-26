from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_distillation_ppo_loss_skips_ppo_path_for_supervised_distillation(monkeypatch):
    distill_loss = torch.tensor(2.5)

    def _fake_distillation_loss(*args, **kwargs):
        return distill_loss, {"distillation/mock": 1.0}

    def _unexpected_ppo_loss(*args, **kwargs):
        raise AssertionError("ppo_loss should be skipped for supervised distillation")

    monkeypatch.setattr("verl.trainer.distillation.losses.distillation_loss", _fake_distillation_loss)
    monkeypatch.setattr("verl.trainer.distillation.losses.ppo_loss", _unexpected_ppo_loss)

    config = SimpleNamespace()
    distillation_config = SimpleNamespace(
        enabled=True,
        distillation_loss=SimpleNamespace(
            use_task_rewards=False,
            use_policy_gradient=False,
            distillation_loss_coef=1.0,
        ),
    )

    policy_loss, policy_metrics = distillation_ppo_loss(
        config=config,
        distillation_config=distillation_config,
        model_output={},
        data={},
    )

    assert torch.equal(policy_loss, distill_loss)
    assert "distillation/mock" in policy_metrics
    assert "distillation/loss" in policy_metrics


class _AsyncSkdManagerStub:
    def generate_sequences_with_carryover(self, *args, **kwargs):
        raise NotImplementedError

    def set_async_skd_data_source(self, *args, **kwargs):
        raise NotImplementedError


def _make_trainer_for_shortcut(config_dict: dict) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(config_dict)
    trainer.async_rollout_manager = _AsyncSkdManagerStub()
    return trainer


def test_async_skd_supervised_shortcut_enabled_for_grpo_only():
    trainer = _make_trainer_for_shortcut(
        {
            "algorithm": {
                "adv_estimator": AdvantageEstimator.GRPO,
                "use_kl_in_reward": False,
                "rollout_correction": None,
            },
            "distillation": {
                "enabled": True,
                "distillation_loss": {
                    "use_task_rewards": False,
                    "use_policy_gradient": False,
                },
            },
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {"agent": {"async_skd_mode": "lookahead"}},
            },
        }
    )

    assert trainer._uses_supervised_async_skd_logprob_shortcut() is True


def test_async_skd_supervised_shortcut_stays_off_when_task_rewards_enabled():
    trainer = _make_trainer_for_shortcut(
        {
            "algorithm": {
                "adv_estimator": AdvantageEstimator.GRPO,
                "use_kl_in_reward": False,
                "rollout_correction": None,
            },
            "distillation": {
                "enabled": True,
                "distillation_loss": {
                    "use_task_rewards": True,
                    "use_policy_gradient": False,
                },
            },
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {"agent": {"async_skd_mode": "lookahead"}},
            },
        }
    )

    assert trainer._uses_supervised_async_skd_logprob_shortcut() is False
