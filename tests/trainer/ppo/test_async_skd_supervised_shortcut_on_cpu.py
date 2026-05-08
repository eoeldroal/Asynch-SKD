from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.distillation.losses import compute_distillation_loss_range
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


def test_distillation_loss_range_handles_padding_only_rows():
    metrics = compute_distillation_loss_range(
        distillation_losses=torch.ones(1, 4),
        response_mask=torch.zeros(1, 4, dtype=torch.bool),
    )

    assert metrics["distillation/loss_min"].aggregate() == 0.0
    assert metrics["distillation/loss_max"].aggregate() == 0.0


class _AsyncSkdManagerStub:
    def __init__(self) -> None:
        self.last_generate_with_carryover_kwargs = None
        self.last_generate_kwargs = None

    def generate_sequences_with_carryover(self, *args, **kwargs):
        self.last_generate_with_carryover_kwargs = kwargs
        return SimpleNamespace(meta_info={"timing": {}})

    def generate_sequences(self, *args, **kwargs):
        self.last_generate_kwargs = kwargs
        return SimpleNamespace(meta_info={"timing": {}})

    def set_async_skd_data_source(self, *args, **kwargs):
        raise NotImplementedError

    def set_async_skd_pad_token_id(self, *args, **kwargs):
        raise NotImplementedError

    def flush_async_skd_lookahead(self):
        self.flush_calls += 1


class _CheckpointManagerStub:
    def __init__(self) -> None:
        self.sleep_calls = 0

    def sleep_replicas(self):
        self.sleep_calls += 1


def _make_trainer_for_shortcut(config_dict: dict) -> RayPPOTrainer:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(config_dict)
    trainer.async_rollout_manager = _AsyncSkdManagerStub()
    trainer.checkpoint_manager = _CheckpointManagerStub()
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


def test_async_skd_sleep_helper_only_sleeps_replicas():
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

    trainer._sleep_replicas_after_async_skd_step_end()

    assert trainer.checkpoint_manager.sleep_calls == 1


def test_async_skd_generate_with_carryover_can_request_step_end_flush():
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

    trainer.async_rollout_manager.generate_sequences_with_carryover(
        fresh_prompts=None,
        carryover_partials=[],
        flush_lookahead_before_return=True,
    )

    assert trainer.async_rollout_manager.last_generate_with_carryover_kwargs["flush_lookahead_before_return"] is True


def test_async_skd_resume_drops_web_skd_carryover_but_keeps_promoted():
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
                "rollout": {
                    "agent": {
                        "async_skd_mode": "lookahead",
                        "default_agent_loop": "web_skd_agent",
                    }
                },
            },
        }
    )

    loaded_state = {
        "fresh_buffer": "fresh",
        "fresh_cursor": 3,
        "carryover_partials": ["partial-a", "partial-b"],
        "carryover_input_batches": ["input-a", "input-b"],
        "reserved_input_batches": {"reserved": "value"},
        "promoted_input_batches": ["promoted-in"],
        "promoted_output_batches": ["promoted-out"],
        "trained_reserved_sample_ids": ["uid-0"],
    }

    sanitized = trainer._drop_async_skd_carryover_from_loaded_state(loaded_state)

    assert sanitized["carryover_partials"] == []
    assert sanitized["carryover_input_batches"] == []
    assert sanitized["promoted_input_batches"] == ["promoted-in"]
    assert sanitized["promoted_output_batches"] == ["promoted-out"]
    assert sanitized["reserved_input_batches"] == {"reserved": "value"}
    assert loaded_state["carryover_partials"] == ["partial-a", "partial-b"]


def test_async_skd_resume_keeps_carryover_for_non_web_agent_loops():
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
                "rollout": {
                    "agent": {
                        "async_skd_mode": "lookahead",
                        "default_agent_loop": "single_turn_agent",
                    }
                },
            },
        }
    )

    loaded_state = {
        "carryover_partials": ["partial-a"],
        "carryover_input_batches": ["input-a"],
    }

    sanitized = trainer._drop_async_skd_carryover_from_loaded_state(loaded_state)

    assert sanitized == loaded_state
