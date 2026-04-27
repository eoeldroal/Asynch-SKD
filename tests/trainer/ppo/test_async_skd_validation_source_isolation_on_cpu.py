from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _object_array(values: list[object]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


class _TrackingAsyncSkdSource:
    def __init__(self) -> None:
        self.validation_leak_count = 0


class _ValidationSourceAwareManager:
    def __init__(self, source: _TrackingAsyncSkdSource) -> None:
        self.current_source = source
        self.generate_seen_sources: list[object | None] = []

    def set_async_skd_data_source(self, source) -> None:
        self.current_source = source

    def generate_sequences(self, batch: DataProto) -> DataProto:
        self.generate_seen_sources.append(self.current_source)
        if self.current_source is not None:
            self.current_source.validation_leak_count += 1

        batch_size = len(batch)
        return DataProto.from_dict(
            tensors={
                "responses": torch.full((batch_size, 2), 7, dtype=torch.long),
            },
            meta_info={"timing": {}},
        )


class _ExplodingValidationManager(_ValidationSourceAwareManager):
    def generate_sequences(self, batch: DataProto) -> DataProto:
        self.generate_seen_sources.append(self.current_source)
        raise RuntimeError("validation boom")


class _TokenizerStub:
    eos_token_id = 42
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "decoded"


def _make_validation_batch_dict() -> dict[str, torch.Tensor | np.ndarray]:
    return {
        "prompts": torch.tensor([[101, 102]], dtype=torch.long),
        "agent_name": _object_array(["skd_agent"]),
        "data_source": _object_array(["dummy"]),
        "reward_model": _object_array([{"ground_truth": "gt"}]),
    }


def _make_trainer_for_validation_source_isolation() -> tuple[
    RayPPOTrainer, _TrackingAsyncSkdSource, _ValidationSourceAwareManager
]:
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "logger": [],
                "log_val_generations": 0,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {
                        "num_workers": 1,
                        "default_agent_loop": "skd_agent",
                    },
                    "val_kwargs": {
                        "n": 4,
                        "do_sample": True,
                    },
                }
            },
        }
    )
    trainer.tokenizer = _TokenizerStub()
    trainer.global_steps = 10
    trainer.use_rm = False
    trainer.validation_generations_logger = SimpleNamespace(log=lambda *args, **kwargs: None)
    trainer.val_dataloader = [_make_validation_batch_dict()]
    trainer._val_metrics_update = lambda *args, **kwargs: {"validation/mock": 1.0}

    source = _TrackingAsyncSkdSource()
    manager = _ValidationSourceAwareManager(source)
    trainer._async_skd_data_source = source
    trainer.async_rollout_manager = manager
    return trainer, source, manager


def test_validate_temporarily_detaches_training_async_skd_source(monkeypatch):
    trainer, source, manager = _make_trainer_for_validation_source_isolation()

    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.extract_reward",
        lambda batch: (torch.ones(len(batch), 1, dtype=torch.float32), {}),
    )

    metrics = trainer._validate()

    assert metrics == {"validation/mock": 1.0}
    assert manager.generate_seen_sources == [None]
    assert source.validation_leak_count == 0
    assert trainer._async_skd_data_source is source
    assert manager.current_source is source


def test_validate_restores_training_async_skd_source_after_exception(monkeypatch):
    trainer, source, _ = _make_trainer_for_validation_source_isolation()
    exploding_manager = _ExplodingValidationManager(source)
    trainer.async_rollout_manager = exploding_manager

    monkeypatch.setattr(
        "verl.trainer.ppo.ray_trainer.extract_reward",
        lambda batch: (torch.ones(len(batch), 1, dtype=torch.float32), {}),
    )

    try:
        trainer._validate()
        raise AssertionError("validation should have raised")
    except RuntimeError as exc:
        assert str(exc) == "validation boom"

    assert exploding_manager.generate_seen_sources == [None]
    assert trainer._async_skd_data_source is source
    assert exploding_manager.current_source is source
