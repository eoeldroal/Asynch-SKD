from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.batch_decode_calls: list[bool] = []

    def batch_decode(self, ids, skip_special_tokens=True):
        self.batch_decode_calls.append(skip_special_tokens)
        prefix = "clean" if skip_special_tokens else "raw"
        return [f"{prefix}-{idx}" for idx in range(len(ids))]


class _RolloutBatchStub:
    def __init__(self) -> None:
        self.batch = {
            "prompts": torch.tensor([[11, 12]], dtype=torch.long),
            "responses": torch.tensor([[21, 22]], dtype=torch.long),
            "token_level_scores": torch.tensor([[1.0]], dtype=torch.float32),
        }
        self.non_tensor_batch = {
            "request_id": np.array(["req-1"], dtype=object),
        }
        self._items = [
            SimpleNamespace(non_tensor_batch={"reward_model": {"ground_truth": "gt-1"}}),
        ]

    def __iter__(self):
        return iter(self._items)


def test_log_rollout_data_keeps_special_tokens_in_outputs():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.tokenizer = _RecordingTokenizer()

    captured: dict[str, object] = {}

    def _capture_dump(*, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        captured["inputs"] = inputs
        captured["outputs"] = outputs
        captured["gts"] = gts
        captured["scores"] = scores
        captured["reward_extra_infos_dict"] = reward_extra_infos_dict
        captured["dump_path"] = dump_path

    trainer._dump_generations = _capture_dump

    batch = _RolloutBatchStub()
    reward_extra_infos_dict = {"web_osgym_termination_reason": ["system_stop"]}
    timing_raw: dict[str, float] = {}

    trainer._log_rollout_data(
        batch=batch,
        reward_extra_infos_dict=reward_extra_infos_dict,
        timing_raw=timing_raw,
        rollout_data_dir="/tmp/rollout-dump",
    )

    assert captured["inputs"] == ["clean-0"]
    assert captured["outputs"] == ["raw-0"]
    assert trainer.tokenizer.batch_decode_calls == [True, False]


def test_webgym_rl_rollout_dump_is_skipped():
    batch = _RolloutBatchStub()
    reward_extra_infos_dict = {"web_osgym_termination_reason": ["system_stop"]}

    assert RayPPOTrainer._should_skip_rollout_data_dump_for_webgym_rl(batch, reward_extra_infos_dict) is True


def test_non_webgym_rollout_dump_is_not_skipped():
    batch = _RolloutBatchStub()
    reward_extra_infos_dict = {"request_id": ["req-1"]}

    assert RayPPOTrainer._should_skip_rollout_data_dump_for_webgym_rl(batch, reward_extra_infos_dict) is False
