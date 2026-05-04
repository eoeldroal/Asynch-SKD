from __future__ import annotations

import numpy as np
import torch

from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    assemble_batch_from_rollout_samples,
)
from verl.protocol import DataProto


class _FakeTokenizer:
    pad_token_id = 0


class _FakeTrainerBalance:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.actor_rollout_wg = object()
        self.seen_lengths: list[int] = []

    def _get_dp_size(self, worker_group, role: str) -> int:
        assert worker_group is self.actor_rollout_wg
        assert role == "actor"
        return self.dp_size

    def __call__(self, batch: DataProto, metrics) -> None:
        del metrics
        self.seen_lengths.append(len(batch))
        assert len(batch) % self.dp_size == 0


def _make_rollout_sample(
    *,
    sample_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    with_optional_seq_tensors: bool = False,
) -> RolloutSample:
    prompt_width = len(prompt_ids)
    response_width = len(response_ids)
    seq_width = prompt_width + response_width

    tensors = {
        "prompts": torch.tensor([prompt_ids], dtype=torch.long),
        "responses": torch.tensor([response_ids], dtype=torch.long),
        "response_mask": torch.ones((1, response_width), dtype=torch.long),
        "input_ids": torch.tensor([prompt_ids + response_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, seq_width), dtype=torch.long),
        "position_ids": torch.arange(seq_width, dtype=torch.long).view(1, 1, seq_width).expand(1, 4, seq_width).clone(),
        "rollout_log_probs": torch.zeros((1, response_width), dtype=torch.float32),
        "rm_scores": torch.zeros((1, response_width), dtype=torch.float32),
    }
    tensors["rm_scores"][0, response_width - 1] = 1.0

    if with_optional_seq_tensors:
        tensors["teacher_ids"] = torch.arange(seq_width * 2, dtype=torch.int32).view(1, seq_width, 2)
        tensors["teacher_logprobs"] = torch.zeros((1, seq_width, 2), dtype=torch.float32)
        tensors["routed_experts"] = torch.ones((1, seq_width, 3, 2), dtype=torch.int64)

    batch = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "min_global_steps": np.array([0], dtype=np.int64),
            "max_global_steps": np.array([0], dtype=np.int64),
            "uid": np.array([sample_id], dtype=object),
            "index": np.array([0], dtype=object),
        },
        meta_info={
            "metrics": [
                {
                    "generate_sequences": 1.0,
                    "tool_calls": 0.1,
                }
            ]
        },
    )
    return RolloutSample(
        full_batch=batch,
        sample_id=sample_id,
        epoch=0,
        rollout_status={"queue_wait_s": 0.0},
    )


def test_assemble_batch_from_rollout_samples_repad_to_batch_local_max_width():
    batch = assemble_batch_from_rollout_samples(
        [
            _make_rollout_sample(sample_id="s1", prompt_ids=[1, 2, 3, 4], response_ids=[11, 12, 13]),
            _make_rollout_sample(sample_id="s2", prompt_ids=[5, 6], response_ids=[21, 22]),
        ],
        tokenizer=_FakeTokenizer(),
        config=None,
        balance_batch=None,
    )

    assert tuple(batch.batch["prompts"].shape) == (2, 4)
    assert tuple(batch.batch["responses"].shape) == (2, 3)
    assert tuple(batch.batch["input_ids"].shape) == (2, 7)
    assert tuple(batch.batch["attention_mask"].shape) == (2, 7)
    assert tuple(batch.batch["position_ids"].shape) == (2, 4, 7)
    assert tuple(batch.batch["rollout_log_probs"].shape) == (2, 3)
    assert tuple(batch.batch["rm_scores"].shape) == (2, 3)

    assert batch.batch["prompts"][1].tolist() == [0, 0, 5, 6]
    assert batch.batch["responses"][1].tolist() == [21, 22, 0]
    assert batch.batch["input_ids"][1].tolist() == [0, 0, 5, 6, 21, 22, 0]
    assert batch.batch["attention_mask"][1].tolist() == [0, 0, 1, 1, 1, 1, 0]
    assert batch.batch["position_ids"][1, 0].tolist() == [0, 0, 0, 1, 2, 3, 0]


def test_assemble_batch_from_rollout_samples_repad_optional_seq_tensors():
    batch = assemble_batch_from_rollout_samples(
        [
            _make_rollout_sample(
                sample_id="s1",
                prompt_ids=[1, 2, 3, 4],
                response_ids=[11, 12, 13],
                with_optional_seq_tensors=True,
            ),
            _make_rollout_sample(
                sample_id="s2",
                prompt_ids=[5, 6],
                response_ids=[21, 22],
                with_optional_seq_tensors=True,
            ),
        ],
        tokenizer=_FakeTokenizer(),
        config=None,
        balance_batch=None,
    )

    assert tuple(batch.batch["teacher_ids"].shape) == (2, 7, 2)
    assert tuple(batch.batch["teacher_logprobs"].shape) == (2, 7, 2)
    assert tuple(batch.batch["routed_experts"].shape) == (2, 7, 3, 2)

    assert batch.batch["teacher_ids"][1, :2].tolist() == [[0, 0], [0, 0]]
    assert batch.batch["teacher_ids"][1, -1].tolist() == [0, 0]
    assert torch.all(batch.batch["teacher_logprobs"][1, :2] == 0)
    assert torch.all(batch.batch["teacher_logprobs"][1, -1] == 0)
    assert torch.all(batch.batch["routed_experts"][1, :2] == 0)
    assert torch.all(batch.batch["routed_experts"][1, -1] == 0)


def test_assemble_batch_from_rollout_samples_pads_to_actor_dp_multiple_before_balance():
    balance = _FakeTrainerBalance(dp_size=2)

    batch = assemble_batch_from_rollout_samples(
        [
            _make_rollout_sample(sample_id="s1", prompt_ids=[1, 2, 3], response_ids=[11, 12]),
            _make_rollout_sample(sample_id="s2", prompt_ids=[4, 5, 6], response_ids=[21, 22]),
            _make_rollout_sample(sample_id="s3", prompt_ids=[7, 8, 9], response_ids=[31, 32]),
        ],
        tokenizer=_FakeTokenizer(),
        config=None,
        balance_batch=balance.__call__,
    )

    assert balance.seen_lengths == [4]
    assert len(batch) == 4
    assert batch.non_tensor_batch["uid"][-1].startswith("fully_async_pad_")
    assert batch.non_tensor_batch["index"][-1] == -1
    assert torch.all(batch.batch["response_mask"][-1] == 0)
    assert torch.all(batch.batch["rm_scores"][-1] == 0)
    assert torch.all(batch.batch["rollout_log_probs"][-1] == 0)
