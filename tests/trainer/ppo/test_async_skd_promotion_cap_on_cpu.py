from __future__ import annotations

import numpy as np
import torch

from verl.experimental.async_skd.data_source import AsyncSkdDataSource
from verl.experimental.async_skd.state import AsyncSkdSample
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import _assemble_async_skd_training_batch


class _UidFactory:
    def __init__(self) -> None:
        self._next = 0

    def __call__(self) -> str:
        value = f"uid-{self._next}"
        self._next += 1
        return value


class _EmptyIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


def _single_sample_batch(uid: str, value: int) -> DataProto:
    seq_len = 4
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[100 + value, 200 + value]], dtype=torch.long),
            "responses": torch.tensor([[value, value + 1]], dtype=torch.long),
            "input_ids": torch.tensor([[100 + value, 200 + value, value, value + 1]], dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        non_tensors={
            "uid": np.array([uid], dtype=object),
            "input_pos": np.array([value], dtype=object),
        },
        meta_info={"metrics": [{}]},
    )


def _completed(sample_id: str, batch: DataProto) -> AsyncSkdSample:
    return AsyncSkdSample.from_completed(
        sample_id=sample_id,
        logical_step=4,
        source_type="lookahead",
        batch=batch,
    )


def _base_batch(start: int, count: int) -> DataProto:
    batches = [_single_sample_batch(f"base-{i}", start + i) for i in range(count)]
    return DataProto.concat(batches)


def test_async_skd_training_batch_respects_promoted_cap_and_preserves_fifo_overflow():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())

    promoted_inputs = []
    promoted_samples = []
    for i in range(64):
        batch = _single_sample_batch(f"promoted-{i}", i)
        promoted_inputs.append(batch)
        source._reserved_input_batches[f"promoted-{i}"] = batch
        promoted_samples.append(_completed(f"promoted-{i}", batch))
    source.record_promoted(promoted_samples)

    base_input_batch = _base_batch(1000, 64)
    base_output_batch = _base_batch(2000, 64)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=4,
        max_promoted_count=48,
    )

    assert len(merged_input) == 112
    assert len(merged_output) == 112
    assert source.promoted_count() == 16

    merged_uids = merged_input.non_tensor_batch["uid"].tolist()
    assert merged_uids[:64] == [f"base-{i}" for i in range(64)]
    assert merged_uids[64:] == [f"promoted-{i}" for i in range(48)]

    remaining_inputs, remaining_outputs = source.pop_promoted_pairs(max_count=32)
    assert [batch.non_tensor_batch["uid"].tolist()[0] for batch in remaining_inputs] == [
        f"promoted-{i}" for i in range(48, 64)
    ]
    assert [batch.non_tensor_batch["uid"].tolist()[0] for batch in remaining_outputs] == [
        f"promoted-{i}" for i in range(48, 64)
    ]
