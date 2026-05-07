from __future__ import annotations

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.async_skd.data_source import AsyncSkdDataSource
from verl.experimental.async_skd.state import AsyncSkdSample
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import _assemble_async_skd_training_batch
from verl.utils.tensordict_utils import chunk_tensordict


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


def _single_sample_batch(uid: str, value: int, prompt_width: int = 2, *, with_teacher: bool = False) -> DataProto:
    response_width = 2
    seq_len = prompt_width + response_width
    prompts = torch.arange(100 + value, 100 + value + prompt_width, dtype=torch.long).unsqueeze(0)
    responses = torch.tensor([[value, value + 1]], dtype=torch.long)
    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": torch.cat([prompts, responses], dim=1),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
    }
    if with_teacher:
        tensors["teacher_ids"] = torch.arange(seq_len * 2, dtype=torch.long).view(1, seq_len, 2)
        tensors["teacher_logprobs"] = -torch.arange(seq_len * 2, dtype=torch.float32).view(1, seq_len, 2)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "uid": np.array([uid], dtype=object),
            "input_pos": np.array([value], dtype=object),
        },
        meta_info={"metrics": [{}]},
    )


def _windowed_output_batch(uid: str, values: list[int]) -> DataProto:
    batch = DataProto.concat([_single_sample_batch(uid, value) for value in values])
    batch.meta_info["metrics"] = [{} for _ in values]
    batch.meta_info["metrics"][0].update(
        {
            "window/num_samples": float(len(values)),
            "window/avg_images": 1.0,
        }
    )
    return batch


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


def _raw_input_batch(uid: str, value: int) -> DataProto:
    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.tensor([[value]], dtype=torch.long)},
        non_tensors={
            "uid": np.array([uid], dtype=object),
            "input_pos": np.array([value], dtype=object),
        },
        meta_info={"metrics": [{}]},
    )


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


def test_async_skd_training_batch_aligns_output_non_tensor_keys_before_concat():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _single_sample_batch("promoted-0", 0)
    promoted_input.non_tensor_batch["agent_name"] = np.array(["web_skd_agent"], dtype=object)
    promoted_output = _single_sample_batch("promoted-0", 10)
    promoted_output.non_tensor_batch["agent_name"] = np.array(["web_skd_agent"], dtype=object)
    promoted_output.non_tensor_batch["multi_modal_inputs"] = np.array([{"image_grid_thw": torch.tensor([[1, 2, 3]])}], dtype=object)
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _base_batch(1000, 1)
    base_output_batch = _base_batch(2000, 1)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
    )

    assert len(merged_input) == 2
    assert len(merged_output) == 2
    assert merged_output.non_tensor_batch["agent_name"].tolist() == [None, "web_skd_agent"]
    assert merged_output.non_tensor_batch["multi_modal_inputs"][0] == {}
    assert "image_grid_thw" in merged_output.non_tensor_batch["multi_modal_inputs"][1]


def test_async_skd_training_batch_keeps_union_metadata_consistent():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _single_sample_batch("promoted-0", 0)
    promoted_input.non_tensor_batch["agent_name"] = np.array(["web_skd_agent"], dtype=object)
    promoted_output = _single_sample_batch("promoted-0", 0)
    promoted_output.non_tensor_batch["agent_name"] = np.array(["web_skd_agent"], dtype=object)
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _base_batch(1000, 1)
    base_input_batch.non_tensor_batch["agent_name"] = np.array(["web_skd_agent"], dtype=object)
    base_output_batch = _base_batch(1000, 1)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
    )

    assert merged_input.non_tensor_batch["agent_name"].tolist() == ["web_skd_agent", "web_skd_agent"]
    assert merged_output.non_tensor_batch["agent_name"].tolist() == ["web_skd_agent", "web_skd_agent"]
    merged_input.union(merged_output)


def test_async_skd_training_batch_expands_promoted_input_to_windowed_output_rows():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _single_sample_batch("promoted-0", 0)
    promoted_output = _windowed_output_batch("promoted-0", [10, 11, 12])
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _single_sample_batch("base-0", 1000)
    base_output_batch = _windowed_output_batch("base-0", [2000, 2001])

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
    )

    assert len(merged_input) == 5
    assert len(merged_output) == 5
    assert merged_input.non_tensor_batch["uid"].tolist() == [
        "base-0",
        "base-0",
        "promoted-0",
        "promoted-0",
        "promoted-0",
    ]
    assert merged_output.non_tensor_batch["uid"].tolist() == [
        "base-0",
        "base-0",
        "promoted-0",
        "promoted-0",
        "promoted-0",
    ]
    assert "window_metrics" not in merged_output.meta_info
    assert merged_output.meta_info["metrics"]["window/num_samples"] == [2.0, 3.0]
    assert source.promoted_count() == 0


def test_async_skd_training_batch_aligns_prompt_width_before_promoted_concat():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _single_sample_batch("promoted-0", 0, prompt_width=2, with_teacher=True)
    promoted_output = _single_sample_batch("promoted-0", 10, prompt_width=5, with_teacher=True)
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _single_sample_batch("base-0", 1000, prompt_width=2, with_teacher=True)
    base_output_batch = _single_sample_batch("base-0", 2000, prompt_width=2, with_teacher=True)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
        pad_token_id=99,
    )

    assert merged_input.batch["prompts"].shape == (2, 5)
    assert merged_output.batch["prompts"].shape == (2, 5)
    assert merged_output.batch["input_ids"].shape == (2, 7)
    assert merged_output.batch["attention_mask"][0, :3].tolist() == [0, 0, 0]
    assert merged_output.batch["input_ids"][0, :3].tolist() == [99, 99, 99]
    assert merged_output.batch["teacher_ids"].shape == (2, 7, 2)
    assert merged_output.batch["teacher_ids"][0, :3].eq(99).all()
    assert merged_output.batch["teacher_logprobs"][0, :3].eq(0.0).all()
    merged_input.union(merged_output)


def test_async_skd_training_batch_aligns_routed_experts_before_promoted_concat():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _single_sample_batch("promoted-0", 0, prompt_width=2)
    promoted_output = _single_sample_batch("promoted-0", 10, prompt_width=5)
    promoted_output.batch["routed_experts"] = torch.full((1, 7, 1, 1), 4, dtype=torch.long)
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _single_sample_batch("base-0", 1000, prompt_width=2)
    base_output_batch = _single_sample_batch("base-0", 2000, prompt_width=2)
    base_output_batch.batch["routed_experts"] = torch.full((1, 4, 1, 1), 3, dtype=torch.long)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
        pad_token_id=99,
    )

    assert merged_output.batch["routed_experts"].shape == (2, 7, 1, 1)
    assert merged_output.batch["routed_experts"][0, :3].eq(0).all()
    assert merged_output.batch["routed_experts"][0, 3:].eq(3).all()
    assert merged_output.batch["routed_experts"][1].eq(4).all()


def test_async_skd_training_batch_rejects_missing_prompts_before_concat():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    base_input_batch = _single_sample_batch("base-0", 1000)
    base_output_batch = _single_sample_batch("base-0", 2000)
    del base_output_batch.batch["prompts"]

    with pytest.raises(ValueError, match="requires 'prompts'"):
        _assemble_async_skd_training_batch(
            base_input_batch,
            base_output_batch,
            async_skd_data_source=source,
            validate=False,
            required_multiple=None,
            max_promoted_count=0,
            pad_token_id=0,
        )


def test_async_skd_training_batch_allows_raw_input_batches_without_prompts():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    promoted_input = _raw_input_batch("promoted-0", 0)
    promoted_output = _single_sample_batch("promoted-0", 10, prompt_width=5)
    source._reserved_input_batches["promoted-0"] = promoted_input
    source.record_promoted([_completed("promoted-0", promoted_output)])

    base_input_batch = _raw_input_batch("base-0", 1000)
    base_output_batch = _single_sample_batch("base-0", 2000, prompt_width=2)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=1,
        pad_token_id=99,
    )

    assert "prompts" not in merged_input.batch
    assert merged_input.batch["dummy_tensor"].shape == (2, 1)
    assert merged_output.batch["prompts"].shape == (2, 5)
    merged_input.union(merged_output)


def test_async_skd_training_batch_expands_input_to_windowed_output_rows():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    base_input_batch = _single_sample_batch("base-0", 1000)
    base_output_batch = DataProto.concat(
        [
            _single_sample_batch("base-0", 1000),
            _single_sample_batch("base-0", 1000),
        ]
    )

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=0,
    )

    assert len(merged_input) == 2
    assert len(merged_output) == 2
    assert merged_input.non_tensor_batch["uid"].tolist() == ["base-0", "base-0"]
    assert merged_output.non_tensor_batch["uid"].tolist() == ["base-0", "base-0"]


def test_async_skd_base_output_expands_input_rows_by_uid_when_windowed():
    base_input = _single_sample_batch("base-0", 10, with_teacher=False)
    base_output = _windowed_output_batch("base-0", [101, 102, 103])

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input,
        base_output,
        async_skd_data_source=AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory()),
        validate=False,
        required_multiple=None,
        max_promoted_count=0,
        pad_token_id=0,
    )

    assert len(merged_input) == 3
    assert len(merged_output) == 3
    assert merged_input.non_tensor_batch["uid"].tolist() == ["base-0", "base-0", "base-0"]
    assert merged_output.non_tensor_batch["uid"].tolist() == ["base-0", "base-0", "base-0"]


def test_async_skd_base_output_rejects_missing_window_identity_with_precise_error():
    base_input = _single_sample_batch("base-0", 10, with_teacher=False)
    base_output = _windowed_output_batch("base-0", [101, 102])
    base_output.non_tensor_batch["uid"] = np.array([None, "base-0"], dtype=object)

    with pytest.raises(ValueError, match="Cannot expand async-SKD input batch"):
        _assemble_async_skd_training_batch(
            base_input,
            base_output,
            async_skd_data_source=AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory()),
            validate=False,
            required_multiple=None,
            max_promoted_count=0,
            pad_token_id=0,
        )


def test_async_skd_training_batch_handles_mixed_base_completed_rows():
    fresh_input = _single_sample_batch("fresh-0", 10, with_teacher=False)
    carry_input = _single_sample_batch("carry-0", 20, with_teacher=False)
    base_input = DataProto.concat([fresh_input, carry_input])

    fresh_output = _windowed_output_batch("fresh-0", [101, 102])
    carry_output = _windowed_output_batch("carry-0", [201, 202, 203])
    base_output = DataProto.concat([fresh_output, carry_output])

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input,
        base_output,
        async_skd_data_source=AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory()),
        validate=False,
        required_multiple=None,
        max_promoted_count=0,
        pad_token_id=0,
    )

    assert merged_input.non_tensor_batch["uid"].tolist() == [
        "fresh-0",
        "fresh-0",
        "carry-0",
        "carry-0",
        "carry-0",
    ]
    assert merged_output.non_tensor_batch["uid"].tolist() == [
        "fresh-0",
        "fresh-0",
        "carry-0",
        "carry-0",
        "carry-0",
    ]


def test_async_skd_training_batch_uses_windowed_prompts_before_union():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    base_input_batch = DataProto(
        batch=TensorDict({"prompts": torch.tensor([[1, 2]])}, batch_size=1),
        non_tensor_batch={"uid": np.array(["base-0"], dtype=object)},
        meta_info={},
    )
    base_output_batch = _windowed_output_batch("base-0", [2000, 2001])
    base_output_batch.batch["prompts"] = torch.tensor([[0, 3], [4, 5]], dtype=torch.long)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=None,
        max_promoted_count=0,
    )

    assert merged_input.batch["prompts"].equal(merged_output.batch["prompts"])
    merged_input.union(merged_output)


def test_async_skd_training_batch_pads_windowed_rows_to_required_multiple():
    source = AsyncSkdDataSource(_EmptyIterator(), uid_fn=_UidFactory())
    base_input_batch = _single_sample_batch("base-0", 1000, with_teacher=False)
    base_input_batch.meta_info = {}
    base_output_batch = _windowed_output_batch("base-0", [2000, 2001, 2002])
    base_output_batch.batch["teacher_ids"] = torch.arange(3 * 4 * 2, dtype=torch.long).view(3, 4, 2)
    base_output_batch.batch["teacher_logprobs"] = -torch.arange(3 * 4 * 2, dtype=torch.float32).view(3, 4, 2)

    merged_input, merged_output = _assemble_async_skd_training_batch(
        base_input_batch,
        base_output_batch,
        async_skd_data_source=source,
        validate=False,
        required_multiple=4,
        max_promoted_count=0,
    )

    assert len(merged_input) == 4
    assert len(merged_output) == 4
    assert merged_output.non_tensor_batch["uid"][-1].startswith("async_skd_pad_")
    assert merged_output.batch["response_mask"][-1].sum().item() == 0
    assert merged_output.batch["teacher_ids"][-1].equal(merged_output.batch["teacher_ids"][-2])
    assert merged_output.batch["teacher_logprobs"][-1].equal(merged_output.batch["teacher_logprobs"][-2])
    training_batch = merged_input.union(merged_output)
    chunk_tensordict(training_batch.to_tensordict(), chunks=4)
