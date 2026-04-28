"""Metadata helpers for async-SKD DataProto boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from verl.protocol import DataProto


def missing_non_tensor_value(key: str) -> Any:
    """Return a semantic default for absent optional non-tensor metadata."""
    if key == "multi_modal_inputs":
        return {}
    return None


def align_non_tensor_keys_for_concat(data: list[DataProto]) -> list[DataProto]:
    """Make DataProto.concat safe across async-SKD scheduling boundaries.

    Async-SKD splits a logical rollout batch into fresh, carryover, and promoted
    single-sample outputs. Those paths can legitimately carry different optional
    metadata keys. DataProto.concat, however, requires every item to expose the
    same non_tensor_batch schema and the same per-item batch length. Normalize the
    schema at the async-SKD boundary instead of relying on whichever sample
    happens to appear first.
    """
    if not data:
        return data

    all_keys: set[str] = set()
    for item in data:
        all_keys.update(item.non_tensor_batch.keys())

    for item in data:
        batch_size = len(item)
        for key in all_keys:
            if key in item.non_tensor_batch:
                continue
            values = np.empty(batch_size, dtype=object)
            values[:] = [missing_non_tensor_value(key) for _ in range(batch_size)]
            item.non_tensor_batch[key] = values
    return data


def sync_output_non_tensor_with_input(input_batch: DataProto, output_batch: DataProto) -> DataProto:
    """Use input metadata as canonical before DataProto.union.

    Rollout outputs may echo input-owned metadata such as agent_name, uid, index,
    reward_model, or tools_kwargs. If both input and output carry the same key,
    DataProto.union requires exact equality. The input batch is the canonical
    owner for these fields; rollout-owned fields are left intact.
    """
    if len(input_batch) != len(output_batch):
        raise ValueError(
            "input/output batch size must match before syncing non-tensor metadata: "
            f"{len(input_batch)} != {len(output_batch)}"
        )

    output_owned_keys = {"multi_modal_inputs", "__num_turns__"}
    for key, value in input_batch.non_tensor_batch.items():
        if key in output_owned_keys:
            continue
        if key in output_batch.non_tensor_batch:
            output_batch.non_tensor_batch[key] = value.copy()
    return output_batch
