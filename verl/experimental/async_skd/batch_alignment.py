"""Tensor width alignment helpers for async SKD batch boundaries."""

from __future__ import annotations

import torch

from verl.protocol import DataProto


def require_prompt_batch(batch: DataProto, *, context_name: str) -> None:
    if batch.batch is None:
        raise ValueError(f"Cannot align {context_name}: batch tensor payload is missing.")
    if "prompts" not in batch.batch:
        raise ValueError(f"Cannot align {context_name}: batch requires 'prompts'.")


def left_pad_dim(tensor: torch.Tensor, dim: int, pad_size: int, value: int | float) -> torch.Tensor:
    if pad_size <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(
        pad_shape,
        value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([pad, tensor], dim=dim)


def pad_batch_prompt_width(
    batch: DataProto,
    target_prompt_width: int,
    *,
    pad_token_id: int,
    context_name: str,
) -> DataProto:
    require_prompt_batch(batch, context_name=context_name)
    prompt_width = int(batch.batch["prompts"].size(1))
    pad_size = target_prompt_width - prompt_width
    if pad_size <= 0:
        return batch

    batch.batch["prompts"] = left_pad_dim(batch.batch["prompts"], 1, pad_size, pad_token_id)
    if "responses" not in batch.batch:
        return batch

    response_width = int(batch.batch["responses"].size(1))
    old_seq_width = prompt_width + response_width
    for key, value in (("input_ids", pad_token_id), ("attention_mask", 0)):
        if key in batch.batch:
            tensor = batch.batch[key]
            if int(tensor.size(1)) != old_seq_width:
                raise ValueError(
                    f"Cannot align {context_name} key {key!r}: "
                    f"width={tensor.size(1)} expected={old_seq_width}"
                )
            batch.batch[key] = left_pad_dim(tensor, 1, pad_size, value)
    if "position_ids" in batch.batch:
        tensor = batch.batch["position_ids"]
        if int(tensor.size(-1)) != old_seq_width:
            raise ValueError(
                f"Cannot align {context_name} key 'position_ids': "
                f"width={tensor.size(-1)} expected={old_seq_width}"
            )
        batch.batch["position_ids"] = left_pad_dim(tensor, tensor.dim() - 1, pad_size, 0)
    for key, value in (("teacher_ids", pad_token_id), ("teacher_logprobs", 0.0), ("routed_experts", 0)):
        if key in batch.batch:
            tensor = batch.batch[key]
            if int(tensor.size(1)) != old_seq_width:
                raise ValueError(
                    f"Cannot align {context_name} key {key!r}: "
                    f"width={tensor.size(1)} expected={old_seq_width}"
                )
            batch.batch[key] = left_pad_dim(tensor, 1, pad_size, value)
    return batch


def align_prompt_width_for_concat(
    batches: list[DataProto],
    *,
    pad_token_id: int,
    context_name: str,
    target_prompt_width: int | None = None,
) -> list[DataProto]:
    if not batches:
        return batches
    for batch in batches:
        require_prompt_batch(batch, context_name=context_name)
    if target_prompt_width is None:
        prompt_widths = [int(batch.batch["prompts"].size(1)) for batch in batches]
        target_prompt_width = max(prompt_widths)
    return [
        pad_batch_prompt_width(
            batch,
            target_prompt_width,
            pad_token_id=pad_token_id,
            context_name=context_name,
        )
        for batch in batches
    ]
