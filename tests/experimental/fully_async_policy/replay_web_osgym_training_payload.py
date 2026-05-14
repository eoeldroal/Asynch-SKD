from __future__ import annotations

import argparse
import ast
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    _summarize_training_payload,
    assemble_batch_from_rollout_samples,
)
from verl.utils.model import extract_multi_modal_inputs
from verl.workers.utils.padding import left_right_2_no_padding


@dataclass(frozen=True)
class ReplayShape:
    required_samples: int
    rollout_n: int
    turns_per_trajectory: int
    block_size: int
    rows_per_sample_group: int
    total_rows: int
    images_per_row: int


class _Tokenizer:
    pad_token_id = 0


def _parse_overrides_from_log(path: Path) -> list[str]:
    text = path.read_text(errors="replace")
    marker = "Error executing job with overrides:"
    idx = text.rfind(marker)
    if idx < 0:
        return []
    line = text[idx:].splitlines()[0]
    raw = line.split(marker, 1)[1].strip()
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _override_value(overrides: list[str], key: str, default: int) -> int:
    prefixes = (f"{key}=", f"+{key}=")
    for item in reversed(overrides):
        if item.startswith(prefixes):
            value = item.split("=", 1)[1].strip("\"'")
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _max_unit_trace_value(path: Path, key: str) -> int | None:
    pattern = re.compile(r"\[WebOsGymTool\]\[UnitTrace\]\s+({.*})")
    max_value: int | None = None
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        value = payload.get(key)
        if isinstance(value, int):
            max_value = value if max_value is None else max(max_value, value)
    return max_value


def _infer_shape(args: argparse.Namespace) -> ReplayShape:
    overrides = _parse_overrides_from_log(args.log)
    required_samples = args.required_samples or _override_value(overrides, "async_training.require_batches", 1) * 16
    rollout_n = args.rollout_n or _override_value(overrides, "actor_rollout_ref.rollout.n", 1)
    max_assistant_turns = _override_value(
        overrides, "actor_rollout_ref.rollout.multi_turn.max_assistant_turns", args.turns
    )
    max_seen_turns = _max_unit_trace_value(args.log, "generation_window_count") or 0
    if args.turn_source == "log":
        turns = max_seen_turns
    elif args.turn_source == "config":
        turns = max_assistant_turns
    else:
        turns = max(max_seen_turns, max_assistant_turns, args.turns)
    if args.turns:
        turns = max(turns, args.turns)

    block_size = max(1, args.block_size)
    rows_per_trajectory = math.ceil(turns / block_size)
    rows_per_sample_group = rollout_n * rows_per_trajectory
    return ReplayShape(
        required_samples=required_samples,
        rollout_n=rollout_n,
        turns_per_trajectory=turns,
        block_size=block_size,
        rows_per_sample_group=rows_per_sample_group,
        total_rows=required_samples * rows_per_sample_group,
        images_per_row=args.images_per_row,
    )


def _make_rollout_sample(sample_idx: int, shape: ReplayShape, args: argparse.Namespace) -> RolloutSample:
    rows = shape.rows_per_sample_group
    seq_len = args.prompt_len + args.response_len
    prompt_positions = torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).expand(rows, 3, seq_len).clone()
    tensors = {
        "prompts": torch.ones((rows, args.prompt_len), dtype=torch.long),
        "responses": torch.ones((rows, args.response_len), dtype=torch.long),
        "input_ids": torch.ones((rows, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((rows, seq_len), dtype=torch.long),
        "position_ids": prompt_positions,
        "response_mask": torch.ones((rows, args.response_len), dtype=torch.long),
        "rm_scores": torch.zeros((rows, args.response_len), dtype=torch.float32),
        "rollout_log_probs": torch.zeros((rows, args.response_len), dtype=torch.float32),
    }

    multi_modal_inputs = np.empty(rows, dtype=object)
    for row_idx in range(rows):
        multi_modal_inputs[row_idx] = {
            "pixel_values": torch.zeros(
                (shape.images_per_row * args.image_tokens, args.vision_width),
                dtype=torch.float16,
            ),
            "image_grid_thw": torch.ones((shape.images_per_row, 3), dtype=torch.long),
            "images_seqlens": torch.full((shape.images_per_row,), args.image_tokens, dtype=torch.long),
        }

    non_tensors = {
        "__num_turns__": np.array([shape.turns_per_trajectory] * rows, dtype=np.int32),
        "uid": np.array([f"synthetic_group_{sample_idx}"] * rows, dtype=object),
        "index": np.array([sample_idx] * rows, dtype=object),
        "min_global_steps": np.array([0] * rows, dtype=object),
        "max_global_steps": np.array([0] * rows, dtype=object),
        "web_osgym_window_row": np.array([True] * rows, dtype=object),
        "web_osgym_window_supervision_block_size": np.array([shape.block_size] * rows, dtype=object),
        "multi_modal_inputs": multi_modal_inputs,
    }
    batch = DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={"metrics": [{"generate_sequences": 0.0, "tool_calls": 0.0}] * rows},
    )
    return RolloutSample(
        full_batch=batch,
        sample_id=f"synthetic_group_{sample_idx}",
        epoch=0,
        rollout_status={"count/total_generated_samples": sample_idx + 1},
    )


def _cuda_used_mib() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / (1024**2), 1)


def _print_event(event: str, **payload: Any) -> None:
    print({"event": event, **payload}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a synthetic Web/OSGym windowed training payload from logs.")
    parser.add_argument("--log", type=Path, default=Path("logs/qwen35_webgym_fully_async_rl.out"))
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--turn-source", choices=("log", "config", "max"), default="max")
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument("--required-samples", type=int, default=0)
    parser.add_argument("--rollout-n", type=int, default=0)
    parser.add_argument("--images-per-row", type=int, default=6)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--vision-width", type=int, default=1176)
    parser.add_argument("--prompt-len", type=int, default=4200)
    parser.add_argument("--response-len", type=int, default=64)
    parser.add_argument("--to-cuda", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    shape = _infer_shape(args)
    _print_event(
        "shape",
        shape=shape.__dict__,
        cuda_available=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    samples = []
    started = time.time()
    for sample_idx in range(shape.required_samples):
        samples.append(_make_rollout_sample(sample_idx, shape, args))
    _print_event("samples_built", elapsed_s=round(time.time() - started, 2), cuda_used_mib=_cuda_used_mib())

    started = time.time()
    batch = assemble_batch_from_rollout_samples(samples, _Tokenizer(), config=None, balance_batch=None)
    _print_event(
        "assembled",
        elapsed_s=round(time.time() - started, 2),
        summary=_summarize_training_payload(batch),
        cuda_used_mib=_cuda_used_mib(),
    )

    started = time.time()
    td = left_right_2_no_padding(batch.to_tensordict())
    _print_event("to_tensordict_left_right_2_no_padding", elapsed_s=round(time.time() - started, 2), cuda_used_mib=_cuda_used_mib())

    started = time.time()
    multi_modal_inputs = extract_multi_modal_inputs(td.get("multi_modal_inputs", []))
    if args.to_cuda:
        device = torch.device(args.device)
        multi_modal_inputs = {
            key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
            for key, value in multi_modal_inputs.items()
        }
        torch.cuda.synchronize(device)
    pixel_values = multi_modal_inputs.get("pixel_values")
    _print_event(
        "extract_multi_modal_inputs",
        elapsed_s=round(time.time() - started, 2),
        pixel_values_shape=tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None,
        pixel_values_mib=(
            round(pixel_values.numel() * pixel_values.element_size() / (1024**2), 2)
            if isinstance(pixel_values, torch.Tensor)
            else 0.0
        ),
        cuda_used_mib=_cuda_used_mib(),
    )


if __name__ == "__main__":
    main()
