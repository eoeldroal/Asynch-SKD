from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.web_osgym_windowing import contiguous_one_spans, normalize_image_spans


@dataclass(frozen=True)
class WindowedSkdConfig:
    enabled: bool = False
    history_n: int = 5
    max_images_per_sample: int | None = 6


def _slice_multi_modal_data(
    multi_modal_data: dict[str, Any] | None,
    *,
    image_start: int,
    image_end: int,
    include_images: bool,
) -> dict[str, Any]:
    if not multi_modal_data:
        return {}

    sliced = dict(multi_modal_data)
    images = multi_modal_data.get("images")
    if images is not None:
        if include_images:
            sliced["images"] = list(images)[image_start:image_end]
        else:
            sliced.pop("images", None)
    return sliced


def _vision_token_ids(tokenizer: Any) -> tuple[int | None, int | None]:
    if tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None, None
    start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if not isinstance(start_id, int) or start_id < 0:
        start_id = None
    if not isinstance(end_id, int) or end_id < 0:
        end_id = None
    return start_id, end_id


def _drop_first_vision_blocks(prompt_ids: list[int], count: int, tokenizer: Any) -> list[int]:
    if count <= 0:
        return list(prompt_ids)
    start_id, end_id = _vision_token_ids(tokenizer)
    if start_id is None or end_id is None:
        return list(prompt_ids)

    keep = [True] * len(prompt_ids)
    dropped = 0
    idx = 0
    while idx < len(prompt_ids) and dropped < count:
        if int(prompt_ids[idx]) != start_id:
            idx += 1
            continue
        end = idx + 1
        while end < len(prompt_ids) and int(prompt_ids[end]) != end_id:
            end += 1
        if end >= len(prompt_ids):
            break
        for drop_idx in range(idx, end + 1):
            keep[drop_idx] = False
        dropped += 1
        idx = end + 1
    return [token for token, should_keep in zip(prompt_ids, keep, strict=True) if should_keep]


def _window_metrics(
    *,
    sample_count: int,
    target_lengths: list[int],
    image_counts: list[int],
    recent_counts: list[int],
    skipped_old_steps: list[int],
) -> dict[str, float]:
    if sample_count <= 0:
        return {"window/num_samples": 0.0}
    return {
        "window/num_samples": float(sample_count),
        "window/avg_target_tokens": float(np.mean(target_lengths)) if target_lengths else 0.0,
        "window/max_target_tokens": float(max(target_lengths)) if target_lengths else 0.0,
        "window/avg_images": float(np.mean(image_counts)) if image_counts else 0.0,
        "window/max_images": float(max(image_counts)) if image_counts else 0.0,
        "window/avg_recent_steps": float(np.mean(recent_counts)) if recent_counts else 0.0,
        "window/skipped_old_steps": float(sum(skipped_old_steps)),
    }


def build_windowed_agent_loop_outputs(
    output: AgentLoopOutput,
    *,
    config: WindowedSkdConfig,
    tokenizer: Any = None,
) -> tuple[list[AgentLoopOutput], dict[str, float]]:
    """Convert one full SKD trajectory output into mini-step training outputs.

    This runs before ``_agent_loop_postprocess()``, while teacher rows are still
    response-relative and images are still raw objects. Target boundaries are
    contiguous ``response_mask == 1`` runs. Context tokens are original response
    slices with loss mask zeroed except for the current target run.
    """

    if not config.enabled:
        return [output], {}

    teacher_ids = output.extra_fields.get("teacher_ids_list")
    teacher_logprobs = output.extra_fields.get("teacher_logprobs_list")
    if teacher_ids is None or teacher_logprobs is None:
        return [output], {}

    if len(output.response_mask) != len(teacher_ids) or len(output.response_mask) != len(teacher_logprobs):
        raise ValueError(
            "Windowed SKD requires response-relative teacher rows: "
            f"response_mask={len(output.response_mask)}, "
            f"teacher_ids={len(teacher_ids)}, teacher_logprobs={len(teacher_logprobs)}"
        )

    assistant_spans = contiguous_one_spans(output.response_mask)
    if not assistant_spans:
        return [output], _window_metrics(
            sample_count=0,
            target_lengths=[],
            image_counts=[],
            recent_counts=[],
            skipped_old_steps=[],
        )

    images = list((output.multi_modal_data or {}).get("images") or [])
    image_by_step: dict[int, int] = {}
    for item in normalize_image_spans(output.extra_fields.get("mini_step_image_spans")):
        step_idx = int(item["step_idx"])
        image_start = int(item["image_start"])
        image_end = int(item["image_end"])
        if image_end != image_start + 1:
            raise ValueError(
                "Windowed WebSKD parser expects visual step metadata to point to exactly one image, "
                f"got step_idx={step_idx}, image_start={image_start}, image_end={image_end}"
            )
        image_by_step[step_idx] = image_start

    topk = len(teacher_ids[0]) if teacher_ids else 0
    zero_ids = [0] * topk
    zero_logprobs = [0.0] * topk

    windows: list[AgentLoopOutput] = []
    target_lengths: list[int] = []
    image_counts: list[int] = []
    recent_counts: list[int] = []
    skipped_old_steps: list[int] = []

    for step_zero, (target_start, target_end) in enumerate(assistant_spans):
        target_step = step_zero + 1
        recent_start = max(1, target_step - max(0, int(config.history_n)))

        while True:
            selected_image_indices = [
                image_by_step[step_idx]
                for step_idx in range(recent_start, target_step + 1)
                if step_idx in image_by_step
            ]
            if config.max_images_per_sample is None:
                break
            if len(selected_image_indices) <= int(config.max_images_per_sample):
                break
            if recent_start >= target_step:
                break
            recent_start += 1

        response_start = 0 if recent_start <= 1 else assistant_spans[recent_start - 2][1]

        if selected_image_indices:
            image_start = min(selected_image_indices)
            image_end = max(selected_image_indices) + 1
            has_images = True
        else:
            image_start = len(images)
            image_end = len(images)
            has_images = False

        local_target_start = target_start - response_start
        local_target_end = target_end - response_start
        response_len = target_end - response_start
        response_mask = [0] * response_len
        response_mask[local_target_start:local_target_end] = [1] * (target_end - target_start)

        response_teacher_ids = [list(zero_ids) for _ in range(response_len)]
        response_teacher_logprobs = [list(zero_logprobs) for _ in range(response_len)]
        for src_idx in range(target_start, target_end):
            dst_idx = src_idx - response_start
            response_teacher_ids[dst_idx] = list(teacher_ids[src_idx])
            response_teacher_logprobs[dst_idx] = list(teacher_logprobs[src_idx])

        extra_fields = dict(output.extra_fields)
        extra_fields["teacher_ids_list"] = response_teacher_ids
        extra_fields["teacher_logprobs_list"] = response_teacher_logprobs
        extra_fields["window_source_response_start"] = response_start
        extra_fields["window_target_start"] = target_start
        extra_fields["window_target_end"] = target_end
        extra_fields["window_step_idx"] = target_step
        extra_fields["window_recent_start"] = recent_start
        extra_fields["window_image_start"] = image_start
        extra_fields["window_image_end"] = image_end
        prompt_ids = _drop_first_vision_blocks(output.prompt_ids, image_start, tokenizer)

        window = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=list(output.response_ids[response_start:target_end]),
            response_mask=response_mask,
            response_logprobs=(
                list(output.response_logprobs[response_start:target_end])
                if output.response_logprobs is not None
                else None
            ),
            routed_experts=None,
            multi_modal_data=_slice_multi_modal_data(
                output.multi_modal_data,
                image_start=image_start,
                image_end=image_end,
                include_images=has_images,
            ),
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )
        windows.append(window)
        target_lengths.append(target_end - target_start)
        image_counts.append(image_end - image_start)
        recent_counts.append(target_step - recent_start)
        skipped_old_steps.append(max(0, recent_start - 1))

    return windows, _window_metrics(
        sample_count=len(windows),
        target_lengths=target_lengths,
        image_counts=image_counts,
        recent_counts=recent_counts,
        skipped_old_steps=skipped_old_steps,
    )
