from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.web_osgym_windowing import contiguous_one_spans, normalize_web_osgym_steps


def _slice_multi_modal_data(
    multi_modal_data: dict[str, Any] | None,
    *,
    image_indices: list[int],
) -> dict[str, Any]:
    if not multi_modal_data:
        return {}

    sliced = dict(multi_modal_data)
    images = multi_modal_data.get("images")
    if images is None:
        return sliced

    image_list = list(images)
    selected_images = [image_list[idx] for idx in image_indices if 0 <= idx < len(image_list)]
    if selected_images:
        sliced["images"] = selected_images
    else:
        sliced.pop("images", None)
    return sliced


def _window_metrics(
    *,
    sample_count: int,
    target_lengths: list[int],
    image_counts: list[int],
    prompt_lengths: list[int],
) -> dict[str, float]:
    if sample_count <= 0:
        return {"web_osgym/window_update_num_samples": 0.0}
    return {
        "web_osgym/window_update_num_samples": float(sample_count),
        "web_osgym/window_update_avg_target_tokens": float(np.mean(target_lengths)) if target_lengths else 0.0,
        "web_osgym/window_update_max_target_tokens": float(max(target_lengths)) if target_lengths else 0.0,
        "web_osgym/window_update_avg_images": float(np.mean(image_counts)) if image_counts else 0.0,
        "web_osgym/window_update_max_images": float(max(image_counts)) if image_counts else 0.0,
        "web_osgym/window_update_avg_prompt_tokens": float(np.mean(prompt_lengths)) if prompt_lengths else 0.0,
        "web_osgym/window_update_max_prompt_tokens": float(max(prompt_lengths)) if prompt_lengths else 0.0,
    }


def _compact_generation_window(meta: dict[str, Any]) -> dict[str, Any]:
    compact = dict(meta)
    compact.pop("prompt_ids", None)
    return compact


def _group_span_indices(num_spans: int, block_size: int) -> list[tuple[int, int]]:
    if num_spans <= 0:
        return []

    groups: list[tuple[int, int]] = []
    # Keep the trailing rows full-sized so the most recent assistant turns stay in
    # the supervised block when the total count is not divisible by block_size.
    first_group_size = num_spans % block_size or block_size
    start = 0
    end = first_group_size
    while start < num_spans:
        groups.append((start, end))
        start = end
        end = min(num_spans, start + block_size)
    return groups


def _mask_prefix_before_supervised_start(mask_slice: list[int], supervised_prefix_len: int) -> list[int]:
    masked = list(mask_slice)
    for idx in range(min(supervised_prefix_len, len(masked))):
        masked[idx] = 0
    return masked


def _step_image_index_map(extra_fields: dict[str, Any]) -> dict[int, list[int]]:
    steps = normalize_web_osgym_steps(extra_fields.get("web_osgym_steps") or [])
    mapping: dict[int, list[int]] = {}
    for step in steps:
        step_idx = int(step["step_idx"])
        mapping[step_idx] = list(range(int(step["image_start"]), int(step["image_end"])))
    return mapping


def _assistant_current_step_idx(generation_window: dict[str, Any]) -> int | None:
    selected_step_indices = list(generation_window.get("selected_step_indices") or [])
    if not selected_step_indices:
        return None
    return int(selected_step_indices[-1])


def _block_image_indices(
    generation_windows: list[dict[str, Any]],
    step_image_indices: dict[int, list[int]],
    *,
    block_start_idx: int,
    block_end_idx: int,
) -> list[int]:
    image_indices = list(generation_windows[block_start_idx].get("prompt_image_indices") or [])
    for span_idx in range(block_start_idx + 1, block_end_idx + 1):
        step_idx = _assistant_current_step_idx(generation_windows[span_idx])
        if step_idx is None:
            continue
        image_indices.extend(step_image_indices.get(step_idx, []))
    return image_indices


def build_web_osgym_windowed_agent_loop_outputs(
    output: AgentLoopOutput,
    *,
    enabled: bool,
    supervision_block_size: int = 1,
    carry_turn_budget: int | None = None,
) -> tuple[list[AgentLoopOutput], dict[str, float]]:
    """Split a Web/OSGym trajectory into per-generation training rows.

    The rollout prompt window records the exact prompt and image slice used for
    each model generation. This helper turns each assistant generation into one
    update sample with that same prompt context and only that generation as the
    supervised/RL response target.
    """

    if not enabled:
        return [output], {}

    generation_windows = output.extra_fields.get("web_osgym_generation_windows")
    if not generation_windows:
        return [output], {}

    assistant_spans = contiguous_one_spans(output.response_mask)
    if len(generation_windows) != len(assistant_spans):
        raise ValueError(
            "Web/OSGym windowed update requires one generation window per assistant response span, "
            f"got windows={len(generation_windows)}, assistant_spans={len(assistant_spans)}."
        )

    windows: list[AgentLoopOutput] = []
    target_lengths: list[int] = []
    image_counts: list[int] = []
    prompt_lengths: list[int] = []
    block_size = max(1, int(supervision_block_size))
    step_image_indices = _step_image_index_map(output.extra_fields)

    if block_size == 1:
        groups = [(idx, idx + 1) for idx in range(len(assistant_spans))]
        effective_budget = 1
    else:
        groups = _group_span_indices(len(assistant_spans), block_size)
        effective_budget = max(1, int(carry_turn_budget)) if carry_turn_budget is not None else block_size

    for row_idx, (supervised_start_idx, supervised_end_idx) in enumerate(groups, start=1):
        supervised_count = supervised_end_idx - supervised_start_idx
        warmup_count = 0 if block_size == 1 else min(supervised_start_idx, max(0, effective_budget - supervised_count))
        block_start_idx = supervised_start_idx - warmup_count
        block_end_idx = supervised_end_idx - 1

        generation_window = generation_windows[block_start_idx]
        if not isinstance(generation_window, dict):
            raise ValueError(f"Web/OSGym generation window must be a dict, got {type(generation_window)!r}.")
        prompt_ids = list(generation_window.get("prompt_ids") or [])
        if not prompt_ids:
            raise ValueError("Web/OSGym generation window is missing prompt_ids.")

        response_start = int(assistant_spans[block_start_idx][0])
        response_end = int(assistant_spans[block_end_idx][1])
        supervised_response_start = int(assistant_spans[supervised_start_idx][0])
        response_ids = list(output.response_ids[response_start:response_end])
        response_logprobs = (
            list(output.response_logprobs[response_start:response_end]) if output.response_logprobs is not None else None
        )
        response_mask = list(output.response_mask[response_start:response_end])
        if warmup_count > 0:
            response_mask = _mask_prefix_before_supervised_start(response_mask, supervised_response_start - response_start)
        else:
            response_mask = list(response_mask)

        if len(response_ids) != len(response_mask):
            raise ValueError(
                "Web/OSGym window row has misaligned response_ids and response_mask lengths: "
                f"response_ids={len(response_ids)} response_mask={len(response_mask)} "
                f"row_idx={row_idx} block_start={block_start_idx + 1} block_end={block_end_idx + 1}."
            )
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise ValueError(
                "Web/OSGym window row has misaligned response_logprobs and response_ids lengths: "
                f"response_logprobs={len(response_logprobs)} response_ids={len(response_ids)} "
                f"row_idx={row_idx} block_start={block_start_idx + 1} block_end={block_end_idx + 1}."
            )
        if int(sum(response_mask)) <= 0:
            raise ValueError(
                "Web/OSGym window row has no supervised response tokens after block grouping: "
                f"row_idx={row_idx} block_start={block_start_idx + 1} supervised_start={supervised_start_idx + 1} "
                f"supervised_end={supervised_end_idx}."
            )

        image_indices = (
            [int(idx) for idx in (generation_window.get("prompt_image_indices") or generation_window.get("image_indices") or [])]
            if block_size == 1
            else _block_image_indices(
                generation_windows,
                step_image_indices,
                block_start_idx=block_start_idx,
                block_end_idx=block_end_idx,
            )
        )

        extra_fields = deepcopy(output.extra_fields)
        extra_fields.pop("web_osgym_generation_windows", None)
        extra_fields["web_osgym_window_row"] = True
        extra_fields["web_osgym_window_row_idx"] = row_idx
        extra_fields["web_osgym_window_row_count"] = len(groups)
        extra_fields["web_osgym_window_block_response_start"] = response_start
        extra_fields["web_osgym_window_block_response_end"] = response_end
        extra_fields["web_osgym_window_prompt_tokens"] = len(prompt_ids)
        extra_fields["web_osgym_window_prompt_image_count"] = len(image_indices)
        extra_fields["web_osgym_window_old_summary_turn_indices"] = list(
            generation_window.get("old_summary_turn_indices") or []
        )
        extra_fields["web_osgym_window_recent_observation_step_indices"] = list(
            generation_window.get("recent_observation_step_indices") or []
        )
        extra_fields["web_osgym_window_recent_assistant_turn_indices"] = list(
            generation_window.get("recent_assistant_turn_indices") or []
        )
        extra_fields["web_osgym_window_text_only_recent_step_count"] = int(
            generation_window.get("text_only_recent_step_count", 0)
        )
        extra_fields["web_osgym_window_supervision_block_size"] = block_size
        extra_fields["web_osgym_window_supervised_span_start_idx"] = supervised_start_idx + 1
        extra_fields["web_osgym_window_supervised_span_end_idx"] = supervised_end_idx
        extra_fields["web_osgym_window_warmup_span_count"] = warmup_count
        extra_fields["web_osgym_generation_window"] = _compact_generation_window(generation_window)

        windows.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                routed_experts=None,
                multi_modal_data=_slice_multi_modal_data(output.multi_modal_data, image_indices=image_indices),
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=extra_fields,
            )
        )
        target_lengths.append(int(sum(response_mask)))
        image_counts.append(len(image_indices))
        prompt_lengths.append(len(prompt_ids))

    return windows, _window_metrics(
        sample_count=len(windows),
        target_lengths=target_lengths,
        image_counts=image_counts,
        prompt_lengths=prompt_lengths,
    )
