from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.web_osgym_windowing import contiguous_one_spans


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


def build_web_osgym_windowed_agent_loop_outputs(
    output: AgentLoopOutput,
    *,
    enabled: bool,
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

    for step_zero, ((target_start, target_end), generation_window) in enumerate(
        zip(assistant_spans, generation_windows, strict=True)
    ):
        if not isinstance(generation_window, dict):
            raise ValueError(f"Web/OSGym generation window must be a dict, got {type(generation_window)!r}.")
        prompt_ids = list(generation_window.get("prompt_ids") or [])
        if not prompt_ids:
            raise ValueError("Web/OSGym generation window is missing prompt_ids.")

        prompt_image_indices = generation_window.get("prompt_image_indices")
        if prompt_image_indices is None:
            prompt_image_indices = generation_window.get("image_indices") or []
        image_indices = [int(idx) for idx in prompt_image_indices]
        target_response_ids = list(output.response_ids[target_start:target_end])
        target_logprobs = (
            list(output.response_logprobs[target_start:target_end]) if output.response_logprobs is not None else None
        )

        extra_fields = deepcopy(output.extra_fields)
        extra_fields.pop("web_osgym_generation_windows", None)
        extra_fields["web_osgym_window_row"] = True
        extra_fields["web_osgym_window_row_idx"] = step_zero + 1
        extra_fields["web_osgym_window_row_count"] = len(assistant_spans)
        extra_fields["web_osgym_window_target_start"] = int(target_start)
        extra_fields["web_osgym_window_target_end"] = int(target_end)
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
        extra_fields["web_osgym_generation_window"] = _compact_generation_window(generation_window)

        windows.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=target_response_ids,
                response_mask=[1] * len(target_response_ids),
                response_logprobs=target_logprobs,
                routed_experts=None,
                multi_modal_data=_slice_multi_modal_data(output.multi_modal_data, image_indices=image_indices),
                reward_score=output.reward_score,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra_fields=extra_fields,
            )
        )
        target_lengths.append(target_end - target_start)
        image_counts.append(len(image_indices))
        prompt_lengths.append(len(prompt_ids))

    return windows, _window_metrics(
        sample_count=len(windows),
        target_lengths=target_lengths,
        image_counts=image_counts,
        prompt_lengths=prompt_lengths,
    )
