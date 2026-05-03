from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def contiguous_one_spans(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if int(value) == 1 and start is None:
            start = idx
        elif int(value) != 1 and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "no", "n", "off", "none", "null"}:
            return False
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
    return bool(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_image_bounds(image_start: Any, image_end: Any) -> tuple[int, int]:
    start = max(0, _coerce_int(image_start, default=0))
    end = max(start, max(0, _coerce_int(image_end, default=start)))
    return start, end


def _normalize_string_list(value: Any) -> list[str]:
    items = _as_list(value)
    return [str(item) for item in items if item is not None]


def _normalize_action_list(value: Any) -> list[dict[str, Any]]:
    items = _as_list(value)
    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
    return normalized


def _flatten_previous_actions(value: Any) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            continue
        if item.get("action_type") is not None or item.get("name") is not None:
            flattened.append(dict(item))
            continue
        flattened.extend(_normalize_action_list(item.get("actions")))
    return flattened


def normalize_image_spans(value: Any) -> list[dict[str, int | bool]]:
    spans: list[dict[str, int | bool]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            continue
        image_start, image_end = _normalize_image_bounds(item.get("image_start", 0), item.get("image_end", 0))
        spans.append(
            {
                "step_idx": _coerce_int(item.get("step_idx"), default=len(spans) + 1),
                "image_start": image_start,
                "image_end": image_end,
                "terminal": _coerce_bool(item.get("terminal"), default=False),
            }
        )
    spans.sort(key=lambda item: int(item["step_idx"]))
    return spans


def normalize_web_osgym_steps(value: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(_as_list(value), start=1):
        if not isinstance(item, Mapping):
            continue
        step_idx = _coerce_int(item.get("step_idx"), default=index)
        assistant_turn = _coerce_int(item.get("assistant_turn"), default=step_idx)
        user_turn = _coerce_int(item.get("user_turn"), default=0)
        raw_text_len = item.get("text_len")
        if raw_text_len is None and item.get("text") is not None:
            text_len = len(str(item.get("text") or ""))
        else:
            text_len = _coerce_int(raw_text_len, default=0)
        image_start, image_end = _normalize_image_bounds(item.get("image_start", 0), item.get("image_end", 0))
        steps.append(
            {
                "step_idx": step_idx,
                "assistant_turn": assistant_turn,
                "user_turn": user_turn,
                "phase": str(item.get("phase", "")),
                "text_len": text_len,
                "action_names": _normalize_string_list(item.get("action_names")),
                "actions": _normalize_action_list(item.get("actions")),
                "image_start": image_start,
                "image_end": image_end,
                "terminal": _coerce_bool(item.get("terminal"), default=False),
                "termination_reason": _coerce_optional_str(item.get("termination_reason")),
            }
        )
    steps.sort(key=lambda item: int(item["step_idx"]))
    return steps


def build_mini_step_image_spans(value: Any) -> list[dict[str, int | bool]]:
    spans: list[dict[str, int | bool]] = []
    for item in normalize_web_osgym_steps(value):
        image_start = int(item["image_start"])
        image_end = int(item["image_end"])
        if image_end <= image_start:
            continue
        spans.append(
            {
                "step_idx": int(item["step_idx"]),
                "image_start": image_start,
                "image_end": image_end,
                "terminal": bool(item["terminal"]),
            }
        )
    return spans


def select_recent_web_osgym_steps(
    value: Any,
    *,
    target_step_idx: int | None = None,
    history_n: int = 5,
    max_images_per_sample: int | None = None,
) -> list[dict[str, Any]]:
    steps = normalize_web_osgym_steps(value)
    if not steps:
        return []

    if target_step_idx is None:
        filtered = steps
    else:
        filtered = [step for step in steps if int(step["step_idx"]) <= int(target_step_idx)]
    if not filtered:
        return []

    keep_count = max(0, int(history_n)) + 1
    selected = filtered[-keep_count:]

    if max_images_per_sample is None:
        return selected

    image_cap = int(max_images_per_sample)
    while len(selected) > 1:
        image_count = sum(max(0, int(step["image_end"]) - int(step["image_start"])) for step in selected)
        if image_count <= image_cap:
            break
        selected = selected[1:]
    return selected


def format_previous_actions(actions: Sequence[Mapping[str, Any]] | np.ndarray | None) -> str:
    formatted: list[str] = []
    for index, action in enumerate(_flatten_previous_actions(actions), start=1):
        action_name = action.get("action_type") or action.get("name")
        if action_name is None:
            continue
        arguments = [
            f"{key}={value!r}"
            for key, value in action.items()
            if key not in {"action_type", "name"} and value is not None
        ]
        formatted.append(f"{index}. {str(action_name).upper()}({', '.join(arguments)})")
    return "\n".join(formatted) if formatted else "None"
