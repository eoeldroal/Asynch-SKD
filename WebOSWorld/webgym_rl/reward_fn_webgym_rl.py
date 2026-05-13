"""Reward function for webgym-rl runs."""

from __future__ import annotations

from typing import Any


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_env_reward(extra_info: dict[str, Any]) -> float | None:
    for key in ("web_osgym_env_reward_score", "reward_score", "env_reward", "reward"):
        score = _as_float(extra_info.get(key))
        if score is not None:
            return score

    rollout_scores = extra_info.get("rollout_reward_scores")
    if isinstance(rollout_scores, dict):
        for key in ("web_osgym_env_reward_score", "reward_score", "env_reward", "reward", "computer"):
            score = _as_float(rollout_scores.get(key))
            if score is not None:
                return score
    return None


def _compute_format_reward(extra_info: dict[str, Any], *, min_denominator: int) -> tuple[float, int, int]:
    attempted_tool_calls = _as_int(extra_info.get("web_osgym_attempted_tool_calls"))
    valid_tool_calls = _as_int(extra_info.get("web_osgym_valid_tool_calls"))
    if attempted_tool_calls is None or valid_tool_calls is None or attempted_tool_calls <= 0:
        return 0.0, 0, 0
    normalized_denominator = max(int(min_denominator), 1)
    format_reward = float(valid_tool_calls) / float(max(int(attempted_tool_calls), normalized_denominator))
    return format_reward, int(attempted_tool_calls), int(valid_tool_calls)


def compute_score_webgym_rl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_reward_alpha: float = 0.0,
    format_reward_min_denominator: int = 5,
    **kwargs,
) -> float | dict[str, float | int]:
    del data_source, solution_str, ground_truth, kwargs
    if not isinstance(extra_info, dict):
        extra_info = {}
    env_reward = _extract_env_reward(extra_info)
    env_reward = 0.0 if env_reward is None else env_reward
    format_reward, attempted_tool_calls, valid_tool_calls = _compute_format_reward(
        extra_info,
        min_denominator=format_reward_min_denominator,
    )

    final_reward = env_reward + float(format_reward_alpha) * float(format_reward)
    return {
        "score": final_reward,
        "score/sum": final_reward,
        "score/env": env_reward,
        "score/format": float(format_reward),
        "web_osgym_env_reward_score": env_reward,
        "web_osgym_format_reward": float(format_reward),
        "web_osgym_attempted_tool_calls": attempted_tool_calls,
        "web_osgym_valid_tool_calls": valid_tool_calls,
    }
