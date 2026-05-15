"""Reward function for webgym-rl runs."""

from __future__ import annotations

import math
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


def _compute_format_reward(
    extra_info: dict[str, Any], *, tau: float, budget_exhausted_penalty: float
) -> tuple[float, int, int, int]:
    attempted_tool_calls = _as_int(extra_info.get("web_osgym_attempted_tool_calls"))
    valid_tool_calls = _as_int(extra_info.get("web_osgym_valid_tool_calls"))
    first_valid_tool_call_index = _as_int(extra_info.get("web_osgym_first_valid_tool_call_index"))
    if attempted_tool_calls is None or valid_tool_calls is None or attempted_tool_calls <= 0:
        return 0.0, 0, 0, 0

    attempted_tool_calls = int(attempted_tool_calls)
    valid_tool_calls = int(valid_tool_calls)
    tau = max(float(tau), 1e-6)

    precision = float(valid_tool_calls) / float(max(attempted_tool_calls, 1))
    if valid_tool_calls <= 0:
        latency = 0.0
        first_valid_tool_call_index = 0
    else:
        first_valid_tool_call_index = (
            int(first_valid_tool_call_index)
            if first_valid_tool_call_index is not None and int(first_valid_tool_call_index) > 0
            else 1
        )
        latency = math.exp(-float(first_valid_tool_call_index - 1) / tau)

    format_reward = precision * latency
    if extra_info.get("web_osgym_termination_reason") == "tool_response_budget_exhausted":
        format_reward -= float(budget_exhausted_penalty)

    return format_reward, attempted_tool_calls, valid_tool_calls, first_valid_tool_call_index


def _compute_non_grounding_adjacency_ratio(extra_info: dict[str, Any]) -> tuple[float, int, int]:
    executed_action_count = _as_int(extra_info.get("web_osgym_executed_action_count"))
    adjacent_pair_count = _as_int(extra_info.get("web_osgym_non_grounding_adjacent_pair_count"))
    if executed_action_count is None or executed_action_count <= 1:
        return 0.0, max(int(executed_action_count or 0), 0), max(int(adjacent_pair_count or 0), 0)
    if adjacent_pair_count is None or adjacent_pair_count <= 0:
        return 0.0, int(executed_action_count), 0

    denominator = max(int(executed_action_count) - 1, 1)
    adjacent_pair_count = min(max(int(adjacent_pair_count), 0), denominator)
    return float(adjacent_pair_count) / float(denominator), int(executed_action_count), adjacent_pair_count


def compute_score_webgym_rl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_reward_alpha: float = 0.0,
    format_reward_tau: float = 2.0,
    format_reward_budget_exhausted_penalty: float = 0.15,
    format_reward_gate_by_env_score: bool = False,
    **kwargs,
) -> float | dict[str, float | int]:
    del data_source, solution_str, ground_truth, kwargs
    if not isinstance(extra_info, dict):
        extra_info = {}
    env_reward = _extract_env_reward(extra_info)
    env_reward = 0.0 if env_reward is None else env_reward
    raw_format_reward, attempted_tool_calls, valid_tool_calls, first_valid_tool_call_index = _compute_format_reward(
        extra_info,
        tau=format_reward_tau,
        budget_exhausted_penalty=format_reward_budget_exhausted_penalty,
    )
    non_grounding_adjacency_ratio, executed_action_count, non_grounding_adjacent_pair_count = (
        _compute_non_grounding_adjacency_ratio(extra_info)
    )
    format_reward = raw_format_reward
    if raw_format_reward > 0.0:
        format_reward = (1.0 - non_grounding_adjacency_ratio) * raw_format_reward
    if format_reward_gate_by_env_score and env_reward <= 0.0:
        format_reward = 0.0

    final_reward = env_reward + float(format_reward_alpha) * float(format_reward)
    return {
        "score": final_reward,
        "web_osgym_env_reward_score": env_reward,
        "web_osgym_format_reward": float(format_reward),
        "web_osgym_raw_format_reward": float(raw_format_reward),
        "web_osgym_attempted_tool_calls": attempted_tool_calls,
        "web_osgym_first_valid_tool_call_index": first_valid_tool_call_index,
        "web_osgym_valid_tool_calls": valid_tool_calls,
        "web_osgym_executed_action_count": executed_action_count,
        "web_osgym_non_grounding_adjacent_pair_count": non_grounding_adjacent_pair_count,
        "web_osgym_non_grounding_adjacency_ratio": float(non_grounding_adjacency_ratio),
    }
