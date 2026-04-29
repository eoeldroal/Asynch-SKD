"""Reward fallback for webgym-rl runs.

The web_skd_agent normally writes the environment reward from webgym-rl into
``rm_scores`` before the reward manager calls this function. This function keeps
the configuration explicit and provides a conservative fallback if a caller
passes the environment reward through metadata instead.
"""

from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_rollout_reward(extra_info: dict[str, Any]) -> float | None:
    for key in ("web_osgym_reward_score", "env_reward", "reward"):
        score = _as_float(extra_info.get(key))
        if score is not None:
            return score

    rollout_scores = extra_info.get("rollout_reward_scores")
    if isinstance(rollout_scores, dict):
        for key in ("web_osgym_reward_score", "env_reward", "reward", "computer"):
            score = _as_float(rollout_scores.get(key))
            if score is not None:
                return score
    return None


def compute_score_webgym_rl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    del data_source, solution_str, ground_truth, kwargs
    if not isinstance(extra_info, dict):
        return 0.0
    score = _extract_rollout_reward(extra_info)
    return 0.0 if score is None else score

