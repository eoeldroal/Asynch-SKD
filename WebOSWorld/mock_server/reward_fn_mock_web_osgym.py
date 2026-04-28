"""Minimal reward function for mock Web/OSGym trainer smoke runs."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_REWARD_DEBUG = os.getenv("VERL_MOCK_WEB_REWARD_DEBUG", "0") == "1"


def compute_score_mock_web_osgym(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Score whether the rollout emitted the expected terminal mock action.

    The mock Web/OSGym run is a wire-contract smoke test, not a task-quality
    benchmark. Reward is therefore intentionally simple: DONE satisfies the
    default mock ground truth, FAIL and non-terminal outputs do not.
    """

    expected = str(ground_truth or "done").strip().upper()
    output = str(solution_str or "").upper()
    if expected == "DONE":
        score = 1.0 if "DONE" in output and "FAIL" not in output else 0.0
    else:
        score = 1.0 if expected and expected in output else 0.0

    if _REWARD_DEBUG:
        extra_info = extra_info if isinstance(extra_info, dict) else {}
        logger.warning(
            "MOCK_WEB_REWARD data_source=%s task_id=%s expected=%s score=%s output_chars=%s",
            data_source,
            extra_info.get("task_id"),
            expected,
            score,
            len(solution_str or ""),
        )
    return score
