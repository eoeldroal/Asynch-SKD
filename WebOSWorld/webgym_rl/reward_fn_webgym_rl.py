"""Reward function for webgym-rl runs."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    field_validator,
)

from verl import DataProto

logger = logging.getLogger(__name__)

_DEFAULT_LLM_JUDGE_MODEL = "gpt-5.4-mini"
_DEFAULT_LLM_JUDGE_REASONING_EFFORT = "medium"
_DEFAULT_LLM_JUDGE_IMAGE_DETAIL = "auto"
_DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS = 300.0
_DEFAULT_LLM_JUDGE_MAX_OUTPUT_TOKENS = 100000
_COMPARE_SCREENSHOT_COUNT = 5

COMPARATIVE_JUDGE_SYSTEM_PROMPT = """\
You compare candidate trajectories for the same task.

Use only the evidence inside each candidate block.
Do not mix screenshots or turns across candidates.

Assign integer ranks where 1 is best and larger numbers are worse.
Ties are allowed.
Assign different ranks only when there is clear task-relevant evidence of different progress toward completing the task instruction.
If the evidence is ambiguous, doubtful, or not clearly distinguishable, give the candidates the same rank.
Use dense ranks without gaps.

Return only the structured output.
"""

COMPARATIVE_JUDGE_USER_PROMPT_TEMPLATE = """\
Task instruction:
{task_instruction}

Judge standards:
{judge_standards}

Ranking rule:
- Rank by meaningful progress toward completing the task instruction.
- Meaningful progress means observable progress toward the final state that would satisfy the task and earn the environment reward.
- Use only the evidence inside each candidate block.
- Assign different ranks only when the difference in task-relevant progress is clear.
- If the evidence is ambiguous, uncertain, or not clearly distinguishable, prefer a tie.
- Do not reward generic activity or repeated attempts unless they create task-relevant progress.
- If candidates are not meaningfully distinguishable in task completion progress, assign the same rank.

Candidates:

{candidate_blocks}
"""

COMPARATIVE_JUDGE_CANDIDATE_BLOCK_TEMPLATE = """\
Candidate {label}
Trajectory turns:
{trajectory_turns}

Attached screenshots:
{screenshot_lines}
"""

COMPARATIVE_JUDGE_SCHEMA_NAME = "webgym_zero_group_compare"

_OPENAI_CLIENT = None
_ASYNC_OPENAI_CLIENT = None


class JudgeStandardItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("id", "text")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class WebGymRewardExtraInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    web_osgym_env_reward_score: float = 0.0
    web_osgym_format_reward: float = 0.0
    web_osgym_raw_format_reward: float = 0.0
    web_osgym_attempted_tool_calls: int = 0
    web_osgym_first_valid_tool_call_index: int = 0
    web_osgym_valid_tool_calls: int = 0
    web_osgym_executed_action_count: int = 0
    web_osgym_non_grounding_adjacent_pair_count: int = 0
    web_osgym_non_grounding_adjacency_ratio: float = 0.0
    web_osgym_llm_judge_used: bool = False
    web_osgym_llm_judge_score: float = 0.0
    web_osgym_llm_judge_rank: int | None = None
    web_osgym_llm_judge_reason: str = ""

    @field_validator("web_osgym_llm_judge_reason")
    @classmethod
    def _normalize_reason(cls, value: str) -> str:
        return value.strip()


class WebGymRewardExtraInfoPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    web_osgym_env_reward_score: float | None = None
    web_osgym_format_reward: float | None = None
    web_osgym_raw_format_reward: float | None = None
    web_osgym_attempted_tool_calls: int | None = None
    web_osgym_first_valid_tool_call_index: int | None = None
    web_osgym_valid_tool_calls: int | None = None
    web_osgym_executed_action_count: int | None = None
    web_osgym_non_grounding_adjacent_pair_count: int | None = None
    web_osgym_non_grounding_adjacency_ratio: float | None = None
    web_osgym_llm_judge_used: bool | None = None
    web_osgym_llm_judge_score: float | None = None
    web_osgym_llm_judge_rank: int | None = None
    web_osgym_llm_judge_reason: str | None = None

    @field_validator("web_osgym_llm_judge_reason")
    @classmethod
    def _normalize_optional_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()


class CompareScreenshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_idx: int = Field(ge=0)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _strip_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ZeroGroupCompareCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    trajectory_turns: str
    screenshots: list[CompareScreenshot] = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("must not be blank")
        return value


class ZeroGroupCompareExtraInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    judge_standard: list[JudgeStandardItem] = Field(min_length=1)
    web_osgym_trajectory_dir: str = Field(min_length=1)
    task_instruction: str | None = Field(
        default=None,
        validation_alias=AliasChoices("task_instruction", "instruction", "task_name"),
    )
    web_osgym_env_reward_score: float | None = None

    @field_validator("web_osgym_trajectory_dir")
    @classmethod
    def _strip_trajectory_dir(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("task_instruction")
    @classmethod
    def _strip_optional_task_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


_WEBGYM_REWARD_EXTRA_INFO_ADAPTER = TypeAdapter(WebGymRewardExtraInfo)
_WEBGYM_REWARD_EXTRA_INFO_PATCH_ADAPTER = TypeAdapter(WebGymRewardExtraInfoPatch)
_ZERO_GROUP_EXTRA_INFOS_ADAPTER = TypeAdapter(list[ZeroGroupCompareExtraInfo])
_COMPARE_CANDIDATES_ADAPTER = TypeAdapter(list[ZeroGroupCompareCandidate])
_JUDGE_STANDARD_ADAPTER = TypeAdapter(list[JudgeStandardItem])

JudgeStandardItem.model_rebuild()
WebGymRewardExtraInfo.model_rebuild()
WebGymRewardExtraInfoPatch.model_rebuild()
CompareScreenshot.model_rebuild()
ZeroGroupCompareCandidate.model_rebuild()
ZeroGroupCompareExtraInfo.model_rebuild()
_WEBGYM_REWARD_EXTRA_INFO_ADAPTER.rebuild()
_WEBGYM_REWARD_EXTRA_INFO_PATCH_ADAPTER.rebuild()
_ZERO_GROUP_EXTRA_INFOS_ADAPTER.rebuild()
_COMPARE_CANDIDATES_ADAPTER.rebuild()
_JUDGE_STANDARD_ADAPTER.rebuild()


def build_compare_labels(count: int) -> tuple[str, ...]:
    if int(count) <= 0:
        raise ValueError("count must be positive")

    labels: list[str] = []
    for index in range(int(count)):
        value = index + 1
        chars: list[str] = []
        while value > 0:
            value, remainder = divmod(value - 1, 26)
            chars.append(chr(ord("A") + remainder))
        labels.append("".join(reversed(chars)))
    return tuple(labels)


@lru_cache(maxsize=None)
def build_zero_group_compare_schema(labels: tuple[str, ...]) -> dict[str, Any]:
    max_rank = len(labels)
    rank_keys = [f"{label}_rank" for label in labels]
    properties = {
        rank_key: {"type": "integer", "minimum": 1, "maximum": max_rank}
        for rank_key in rank_keys
    }
    properties["reason"] = {"type": "string"}
    return {
        "type": "json_schema",
        "name": COMPARATIVE_JUDGE_SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": [*rank_keys, "reason"],
        },
    }


@lru_cache(maxsize=None)
def _build_zero_group_compare_response_model(labels: tuple[str, ...]) -> type[BaseModel]:
    max_rank = len(labels)
    fields: dict[str, tuple[Any, Any]] = {
        f"{label}_rank": (int, Field(..., ge=1, le=max_rank))
        for label in labels
    }
    fields["reason"] = (str, ...)
    model = create_model(
        f"ZeroGroupCompareResponse_{'_'.join(labels)}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    model.model_rebuild()
    return model


def validate_zero_group_compare_response(
    response_json: dict[str, Any],
    *,
    labels: Sequence[str],
) -> BaseModel:
    normalized_labels = tuple(str(label).strip().upper() for label in labels)
    if not normalized_labels:
        raise ValueError("labels must not be empty")
    response_model = _build_zero_group_compare_response_model(normalized_labels)
    return response_model.model_validate(response_json)


def validate_zero_group_compare_extra_infos(
    extra_infos: list[dict[str, Any]],
) -> list[ZeroGroupCompareExtraInfo]:
    return _ZERO_GROUP_EXTRA_INFOS_ADAPTER.validate_python(extra_infos)


def validate_webgym_reward_extra_info(extra_info: dict[str, Any] | WebGymRewardExtraInfo) -> WebGymRewardExtraInfo:
    if isinstance(extra_info, WebGymRewardExtraInfo):
        return extra_info
    return _WEBGYM_REWARD_EXTRA_INFO_ADAPTER.validate_python(extra_info)


def validate_webgym_reward_extra_info_patch(
    extra_info: dict[str, Any] | WebGymRewardExtraInfoPatch,
) -> WebGymRewardExtraInfoPatch:
    if isinstance(extra_info, WebGymRewardExtraInfoPatch):
        return extra_info
    return _WEBGYM_REWARD_EXTRA_INFO_PATCH_ADAPTER.validate_python(extra_info)


def merge_webgym_reward_extra_info(
    base: dict[str, Any] | WebGymRewardExtraInfo,
    patch: dict[str, Any] | WebGymRewardExtraInfoPatch,
) -> dict[str, Any]:
    base_model = validate_webgym_reward_extra_info(base)
    patch_model = validate_webgym_reward_extra_info_patch(patch)

    merged = base_model.model_dump()
    for field_name in patch_model.model_fields_set:
        merged[field_name] = getattr(patch_model, field_name)
    return validate_webgym_reward_extra_info(merged).model_dump()


def pack_webgym_reward_extra_infos(
    reward_extra_infos: Sequence[dict[str, Any] | WebGymRewardExtraInfo],
    *,
    template_non_tensor_batch: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    normalized_rows = [validate_webgym_reward_extra_info(item).model_dump() for item in reward_extra_infos]
    reward_extra_keys = list(WebGymRewardExtraInfo.model_fields.keys())
    non_tensor_batch: dict[str, np.ndarray] = {}
    template_non_tensor_batch = template_non_tensor_batch or {}

    for key in reward_extra_keys:
        existing_value = template_non_tensor_batch.get(key)
        target_dtype = existing_value.dtype if isinstance(existing_value, np.ndarray) else object
        non_tensor_batch[key] = np.array([row[key] for row in normalized_rows], dtype=target_dtype)
    return non_tensor_batch, reward_extra_keys


def extract_webgym_reward_extra_infos(
    non_tensor_batch: dict[str, Any],
    *,
    reward_extra_keys: Sequence[str] | None = None,
) -> list[WebGymRewardExtraInfo]:
    reward_extra_keys = tuple(reward_extra_keys or WebGymRewardExtraInfo.model_fields.keys())
    batch_size = None
    for key in reward_extra_keys:
        values = non_tensor_batch.get(key)
        if isinstance(values, np.ndarray):
            batch_size = len(values)
            break
    if batch_size is None:
        return []

    rows: list[WebGymRewardExtraInfo] = []
    for index in range(batch_size):
        row: dict[str, Any] = {}
        for key in reward_extra_keys:
            values = non_tensor_batch.get(key)
            if values is None:
                continue
            if not isinstance(values, np.ndarray):
                raise TypeError(f"Reward extra info column {key} must be a numpy array")
            if len(values) != batch_size:
                raise ValueError(f"Reward extra info column {key} length mismatch: expected {batch_size}, got {len(values)}")
            row[key] = values[index]
        rows.append(validate_webgym_reward_extra_info(row))
    return rows


def extract_webgym_reward_extra_infos_from_batch(batch: DataProto) -> list[WebGymRewardExtraInfo]:
    reward_extra_keys = batch.meta_info.get("reward_extra_keys")
    if not isinstance(reward_extra_keys, Sequence) or isinstance(reward_extra_keys, (str, bytes)):
        reward_extra_keys = list(WebGymRewardExtraInfo.model_fields.keys())
    return extract_webgym_reward_extra_infos(batch.non_tensor_batch, reward_extra_keys=list(reward_extra_keys))


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


def _normalize_optional_judge_standard(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        return [item.model_dump() for item in _JUDGE_STANDARD_ADAPTER.validate_python(value)]
    except ValidationError:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _format_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or action.get("name") or "UNKNOWN").upper()
    arguments = [f"{key}={value!r}" for key, value in action.items() if key not in {"action_type", "name"}]
    return f"{action_type}({', '.join(arguments)})" if arguments else f"{action_type}()"


def _turn_result_passed(event: dict[str, Any]) -> bool:
    if event.get("parse_error") is not None:
        return False
    result = event.get("result")
    if not isinstance(result, dict):
        return False
    if bool(result.get("invalid_action")):
        return False
    action_count = _as_int(result.get("action_count"))
    return action_count is not None and action_count > 0


def _render_turns(events: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for event in events:
        if event.get("event_type") != "assistant_turn":
            continue
        turn_idx = _as_int(event.get("assistant_turn"))
        if turn_idx is None or turn_idx <= 0:
            continue
        lines = [f"Turn {turn_idx}: result_passed={'true' if _turn_result_passed(event) else 'false'}"]
        if event.get("parse_error") is not None:
            lines.append("  Parse status: parse_error")
        elif bool((event.get("result") or {}).get("invalid_action")):
            lines.append("  Execution status: invalid_action")
        actions = _normalize_actions(event.get("actions"))
        if actions:
            for index, action in enumerate(actions, start=1):
                lines.append(f"  {index}. {_format_action(action)}")
        else:
            lines.append("  actions: none")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "No assistant turns."


def _select_compare_images(events: list[dict[str, Any]], *, trajectory_dir: Path) -> list[CompareScreenshot]:
    initial_event = next(
        (
            event
            for event in events
            if event.get("event_type") == "initial_observation"
            and isinstance(event.get("image_paths"), list)
            and event["image_paths"]
        ),
        None,
    )
    if initial_event is None:
        raise ValueError(f"Initial observation image is missing in {trajectory_dir}")

    assistant_events = [
        event
        for event in events
        if event.get("event_type") == "assistant_turn"
        and isinstance(event.get("image_paths"), list)
        and event["image_paths"]
    ]
    def _to_image_item(event: dict[str, Any], *, fallback_turn_idx: int) -> CompareScreenshot:
        paths = event.get("image_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Image paths are missing in {trajectory_dir}")
        return CompareScreenshot(
            turn_idx=_as_int(event.get("assistant_turn")) or fallback_turn_idx,
            path=str(trajectory_dir / str(paths[0])),
        )

    if assistant_events:
        selected_events = assistant_events[-_COMPARE_SCREENSHOT_COUNT:]
        return [_to_image_item(event, fallback_turn_idx=0) for event in selected_events]

    return [_to_image_item(initial_event, fallback_turn_idx=0)]


def _extract_task_instruction(extra_info: ZeroGroupCompareExtraInfo, task_instruction: str | None) -> str:
    if isinstance(task_instruction, str) and task_instruction.strip():
        return task_instruction.strip()
    if isinstance(extra_info.task_instruction, str) and extra_info.task_instruction.strip():
        return extra_info.task_instruction.strip()
    return "Complete the requested task."


def _path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_compare_candidate(*, label: str, trajectory_dir: Path) -> ZeroGroupCompareCandidate:
    events = _load_jsonl(trajectory_dir / "trajectory.jsonl")
    screenshots = _select_compare_images(events, trajectory_dir=trajectory_dir)
    return ZeroGroupCompareCandidate(
        label=label,
        trajectory_turns=_render_turns(events),
        screenshots=screenshots,
    )


def build_zero_group_compare_request(
    *,
    task_instruction: str,
    judge_standard: list[dict[str, str]],
    candidates: list[ZeroGroupCompareCandidate] | list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    image_detail: str,
    max_output_tokens: int = _DEFAULT_LLM_JUDGE_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    normalized_judge_standard = _JUDGE_STANDARD_ADAPTER.validate_python(judge_standard)
    normalized_candidates = _COMPARE_CANDIDATES_ADAPTER.validate_python(candidates)
    labels = tuple(candidate.label for candidate in normalized_candidates)
    if len(set(labels)) != len(labels):
        raise ValueError("Candidate labels must be unique")
    judge_standards = "\n".join(f"- {item.id}: {item.text}" for item in normalized_judge_standard)
    candidate_blocks = "\n\n".join(
        COMPARATIVE_JUDGE_CANDIDATE_BLOCK_TEMPLATE.format(
            label=candidate.label,
            trajectory_turns=candidate.trajectory_turns,
            screenshot_lines="\n".join(
                (
                    f"- Candidate {candidate.label} screenshot after assistant turn {screenshot.turn_idx}"
                    if screenshot.turn_idx > 0
                    else f"- Candidate {candidate.label} initial screenshot"
                )
                for screenshot in candidate.screenshots
            ),
        )
        for candidate in normalized_candidates
    )
    user_text = COMPARATIVE_JUDGE_USER_PROMPT_TEMPLATE.format(
        task_instruction=task_instruction,
        judge_standards=judge_standards,
        candidate_blocks=candidate_blocks,
    )

    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_text}]
    for candidate in normalized_candidates:
        for screenshot in candidate.screenshots:
            turn_idx = screenshot.turn_idx
            caption = (
                f"Candidate {candidate.label} screenshot after assistant turn {turn_idx}."
                if turn_idx > 0
                else f"Candidate {candidate.label} initial screenshot."
            )
            content.append({"type": "input_text", "text": caption})
            content.append(
                {
                    "type": "input_image",
                    "image_url": _path_to_data_url(Path(screenshot.path)),
                    "detail": image_detail,
                }
            )

    return {
        "model": model,
        "store": False,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": int(max_output_tokens),
        "input": [
            {"role": "system", "content": COMPARATIVE_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "text": {"format": build_zero_group_compare_schema(labels)},
    }


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _OPENAI_CLIENT


def _get_async_openai_client():
    global _ASYNC_OPENAI_CLIENT
    if _ASYNC_OPENAI_CLIENT is None:
        from openai import AsyncOpenAI

        _ASYNC_OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _ASYNC_OPENAI_CLIENT


def _extract_response_json(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return json.loads(output_text)

    if hasattr(response, "model_dump"):
        response = response.model_dump()

    if isinstance(response, dict):
        for output in response.get("output", []):
            if not isinstance(output, dict):
                continue
            for content in output.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return json.loads(text)
    raise ValueError("Could not extract structured JSON from comparative judge response")


def _normalize_dense_ranks(raw_ranks: list[int]) -> list[int]:
    unique_ranks = sorted(set(int(rank) for rank in raw_ranks))
    remap = {rank: index + 1 for index, rank in enumerate(unique_ranks)}
    return [remap[int(rank)] for rank in raw_ranks]


def _ranks_to_scores(ranks: list[int]) -> list[float]:
    unique_ranks = sorted(set(ranks))
    if len(unique_ranks) <= 1:
        return [0.0 for _ in ranks]
    max_index = len(unique_ranks) - 1
    rank_to_score = {
        rank: float(max_index - index) / float(max_index)
        for index, rank in enumerate(unique_ranks)
    }
    return [rank_to_score[rank] for rank in ranks]


def _should_log_llm_judge_timing(model: str) -> bool:
    return str(model).startswith("gpt-5.4")


def _log_llm_judge_timing(*, event: str, model: str, candidate_count: int, elapsed_seconds: float | None = None) -> None:
    if not _should_log_llm_judge_timing(model):
        return
    if elapsed_seconds is None:
        print(f"[WebGymReward][LLMJudge] event={event} model={model} candidates={candidate_count}")
        return
    print(
        f"[WebGymReward][LLMJudge] event={event} model={model} "
        f"candidates={candidate_count} elapsed_seconds={elapsed_seconds:.2f}"
    )


def _normalize_zero_group_compare_extra_infos(
    extra_infos: list[dict[str, Any]] | list[ZeroGroupCompareExtraInfo],
) -> list[ZeroGroupCompareExtraInfo]:
    if extra_infos and all(isinstance(item, ZeroGroupCompareExtraInfo) for item in extra_infos):
        return list(extra_infos)
    return _ZERO_GROUP_EXTRA_INFOS_ADAPTER.validate_python(extra_infos)


def _prepare_zero_group_compare_request(
    *,
    extra_infos: list[dict[str, Any]] | list[ZeroGroupCompareExtraInfo],
    task_instructions: list[str | None] | None,
    llm_judge_model: str,
    llm_judge_reasoning_effort: str,
    llm_judge_image_detail: str,
) -> tuple[tuple[str, ...], list[ZeroGroupCompareExtraInfo], dict[str, Any]]:
    normalized_extra_infos = _normalize_zero_group_compare_extra_infos(extra_infos)
    labels = build_compare_labels(len(normalized_extra_infos))
    judge_standard = [item.model_dump() for item in normalized_extra_infos[0].judge_standard]

    candidate_instructions = task_instructions or [None] * len(normalized_extra_infos)
    task_instruction = "Complete the requested task."
    for extra_info_model, candidate_instruction in zip(normalized_extra_infos, candidate_instructions, strict=True):
        resolved_instruction = _extract_task_instruction(extra_info_model, candidate_instruction)
        if resolved_instruction.strip():
            task_instruction = resolved_instruction
            break

    candidates: list[ZeroGroupCompareCandidate] = []
    for label, extra_info in zip(labels, normalized_extra_infos, strict=True):
        candidates.append(_build_compare_candidate(label=label, trajectory_dir=Path(extra_info.web_osgym_trajectory_dir)))

    request = build_zero_group_compare_request(
        task_instruction=task_instruction,
        judge_standard=judge_standard,
        candidates=candidates,
        model=llm_judge_model,
        reasoning_effort=llm_judge_reasoning_effort,
        image_detail=llm_judge_image_detail,
    )
    return labels, normalized_extra_infos, request


def _success_zero_group_compare_outputs(
    *,
    labels: tuple[str, ...],
    normalized_extra_infos: list[ZeroGroupCompareExtraInfo],
    parsed_response: BaseModel,
) -> list[dict[str, Any]]:
    raw_ranks = [int(getattr(parsed_response, f"{label}_rank")) for label in labels]
    ranks = _normalize_dense_ranks(raw_ranks)
    scores = _ranks_to_scores(ranks)
    reason = str(getattr(parsed_response, "reason", "")).strip()
    return [
        {
            "reward_score": float(score),
            "reward_extra_info": validate_webgym_reward_extra_info_patch(
                {
                    "web_osgym_env_reward_score": float(extra_info.web_osgym_env_reward_score or 0.0),
                    "web_osgym_llm_judge_used": True,
                    "web_osgym_llm_judge_score": float(score),
                    "web_osgym_llm_judge_rank": int(rank),
                    "web_osgym_llm_judge_reason": reason,
                }
            ).model_dump(exclude_unset=True),
        }
        for extra_info, score, rank in zip(normalized_extra_infos, scores, ranks, strict=True)
    ]


def _fallback_zero_group_compare_outputs(
    normalized_extra_infos: list[ZeroGroupCompareExtraInfo],
) -> list[dict[str, Any]]:
    return [
        {
            "reward_score": 0.0,
            "reward_extra_info": validate_webgym_reward_extra_info_patch(
                {
                    "web_osgym_env_reward_score": float(extra_info.web_osgym_env_reward_score or 0.0),
                    "web_osgym_llm_judge_used": False,
                    "web_osgym_llm_judge_score": 0.0,
                    "web_osgym_llm_judge_rank": None,
                    "web_osgym_llm_judge_reason": "",
                }
            ).model_dump(exclude_unset=True),
        }
        for extra_info in normalized_extra_infos
    ]


def compare_zero_group_webgym_rl(
    *,
    extra_infos: list[dict[str, Any]] | list[ZeroGroupCompareExtraInfo],
    task_instructions: list[str | None] | None = None,
    llm_judge_model: str = _DEFAULT_LLM_JUDGE_MODEL,
    llm_judge_reasoning_effort: str = _DEFAULT_LLM_JUDGE_REASONING_EFFORT,
    llm_judge_image_detail: str = _DEFAULT_LLM_JUDGE_IMAGE_DETAIL,
    llm_judge_timeout_seconds: float = _DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS,
    **kwargs,
) -> list[dict[str, Any]]:
    del kwargs
    labels, normalized_extra_infos, request = _prepare_zero_group_compare_request(
        extra_infos=extra_infos,
        task_instructions=task_instructions,
        llm_judge_model=llm_judge_model,
        llm_judge_reasoning_effort=llm_judge_reasoning_effort,
        llm_judge_image_detail=llm_judge_image_detail,
    )

    try:
        started_at = time.monotonic()
        _log_llm_judge_timing(event="start", model=llm_judge_model, candidate_count=len(labels))
        client = _get_openai_client()
        if llm_judge_timeout_seconds is not None:
            client = client.with_options(timeout=float(llm_judge_timeout_seconds))
        response = client.responses.create(**request)
        parsed = validate_zero_group_compare_response(_extract_response_json(response), labels=labels)
        _log_llm_judge_timing(
            event="done",
            model=llm_judge_model,
            candidate_count=len(labels),
            elapsed_seconds=time.monotonic() - started_at,
        )
        return _success_zero_group_compare_outputs(
            labels=labels,
            normalized_extra_infos=normalized_extra_infos,
            parsed_response=parsed,
        )
    except Exception as exc:
        logger.warning("[WebGymReward][ZeroGroupCompareError] error=%s", exc, exc_info=True)
        return _fallback_zero_group_compare_outputs(normalized_extra_infos)


async def compare_zero_group_webgym_rl_async(
    *,
    extra_infos: list[dict[str, Any]] | list[ZeroGroupCompareExtraInfo],
    task_instructions: list[str | None] | None = None,
    llm_judge_model: str = _DEFAULT_LLM_JUDGE_MODEL,
    llm_judge_reasoning_effort: str = _DEFAULT_LLM_JUDGE_REASONING_EFFORT,
    llm_judge_image_detail: str = _DEFAULT_LLM_JUDGE_IMAGE_DETAIL,
    llm_judge_timeout_seconds: float = _DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS,
    **kwargs,
) -> list[dict[str, Any]]:
    del kwargs
    labels, normalized_extra_infos, request = _prepare_zero_group_compare_request(
        extra_infos=extra_infos,
        task_instructions=task_instructions,
        llm_judge_model=llm_judge_model,
        llm_judge_reasoning_effort=llm_judge_reasoning_effort,
        llm_judge_image_detail=llm_judge_image_detail,
    )

    try:
        started_at = time.monotonic()
        _log_llm_judge_timing(event="start", model=llm_judge_model, candidate_count=len(labels))
        client = _get_async_openai_client()
        if llm_judge_timeout_seconds is not None:
            client = client.with_options(timeout=float(llm_judge_timeout_seconds))
        response = await client.responses.create(**request)
        parsed = validate_zero_group_compare_response(_extract_response_json(response), labels=labels)
        _log_llm_judge_timing(
            event="done",
            model=llm_judge_model,
            candidate_count=len(labels),
            elapsed_seconds=time.monotonic() - started_at,
        )
        return _success_zero_group_compare_outputs(
            labels=labels,
            normalized_extra_infos=normalized_extra_infos,
            parsed_response=parsed,
        )
    except Exception as exc:
        logger.warning("[WebGymReward][ZeroGroupCompareAsyncError] error=%s", exc, exc_info=True)
        return _fallback_zero_group_compare_outputs(normalized_extra_infos)


def compute_score_webgym_rl(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    format_reward_alpha: float = 0.0,
    format_reward_tau: float = 2.0,
    format_reward_budget_exhausted_penalty: float = 0.15,
    format_reward_gate_by_env_score: bool = False,
    llm_judge_enable: bool = False,
    llm_judge_only_zerogroup: bool = False,
    llm_judge_model: str = _DEFAULT_LLM_JUDGE_MODEL,
    llm_judge_reasoning_effort: str = _DEFAULT_LLM_JUDGE_REASONING_EFFORT,
    llm_judge_image_detail: str = _DEFAULT_LLM_JUDGE_IMAGE_DETAIL,
    llm_judge_timeout_seconds: float = _DEFAULT_LLM_JUDGE_TIMEOUT_SECONDS,
    **kwargs,
) -> float | dict[str, float | int | bool | None | str]:
    del data_source
    del solution_str
    del ground_truth
    del llm_judge_enable
    del llm_judge_only_zerogroup
    del llm_judge_model
    del llm_judge_reasoning_effort
    del llm_judge_image_detail
    del llm_judge_timeout_seconds
    del kwargs

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
    reward_extra_info = validate_webgym_reward_extra_info(
        {
            "web_osgym_env_reward_score": env_reward,
            "web_osgym_format_reward": float(format_reward),
            "web_osgym_raw_format_reward": float(raw_format_reward),
            "web_osgym_attempted_tool_calls": attempted_tool_calls,
            "web_osgym_first_valid_tool_call_index": first_valid_tool_call_index,
            "web_osgym_valid_tool_calls": valid_tool_calls,
            "web_osgym_executed_action_count": executed_action_count,
            "web_osgym_non_grounding_adjacent_pair_count": non_grounding_adjacent_pair_count,
            "web_osgym_non_grounding_adjacency_ratio": float(non_grounding_adjacency_ratio),
            "web_osgym_llm_judge_used": False,
            "web_osgym_llm_judge_score": 0.0,
            "web_osgym_llm_judge_rank": None,
            "web_osgym_llm_judge_reason": "",
        }
    ).model_dump()
    return {
        "score": final_reward,
        **reward_extra_info,
    }
