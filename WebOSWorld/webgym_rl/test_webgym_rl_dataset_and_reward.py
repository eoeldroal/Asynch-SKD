import json
from pathlib import Path

import numpy as np
import openai
import pytest
from pydantic import ValidationError

try:
    from hypothesis import given, settings, strategies as st
except ImportError:  # pragma: no cover - optional test dependency
    given = settings = st = None

from WebOSWorld.webgym_rl.create_webgym_rl_dataset import build_rows
from WebOSWorld.webgym_rl import reward_fn_webgym_rl
from WebOSWorld.webgym_rl.reward_fn_webgym_rl import compute_score_webgym_rl


def _write_candidate_trajectory(
    root: Path,
    name: str,
    *,
    include_parse_error: bool = False,
    include_invalid_action: bool = False,
) -> Path:
    trajectory_dir = root / name
    image_dir = trajectory_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for image_name in ["init.png", "turn1.png", "turn2.png", "turn3.png", "turn4.png", "turn5.png", "turn6.png"]:
        (image_dir / image_name).write_bytes(f"fake-{image_name}".encode("utf-8"))

    events = [
        {"event_type": "initial_observation", "image_paths": ["images/init.png"]},
        {
            "event_type": "assistant_turn",
            "assistant_turn": 1,
            "actions": [{"action_type": "CLICK", "x": 10, "y": 20}],
            "result": {"invalid_action": False, "action_count": 1},
            "parse_error": None,
            "image_paths": ["images/turn1.png"],
        },
        {
            "event_type": "assistant_turn",
            "assistant_turn": 2,
            "actions": [] if include_parse_error else [{"action_type": "DOUBLE_CLICK", "x": 30, "y": 40}],
            "result": {"invalid_action": False, "action_count": 0 if include_parse_error else 1},
            "parse_error": {"kind": "tool_parse_error"} if include_parse_error else None,
            "image_paths": ["images/turn2.png"],
        },
        {
            "event_type": "assistant_turn",
            "assistant_turn": 3,
            "actions": [{"action_type": "CLICK", "x": 50, "y": 60}],
            "result": {"invalid_action": include_invalid_action, "action_count": 0 if include_invalid_action else 1},
            "parse_error": None,
            "image_paths": ["images/turn3.png"],
        },
        {
            "event_type": "assistant_turn",
            "assistant_turn": 4,
            "actions": [{"action_type": "CLICK", "x": 70, "y": 80}],
            "result": {"invalid_action": False, "action_count": 1},
            "parse_error": None,
            "image_paths": ["images/turn4.png"],
        },
        {
            "event_type": "assistant_turn",
            "assistant_turn": 5,
            "actions": [{"action_type": "CLICK", "x": 90, "y": 100}],
            "result": {"invalid_action": False, "action_count": 1},
            "parse_error": None,
            "image_paths": ["images/turn5.png"],
        },
        {
            "event_type": "assistant_turn",
            "assistant_turn": 6,
            "actions": [{"action_type": "CLICK", "x": 110, "y": 120}],
            "result": {"invalid_action": False, "action_count": 1},
            "parse_error": None,
            "image_paths": ["images/turn6.png"],
        },
    ]
    (trajectory_dir / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return trajectory_dir


class _FakeJudgeResponse:
    def __init__(self, output_text: str):
        self.output_text = output_text


class _FakeJudgeResponsesAPI:
    def __init__(self, requests: list[dict], output_text: str):
        self._requests = requests
        self._output_text = output_text

    def create(self, **request):
        self._requests.append(request)
        return _FakeJudgeResponse(self._output_text)


class _FakeJudgeClientWithOptions:
    def __init__(self, requests: list[dict], timeouts: list[float], timeout: float, output_text: str):
        self.responses = _FakeJudgeResponsesAPI(requests, output_text)
        self._timeouts = timeouts
        self._timeouts.append(timeout)


class _FakeJudgeClient:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        return _FakeJudgeClientWithOptions(self.requests, self.timeouts, timeout, self.output_text)


class _FakeAsyncJudgeResponsesAPI:
    def __init__(self, requests: list[dict], output_text: str):
        self._requests = requests
        self._output_text = output_text

    async def create(self, **request):
        self._requests.append(request)
        return _FakeJudgeResponse(self._output_text)


class _FakeAsyncJudgeClientWithOptions:
    def __init__(self, requests: list[dict], timeouts: list[float], timeout: float, output_text: str):
        self.responses = _FakeAsyncJudgeResponsesAPI(requests, output_text)
        self._timeouts = timeouts
        self._timeouts.append(timeout)


class _FakeAsyncJudgeClient:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.requests: list[dict] = []
        self.timeouts: list[float] = []

    def with_options(self, *, timeout: float):
        return _FakeAsyncJudgeClientWithOptions(self.requests, self.timeouts, timeout, self.output_text)


@pytest.fixture(autouse=True)
def _block_live_openai_client(monkeypatch):
    def _raise_unexpected_openai(*args, **kwargs):
        raise AssertionError("Unexpected live OpenAI client construction in test")

    monkeypatch.setattr(openai, "OpenAI", _raise_unexpected_openai)
    monkeypatch.setattr(openai, "AsyncOpenAI", _raise_unexpected_openai)
    monkeypatch.setattr(reward_fn_webgym_rl, "_OPENAI_CLIENT", None)
    monkeypatch.setattr(reward_fn_webgym_rl, "_ASYNC_OPENAI_CLIENT", None)


HYPOTHESIS_AVAILABLE = st is not None

if HYPOTHESIS_AVAILABLE:
    _REWARD_EXTRA_FLOATS = st.floats(allow_nan=False, allow_infinity=False, width=32)
    _REWARD_EXTRA_INTS = st.integers(min_value=0, max_value=100)
    _REWARD_EXTRA_REASON = st.text(max_size=32)

    @st.composite
    def _reward_extra_info_strategy(draw):
        return {
            "web_osgym_env_reward_score": draw(_REWARD_EXTRA_FLOATS),
            "web_osgym_format_reward": draw(_REWARD_EXTRA_FLOATS),
            "web_osgym_raw_format_reward": draw(_REWARD_EXTRA_FLOATS),
            "web_osgym_attempted_tool_calls": draw(_REWARD_EXTRA_INTS),
            "web_osgym_first_valid_tool_call_index": draw(_REWARD_EXTRA_INTS),
            "web_osgym_valid_tool_calls": draw(_REWARD_EXTRA_INTS),
            "web_osgym_executed_action_count": draw(_REWARD_EXTRA_INTS),
            "web_osgym_non_grounding_adjacent_pair_count": draw(_REWARD_EXTRA_INTS),
            "web_osgym_non_grounding_adjacency_ratio": draw(_REWARD_EXTRA_FLOATS),
            "web_osgym_llm_judge_used": draw(st.booleans()),
            "web_osgym_llm_judge_score": draw(_REWARD_EXTRA_FLOATS),
            "web_osgym_llm_judge_rank": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=8))),
            "web_osgym_llm_judge_reason": draw(_REWARD_EXTRA_REASON),
        }

    @st.composite
    def _reward_extra_info_patch_strategy(draw):
        patch: dict[str, object] = {}
        maybe = st.booleans()
        if draw(maybe):
            patch["web_osgym_env_reward_score"] = draw(_REWARD_EXTRA_FLOATS)
        if draw(maybe):
            patch["web_osgym_format_reward"] = draw(_REWARD_EXTRA_FLOATS)
        if draw(maybe):
            patch["web_osgym_raw_format_reward"] = draw(_REWARD_EXTRA_FLOATS)
        if draw(maybe):
            patch["web_osgym_attempted_tool_calls"] = draw(_REWARD_EXTRA_INTS)
        if draw(maybe):
            patch["web_osgym_first_valid_tool_call_index"] = draw(_REWARD_EXTRA_INTS)
        if draw(maybe):
            patch["web_osgym_valid_tool_calls"] = draw(_REWARD_EXTRA_INTS)
        if draw(maybe):
            patch["web_osgym_executed_action_count"] = draw(_REWARD_EXTRA_INTS)
        if draw(maybe):
            patch["web_osgym_non_grounding_adjacent_pair_count"] = draw(_REWARD_EXTRA_INTS)
        if draw(maybe):
            patch["web_osgym_non_grounding_adjacency_ratio"] = draw(_REWARD_EXTRA_FLOATS)
        if draw(maybe):
            patch["web_osgym_llm_judge_used"] = draw(st.booleans())
        if draw(maybe):
            patch["web_osgym_llm_judge_score"] = draw(_REWARD_EXTRA_FLOATS)
        if draw(maybe):
            patch["web_osgym_llm_judge_rank"] = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=8)))
        if draw(maybe):
            patch["web_osgym_llm_judge_reason"] = draw(_REWARD_EXTRA_REASON)
        return patch


def test_build_rows_carries_optional_judge_standard():
    tasks = [
        {
            "task_id": "terminal_task",
            "task_name": "Open the terminal.",
            "website": "https://example.test/terminal",
            "judge_standard": [{"id": "terminal_visible", "text": "The terminal is visible."}],
        }
    ]

    row = build_rows(split="train", tasks=tasks, num_samples=1, agent_name="web_tool_agent")[0]

    assert row["judge_standard"] == [{"id": "terminal_visible", "text": "The terminal is visible."}]
    assert row["extra_info"]["judge_standard"] == row["judge_standard"]


def test_compute_score_webgym_rl_returns_env_and_format_fields_without_using_llm_judge():
    score = compute_score_webgym_rl(
        data_source="webgym_rl",
        solution_str="irrelevant",
        ground_truth="env_reward",
        extra_info={
            "web_osgym_env_reward_score": 1.0,
            "web_osgym_attempted_tool_calls": 2,
            "web_osgym_valid_tool_calls": 2,
            "web_osgym_first_valid_tool_call_index": 1,
        },
        llm_judge_enable=True,
        llm_judge_only_zerogroup=True,
    )

    assert score["score"] == 1.0
    assert score["web_osgym_llm_judge_used"] is False
    assert score["web_osgym_llm_judge_score"] == 0.0


def test_build_zero_group_compare_request_renders_full_trajectory_and_candidate_boundaries(tmp_path: Path):
    judge_standard = [
        {"id": "check_1", "text": "The target app is open."},
        {"id": "check_2", "text": "The final state matches the task."},
    ]
    candidates = []
    for label, parse_error, invalid_action in [
        ("A", False, False),
        ("B", True, False),
        ("C", False, True),
        ("D", False, False),
    ]:
        trajectory_dir = _write_candidate_trajectory(
            tmp_path,
            f"candidate_{label}",
            include_parse_error=parse_error,
            include_invalid_action=invalid_action,
        )
        candidates.append(reward_fn_webgym_rl._build_compare_candidate(label=label, trajectory_dir=trajectory_dir))

    request = reward_fn_webgym_rl.build_zero_group_compare_request(
        task_instruction="Complete the requested task.",
        judge_standard=judge_standard,
        candidates=candidates,
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        image_detail="auto",
        max_output_tokens=100000,
    )

    content = request["input"][1]["content"]
    user_text = content[0]["text"]
    assert "Candidate A" in user_text
    assert "Candidate B" in user_text
    assert "Trajectory turns:" in user_text
    assert "Turn 1: result_passed=true" in user_text
    assert "Turn 2: result_passed=false" in user_text
    assert "Parse status: parse_error" in user_text
    assert "Execution status: invalid_action" in user_text
    assert "Candidate A screenshot after assistant turn 2" in user_text
    assert "Candidate A screenshot after assistant turn 6" in user_text
    assert "Candidate D screenshot after assistant turn 6" in user_text
    assert "Assign different ranks only when the difference in task-relevant progress is clear." in user_text
    assert "If the evidence is ambiguous, uncertain, or not clearly distinguishable, prefer a tie." in user_text
    assert [item["type"] for item in content[:11]] == [
        "input_text",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
        "input_text",
        "input_image",
    ]
    assert content[1]["text"] == "Candidate A screenshot after assistant turn 2."
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["name"] == reward_fn_webgym_rl.COMPARATIVE_JUDGE_SCHEMA_NAME
    assert request["text"]["format"]["strict"] is True
    assert request["text"]["format"]["schema"]["properties"]["D_rank"]["maximum"] == 4
    assert "If the evidence is ambiguous, doubtful, or not clearly distinguishable, give the candidates the same rank." in request["input"][0]["content"]


def test_build_zero_group_compare_request_supports_six_candidates(tmp_path: Path):
    judge_standard = [{"id": "check_1", "text": "The target app is open."}]
    labels = reward_fn_webgym_rl.build_compare_labels(6)
    candidates = []
    for label in labels:
        trajectory_dir = _write_candidate_trajectory(tmp_path, f"candidate_{label}")
        candidates.append(reward_fn_webgym_rl._build_compare_candidate(label=label, trajectory_dir=trajectory_dir))

    request = reward_fn_webgym_rl.build_zero_group_compare_request(
        task_instruction="Complete the requested task.",
        judge_standard=judge_standard,
        candidates=candidates,
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        image_detail="auto",
        max_output_tokens=100000,
    )

    schema = request["text"]["format"]["schema"]
    assert schema["required"] == [f"{label}_rank" for label in labels] + ["reason"]
    assert schema["properties"]["F_rank"]["maximum"] == 6
    user_text = request["input"][1]["content"][0]["text"]
    assert "Candidate F" in user_text


def test_compare_zero_group_webgym_rl_maps_dense_ranks_with_ties(monkeypatch, tmp_path: Path):
    judge_standard = [{"id": "check_1", "text": "The target app is open."}]
    extra_infos = []
    for label in ["a", "b", "c", "d"]:
        trajectory_dir = _write_candidate_trajectory(tmp_path, label)
        extra_infos.append(
            {
                "judge_standard": judge_standard,
                "web_osgym_trajectory_dir": str(trajectory_dir),
                "task_name": "Open the target app.",
                "web_osgym_env_reward_score": 0.0,
            }
        )

    fake_client = _FakeJudgeClient(
        json.dumps(
            {
                "A_rank": 1,
                "B_rank": 1,
                "C_rank": 2,
                "D_rank": 3,
                "reason": "A and B are tied for best progress.",
            }
        )
    )
    monkeypatch.setattr(reward_fn_webgym_rl, "_get_openai_client", lambda: fake_client)

    outputs = reward_fn_webgym_rl.compare_zero_group_webgym_rl(
        extra_infos=extra_infos,
        llm_judge_model="gpt-5.4-mini",
        llm_judge_reasoning_effort="medium",
        llm_judge_image_detail="auto",
        llm_judge_timeout_seconds=17,
    )

    assert [item["reward_score"] for item in outputs] == pytest.approx([1.0, 1.0, 0.5, 0.0])
    assert [item["reward_extra_info"]["web_osgym_llm_judge_rank"] for item in outputs] == [1, 1, 2, 3]
    assert all(item["reward_extra_info"]["web_osgym_llm_judge_used"] for item in outputs)
    assert fake_client.timeouts == [17.0]
    assert len(fake_client.requests) == 1


def test_compare_zero_group_webgym_rl_returns_zero_scores_when_all_candidates_tie(monkeypatch, tmp_path: Path):
    judge_standard = [{"id": "check_1", "text": "The target app is open."}]
    extra_infos = []
    for label in ["a", "b", "c", "d"]:
        trajectory_dir = _write_candidate_trajectory(tmp_path, label)
        extra_infos.append(
            {
                "judge_standard": judge_standard,
                "web_osgym_trajectory_dir": str(trajectory_dir),
                "task_name": "Open the target app.",
                "web_osgym_env_reward_score": 0.0,
            }
        )

    fake_client = _FakeJudgeClient(
        json.dumps(
            {
                "A_rank": 1,
                "B_rank": 1,
                "C_rank": 1,
                "D_rank": 1,
                "reason": "No candidate shows distinguishable progress.",
            }
        )
    )
    monkeypatch.setattr(reward_fn_webgym_rl, "_get_openai_client", lambda: fake_client)

    outputs = reward_fn_webgym_rl.compare_zero_group_webgym_rl(
        extra_infos=extra_infos,
        llm_judge_model="gpt-5.4-mini",
        llm_judge_reasoning_effort="medium",
        llm_judge_image_detail="auto",
        llm_judge_timeout_seconds=17,
    )

    assert [item["reward_score"] for item in outputs] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert [item["reward_extra_info"]["web_osgym_llm_judge_rank"] for item in outputs] == [1, 1, 1, 1]


def test_validate_zero_group_compare_extra_infos_rejects_missing_trajectory_dir(tmp_path: Path):
    trajectory_dir = _write_candidate_trajectory(tmp_path, "a")
    extra_infos = [
        {
            "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
            "web_osgym_trajectory_dir": str(trajectory_dir),
            "task_name": "Open the target app.",
            "web_osgym_env_reward_score": 0.0,
        },
        {
            "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
            "web_osgym_trajectory_dir": "",
            "task_name": "Open the target app.",
            "web_osgym_env_reward_score": 0.0,
        },
        {
            "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
            "web_osgym_trajectory_dir": str(trajectory_dir),
            "task_name": "Open the target app.",
            "web_osgym_env_reward_score": 0.0,
        },
        {
            "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
            "web_osgym_trajectory_dir": str(trajectory_dir),
            "task_name": "Open the target app.",
            "web_osgym_env_reward_score": 0.0,
        },
    ]

    with pytest.raises(ValidationError):
        reward_fn_webgym_rl._ZERO_GROUP_EXTRA_INFOS_ADAPTER.validate_python(extra_infos)


def test_validate_zero_group_compare_response_rejects_wrong_rank_type():
    with pytest.raises(ValidationError):
        reward_fn_webgym_rl.validate_zero_group_compare_response(
            {
                "A_rank": 1,
                "B_rank": "bad",
                "C_rank": 2,
                "D_rank": 3,
                "reason": "malformed",
            },
            labels=("A", "B", "C", "D"),
        )


def test_validate_zero_group_compare_response_supports_six_candidates():
    parsed = reward_fn_webgym_rl.validate_zero_group_compare_response(
        {
            "A_rank": 1,
            "B_rank": 2,
            "C_rank": 2,
            "D_rank": 3,
            "E_rank": 4,
            "F_rank": 5,
            "reason": "six-way ranking",
        },
        labels=reward_fn_webgym_rl.build_compare_labels(6),
    )

    assert parsed.model_dump()["F_rank"] == 5


def test_validate_zero_group_compare_extra_infos_normalizes_instruction_and_drops_unknown_fields(tmp_path: Path):
    trajectory_dir = _write_candidate_trajectory(tmp_path, "a")
    validated = reward_fn_webgym_rl.validate_zero_group_compare_extra_infos(
        [
            {
                "judge_standard": [{"id": " check_1 ", "text": " The target app is open. "}],
                "web_osgym_trajectory_dir": f"  {trajectory_dir}  ",
                "task_name": " Open the target app. ",
                "web_osgym_env_reward_score": 0.0,
                "request_id": "req-1",
            }
        ]
        * 4
    )

    assert all(isinstance(item, reward_fn_webgym_rl.ZeroGroupCompareExtraInfo) for item in validated)
    assert all(
        [entry.model_dump()["judge_standard"] == [{"id": "check_1", "text": "The target app is open."}] for entry in validated]
    )
    assert all(item.web_osgym_trajectory_dir == str(trajectory_dir) for item in validated)
    assert all(item.task_instruction == "Open the target app." for item in validated)
    assert all("request_id" not in item.model_dump() for item in validated)


def test_merge_webgym_reward_extra_info_preserves_base_fields_and_applies_patch():
    merged = reward_fn_webgym_rl.merge_webgym_reward_extra_info(
        {
            "web_osgym_env_reward_score": 0.0,
            "web_osgym_format_reward": 0.8,
            "web_osgym_raw_format_reward": 1.0,
            "web_osgym_attempted_tool_calls": 4,
        },
        {
            "web_osgym_llm_judge_used": True,
            "web_osgym_llm_judge_score": 0.5,
            "web_osgym_llm_judge_rank": 2,
            "web_osgym_llm_judge_reason": "partial progress",
        },
    )

    assert merged["web_osgym_env_reward_score"] == pytest.approx(0.0)
    assert merged["web_osgym_format_reward"] == pytest.approx(0.8)
    assert merged["web_osgym_raw_format_reward"] == pytest.approx(1.0)
    assert merged["web_osgym_attempted_tool_calls"] == 4
    assert merged["web_osgym_llm_judge_used"] is True
    assert merged["web_osgym_llm_judge_score"] == pytest.approx(0.5)
    assert merged["web_osgym_llm_judge_rank"] == 2
    assert merged["web_osgym_llm_judge_reason"] == "partial progress"


def test_pack_webgym_reward_extra_infos_builds_dense_columns():
    non_tensor_batch, reward_extra_keys = reward_fn_webgym_rl.pack_webgym_reward_extra_infos(
        [
            {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_llm_judge_used": False,
            },
            {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.5,
                "web_osgym_llm_judge_rank": 2,
            },
        ],
        template_non_tensor_batch={"web_osgym_env_reward_score": np.array([0.0, 0.0], dtype=np.float32)},
    )

    assert reward_extra_keys == list(reward_fn_webgym_rl.WebGymRewardExtraInfo.model_fields.keys())
    assert non_tensor_batch["web_osgym_env_reward_score"].dtype == np.float32
    assert non_tensor_batch["web_osgym_env_reward_score"].tolist() == pytest.approx([1.0, 0.0])
    assert non_tensor_batch["web_osgym_format_reward"].tolist() == pytest.approx([0.8, 0.4])
    assert non_tensor_batch["web_osgym_llm_judge_used"].tolist() == [False, True]
    assert non_tensor_batch["web_osgym_llm_judge_rank"].tolist() == [None, 2]


def test_extract_webgym_reward_extra_infos_roundtrips_packed_columns():
    packed_non_tensor_batch, reward_extra_keys = reward_fn_webgym_rl.pack_webgym_reward_extra_infos(
        [
            {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_llm_judge_used": False,
                "web_osgym_llm_judge_reason": "base",
            },
            {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.5,
                "web_osgym_llm_judge_rank": 2,
                "web_osgym_llm_judge_reason": "judge",
            },
        ]
    )

    restored = reward_fn_webgym_rl.extract_webgym_reward_extra_infos(
        packed_non_tensor_batch,
        reward_extra_keys=reward_extra_keys,
    )

    assert [item.model_dump() for item in restored] == [
        reward_fn_webgym_rl.validate_webgym_reward_extra_info(
            {
                "web_osgym_env_reward_score": 1.0,
                "web_osgym_format_reward": 0.8,
                "web_osgym_llm_judge_used": False,
                "web_osgym_llm_judge_reason": "base",
            }
        ).model_dump(),
        reward_fn_webgym_rl.validate_webgym_reward_extra_info(
            {
                "web_osgym_env_reward_score": 0.0,
                "web_osgym_format_reward": 0.4,
                "web_osgym_llm_judge_used": True,
                "web_osgym_llm_judge_score": 0.5,
                "web_osgym_llm_judge_rank": 2,
                "web_osgym_llm_judge_reason": "judge",
            }
        ).model_dump(),
    ]


def test_extract_webgym_reward_extra_infos_rejects_mismatched_column_lengths():
    with pytest.raises(ValueError):
        reward_fn_webgym_rl.extract_webgym_reward_extra_infos(
            {
                "web_osgym_env_reward_score": np.array([1.0, 0.0], dtype=np.float32),
                "web_osgym_format_reward": np.array([0.5], dtype=np.float32),
            }
        )


if HYPOTHESIS_AVAILABLE:

    @settings(max_examples=60, deadline=None)
    @given(base=_reward_extra_info_strategy(), patch=_reward_extra_info_patch_strategy())
    def test_merge_webgym_reward_extra_info_property(base, patch):
        merged = reward_fn_webgym_rl.merge_webgym_reward_extra_info(base, patch)

        base_model = reward_fn_webgym_rl.validate_webgym_reward_extra_info(base).model_dump()
        patch_model = reward_fn_webgym_rl.validate_webgym_reward_extra_info_patch(patch)
        expected = dict(base_model)
        for field_name in patch_model.model_fields_set:
            expected[field_name] = getattr(patch_model, field_name)
        expected = reward_fn_webgym_rl.validate_webgym_reward_extra_info(expected).model_dump()

        assert merged == expected


    @settings(max_examples=40, deadline=None)
    @given(rows=st.lists(_reward_extra_info_strategy(), min_size=1, max_size=4))
    def test_pack_extract_webgym_reward_extra_infos_property(rows):
        packed_non_tensor_batch, reward_extra_keys = reward_fn_webgym_rl.pack_webgym_reward_extra_infos(rows)
        restored = reward_fn_webgym_rl.extract_webgym_reward_extra_infos(
            packed_non_tensor_batch,
            reward_extra_keys=reward_extra_keys,
        )
        expected = [reward_fn_webgym_rl.validate_webgym_reward_extra_info(row).model_dump() for row in rows]

        assert [item.model_dump() for item in restored] == expected

else:

    @pytest.mark.skip(reason="hypothesis is not installed in this test environment")
    def test_merge_webgym_reward_extra_info_property():
        pass


    @pytest.mark.skip(reason="hypothesis is not installed in this test environment")
    def test_pack_extract_webgym_reward_extra_infos_property():
        pass


def test_build_zero_group_compare_request_rejects_malformed_candidate(tmp_path: Path):
    trajectory_dir = _write_candidate_trajectory(tmp_path, "candidate_A")
    candidate = reward_fn_webgym_rl._build_compare_candidate(label="A", trajectory_dir=trajectory_dir).model_dump()
    candidate["screenshots"] = []

    with pytest.raises(ValidationError):
        reward_fn_webgym_rl.build_zero_group_compare_request(
            task_instruction="Complete the requested task.",
            judge_standard=[{"id": "check_1", "text": "The target app is open."}],
            candidates=[candidate] * 4,
            model="gpt-5.4-mini",
            reasoning_effort="medium",
            image_detail="auto",
            max_output_tokens=100000,
        )


def test_compare_zero_group_webgym_rl_returns_zero_scores_for_out_of_range_rank(monkeypatch, tmp_path: Path):
    extra_infos = []
    for label in ["a", "b", "c", "d"]:
        trajectory_dir = _write_candidate_trajectory(tmp_path, label)
        extra_infos.append(
            {
                "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
                "web_osgym_trajectory_dir": str(trajectory_dir),
                "task_name": "Open the target app.",
                "web_osgym_env_reward_score": 0.0,
            }
        )

    fake_client = _FakeJudgeClient(
        json.dumps(
            {
                "A_rank": 0,
                "B_rank": 1,
                "C_rank": 2,
                "D_rank": 3,
                "reason": "invalid rank range",
            }
        )
    )
    monkeypatch.setattr(reward_fn_webgym_rl, "_get_openai_client", lambda: fake_client)

    outputs = reward_fn_webgym_rl.compare_zero_group_webgym_rl(
        extra_infos=extra_infos,
        llm_judge_model="gpt-5.4-mini",
        llm_judge_reasoning_effort="medium",
        llm_judge_image_detail="auto",
        llm_judge_timeout_seconds=17,
    )

    assert [item["reward_score"] for item in outputs] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert [item["reward_extra_info"]["web_osgym_llm_judge_rank"] for item in outputs] == [None, None, None, None]
    assert [item["reward_extra_info"]["web_osgym_llm_judge_used"] for item in outputs] == [False, False, False, False]


@pytest.mark.asyncio
async def test_compare_zero_group_webgym_rl_async_supports_six_candidates(monkeypatch, tmp_path: Path):
    labels = reward_fn_webgym_rl.build_compare_labels(6)
    extra_infos = []
    for label in labels:
        trajectory_dir = _write_candidate_trajectory(tmp_path, label.lower())
        extra_infos.append(
            {
                "judge_standard": [{"id": "check_1", "text": "The target app is open."}],
                "web_osgym_trajectory_dir": str(trajectory_dir),
                "task_name": "Open the target app.",
                "web_osgym_env_reward_score": 0.0,
            }
        )

    fake_client = _FakeAsyncJudgeClient(
        json.dumps(
            {
                "A_rank": 1,
                "B_rank": 1,
                "C_rank": 2,
                "D_rank": 3,
                "E_rank": 4,
                "F_rank": 5,
                "reason": "six-way async ranking",
            }
        )
    )
    monkeypatch.setattr(reward_fn_webgym_rl, "_get_async_openai_client", lambda: fake_client)

    outputs = await reward_fn_webgym_rl.compare_zero_group_webgym_rl_async(
        extra_infos=extra_infos,
        llm_judge_model="gpt-5.4-mini",
        llm_judge_reasoning_effort="medium",
        llm_judge_image_detail="auto",
        llm_judge_timeout_seconds=17,
    )

    assert [item["reward_score"] for item in outputs] == pytest.approx([1.0, 1.0, 0.75, 0.5, 0.25, 0.0])
    assert [item["reward_extra_info"]["web_osgym_llm_judge_rank"] for item in outputs] == [1, 1, 2, 3, 4, 5]
    assert fake_client.timeouts == [17.0]
    assert len(fake_client.requests) == 1
