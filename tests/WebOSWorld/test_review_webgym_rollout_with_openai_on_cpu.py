from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/home/sogang_nlpy/verl")
sys.path.insert(0, str(ROOT))

from WebOSWorld.review_webgym_rollout_with_openai import (
    DEFAULT_REVIEW_MODEL,
    build_batch_requests,
    build_session_review_input,
    collect_session_dirs,
    parse_batch_output_jsonl,
    review_response_format,
)


def _write_session(root: Path, name: str, *, reward_score: float, model_output_text: str, mtime: int) -> Path:
    session_dir = root / name
    session_dir.mkdir(parents=True)
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "task_id": "prozilla_explorer_10",
                "sample_uid": "index_41",
                "session_id": 101,
                "reward_score": reward_score,
                "termination_reason": "system_stop",
                "num_turns": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (session_dir / "trajectory.jsonl").write_text(
        json.dumps(
            {
                "event_type": "assistant_turn",
                "assistant_turn": 1,
                "model_output_text": model_output_text,
                "tool_call_count": 0,
                "tool_calls_raw": [],
                "actions": [],
                "result": {"termination_reason": "system_stop"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "summary.json").touch()
    (session_dir / "summary.json").stat()
    import os

    os.utime(session_dir / "summary.json", (mtime, mtime))
    return session_dir


def test_collect_session_dirs_returns_latest_completed_sessions(tmp_path: Path):
    newest = _write_session(tmp_path, "task_a___uid_a___1", reward_score=1.0, model_output_text="A", mtime=200)
    older = _write_session(tmp_path, "task_b___uid_b___2", reward_score=0.0, model_output_text="B", mtime=100)
    ignored = tmp_path / "broken___uid___3"
    ignored.mkdir()
    (ignored / "trajectory.jsonl").write_text("", encoding="utf-8")

    session_dirs = collect_session_dirs(tmp_path, limit=2)

    assert session_dirs == [newest, older]


def test_build_session_review_input_includes_summary_and_trajectory_fields(tmp_path: Path):
    session_dir = _write_session(
        tmp_path,
        "prozilla_explorer_10___index_41___101",
        reward_score=0.0,
        model_output_text="The Documents folder is visible.<|im_end|>",
        mtime=100,
    )

    review_input = build_session_review_input(session_dir)

    assert "prozilla_explorer_10___index_41___101" in review_input
    assert '"reward_score": 0.0' in review_input
    assert "The Documents folder is visible.<|im_end|>" in review_input
    assert '"session_count": 1' in review_input


def test_build_session_review_input_is_scoped_to_one_session(tmp_path: Path):
    session_dir = _write_session(
        tmp_path,
        "prozilla_explorer_10___index_41___101",
        reward_score=0.0,
        model_output_text="only this session",
        mtime=100,
    )

    review_input = build_session_review_input(session_dir)

    assert '"session_count": 1' in review_input
    assert "only this session" in review_input


def test_build_batch_requests_emits_one_response_request_per_session(tmp_path: Path):
    first = _write_session(
        tmp_path,
        "prozilla_explorer_10___index_41___101",
        reward_score=0.0,
        model_output_text="first",
        mtime=200,
    )
    second = _write_session(
        tmp_path,
        "prozilla_terminal_03___index_11___202",
        reward_score=1.0,
        model_output_text="second",
        mtime=100,
    )

    requests = build_batch_requests([first, second], rollout_dir=tmp_path)

    assert len(requests) == 2
    assert requests[0]["custom_id"] == first.name
    assert requests[0]["method"] == "POST"
    assert requests[0]["url"] == "/v1/responses"
    assert requests[0]["body"]["model"] == DEFAULT_REVIEW_MODEL
    assert requests[0]["body"]["text"]["format"]["type"] == "json_schema"
    assert requests[1]["custom_id"] == second.name


def test_review_response_format_uses_minimal_schema():
    response_format = review_response_format()

    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "webgym_rollout_review"
    assert response_format["strict"] is True
    schema = response_format["schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "summary_text",
        "goal_reaching_level",
        "scoring_logic_ok",
        "scoring_logic_confidence",
    }
    assert schema["properties"]["summary_text"]["type"] == "string"
    assert schema["properties"]["goal_reaching_level"]["enum"] == ["high", "medium", "low", "none"]
    assert schema["properties"]["scoring_logic_ok"]["type"] == "boolean"
    assert schema["properties"]["scoring_logic_confidence"]["enum"] == ["high", "medium", "low"]


def test_parse_batch_output_jsonl_extracts_structured_review():
    output_text = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "session-a",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "output_text": json.dumps(
                                {
                                    "summary_text": "Looks correct",
                                    "goal_reaching_level": "high",
                                    "scoring_logic_ok": True,
                                    "scoring_logic_confidence": "high",
                                },
                                ensure_ascii=False,
                            )
                        },
                    },
                    "error": None,
                },
                ensure_ascii=False,
            )
        ]
    )

    rows = parse_batch_output_jsonl(output_text)

    assert rows == [
        {
            "custom_id": "session-a",
            "status_code": 200,
            "error": None,
            "review": {
                "summary_text": "Looks correct",
                "goal_reaching_level": "high",
                "scoring_logic_ok": True,
                "scoring_logic_confidence": "high",
            },
            "response_body": {
                "output_text": json.dumps(
                    {
                        "summary_text": "Looks correct",
                        "goal_reaching_level": "high",
                        "scoring_logic_ok": True,
                        "scoring_logic_confidence": "high",
                    },
                    ensure_ascii=False,
                )
            },
        }
    ]


def test_parse_batch_output_jsonl_extracts_review_from_response_output_items():
    review = {
        "summary_text": "Looks correct",
        "goal_reaching_level": "high",
        "scoring_logic_ok": True,
        "scoring_logic_confidence": "high",
    }
    output_text = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "session-b",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": json.dumps(review, ensure_ascii=False),
                                        }
                                    ],
                                }
                            ]
                        },
                    },
                    "error": None,
                },
                ensure_ascii=False,
            )
        ]
    )

    rows = parse_batch_output_jsonl(output_text)

    assert rows[0]["review"] == review
