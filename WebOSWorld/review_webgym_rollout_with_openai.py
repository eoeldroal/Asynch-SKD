from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_REVIEW_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You review Web/OSGym RL rollout logs.
Focus on concrete failure modes and reward/scoring reliability.
Be concise and specific."""


def review_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "webgym_rollout_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary_text": {"type": "string"},
                "goal_reaching_level": {"type": "string", "enum": ["high", "medium", "low", "none"]},
                "scoring_logic_ok": {"type": "boolean"},
                "scoring_logic_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": [
                "summary_text",
                "goal_reaching_level",
                "scoring_logic_ok",
                "scoring_logic_confidence",
            ],
        },
    }


def collect_session_dirs(rollout_dir: Path, limit: int) -> list[Path]:
    session_dirs: list[Path] = []
    for child in rollout_dir.iterdir():
        if not child.is_dir():
            continue
        if not (child / "summary.json").exists():
            continue
        if not (child / "trajectory.jsonl").exists():
            continue
        session_dirs.append(child)
    session_dirs.sort(key=lambda path: (path / "summary.json").stat().st_mtime, reverse=True)
    return session_dirs[:limit]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_session_review_input(session_dir: Path) -> str:
    payload = {
        "session_count": 1,
        "sessions": [
            {
                "session_dir": session_dir.name,
                "summary": _load_json(session_dir / "summary.json"),
                "trajectory": _load_jsonl(session_dir / "trajectory.jsonl"),
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_batch_requests(session_dirs: list[Path], *, rollout_dir: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for session_dir in session_dirs:
        requests.append(
            {
                "custom_id": session_dir.name,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": DEFAULT_REVIEW_MODEL,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Review the following Web/OSGym rollout session from {rollout_dir}.\n"
                                "Assess this one session only.\n\n"
                                f"{build_session_review_input(session_dir)}"
                            ),
                        },
                    ],
                    "text": {"format": review_response_format()},
                },
            }
        )
    return requests


def write_batch_input_file(rollout_dir: Path, requests: list[dict[str, Any]]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_input_path = rollout_dir / f"openai_rollout_review_batch_{timestamp}.jsonl"
    with batch_input_path.open("w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return batch_input_path


def _extract_review_from_response_body(body: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        try:
            return json.loads(output_text)
        except json.JSONDecodeError:
            return None
    output_items = body.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        continue
    return None


def parse_batch_output_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response = record.get("response") or {}
        body = response.get("body") or {}
        rows.append(
            {
                "custom_id": record.get("custom_id"),
                "status_code": response.get("status_code"),
                "error": record.get("error"),
                "review": _extract_review_from_response_body(body),
                "response_body": body,
            }
        )
    return rows


def wait_for_batch_completion(
    client: OpenAI,
    *,
    batch_id: str,
    poll_interval_seconds: int = 10,
    timeout_seconds: int = 1800,
):
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    started_at = time.time()
    batch = client.batches.retrieve(batch_id)
    while batch.status not in terminal_statuses:
        if time.time() - started_at >= timeout_seconds:
            break
        time.sleep(poll_interval_seconds)
        batch = client.batches.retrieve(batch_id)
    return batch


def save_batch_outputs(client: OpenAI, *, rollout_dir: Path, batch) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: dict[str, Any] = {}

    if getattr(batch, "output_file_id", None):
        output_text = client.files.content(batch.output_file_id).text
        output_jsonl_path = rollout_dir / f"openai_rollout_review_batch_output_{timestamp}.jsonl"
        output_jsonl_path.write_text(output_text, encoding="utf-8")
        parsed_output_path = rollout_dir / f"openai_rollout_review_batch_output_{timestamp}.json"
        parsed_output_path.write_text(
            json.dumps(parse_batch_output_jsonl(output_text), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved["output_file_id"] = batch.output_file_id
        saved["output_jsonl_path"] = str(output_jsonl_path)
        saved["parsed_output_path"] = str(parsed_output_path)

    if getattr(batch, "error_file_id", None):
        error_text = client.files.content(batch.error_file_id).text
        error_jsonl_path = rollout_dir / f"openai_rollout_review_batch_error_{timestamp}.jsonl"
        error_jsonl_path.write_text(error_text, encoding="utf-8")
        saved["error_file_id"] = batch.error_file_id
        saved["error_jsonl_path"] = str(error_jsonl_path)

    return saved


def submit_rollout_review_batch(*, rollout_dir: Path, limit: int) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    session_dirs = collect_session_dirs(rollout_dir, limit=limit)
    if not session_dirs:
        raise RuntimeError(f"No completed session directories found under {rollout_dir}")

    requests = build_batch_requests(session_dirs, rollout_dir=rollout_dir)
    batch_input_path = write_batch_input_file(rollout_dir, requests)

    client = OpenAI(api_key=api_key)
    with batch_input_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )

    batch = wait_for_batch_completion(client, batch_id=batch.id)
    saved_outputs = save_batch_outputs(client, rollout_dir=rollout_dir, batch=batch)

    return {
        "rollout_dir": str(rollout_dir),
        "model": DEFAULT_REVIEW_MODEL,
        "limit": limit,
        "session_count": len(session_dirs),
        "batch_input_path": str(batch_input_path),
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "batch_status": batch.status,
        "session_dirs": [session_dir.name for session_dir in session_dirs],
        **saved_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = submit_rollout_review_batch(rollout_dir=args.rollout_dir, limit=args.limit)

    output_path = args.output_file
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.rollout_dir / f"openai_rollout_review_batch_{timestamp}.json"

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
