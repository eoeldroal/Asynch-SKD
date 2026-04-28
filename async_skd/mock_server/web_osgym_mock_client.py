from __future__ import annotations

import argparse
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:18000"


def _post(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(base_url.rstrip("/") + "/", json=payload, timeout=10.0)
    response.raise_for_status()
    return response.json()


def run_protocol_smoke(*, base_url: str, session_id: int = 1, task_id: str = "12345") -> dict[str, Any]:
    start = _post(
        base_url,
        {
            "op": "start",
            "session_id": session_id,
            "task_id": task_id,
            "include_a11y": True,
        },
    )
    action = _post(
        base_url,
        {
            "op": "action",
            "session_id": session_id,
            "task_id": task_id,
            "include_a11y": True,
            "actions": [
                {
                    "action_type": "CLICK",
                    "button": "left",
                    "x": 100,
                    "y": 200,
                    "num_clicks": 1,
                }
            ],
        },
    )
    done = _post(
        base_url,
        {
            "op": "action",
            "session_id": session_id,
            "task_id": task_id,
            "include_a11y": True,
            "actions": [{"action_type": "DONE"}],
        },
    )
    reward = _post(
        base_url,
        {
            "op": "reward",
            "session_id": session_id,
            "task_id": task_id,
        },
    )
    return {
        "session_id": session_id,
        "task_id": task_id,
        "start_status": start["status"],
        "action_status": action["status"],
        "done_status": done["status"],
        "reward": float(reward["reward"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--session-id", type=int, default=1)
    parser.add_argument("--task-id", default="12345")
    args = parser.parse_args()
    result = run_protocol_smoke(base_url=args.base_url, session_id=args.session_id, task_id=args.task_id)
    print(result)


if __name__ == "__main__":
    main()
