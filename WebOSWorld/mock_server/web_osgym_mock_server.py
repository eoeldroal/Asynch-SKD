from __future__ import annotations

import argparse
import base64
import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, ImageDraw


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18000
DEFAULT_LOG_PATH = "logs/mock_web_osgym_requests.jsonl"

ACTION_REQUIRED_FIELDS = {
    "MOVE_TO": {"x", "y"},
    "CLICK": {"button", "x", "y", "num_clicks"},
    "DOUBLE_CLICK": {"x", "y"},
    "SCROLL": {"dx", "dy"},
    "TYPING": {"text"},
    "PRESS": {"key"},
    "KEY_DOWN": {"key"},
    "KEY_UP": {"key"},
    "HOTKEY": {"keys"},
    "WAIT": set(),
    "DONE": set(),
    "FAIL": set(),
}

ACTION_ALLOWED_FIELDS = {
    action_type: {"action_type", *required_fields}
    for action_type, required_fields in ACTION_REQUIRED_FIELDS.items()
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_png_b64(label: str, step: int) -> str:
    image = Image.new("RGB", (640, 360), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 616, 336), outline=(80, 96, 120), width=3)
    draw.text((48, 56), "Mock Web/OSGym", fill=(20, 30, 45))
    draw.text((48, 96), f"task: {label}", fill=(20, 30, 45))
    draw.text((48, 136), f"step: {step}", fill=(20, 30, 45))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _validate_actions(actions: Any) -> list[dict[str, Any]]:
    if not isinstance(actions, list) or not actions:
        raise HTTPException(status_code=422, detail="actions must be a non-empty list")

    validated_actions = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise HTTPException(status_code=422, detail=f"actions[{index}] must be an object")

        action_type = action.get("action_type")
        if action_type not in ACTION_REQUIRED_FIELDS:
            raise HTTPException(status_code=422, detail=f"actions[{index}].action_type is unsupported: {action_type!r}")

        action_keys = set(action)
        missing = sorted(ACTION_REQUIRED_FIELDS[action_type] - action_keys)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"actions[{index}] missing required field(s): {', '.join(missing)}",
            )

        extra = sorted(action_keys - ACTION_ALLOWED_FIELDS[action_type])
        if extra:
            raise HTTPException(
                status_code=422,
                detail=f"actions[{index}] has unsupported field(s) for {action_type}: {', '.join(extra)}",
            )

        if action_type == "CLICK" and action.get("button") != "left":
            raise HTTPException(status_code=422, detail=f"actions[{index}].button must be 'left'")

        validated_actions.append(action)

    terminal = [action for action in validated_actions if action["action_type"] in {"DONE", "FAIL"}]
    if terminal and len(validated_actions) != 1:
        raise HTTPException(status_code=422, detail="DONE/FAIL must be sent as a standalone action list")

    return validated_actions


class MockWebOsGymState:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.sessions: dict[int, dict[str, Any]] = {}
        self.event_index = 0
        self.lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("")

    def log(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.event_index += 1
            event = {"ts": _now_iso(), "event_index": self.event_index, **payload}
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _require_session(self, session_id: int, task_id: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown session_id={session_id}")
        if session["task_id"] != task_id:
            raise HTTPException(
                status_code=409,
                detail=f"task_id mismatch for session_id={session_id}: expected {session['task_id']}, got {task_id}",
            )
        return session

    def normal_observation(self, *, session_id: int, task_id: str, include_a11y: bool, step: int) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "task_id": task_id,
            "status": "ok",
            "text": f"A11Y_TREE:\nmock_root task={task_id} step={step}" if include_a11y else None,
            "image": {
                "data": _make_png_b64(task_id, step),
                "mimeType": "image/png",
            },
        }


def create_app(log_path: str | Path) -> FastAPI:
    state = MockWebOsGymState(Path(log_path))
    app = FastAPI(title="Mock Web/OSGym Server")
    app.state.mock_web_osgym_state = state

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": len(state.sessions)}

    @app.post("/")
    async def protocol(request: Request) -> dict[str, Any]:
        payload = await request.json()
        op = payload.get("op")
        session_id = int(payload["session_id"])
        task_id = str(payload["task_id"])
        include_a11y = bool(payload.get("include_a11y", False))

        if op == "start":
            if session_id in state.sessions:
                state.log({"op": op, "session_id": session_id, "task_id": task_id, "status": "duplicate_start"})
                raise HTTPException(status_code=409, detail=f"duplicate start for session_id={session_id}")
            state.sessions[session_id] = {
                "task_id": task_id,
                "step": 0,
                "actions": [],
                "terminated": False,
                "reward": 1.0,
            }
            state.log(
                {
                    "op": op,
                    "session_id": session_id,
                    "task_id": task_id,
                    "include_a11y": include_a11y,
                    "step": 0,
                    "status": "ok",
                }
            )
            return state.normal_observation(
                session_id=session_id,
                task_id=task_id,
                include_a11y=include_a11y,
                step=0,
            )

        if op == "action":
            session = state._require_session(session_id, task_id)
            actions = _validate_actions(payload.get("actions"))
            session["actions"].append(actions)
            session["step"] += 1
            terminal = len(actions) == 1 and actions[0].get("action_type") in {"DONE", "FAIL"}
            if terminal:
                session["terminated"] = True
            state.log(
                {
                    "op": op,
                    "session_id": session_id,
                    "task_id": task_id,
                    "include_a11y": include_a11y,
                    "step": session["step"],
                    "action_count": len(actions),
                    "actions": actions,
                    "terminated": terminal,
                    "status": "ok",
                }
            )
            return state.normal_observation(
                session_id=session_id,
                task_id=task_id,
                include_a11y=include_a11y,
                step=session["step"],
            )

        if op == "reward":
            session = state._require_session(session_id, task_id)
            reward = float(session["reward"])
            state.log(
                {
                    "op": op,
                    "session_id": session_id,
                    "task_id": task_id,
                    "reward": reward,
                    "terminated": session["terminated"],
                    "status": "ok",
                }
            )
            return {"session_id": session_id, "task_id": task_id, "status": "ok", "reward": reward}

        raise HTTPException(status_code=400, detail=f"unsupported op={op!r}")

    return app


def _pick_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class RunningMockServer:
    base_url: str
    log_path: Path
    _server: uvicorn.Server
    _thread: threading.Thread

    def shutdown(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def run_mock_server_in_thread(*, host: str, port: int, log_path: str | Path) -> RunningMockServer:
    actual_port = _pick_port(host) if port == 0 else port
    app = create_app(log_path)
    config = uvicorn.Config(app, host=host, port=actual_port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://{host}:{actual_port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=0.5)
            if response.status_code == 200:
                return RunningMockServer(base_url=base_url, log_path=Path(log_path), _server=server, _thread=thread)
        except Exception:
            time.sleep(0.05)
    server.should_exit = True
    thread.join(timeout=2)
    raise RuntimeError(f"Timed out waiting for mock server at {base_url}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    app = create_app(args.log_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
