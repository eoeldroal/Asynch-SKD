import base64
import os
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel

_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    fields = {"pid": os.getpid(), "mono_ns": time.monotonic_ns(), **fields}
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


class WebOsGymAction(BaseModel):
    action_type: str
    x: int | None = None
    y: int | None = None
    button: str | None = None
    num_clicks: int | None = None
    dx: int | None = None
    dy: int | None = None
    text: str | None = None
    key: str | None = None
    keys: list[str] | None = None


class WebOsGymResponse(BaseModel):
    session_id: int
    task_id: str
    status: str
    text: str | None = None
    reward: float | None = None
    image_b64: str | None = None
    image_mime_type: str | None = None

    @property
    def image(self):
        if not self.image_b64:
            return None
        _trace_async_skd(
            "web_osgym_response.image_decode_begin",
            session_id=self.session_id,
            task_id=self.task_id,
            image_b64_len=len(self.image_b64),
        )
        decode_t0 = time.monotonic()
        raw_bytes = base64.b64decode(self.image_b64)
        base64_ms = (time.monotonic() - decode_t0) * 1000
        pil_t0 = time.monotonic()
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        pil_ms = (time.monotonic() - pil_t0) * 1000
        _trace_async_skd(
            "web_osgym_response.image_decode_done",
            session_id=self.session_id,
            task_id=self.task_id,
            base64_ms=round(base64_ms, 1),
            pil_ms=round(pil_ms, 1),
            total_ms=round((time.monotonic() - decode_t0) * 1000, 1),
            image_width=image.width,
            image_height=image.height,
        )
        return image


class WebOsGymClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> WebOsGymResponse:
        image_payload = payload.get("image") or {}
        return WebOsGymResponse(
            session_id=int(payload["session_id"]),
            task_id=str(payload["task_id"]),
            status=payload["status"],
            text=payload.get("text"),
            reward=payload.get("reward"),
            image_b64=image_payload.get("data"),
            image_mime_type=image_payload.get("mimeType"),
        )

    async def _post(self, payload: dict[str, Any]) -> WebOsGymResponse:
        op = payload.get("op")
        request_id = payload.get("session_id")
        task_id = payload.get("task_id")
        _trace_async_skd(
            "web_osgym_client.post_begin",
            op=op,
            request_id=request_id,
            task_id=task_id,
            timeout=self.timeout,
            payload_keys=sorted(payload),
            action_count=len(payload.get("actions") or []),
        )
        post_t0 = time.monotonic()
        async with httpx.AsyncClient() as client:
            client_ms = (time.monotonic() - post_t0) * 1000
            _trace_async_skd(
                "web_osgym_client.client_ready",
                op=op,
                request_id=request_id,
                task_id=task_id,
                elapsed_ms=round(client_ms, 1),
            )
            http_t0 = time.monotonic()
            try:
                response = await client.post(self.base_url, json=payload, timeout=self.timeout)
            except Exception as exc:
                http_ms = (time.monotonic() - http_t0) * 1000
                _trace_async_skd(
                    "web_osgym_client.http_post_error",
                    op=op,
                    request_id=request_id,
                    task_id=task_id,
                    elapsed_ms=round(http_ms, 1),
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                raise
            http_ms = (time.monotonic() - http_t0) * 1000
            _trace_async_skd(
                "web_osgym_client.http_post_done",
                op=op,
                request_id=request_id,
                task_id=task_id,
                elapsed_ms=round(http_ms, 1),
                status_code=response.status_code,
                response_bytes=len(response.content or b""),
            )
            try:
                response.raise_for_status()
            except Exception as exc:
                _trace_async_skd(
                    "web_osgym_client.raise_for_status_error",
                    op=op,
                    request_id=request_id,
                    task_id=task_id,
                    status_code=response.status_code,
                    response_bytes=len(response.content or b""),
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                raise
        json_t0 = time.monotonic()
        try:
            payload_json = response.json()
        except Exception as exc:
            json_ms = (time.monotonic() - json_t0) * 1000
            _trace_async_skd(
                "web_osgym_client.json_error",
                op=op,
                request_id=request_id,
                task_id=task_id,
                elapsed_ms=round(json_ms, 1),
                response_bytes=len(response.content or b""),
                error_type=type(exc).__name__,
                error=repr(exc),
            )
            raise
        json_ms = (time.monotonic() - json_t0) * 1000
        parse_t0 = time.monotonic()
        try:
            parsed = self._parse_response(payload_json)
        except Exception as exc:
            parse_ms = (time.monotonic() - parse_t0) * 1000
            _trace_async_skd(
                "web_osgym_client.parse_error",
                op=op,
                request_id=request_id,
                task_id=task_id,
                elapsed_ms=round(parse_ms, 1),
                payload_keys=sorted(payload_json.keys()) if isinstance(payload_json, dict) else None,
                error_type=type(exc).__name__,
                error=repr(exc),
            )
            raise
        parse_ms = (time.monotonic() - parse_t0) * 1000
        _trace_async_skd(
            "web_osgym_client.post_done",
            op=op,
            request_id=request_id,
            task_id=task_id,
            total_ms=round((time.monotonic() - post_t0) * 1000, 1),
            json_ms=round(json_ms, 1),
            parse_ms=round(parse_ms, 1),
            status=parsed.status,
            text_len=len(parsed.text or ""),
            has_image=parsed.image_b64 is not None,
            image_b64_len=len(parsed.image_b64 or ""),
        )
        return parsed

    async def start(self, *, request_id: int, task_id: str, include_a11y: bool) -> WebOsGymResponse:
        return await self._post(
            {
                "session_id": request_id,
                "task_id": task_id,
                "op": "start",
                "include_a11y": include_a11y,
            }
        )

    async def action(
        self,
        *,
        request_id: int,
        task_id: str,
        include_a11y: bool,
        actions: list[WebOsGymAction],
    ) -> WebOsGymResponse:
        return await self._post(
            {
                "session_id": request_id,
                "task_id": task_id,
                "op": "action",
                "include_a11y": include_a11y,
                "actions": [action.model_dump(exclude_none=True) for action in actions],
            }
        )

    async def reward(self, *, request_id: int, task_id: str) -> float:
        response = await self._post({"session_id": request_id, "task_id": task_id, "op": "reward"})
        assert response.reward is not None
        return float(response.reward)
