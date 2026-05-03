import base64
from io import BytesIO
from typing import Any

import httpx
from PIL import Image
from pydantic import BaseModel


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
    error_type: str | None = None
    message: str | None = None
    image_b64: str | None = None
    image_mime_type: str | None = None

    @property
    def image(self):
        if not self.image_b64:
            return None
        return Image.open(BytesIO(base64.b64decode(self.image_b64))).convert("RGB")


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
            error_type=payload.get("error_type"),
            message=payload.get("message"),
            image_b64=image_payload.get("data"),
            image_mime_type=image_payload.get("mimeType"),
        )

    async def _post(self, payload: dict[str, Any]) -> WebOsGymResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        return self._parse_response(response.json())

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
