import base64
import unittest
from io import BytesIO

from PIL import Image

from verl.experimental.agent_loop.web_osgym_protocol import (
    WebOsGymAction,
    WebOsGymClient,
)


def _png_b64() -> str:
    image = Image.new("RGB", (2, 2), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestWebOsGymProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_start_preserves_session_request_id(self):
        payload = {
            "session_id": 101,
            "task_id": "12345",
            "status": "ok",
            "text": "A11Y_TREE:\nroot",
            "image": {"data": _png_b64(), "mimeType": "image/png"},
        }

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, timeout):
                assert json["session_id"] == 101
                assert json["task_id"] == "12345"
                assert json["op"] == "start"
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            response = await client.start(request_id=101, task_id="12345", include_a11y=True)
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertEqual(response.session_id, 101)
        self.assertEqual(response.task_id, "12345")
        self.assertIsNotNone(response.image)

    async def test_action_reuses_same_session_request_id(self):
        seen = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "session_id": 101,
                    "task_id": "12345",
                    "status": "ok",
                    "text": "A11Y_TREE:\nnext",
                    "image": {},
                }

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, timeout):
                seen["payload"] = json
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            await client.action(
                request_id=101,
                task_id="12345",
                include_a11y=True,
                actions=[WebOsGymAction(action_type="CLICK", x=10, y=20, button="left", num_clicks=1)],
            )
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertEqual(seen["payload"]["session_id"], 101)
        self.assertEqual(seen["payload"]["task_id"], "12345")
        self.assertEqual(seen["payload"]["actions"][0]["action_type"], "CLICK")

    async def test_reward_reuses_same_session_request_id(self):
        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"session_id": 101, "task_id": "12345", "status": "ok", "reward": 1.0}

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, timeout):
                assert json == {"session_id": 101, "task_id": "12345", "op": "reward"}
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            reward = await client.reward(request_id=101, task_id="12345")
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertEqual(reward, 1.0)
