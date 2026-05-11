import asyncio
import base64
import unittest
from io import BytesIO

import httpx
from PIL import Image

from verl.experimental.agent_loop.web_osgym_protocol import (
    WebOsGymAction,
    WebOsGymClient,
    WebOsGymRemoteError,
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

    async def test_action_retries_on_connect_timeout(self):
        attempts = {"count": 0}

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
                attempts["count"] += 1
                if attempts["count"] < 4:
                    raise httpx.ConnectTimeout("timed out")
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original_client = web_osgym_protocol.httpx.AsyncClient
        original_sleep = web_osgym_protocol.asyncio.sleep
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        web_osgym_protocol.asyncio.sleep = _fake_sleep
        try:
            client = WebOsGymClient(base_url="http://env")
            await client.action(
                request_id=101,
                task_id="12345",
                include_a11y=True,
                actions=[WebOsGymAction(action_type="CLICK", x=10, y=20)],
            )
        finally:
            web_osgym_protocol.httpx.AsyncClient = original_client
            web_osgym_protocol.asyncio.sleep = original_sleep

        self.assertEqual(attempts["count"], 4)

    async def test_action_retries_on_gateway_busy_error_response(self):
        attempts = {"count": 0}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                attempts["count"] += 1
                if attempts["count"] < 4:
                    return {
                        "session_id": 101,
                        "task_id": "12345",
                        "status": "error",
                        "error_type": "gateway_busy",
                        "message": "Timed out waiting for gateway capacity.",
                    }
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
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original_client = web_osgym_protocol.httpx.AsyncClient
        original_sleep = web_osgym_protocol.asyncio.sleep
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        web_osgym_protocol.asyncio.sleep = _fake_sleep
        try:
            client = WebOsGymClient(base_url="http://env")
            await client.action(
                request_id=101,
                task_id="12345",
                include_a11y=True,
                actions=[WebOsGymAction(action_type="CLICK", x=10, y=20)],
            )
        finally:
            web_osgym_protocol.httpx.AsyncClient = original_client
            web_osgym_protocol.asyncio.sleep = original_sleep

        self.assertEqual(attempts["count"], 4)

    async def test_start_raises_on_error_status(self):
        payload = {
            "session_id": 101,
            "task_id": "12345",
            "status": "error",
            "error_type": "gateway_busy",
            "message": "Timed out waiting for gateway capacity.",
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
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            with self.assertRaises(WebOsGymRemoteError) as ctx:
                await client.start(request_id=101, task_id="12345", include_a11y=True)
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertIn("start", str(ctx.exception))
        self.assertIn("gateway_busy", str(ctx.exception))
        self.assertIn("12345", str(ctx.exception))

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

    async def test_reward_raises_on_error_status(self):
        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "session_id": 101,
                    "task_id": "12345",
                    "status": "error",
                    "error_type": "fail_request_handle",
                    "message": "Session 101 does not exist",
                }

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, timeout):
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            with self.assertRaises(WebOsGymRemoteError) as ctx:
                await client.reward(request_id=101, task_id="12345")
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertIn("reward", str(ctx.exception))
        self.assertIn("fail_request_handle", str(ctx.exception))

    async def test_reward_raises_when_ok_response_omits_reward(self):
        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "session_id": 101,
                    "task_id": "12345",
                    "status": "ok",
                }

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json, timeout):
                return _FakeResponse()

        from verl.experimental.agent_loop import web_osgym_protocol

        original = web_osgym_protocol.httpx.AsyncClient
        web_osgym_protocol.httpx.AsyncClient = _FakeAsyncClient
        try:
            client = WebOsGymClient(base_url="http://env")
            with self.assertRaises(WebOsGymRemoteError) as ctx:
                await client.reward(request_id=101, task_id="12345")
        finally:
            web_osgym_protocol.httpx.AsyncClient = original

        self.assertIn("missing reward", str(ctx.exception))


async def _fake_sleep(_delay):
    return None
