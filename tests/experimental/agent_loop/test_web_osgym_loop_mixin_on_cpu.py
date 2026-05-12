import asyncio
import unittest

from PIL import Image

from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymRemoteError
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.tools.base_tool import ToolResponse


class _FakeTool:
    name = "computer"
    tool_schema = None

    def __init__(self):
        self.created = []
        self.rewards = []
        self.restored = []
        self.detached_reward_requests = []
        self._instance_dict = {}

    async def create(self, **kwargs):
        self.created.append(kwargs)
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
            "screen_width": 1920,
            "screen_height": 1080,
        }
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0

    def request_reward_detached(self, *, request_id: int, task_id: str):
        self.detached_reward_requests.append((request_id, task_id))
        return asyncio.create_task(asyncio.sleep(0), name=f"reward-close-{request_id}")

    def restore_instance(self, instance_id, **kwargs):
        self.restored.append((instance_id, kwargs))
        self._instance_dict[instance_id] = dict(kwargs)


class _RetryingFakeTool(_FakeTool):
    def __init__(self, *, failures_before_success: int):
        super().__init__()
        self.failures_before_success = failures_before_success
        self.attempts = 0

    async def create(self, **kwargs):
        self.attempts += 1
        self.created.append(kwargs)
        if self.attempts <= self.failures_before_success:
            raise WebOsGymRemoteError(
                op="start",
                session_id=kwargs["request_id"],
                task_id=kwargs["task_id"],
                error_type="fail_request_handle",
                message=f"attempt {self.attempts} failed",
            )
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
            "screen_width": 1920,
            "screen_height": 1080,
        }
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])


class TestWebOsGymLoopMixin(unittest.IsolatedAsyncioTestCase):
    async def test_start_session_stores_instance_id_and_observation(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "12345"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(agent_data.extra_fields["web_osgym_instance_id"], "instance-1")
        self.assertEqual(agent_data.extra_fields["web_osgym_session_id"], 101)
        self.assertEqual(agent_data.extra_fields["web_osgym_screen_width"], 1920)
        self.assertEqual(agent_data.extra_fields["web_osgym_screen_height"], 1080)
        self.assertEqual(response.text, "initial-observation")

    async def test_start_session_reads_shared_web_osgym_create_kwargs(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "shared-task"}}},
        )
        agent_data._active_tools = {"CLICK": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.created[0]["task_id"], "shared-task")

    async def test_start_session_retries_until_third_attempt_succeeds(self):
        loop = WebOsGymLoopMixin()
        tool = _RetryingFakeTool(failures_before_success=2)
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "retry-task"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.attempts, 3)
        self.assertEqual(len(tool.created), 3)
        self.assertEqual(
            [kwargs["request_id"] for kwargs in tool.created],
            [101, 101, 101],
        )
        self.assertEqual(agent_data.extra_fields["web_osgym_instance_id"], "instance-1")
        self.assertEqual(response.text, "initial-observation")

    async def test_start_session_raises_after_three_failed_attempts(self):
        loop = WebOsGymLoopMixin()
        tool = _RetryingFakeTool(failures_before_success=3)
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"web_osgym": {"create_kwargs": {"task_id": "retry-task"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        with self.assertRaises(WebOsGymRemoteError):
            await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(tool.attempts, 3)
        self.assertNotIn("web_osgym_instance_id", agent_data.extra_fields)

    async def test_finalize_with_reward_stores_reward_once(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )
        tool._instance_dict["instance-1"] = {
            "task_id": "12345",
            "request_id": 101,
            "include_a11y": False,
            "reward": None,
        }

        await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")

        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)
        self.assertTrue(agent_data.extra_fields["web_osgym_reward_requested"])
        self.assertEqual(
            agent_data.extra_fields["reward_extra_info"],
            {
                "web_osgym_reward_score": 1.0,
                "web_osgym_termination_reason": "system_stop",
            },
        )
        self.assertEqual(len(tool.rewards), 1)
        self.assertEqual(tool.detached_reward_requests, [])

    async def test_request_web_osgym_reward_best_effort_sends_once_for_unfetched_session(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": False,
            }
        )

        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")
        loop._request_web_osgym_reward_best_effort(agent_data, termination_reason="system_stop")

        self.assertEqual(tool.detached_reward_requests, [(101, "12345")])
        self.assertTrue(agent_data.extra_fields["web_osgym_reward_requested"])
        self.assertEqual(agent_data.extra_fields["web_osgym_termination_reason"], "system_stop")

    def test_ensure_session_restores_missing_local_instance_state(self):
        loop = WebOsGymLoopMixin()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": "instance-1",
                "web_osgym_task_id": "12345",
                "web_osgym_session_id": 101,
                "web_osgym_include_a11y": True,
                "web_osgym_cursor_x": 7,
                "web_osgym_cursor_y": 8,
                "web_osgym_screen_width": 1920,
                "web_osgym_screen_height": 1080,
            }
        )

        loop._ensure_web_osgym_session(agent_data)

        self.assertEqual(
            tool.restored,
            [
                (
                    "instance-1",
                    {
                        "task_id": "12345",
                        "request_id": 101,
                        "include_a11y": True,
                        "reward": None,
                        "cursor_x": 7,
                        "cursor_y": 8,
                        "screen_width": 1920,
                        "screen_height": 1080,
                    },
                )
            ],
        )
