import unittest

from PIL import Image

from verl.experimental.agent_loop.tool_agent_loop import AgentData
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin
from verl.tools.base_tool import ToolResponse


class _FakeTool:
    name = "computer"
    tool_schema = None

    def __init__(self):
        self.created = []
        self.rewards = []
        self.restored = []
        self._instance_dict = {}

    async def create(self, **kwargs):
        self.created.append(kwargs)
        self._instance_dict["instance-1"] = {
            "task_id": kwargs["task_id"],
            "request_id": kwargs["request_id"],
            "include_a11y": kwargs["include_a11y"],
            "reward": None,
        }
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0

    def restore_instance(self, instance_id, **kwargs):
        self.restored.append((instance_id, kwargs))
        self._instance_dict[instance_id] = dict(kwargs)


class _Loop(WebOsGymLoopMixin):
    web_osgym_tool_name = "computer"


class TestWebOsGymLoopMixin(unittest.IsolatedAsyncioTestCase):
    async def test_start_session_stores_instance_id_and_observation(self):
        loop = _Loop()
        tool = _FakeTool()
        agent_data = AgentData(
            messages=[{"role": "user", "content": "task"}],
            image_data=[],
            video_data=[],
            metrics={},
            request_id="loop-req",
            tools_kwargs={"computer": {"create_kwargs": {"task_id": "12345"}}},
        )
        agent_data._active_tools = {"computer": tool}
        agent_data.extra_fields["web_osgym_session_id"] = 101

        response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

        self.assertEqual(agent_data.extra_fields["web_osgym_instance_id"], "instance-1")
        self.assertEqual(agent_data.extra_fields["web_osgym_session_id"], 101)
        self.assertEqual(response.text, "initial-observation")

    async def test_finalize_with_reward_stores_reward_once(self):
        loop = _Loop()
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

        self.assertEqual(agent_data.extra_fields["web_osgym_reward_score"], 1.0)
        self.assertEqual(len(tool.rewards), 1)

    def test_ensure_session_restores_missing_local_instance_state(self):
        loop = _Loop()
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
                    },
                )
            ],
        )
