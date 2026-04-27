# Web / OS Gym Agent Loop Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add WebGym / OSWorld remote-environment support at the agent-loop boundary so standard RL can use `web_tool_agent`, async SKD can use `web_skd_agent`, and the final environment reward flows into `AgentLoopOutput.reward_score` without changing trainer or async scheduler semantics.

**Architecture:** Keep `AgentLoopManager`, async SKD manager/worker, and PPO trainer unchanged. Add a protocol client, a persistent environment tool, and two new loop types: `WebToolAgentLoop(ToolAgentLoop)` for regular RL and `WebSkdAgentLoop(SkdAgentLoop)` for SKD. Reuse the existing `qwen3_coder` XML tool-calling path and translate model-facing Computer 13 actions into the external environment protocol internally.

**Tech Stack:** Python, `httpx`, `pydantic`, existing `ToolParser` (`qwen3_coder`), `BaseTool`, `ToolAgentLoop`, `SkdAgentLoop`, `pytest`

---

## Critical Invariants

Before any task work starts, keep these invariants fixed:

1. **`request_id` means session, not transport request.**
   - For a given trajectory, `start -> action* -> reward` must reuse the same `request_id`.
   - `task_id` identifies the benchmark task; `request_id` identifies one rollout session on that task.

2. **Model-facing interface is Computer 13, not protocol JSON.**
   - The model should emit one `computer` tool call using the current `qwen3_coder` XML format.
   - Internal code is responsible for converting parsed tool args into protocol `action` payloads.

3. **`DONE` / `FAIL` stay inside the model-facing action schema but are treated as terminal actions internally.**
   - `DONE` / `FAIL` trigger reward fetch and loop termination.
   - `max_length` / `max_chunks` also fetch reward and terminate, but without an action request.

4. **SKD uses teacher-only a11y.**
   - `WebSkdAgentLoop` must always request `include_a11y=True`.
   - Student-facing observation text must omit a11y tree.
   - Teacher-facing prompt stream must include a11y tree.
   - Therefore `WebSkdAgentLoop` cannot reuse the student/teacher prompt update path unchanged.

5. **Do not modify async rollout scheduling layers.**
   - `verl/experimental/async_skd/manager.py`
   - `verl/experimental/async_skd/worker.py`
   - `verl/trainer/ppo/ray_trainer.py`

The feature must plug in through `agent_name` selection, not by special-casing Web/OS gym inside the trainer or scheduler.

---

## File Structure

### New files

- `verl/experimental/agent_loop/web_osgym_protocol.py`
  - Protocol models and async HTTP client for `start`, `action`, `reward`
- `verl/tools/web_osgym_tool.py`
  - Persistent session tool exposing Computer 13 schema to the model
- `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
  - Shared session/reward helpers for both Web loops
- `verl/experimental/agent_loop/web_tool_agent_loop.py`
  - Standard RL loop built on `ToolAgentLoop`
- `verl/experimental/agent_loop/web_skd_agent_loop.py`
  - SKD loop built on `SkdAgentLoop`, with teacher-only a11y handling
- `examples/sglang_multiturn/config/tool_config/web_osgym_tool_config.yaml`
  - Tool config for the environment-backed `computer` tool
- `tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py`
- `tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py`
- `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`
- `tests/skd/test_web_skd_agent_loop_on_cpu.py`

### Modified files

- `verl/experimental/agent_loop/__init__.py:15-28`
  - Import and register the new loop classes

### Explicitly not modified

- `verl/experimental/async_skd/manager.py`
- `verl/experimental/async_skd/worker.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/experimental/agent_loop/tool_parser.py`

Reason:
- async rollout and trainer already dispatch by `agent_name`
- reward propagation via `reward_score -> rm_scores` already exists
- Qwen 3.5 already has `qwen3_coder` XML tool parsing support

---

### Task 1: Add protocol client and session-level request semantics

**Files:**
- Create: `verl/experimental/agent_loop/web_osgym_protocol.py`
- Test: `tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py`

- [ ] **Step 1: Write the failing protocol tests**

```python
import base64
from io import BytesIO

import pytest
from PIL import Image

from verl.experimental.agent_loop.web_osgym_protocol import (
    WebOsGymAction,
    WebOsGymClient,
    WebOsGymResponse,
)


def _png_b64() -> str:
    image = Image.new("RGB", (2, 2), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.mark.asyncio
async def test_start_preserves_session_request_id(monkeypatch):
    payload = {
        "request_id": 101,
        "task_id": "12345",
        "status": "ok",
        "text": "A11Y_TREE:\\nroot",
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
            assert json["request_id"] == 101
            assert json["task_id"] == "12345"
            assert json["op"] == "start"
            return _FakeResponse()

    monkeypatch.setattr("verl.experimental.agent_loop.web_osgym_protocol.httpx.AsyncClient", _FakeAsyncClient)

    client = WebOsGymClient(base_url="http://env")
    response = await client.start(request_id=101, task_id="12345", include_a11y=True)

    assert response.request_id == 101
    assert response.task_id == "12345"
    assert response.image is not None


@pytest.mark.asyncio
async def test_action_reuses_same_session_request_id(monkeypatch):
    seen = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": 101, "task_id": "12345", "status": "ok", "text": "A11Y_TREE:\\nnext", "image": {}}

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, timeout):
            seen["payload"] = json
            return _FakeResponse()

    monkeypatch.setattr("verl.experimental.agent_loop.web_osgym_protocol.httpx.AsyncClient", _FakeAsyncClient)

    client = WebOsGymClient(base_url="http://env")
    await client.action(
        request_id=101,
        task_id="12345",
        include_a11y=True,
        actions=[WebOsGymAction(action_type="CLICK", x=10, y=20, button="left", num_clicks=1)],
    )

    assert seen["payload"]["request_id"] == 101
    assert seen["payload"]["task_id"] == "12345"
    assert seen["payload"]["actions"][0]["action_type"] == "CLICK"


@pytest.mark.asyncio
async def test_reward_reuses_same_session_request_id(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"request_id": 101, "task_id": "12345", "status": "ok", "reward": 1.0}

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, timeout):
            assert json == {"request_id": 101, "task_id": "12345", "op": "reward"}
            return _FakeResponse()

    monkeypatch.setattr("verl.experimental.agent_loop.web_osgym_protocol.httpx.AsyncClient", _FakeAsyncClient)

    client = WebOsGymClient(base_url="http://env")
    reward = await client.reward(request_id=101, task_id="12345")
    assert reward == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'verl.experimental.agent_loop.web_osgym_protocol'
```

- [ ] **Step 3: Write minimal protocol client**

```python
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
    request_id: int
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
        return Image.open(BytesIO(base64.b64decode(self.image_b64))).convert("RGB")


class WebOsGymClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _parse_response(payload: dict[str, Any]) -> WebOsGymResponse:
        image_payload = payload.get("image") or {}
        return WebOsGymResponse(
            request_id=int(payload["request_id"]),
            task_id=str(payload["task_id"]),
            status=payload["status"],
            text=payload.get("text"),
            reward=payload.get("reward"),
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
                "request_id": request_id,
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
                "request_id": request_id,
                "task_id": task_id,
                "op": "action",
                "include_a11y": include_a11y,
                "actions": [action.model_dump(exclude_none=True) for action in actions],
            }
        )

    async def reward(self, *, request_id: int, task_id: str) -> float:
        response = await self._post({"request_id": request_id, "task_id": task_id, "op": "reward"})
        assert response.reward is not None
        return float(response.reward)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
cd /home/sogang_nlpy/verl
git add \
  verl/experimental/agent_loop/web_osgym_protocol.py \
  tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py
git commit -m "feat(web-osgym): add protocol client with session semantics"
```

### Task 2: Add the persistent `computer` tool with model-facing Computer 13 schema

**Files:**
- Create: `verl/tools/web_osgym_tool.py`
- Create: `examples/sglang_multiturn/config/tool_config/web_osgym_tool_config.yaml`
- Test: `tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py`

- [ ] **Step 1: Write failing tests for persistent session state and terminal actions**

```python
import pytest

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.web_osgym_tool import WebOsGymTool


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "computer",
                "description": "Apply one low-level computer action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {"type": "string", "description": "Computer 13 action type."},
                        "x": {"type": "integer", "description": "Screen x coordinate."},
                        "y": {"type": "integer", "description": "Screen y coordinate."},
                        "text": {"type": "string", "description": "Typing payload."},
                    },
                    "required": ["action_type"],
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_tool_create_starts_session_and_stores_session_request_id():
    class _FakeClient:
        async def start(self, **kwargs):
            class _Response:
                text = "A11Y_TREE:\\nroot"
                image = None
            return _Response()

    tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
    tool.client = _FakeClient()

    instance_id, response = await tool.create(task_id="12345", request_id=101, include_a11y=True)

    assert response.text == "A11Y_TREE:\\nroot"
    assert tool._instance_dict[instance_id]["request_id"] == 101
    assert tool._instance_dict[instance_id]["task_id"] == "12345"


@pytest.mark.asyncio
async def test_tool_execute_uses_same_session_request_id():
    seen = {}

    class _FakeClient:
        async def action(self, **kwargs):
            seen.update(kwargs)
            class _Response:
                text = "A11Y_TREE:\\nnext"
                image = None
            return _Response()

    tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
    tool.client = _FakeClient()
    tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

    response, reward, metrics = await tool.execute("i1", {"action_type": "CLICK", "x": 1, "y": 2})

    assert response.text == "A11Y_TREE:\\nnext"
    assert reward is None
    assert metrics["terminated"] is False
    assert seen["request_id"] == 101
    assert seen["task_id"] == "12345"


@pytest.mark.asyncio
async def test_tool_execute_marks_done_fail_as_terminal_without_fetching_reward():
    class _FakeClient:
        async def action(self, **kwargs):
            class _Response:
                text = "done"
                image = None
            return _Response()

    tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
    tool.client = _FakeClient()
    tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

    _, _, metrics = await tool.execute("i1", {"action_type": "DONE"})
    assert metrics["terminated"] is True
    assert metrics["termination_reason"] == "model_done"


@pytest.mark.asyncio
async def test_tool_calc_reward_uses_existing_session_request_id():
    class _FakeClient:
        async def reward(self, **kwargs):
            assert kwargs["request_id"] == 101
            assert kwargs["task_id"] == "12345"
            return 1.0

    tool = WebOsGymTool(config={"base_url": "http://env"}, tool_schema=_tool_schema())
    tool.client = _FakeClient()
    tool._instance_dict["i1"] = {"task_id": "12345", "request_id": 101, "include_a11y": False, "reward": None}

    reward = await tool.calc_reward("i1")
    assert reward == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'verl.tools.web_osgym_tool'
```

- [ ] **Step 3: Implement the persistent tool**

```python
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymAction, WebOsGymClient
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class WebOsGymTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.client = WebOsGymClient(base_url=config["base_url"], timeout=config.get("timeout", 30.0))
        self.include_a11y = config.get("include_a11y", False)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        *,
        task_id: str,
        request_id: int,
        include_a11y: bool | None = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        include_a11y = self.include_a11y if include_a11y is None else include_a11y
        response = await self.client.start(request_id=request_id, task_id=task_id, include_a11y=include_a11y)
        self._instance_dict[instance_id] = {
            "task_id": task_id,
            "request_id": request_id,
            "include_a11y": include_a11y,
            "reward": None,
        }
        image = [response.image] if response.image is not None else None
        return instance_id, ToolResponse(text=response.text, image=image)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        state = self._instance_dict[instance_id]
        action = WebOsGymAction(**parameters)
        response = await self.client.action(
            request_id=state["request_id"],
            task_id=state["task_id"],
            include_a11y=state["include_a11y"],
            actions=[action],
        )
        image = [response.image] if response.image is not None else None
        terminated = action.action_type in {"DONE", "FAIL"}
        termination_reason = None
        if action.action_type == "DONE":
            termination_reason = "model_done"
        elif action.action_type == "FAIL":
            termination_reason = "model_fail"
        return ToolResponse(text=response.text, image=image), None, {
            "terminated": terminated,
            "termination_reason": termination_reason,
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        state = self._instance_dict[instance_id]
        if state["reward"] is None:
            state["reward"] = await self.client.reward(request_id=state["request_id"], task_id=state["task_id"])
        return float(state["reward"])

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
```

- [ ] **Step 4: Add the tool config file**

```yaml
tools:
  - class_name: "verl.tools.web_osgym_tool.WebOsGymTool"
    config:
      base_url: "http://127.0.0.1:38081"
      timeout: 30.0
      include_a11y: false
    tool_schema:
      type: "function"
      function:
        name: "computer"
        description: "Apply one low-level computer action to the remote environment."
        parameters:
          type: "object"
          properties:
            action_type:
              type: "string"
              description: "One of MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP, RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY, WAIT, DONE, FAIL."
            x:
              type: "integer"
              description: "Screen x coordinate."
            y:
              type: "integer"
              description: "Screen y coordinate."
            button:
              type: "string"
              description: "Mouse button."
            num_clicks:
              type: "integer"
              description: "Number of clicks."
            dx:
              type: "integer"
              description: "Horizontal scroll delta."
            dy:
              type: "integer"
              description: "Vertical scroll delta."
            text:
              type: "string"
              description: "Typing payload."
            key:
              type: "string"
              description: "Single key."
            keys:
              type: "array"
              description: "Hotkey key list."
          required: ["action_type"]
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit**

```bash
cd /home/sogang_nlpy/verl
git add \
  verl/tools/web_osgym_tool.py \
  examples/sglang_multiturn/config/tool_config/web_osgym_tool_config.yaml \
  tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py
git commit -m "feat(web-osgym): add persistent computer tool"
```

### Task 3: Add shared loop helpers for session lifecycle and reward finalization

**Files:**
- Create: `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
- Test: `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

- [ ] **Step 1: Write failing tests for session start and reward finalization helpers**

```python
import pytest
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

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return "instance-1", ToolResponse(text="initial-observation", image=[Image.new("RGB", (2, 2), "blue")])

    async def calc_reward(self, instance_id, **kwargs):
        self.rewards.append((instance_id, kwargs))
        return 1.0


class _Loop(WebOsGymLoopMixin):
    web_osgym_tool_name = "computer"


@pytest.mark.asyncio
async def test_start_session_stores_instance_id_and_observation():
    loop = _Loop()
    tool = _FakeTool()
    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="loop-req",
        tools_kwargs={"computer": {"create_kwargs": {"task_id": "12345", "request_id": 101, "include_a11y": False}}},
    )
    agent_data._active_tools = {"computer": tool}

    response = await loop._start_web_osgym_session(agent_data, include_a11y=False)

    assert agent_data.extra_fields["web_osgym_instance_id"] == "instance-1"
    assert response.text == "initial-observation"


@pytest.mark.asyncio
async def test_finalize_with_reward_stores_reward_once():
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
    agent_data.extra_fields["web_osgym_instance_id"] = "instance-1"

    await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
    await loop._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")

    assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0
    assert len(tool.rewards) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py::test_start_session_stores_instance_id_and_observation
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'verl.experimental.agent_loop.web_osgym_loop_mixin'
```

- [ ] **Step 3: Implement the shared mixin**

```python
class WebOsGymLoopMixin:
    web_osgym_tool_name = "computer"

    def _get_active_tool(self, agent_data):
        active_tools = getattr(agent_data, "_active_tools", {})
        return active_tools[self.web_osgym_tool_name]

    async def _start_web_osgym_session(self, agent_data, *, include_a11y: bool):
        tool = self._get_active_tool(agent_data)
        kwargs = agent_data.tools_kwargs.get(self.web_osgym_tool_name, {})
        create_kwargs = dict(kwargs.get("create_kwargs", {}))
        create_kwargs["include_a11y"] = include_a11y
        instance_id, start_response = await tool.create(**create_kwargs)
        agent_data.extra_fields["web_osgym_instance_id"] = instance_id
        agent_data.extra_fields["web_osgym_include_a11y"] = include_a11y
        return start_response

    async def _finalize_with_web_osgym_reward(self, agent_data, termination_reason: str) -> None:
        if agent_data.extra_fields.get("web_osgym_reward_fetched"):
            return
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        reward = await tool.calc_reward(instance_id, termination_reason=termination_reason)
        agent_data.extra_fields["web_osgym_reward_fetched"] = True
        agent_data.extra_fields["web_osgym_termination_reason"] = termination_reason
        agent_data.extra_fields["web_osgym_reward_score"] = float(reward)

    async def _release_web_osgym_session(self, agent_data) -> None:
        instance_id = agent_data.extra_fields.get("web_osgym_instance_id")
        if instance_id is None:
            return
        tool = self._get_active_tool(agent_data)
        await tool.release(instance_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py::test_start_session_stores_instance_id_and_observation
pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py::test_finalize_with_reward_stores_reward_once
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
cd /home/sogang_nlpy/verl
git add \
  verl/experimental/agent_loop/web_osgym_loop_mixin.py \
  tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py
git commit -m "feat(web-osgym): add shared session lifecycle helpers"
```

### Task 4: Add `WebToolAgentLoop` using existing `qwen3_coder` XML parsing

**Files:**
- Create: `verl/experimental/agent_loop/web_tool_agent_loop.py`
- Modify: `verl/experimental/agent_loop/__init__.py:15-28`
- Test: `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

- [ ] **Step 1: Write failing loop tests for start, normal action, terminal action, and system stop**

```python
import pytest
from PIL import Image

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.web_tool_agent_loop import WebToolAgentLoop
from verl.tools.base_tool import ToolResponse


class _FakeTool:
    name = "computer"
    tool_schema = None

    def __init__(self):
        self.created = []
        self.executed = []
        self.released = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return "instance-1", ToolResponse(text="initial observation", image=[Image.new("RGB", (2, 2), "blue")])

    async def execute(self, instance_id, parameters, **kwargs):
        self.executed.append((instance_id, parameters))
        return ToolResponse(text="next observation"), None, {"terminated": False, "termination_reason": None}

    async def calc_reward(self, instance_id, **kwargs):
        return 1.0

    async def release(self, instance_id, **kwargs):
        self.released.append(instance_id)


def _build_loop():
    loop = WebToolAgentLoop.__new__(WebToolAgentLoop)
    loop.tools = {"computer": _FakeTool()}
    loop.tool_schemas = []
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 4096
    loop.tool_response_truncate_side = "left"
    loop.response_length = 64
    loop.prompt_length = 64
    loop.tool_parser_name = "qwen3_coder"
    loop.processor = None
    return loop


@pytest.mark.asyncio
async def test_pending_calls_start_and_builds_prompt():
    loop = _build_loop()

    async def _fake_apply_chat_template(messages, **kwargs):
        assert any("initial observation" in str(m.get("content")) for m in messages)
        return [1, 2, 3]

    loop.apply_chat_template = _fake_apply_chat_template

    agent_data = AgentData(
        messages=[{"role": "user", "content": "task instruction"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="loop-req",
        tools_kwargs={"computer": {"create_kwargs": {"task_id": "12345", "request_id": 101}}},
    )
    agent_data._active_tools = loop.tools
    agent_data._active_tool_schemas = []

    state = await WebToolAgentLoop._handle_pending_state(loop, agent_data, {})

    assert state == AgentState.GENERATING
    assert agent_data.extra_fields["web_osgym_instance_id"] == "instance-1"
    assert agent_data.prompt_ids == [1, 2, 3]


@pytest.mark.asyncio
async def test_processing_tools_executes_regular_action_and_returns_to_generating():
    loop = _build_loop()

    async def _fake_apply_chat_template(messages, **kwargs):
        return [10, 11]

    loop.apply_chat_template = _fake_apply_chat_template

    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="loop-req",
        tools_kwargs={},
    )
    agent_data._active_tools = loop.tools
    agent_data.extra_fields["web_osgym_instance_id"] = "instance-1"
    agent_data.tool_calls = [type("Call", (), {"name": "computer", "arguments": '{"action_type":"CLICK","x":1,"y":2}'})()]

    state = await WebToolAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert state == AgentState.GENERATING
    assert loop.tools["computer"].executed[0][1]["action_type"] == "CLICK"


@pytest.mark.asyncio
async def test_processing_tools_treats_done_as_terminal_and_fetches_reward():
    loop = _build_loop()

    async def _execute_done(instance_id, parameters, **kwargs):
        return ToolResponse(text="done"), None, {"terminated": True, "termination_reason": "model_done"}

    loop.tools["computer"].execute = _execute_done

    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="loop-req",
        tools_kwargs={},
    )
    agent_data._active_tools = loop.tools
    agent_data.extra_fields["web_osgym_instance_id"] = "instance-1"
    agent_data.tool_calls = [type("Call", (), {"name": "computer", "arguments": '{"action_type":"DONE"}'})()]

    state = await WebToolAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert state == AgentState.TERMINATED
    assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0


@pytest.mark.asyncio
async def test_generating_system_stop_fetches_reward_before_termination(monkeypatch):
    loop = _build_loop()
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="loop-req",
        tools_kwargs={},
    )
    agent_data._active_tools = loop.tools
    agent_data.extra_fields["web_osgym_instance_id"] = "instance-1"

    async def _base_generating(self, agent_data, sampling_params, ignore_termination=False):
        return AgentState.TERMINATED

    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop._handle_generating_state",
        _base_generating,
    )

    state = await WebToolAgentLoop._handle_generating_state(loop, agent_data, {})

    assert state == AgentState.TERMINATED
    assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'verl.experimental.agent_loop.web_tool_agent_loop'
```

- [ ] **Step 3: Implement `WebToolAgentLoop`**

```python
import json

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin


@register("web_tool_agent")
class WebToolAgentLoop(WebOsGymLoopMixin, ToolAgentLoop):
    async def _handle_pending_state(self, agent_data, sampling_params):
        start_response = await self._start_web_osgym_session(agent_data, include_a11y=False)
        agent_data.messages.append({"role": "tool", "content": start_response.text or ""})
        if start_response.image:
            if agent_data.image_data is None:
                agent_data.image_data = []
            agent_data.image_data.extend(start_response.image)
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination: bool = False):
        next_state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=ignore_termination)
        if next_state == AgentState.TERMINATED and "web_osgym_reward_score" not in agent_data.extra_fields:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        return next_state

    async def _handle_processing_tools_state(self, agent_data):
        tool_call = agent_data.tool_calls[0]
        tool_args = json.loads(tool_call.arguments)
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        tool_response, _, result = await tool.execute(instance_id, tool_args, agent_data=agent_data)
        agent_data.messages.append({"role": "tool", "content": tool_response.text or ""})
        if tool_response.image:
            if agent_data.image_data is None:
                agent_data.image_data = []
            agent_data.image_data.extend(tool_response.image)
        if result.get("terminated"):
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason=result.get("termination_reason") or "model_done",
            )
            return AgentState.TERMINATED
        response_ids = await self.apply_chat_template(
            [agent_data.messages[-1]],
            images=tool_response.image if tool_response.image else None,
            videos=None,
            remove_system_prompt=True,
        )
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        if "web_osgym_reward_score" in output.extra_fields:
            output.reward_score = float(output.extra_fields["web_osgym_reward_score"])
        return output
```

- [ ] **Step 4: Register the loop**

```python
from .web_tool_agent_loop import WebToolAgentLoop

_ = [SingleTurnAgentLoop, ToolAgentLoop, SkdAgentLoop, WebToolAgentLoop]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit**

```bash
cd /home/sogang_nlpy/verl
git add \
  verl/experimental/agent_loop/web_tool_agent_loop.py \
  verl/experimental/agent_loop/__init__.py \
  tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py
git commit -m "feat(web-osgym): add RL web tool agent loop"
```

### Task 5: Add `WebSkdAgentLoop` with teacher-only a11y handling

**Files:**
- Create: `verl/experimental/agent_loop/web_skd_agent_loop.py`
- Modify: `verl/experimental/agent_loop/__init__.py:15-28`
- Test: `tests/skd/test_web_skd_agent_loop_on_cpu.py`

- [ ] **Step 1: Write failing tests for dual observation handling**

```python
import pytest

from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.experimental.agent_loop.web_skd_agent_loop import WebSkdAgentLoop
from verl.tools.base_tool import ToolResponse


def test_web_skd_agent_is_still_skd():
    assert issubclass(WebSkdAgentLoop, SkdAgentLoop)


@pytest.mark.asyncio
async def test_pending_requests_a11y_but_only_teacher_prompt_gets_a11y(monkeypatch):
    class _FakeTool:
        async def create(self, **kwargs):
            assert kwargs["include_a11y"] is True
            return "instance-1", ToolResponse(text="A11Y_TREE:\\nroot", image=None)

    loop = WebSkdAgentLoop.__new__(WebSkdAgentLoop)
    loop.tools = {"computer": _FakeTool()}
    loop.tool_schemas = []
    loop.teacher_key = "data_source"

    async def _fake_tool_pending(self, agent_data, sampling_params):
        return AgentState.GENERATING

    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop._handle_pending_state",
        _fake_tool_pending,
    )

    async def _fake_apply_chat_template(messages, **kwargs):
        return [1, 2, 3]

    loop.apply_chat_template = _fake_apply_chat_template
    loop._build_teacher_messages = lambda messages: list(messages)

    agent_data = AgentData(
        messages=[{"role": "user", "content": "task"}],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="req-1",
        tools_kwargs={"computer": {"create_kwargs": {"task_id": "12345", "request_id": 101}}},
    )
    agent_data._active_tools = loop.tools
    agent_data._active_tool_schemas = []

    state = await WebSkdAgentLoop._handle_pending_state(loop, agent_data, {})

    assert state == AgentState.GENERATING
    assert "A11Y_TREE" not in str(agent_data.messages)
    assert agent_data.extra_fields["web_osgym_teacher_observation_text"].startswith("A11Y_TREE")


@pytest.mark.asyncio
async def test_system_stop_fetches_reward_on_skd_loop(monkeypatch):
    class _FakeTool:
        async def calc_reward(self, instance_id, **kwargs):
            return 1.0

    loop = WebSkdAgentLoop.__new__(WebSkdAgentLoop)
    loop.tools = {"computer": _FakeTool()}

    async def _base_generating(self, agent_data, sampling_params, ignore_termination=False):
        return AgentState.TERMINATED

    monkeypatch.setattr(
        "verl.experimental.agent_loop.skd_agent_loop.SkdAgentLoop._handle_generating_state",
        _base_generating,
    )

    agent_data = AgentData(messages=[], image_data=[], video_data=[], metrics={}, request_id="req-1", tools_kwargs={})
    agent_data._active_tools = loop.tools
    agent_data.extra_fields["web_osgym_instance_id"] = "instance-1"

    state = await WebSkdAgentLoop._handle_generating_state(loop, agent_data, {})

    assert state == AgentState.TERMINATED
    assert agent_data.extra_fields["web_osgym_reward_score"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/skd/test_web_skd_agent_loop_on_cpu.py
```

Expected:

```text
FAIL ... ModuleNotFoundError: No module named 'verl.experimental.agent_loop.web_skd_agent_loop'
```

- [ ] **Step 3: Implement `WebSkdAgentLoop`**

```python
from copy import deepcopy

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.experimental.agent_loop.web_osgym_loop_mixin import WebOsGymLoopMixin


@register("web_skd_agent")
class WebSkdAgentLoop(WebOsGymLoopMixin, SkdAgentLoop):
    def _build_student_observation_text(self, env_text: str | None) -> str:
        return ""

    def _build_teacher_observation_text(self, env_text: str | None) -> str:
        return env_text or ""

    async def _handle_pending_state(self, agent_data, sampling_params):
        start_response = await self._start_web_osgym_session(agent_data, include_a11y=True)
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = self._build_teacher_observation_text(
            start_response.text
        )
        student_obs = self._build_student_observation_text(start_response.text)
        if student_obs:
            agent_data.messages.append({"role": "tool", "content": student_obs})
        if start_response.image:
            if agent_data.image_data is None:
                agent_data.image_data = []
            agent_data.image_data.extend(start_response.image)
        state = await super()._handle_pending_state(agent_data, sampling_params)
        teacher_messages = deepcopy(agent_data.messages)
        teacher_obs = agent_data.extra_fields["web_osgym_teacher_observation_text"]
        if teacher_obs:
            teacher_messages.append({"role": "tool", "content": teacher_obs})
        teacher_prompt_ids = await self.apply_chat_template(
            self._build_teacher_messages(teacher_messages),
            tools=getattr(agent_data, "_active_tool_schemas", self.tool_schemas),
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
        return state

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination: bool = False):
        next_state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=ignore_termination)
        if next_state == AgentState.TERMINATED and "web_osgym_reward_score" not in agent_data.extra_fields:
            await self._finalize_with_web_osgym_reward(agent_data, termination_reason="system_stop")
        return next_state

    async def _handle_processing_tools_state(self, agent_data):
        raise NotImplementedError("Implement student/teacher observation split for env tool responses here.")
```

- [ ] **Step 4: Replace the placeholder `NotImplementedError` with the real dual-update logic**

```python
import json

    async def _handle_processing_tools_state(self, agent_data):
        tool_call = agent_data.tool_calls[0]
        tool_args = json.loads(tool_call.arguments)
        tool = self._get_active_tool(agent_data)
        instance_id = agent_data.extra_fields["web_osgym_instance_id"]
        tool_response, _, result = await tool.execute(instance_id, tool_args, agent_data=agent_data)

        student_obs = self._build_student_observation_text(tool_response.text)
        teacher_obs = self._build_teacher_observation_text(tool_response.text)

        if student_obs:
            student_message = {"role": "tool", "content": student_obs}
            agent_data.messages.append(student_message)
            response_ids = await self.apply_chat_template(
                [student_message],
                images=tool_response.image if tool_response.image else None,
                videos=None,
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            agent_data.user_turns += 1

        teacher_messages = deepcopy(agent_data.messages)
        if teacher_obs:
            teacher_messages.append({"role": "tool", "content": teacher_obs})
        teacher_prompt_ids = await self.apply_chat_template(
            self._build_teacher_messages(teacher_messages),
            tools=getattr(agent_data, "_active_tool_schemas", self.tool_schemas),
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.extra_fields["teacher_prompt_ids"] = teacher_prompt_ids
        agent_data.extra_fields["web_osgym_teacher_observation_text"] = teacher_obs

        if result.get("terminated"):
            await self._finalize_with_web_osgym_reward(
                agent_data,
                termination_reason=result.get("termination_reason") or "model_done",
            )
            return AgentState.TERMINATED

        return AgentState.GENERATING
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/skd/test_web_skd_agent_loop_on_cpu.py
```

Expected:

```text
3 passed
```

- [ ] **Step 6: Commit**

```bash
cd /home/sogang_nlpy/verl
git add \
  verl/experimental/agent_loop/web_skd_agent_loop.py \
  verl/experimental/agent_loop/__init__.py \
  tests/skd/test_web_skd_agent_loop_on_cpu.py
git commit -m "feat(web-osgym): add SKD web agent loop with teacher-only a11y"
```

### Task 6: Add registration smoke tests and loop-selection coverage

**Files:**
- Modify: `tests/experimental/agent_loop/test_basic_agent_loop.py`

- [ ] **Step 1: Add a failing registration smoke test**

```python
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry


def test_web_agent_loops_are_registered():
    assert "web_tool_agent" in _agent_loop_registry
    assert "web_skd_agent" in _agent_loop_registry
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_basic_agent_loop.py::test_web_agent_loops_are_registered
```

Expected:

```text
FAIL ... assertion error until new loops are imported in __init__.py
```

- [ ] **Step 3: Run the smoke test after registration is complete**

Run:

```bash
cd /home/sogang_nlpy/verl
pytest -q tests/experimental/agent_loop/test_basic_agent_loop.py::test_web_agent_loops_are_registered
```

Expected:

```text
1 passed
```

- [ ] **Step 4: Commit**

```bash
cd /home/sogang_nlpy/verl
git add tests/experimental/agent_loop/test_basic_agent_loop.py
git commit -m "test(web-osgym): add agent loop registration smoke coverage"
```

---

## Self-Review

### Spec coverage

- `request_id` as trajectory session id: covered in Task 1 and Task 2
- model-facing `qwen3_coder` XML + Computer 13 action schema: covered in Task 2 and Task 4
- `DONE/FAIL` as terminal actions: covered in Task 2 and Task 4
- `max_length/max_chunks` reward-only finalization: covered in Task 4 and Task 5
- student-only regular RL loop: covered in Task 4
- teacher-only a11y in SKD: covered in Task 5
- reward propagation through `reward_score`: covered in Task 4 and Task 5
- no trainer/manager changes: enforced in File Structure and task scope

### Placeholder scan

No `TODO`, `TBD`, or “fill in later” placeholders remain. The only intermediate placeholder is explicitly replaced within the same task (`NotImplementedError` step followed by replacement step) so the engineer never stops with a partial design.

### Type consistency

Consistent names used throughout:

- protocol client: `WebOsGymClient`
- tool: `WebOsGymTool`
- shared helper: `WebOsGymLoopMixin`
- RL loop: `WebToolAgentLoop`
- SKD loop: `WebSkdAgentLoop`
- final reward field: `reward_score`
- model-facing tool name: `computer`

---

Plan complete and saved to `verl/async_skd/document/web_osgym/implementation_plan.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
