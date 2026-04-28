from collections.abc import Mapping
from typing import Any, Optional
from uuid import uuid4

from pydantic import ValidationError

from verl.experimental.agent_loop.web_osgym_protocol import WebOsGymAction, WebOsGymClient
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class WebOsGymTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.client = WebOsGymClient(base_url=config["base_url"], timeout=config.get("timeout", 30.0))
        self.include_a11y = config.get("include_a11y", False)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def restore_instance(
        self,
        instance_id: str,
        *,
        task_id: str,
        request_id: int,
        include_a11y: bool,
        reward: float | None = None,
    ) -> None:
        self._instance_dict[instance_id] = {
            "task_id": task_id,
            "request_id": request_id,
            "include_a11y": include_a11y,
            "reward": reward,
        }

    async def create(
        self,
        instance_id: Optional[str] = None,
        *,
        task_id: str,
        request_id: int,
        include_a11y: bool | None = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        del kwargs
        instance_id = instance_id or str(uuid4())
        include_a11y = self.include_a11y if include_a11y is None else include_a11y
        response = await self.client.start(request_id=request_id, task_id=task_id, include_a11y=include_a11y)
        self.restore_instance(
            instance_id,
            task_id=task_id,
            request_id=request_id,
            include_a11y=include_a11y,
            reward=None,
        )
        image = [response.image] if response.image is not None else None
        response_kwargs = {"text": response.text}
        if image is not None:
            response_kwargs["image"] = image
        return instance_id, ToolResponse(**response_kwargs)

    def _parse_actions(self, parameters: dict[str, Any]) -> list[WebOsGymAction]:
        if not isinstance(parameters, Mapping):
            raise ValueError(f"computer tool arguments must be an object, got {type(parameters).__name__}")

        raw_actions = parameters.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError("computer tool requires a non-empty actions list")

        actions = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, Mapping):
                raise ValueError(
                    f"computer tool actions[{index}] must be an object matching the action schema, "
                    f"got {type(raw_action).__name__}"
                )
            actions.append(WebOsGymAction(**raw_action))

        terminal_actions = [action for action in actions if action.action_type in {"DONE", "FAIL"}]
        if terminal_actions and len(actions) != 1:
            raise ValueError("DONE/FAIL must be sent as a standalone action list")
        return actions

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float | None, dict]:
        del kwargs
        state = self._instance_dict[instance_id]
        try:
            actions = self._parse_actions(parameters)
        except (TypeError, ValueError, ValidationError) as exc:
            return ToolResponse(text=f"Invalid computer action payload: {exc}"), None, {
                "terminated": False,
                "termination_reason": None,
                "action_count": 0,
                "invalid_action": True,
            }

        response = await self.client.action(
            request_id=state["request_id"],
            task_id=state["task_id"],
            include_a11y=state["include_a11y"],
            actions=actions,
        )
        image = [response.image] if response.image is not None else None
        terminal_action = actions[0] if len(actions) == 1 and actions[0].action_type in {"DONE", "FAIL"} else None
        terminated = terminal_action is not None
        termination_reason = None
        if terminal_action and terminal_action.action_type == "DONE":
            termination_reason = "model_done"
        elif terminal_action and terminal_action.action_type == "FAIL":
            termination_reason = "model_fail"
        response_kwargs = {"text": response.text}
        if image is not None:
            response_kwargs["image"] = image
        return ToolResponse(**response_kwargs), None, {
            "terminated": terminated,
            "termination_reason": termination_reason,
            "action_count": len(actions),
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        del kwargs
        state = self._instance_dict[instance_id]
        if state["reward"] is None:
            state["reward"] = await self.client.reward(request_id=state["request_id"], task_id=state["task_id"])
        return float(state["reward"])

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)
