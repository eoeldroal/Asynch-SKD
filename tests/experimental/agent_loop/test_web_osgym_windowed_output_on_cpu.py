from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.web_osgym_windowed_output import build_web_osgym_windowed_agent_loop_outputs
from verl.protocol import DataProto
from verl.utils.rollout_trace import RolloutTraceConfig
from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _FakeTokenizer


def _web_osgym_output() -> AgentLoopOutput:
    response_ids = [11, 12, 90, 91, 13, 14, 92]
    response_mask = [1, 1, 0, 0, 1, 1, 0]
    return AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=[-0.1] * len(response_ids),
        multi_modal_data={"images": ["initial", "after-click"]},
        reward_score=0.75,
        num_turns=5,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "reward_extra_info": {"web_osgym_reward_score": 0.75},
            "web_osgym_generation_windows": [
                {
                    "assistant_turn": 1,
                    "response_start": 0,
                    "response_end": 2,
                    "prompt_ids": [101, 102, 103],
                    "window_used": True,
                    "image_indices": [0],
                    "selected_step_indices": [1],
                },
                {
                    "assistant_turn": 2,
                    "response_start": 4,
                    "response_end": 6,
                    "prompt_ids": [201, 202, 203, 204],
                    "window_used": True,
                    "image_indices": [1],
                    "selected_step_indices": [2],
                },
            ],
        },
    )


def test_web_osgym_windowed_outputs_use_exact_generation_prompt_and_target_only():
    windows, metrics = build_web_osgym_windowed_agent_loop_outputs(_web_osgym_output(), enabled=True)

    assert len(windows) == 2
    assert metrics["web_osgym/window_update_num_samples"] == 2

    assert windows[0].prompt_ids == [101, 102, 103]
    assert windows[0].response_ids == [11, 12]
    assert windows[0].response_mask == [1, 1]
    assert windows[0].multi_modal_data["images"] == ["initial"]
    assert windows[0].reward_score == 0.75

    assert windows[1].prompt_ids == [201, 202, 203, 204]
    assert windows[1].response_ids == [13, 14]
    assert windows[1].response_mask == [1, 1]
    assert windows[1].multi_modal_data["images"] == ["after-click"]
    assert windows[1].extra_fields["web_osgym_window_row"] is True
    assert "web_osgym_generation_windows" not in windows[1].extra_fields


class _WindowedWorker(AgentLoopWorker):
    reward_loop_worker_handles = None
    distillation_enabled = False
    processor = None

    def __init__(self):
        self.rollout_config = OmegaConf.create(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "calculate_log_probs": False,
                "prompt_length": 8,
                "response_length": 128000,
                "val_kwargs": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
                "agent": {"default_agent_loop": "web_tool_agent"},
                "multi_turn": {"web_osgym_window_enable": True},
            }
        )
        self.tokenizer = _FakeTokenizer()
        self.tokenizer.pad_token_id = 0
        self.raw_calls: list[dict[str, Any]] = []

    async def _run_raw_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> AgentLoopOutput:
        self.raw_calls.append(
            {
                "sampling_params": sampling_params,
                "trajectory": trajectory,
                "agent_name": agent_name,
                "trace": trace,
                "kwargs": kwargs,
            }
        )
        return _web_osgym_output()


def test_agent_loop_worker_expands_web_osgym_window_rows_before_update_batch():
    RolloutTraceConfig.reset()
    raw_prompt = [{"role": "user", "content": "do the task"}]
    batch = DataProto.from_dict(
        non_tensors={
            "raw_prompt": np.array([raw_prompt], dtype=object),
            "agent_name": np.array(["web_tool_agent"], dtype=object),
            "uid": np.array(["sample-a"], dtype=object),
            "index": np.array([7], dtype=object),
        },
        meta_info={"global_steps": 12, "validate": False},
    )

    worker = _WindowedWorker()
    output = asyncio.run(worker.generate_sequences(batch))

    assert len(worker.raw_calls) == 1
    assert len(output) == 2
    assert output.batch["prompts"].shape == (2, 8)
    assert output.batch["responses"].shape == (2, 2)
    assert output.batch["response_mask"].sum(dim=1).tolist() == [2, 2]
    assert output.non_tensor_batch["uid"].tolist() == ["sample-a", "sample-a"]
    assert output.non_tensor_batch["index"].tolist() == [7, 7]
    assert output.non_tensor_batch["web_osgym_window_row"].tolist() == [True, True]
    assert output.meta_info["metrics"][0]["web_osgym/window_update_num_samples"] == 2
