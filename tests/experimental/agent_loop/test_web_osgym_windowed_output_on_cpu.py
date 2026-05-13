from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DeferredAgentLoopOutputs,
)
from verl.experimental.agent_loop.web_osgym_windowed_output import build_web_osgym_windowed_agent_loop_outputs
from verl.experimental.fully_async_policy.detach_utils import RolloutSample, assemble_batch_from_rollout_samples
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
            "reward_extra_info": {"web_osgym_env_reward_score": 0.75},
            "web_osgym_generation_windows": [
                {
                    "assistant_turn": 1,
                    "response_start": 0,
                    "response_end": 2,
                    "prompt_ids": [101, 102, 103],
                    "window_used": True,
                    "prompt_image_indices": [0],
                    "selected_step_indices": [1],
                    "old_summary_turn_indices": [],
                    "recent_observation_step_indices": [1],
                    "recent_assistant_turn_indices": [],
                    "text_only_recent_step_count": 0,
                },
                {
                    "assistant_turn": 2,
                    "response_start": 4,
                    "response_end": 6,
                    "prompt_ids": [201, 202, 203, 204],
                    "window_used": True,
                    "prompt_image_indices": [1],
                    "selected_step_indices": [2],
                    "old_summary_turn_indices": [1],
                    "recent_observation_step_indices": [2],
                    "recent_assistant_turn_indices": [1],
                    "text_only_recent_step_count": 0,
                },
            ],
        },
    )


def _web_osgym_output_five_turns() -> AgentLoopOutput:
    response_ids = [11, 12, 90, 91, 13, 14, 92, 15, 16, 93, 17, 18, 94, 19, 20]
    response_mask = [1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    return AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=[-0.1] * len(response_ids),
        multi_modal_data={"images": ["obs1", "obs2", "obs3", "obs4", "obs5"]},
        reward_score=1.0,
        num_turns=9,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "reward_extra_info": {"web_osgym_env_reward_score": 1.0},
            "web_osgym_steps": [
                {"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False},
                {"step_idx": 2, "image_start": 1, "image_end": 2, "terminal": False},
                {"step_idx": 3, "image_start": 2, "image_end": 3, "terminal": False},
                {"step_idx": 4, "image_start": 3, "image_end": 4, "terminal": False},
                {"step_idx": 5, "image_start": 4, "image_end": 5, "terminal": False},
            ],
            "web_osgym_generation_windows": [
                {
                    "assistant_turn": 1,
                    "response_start": 0,
                    "response_end": 2,
                    "prompt_ids": [101],
                    "window_used": True,
                    "prompt_image_indices": [0],
                    "selected_step_indices": [1],
                    "old_summary_turn_indices": [],
                    "recent_observation_step_indices": [1],
                    "recent_assistant_turn_indices": [],
                    "text_only_recent_step_count": 0,
                },
                {
                    "assistant_turn": 2,
                    "response_start": 4,
                    "response_end": 6,
                    "prompt_ids": [201],
                    "window_used": True,
                    "prompt_image_indices": [0, 1],
                    "selected_step_indices": [1, 2],
                    "old_summary_turn_indices": [],
                    "recent_observation_step_indices": [1, 2],
                    "recent_assistant_turn_indices": [1],
                    "text_only_recent_step_count": 0,
                },
                {
                    "assistant_turn": 3,
                    "response_start": 7,
                    "response_end": 9,
                    "prompt_ids": [301],
                    "window_used": True,
                    "prompt_image_indices": [0, 1, 2],
                    "selected_step_indices": [1, 2, 3],
                    "old_summary_turn_indices": [],
                    "recent_observation_step_indices": [1, 2, 3],
                    "recent_assistant_turn_indices": [1, 2],
                    "text_only_recent_step_count": 0,
                },
                {
                    "assistant_turn": 4,
                    "response_start": 10,
                    "response_end": 12,
                    "prompt_ids": [401],
                    "window_used": True,
                    "prompt_image_indices": [1, 2, 3],
                    "selected_step_indices": [2, 3, 4],
                    "old_summary_turn_indices": [1],
                    "recent_observation_step_indices": [2, 3, 4],
                    "recent_assistant_turn_indices": [2, 3],
                    "text_only_recent_step_count": 0,
                },
                {
                    "assistant_turn": 5,
                    "response_start": 13,
                    "response_end": 15,
                    "prompt_ids": [501],
                    "window_used": True,
                    "prompt_image_indices": [2, 3, 4],
                    "selected_step_indices": [3, 4, 5],
                    "old_summary_turn_indices": [1, 2],
                    "recent_observation_step_indices": [3, 4, 5],
                    "recent_assistant_turn_indices": [3, 4],
                    "text_only_recent_step_count": 0,
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
    assert windows[1].extra_fields["web_osgym_window_old_summary_turn_indices"] == [1]
    assert windows[1].extra_fields["web_osgym_window_recent_observation_step_indices"] == [2]
    assert windows[1].extra_fields["web_osgym_window_recent_assistant_turn_indices"] == [1]
    assert "web_osgym_generation_windows" not in windows[1].extra_fields


def test_web_osgym_windowed_outputs_keep_exact_prompt_image_order_for_live_recent_history():
    output = _web_osgym_output()
    output.multi_modal_data = {"images": ["obs-2", "obs-3", "obs-4"]}
    output.extra_fields["web_osgym_generation_windows"][1].update(
        {
            "prompt_ids": [201, 202, 203, 204, 205],
            "prompt_image_indices": [0, 1, 2],
            "old_summary_turn_indices": [1],
            "recent_observation_step_indices": [2, 3, 4],
            "recent_assistant_turn_indices": [2, 3],
        }
    )

    windows, _ = build_web_osgym_windowed_agent_loop_outputs(output, enabled=True)

    assert windows[1].prompt_ids == [201, 202, 203, 204, 205]
    assert windows[1].multi_modal_data["images"] == ["obs-2", "obs-3", "obs-4"]
    assert windows[1].response_ids == [13, 14]
    assert windows[1].response_mask == [1, 1]


def test_web_osgym_windowed_outputs_omit_images_for_text_only_prompt_window():
    output = _web_osgym_output()
    output.multi_modal_data = {"images": ["obs-1"]}
    output.extra_fields["web_osgym_generation_windows"][0].update(
        {
            "prompt_ids": [101, 102, 103, 104],
            "prompt_image_indices": [],
            "text_only_recent_step_count": 1,
        }
    )

    windows, _ = build_web_osgym_windowed_agent_loop_outputs(output, enabled=True)

    assert windows[0].prompt_ids == [101, 102, 103, 104]
    assert "images" not in windows[0].multi_modal_data
    assert windows[0].response_ids == [11, 12]


def test_web_osgym_windowed_outputs_group_three_supervised_turns_with_zero_loss_warmup():
    windows, metrics = build_web_osgym_windowed_agent_loop_outputs(
        _web_osgym_output_five_turns(),
        enabled=True,
        supervision_block_size=3,
        carry_turn_budget=5,
    )

    assert len(windows) == 2
    assert metrics["web_osgym/window_update_num_samples"] == 2
    assert metrics["web_osgym/window_update_max_target_tokens"] == 6

    assert windows[0].prompt_ids == [101]
    assert windows[0].response_ids == [11, 12, 90, 91, 13, 14]
    assert windows[0].response_mask == [1, 1, 0, 0, 1, 1]
    assert windows[0].multi_modal_data["images"] == ["obs1", "obs2"]
    assert windows[0].extra_fields["web_osgym_window_row_idx"] == 1
    assert windows[0].extra_fields["web_osgym_window_row_count"] == 2
    assert "web_osgym_steps" not in windows[0].extra_fields
    assert "web_osgym_generation_windows" not in windows[0].extra_fields

    assert windows[1].prompt_ids == [101]
    assert windows[1].response_ids == [11, 12, 90, 91, 13, 14, 92, 15, 16, 93, 17, 18, 94, 19, 20]
    assert windows[1].response_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
    assert windows[1].multi_modal_data["images"] == ["obs1", "obs2", "obs3", "obs4", "obs5"]
    assert windows[1].extra_fields["web_osgym_window_row_idx"] == 2
    assert windows[1].extra_fields["web_osgym_window_row_count"] == 2
    assert "web_osgym_steps" not in windows[1].extra_fields
    assert "web_osgym_generation_windows" not in windows[1].extra_fields


class _FakeBatchFeature(dict):
    def convert_to_tensors(self, *_args, **_kwargs):
        return self


class _RecordingProcessor:
    image_token_id = -1
    video_token_id = -2

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, text, images, videos, video_metadata, return_tensors, do_sample_frames):
        del videos, video_metadata, return_tensors, do_sample_frames
        recorded_images = list(images) if images is not None else None
        self.calls.append({"text": list(text), "images": recorded_images})
        image_count = 0 if recorded_images is None else len(recorded_images)
        return _FakeBatchFeature(
            {
                "pixel_values": torch.ones((image_count, 2), dtype=torch.float16),
                "image_grid_thw": torch.ones((image_count, 3), dtype=torch.long),
            }
        )

    def get_rope_index(self, *, input_ids, attention_mask, **_kwargs):
        del attention_mask
        seq_len = input_ids.shape[1]
        return torch.zeros((3, 1, seq_len), dtype=torch.long), None


class _LargeRecordingProcessor(_RecordingProcessor):
    def __call__(self, *, text, images, videos, video_metadata, return_tensors, do_sample_frames):
        del videos, video_metadata, return_tensors, do_sample_frames
        recorded_images = list(images) if images is not None else None
        self.calls.append({"text": list(text), "images": recorded_images})
        image_count = 0 if recorded_images is None else len(recorded_images)
        return _FakeBatchFeature(
            {
                "pixel_values": torch.ones((image_count, 3, 256, 256), dtype=torch.float16),
                "image_grid_thw": torch.ones((image_count, 3), dtype=torch.long),
            }
        )


class _WindowedWorker(AgentLoopWorker):
    reward_loop_worker_handles = None
    distillation_enabled = False
    processor = None

    def __init__(self):
        self.rollout_config = OmegaConf.create(
            {
                "name": "sglang",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
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
        self.output_override: AgentLoopOutput | None = None

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
        if self.output_override is not None:
            return self.output_override
        return _web_osgym_output()


def _trainer_config_for_worker(worker: _WindowedWorker):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": OmegaConf.to_container(worker.rollout_config, resolve=True),
                "model": {},
            }
        }
    )


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


def test_agent_loop_worker_does_not_export_rollout_trace_fields_to_training_tensordict():
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

    output_override = _web_osgym_output_five_turns()
    output_override.extra_fields["mini_step_image_spans"] = [
        {"step_idx": idx, "image_start": idx - 1, "image_end": idx} for idx in range(1, 6)
    ]
    output_override.extra_fields["web_osgym_assistant_turns"] = [
        {
            "assistant_turn": idx,
            "observation_step_idx": idx,
            "response_start": idx * 2,
            "response_end": idx * 2 + 1,
            "response_text": "x" * 1000,
            "actions": [{"action_type": "CLICK", "x": idx, "y": idx}],
        }
        for idx in range(1, 6)
    ]
    output_override.extra_fields["web_osgym_unit_trace"] = {
        "rollout_context": "windowed_prompt",
        "backprop_context": "windowed_generation_rows",
        "generation_window_count": 5,
        "step_count": 5,
    }
    output_override.extra_fields["min_global_steps"] = 12
    output_override.extra_fields["max_global_steps"] = 12

    worker = _WindowedWorker()
    worker.output_override = output_override

    output = asyncio.run(worker.generate_sequences(batch))
    training_tensordict = output.to_tensordict()

    heavy_trace_keys = {
        "web_osgym_generation_windows",
        "web_osgym_steps",
        "mini_step_image_spans",
        "web_osgym_assistant_turns",
    }
    assert heavy_trace_keys.isdisjoint(output.non_tensor_batch)
    assert heavy_trace_keys.isdisjoint(training_tensordict.keys())
    assert output.non_tensor_batch["web_osgym_window_row"].tolist() == [True] * len(output)
    assert "web_osgym_generation_window" in output.non_tensor_batch
    assert "prompt_ids" not in output.non_tensor_batch["web_osgym_generation_window"][0]


def test_windowed_web_osgym_rollout_sample_assembly_keeps_training_tensordict_lean():
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

    output_override = _web_osgym_output_five_turns()
    output_override.extra_fields["web_osgym_steps"] = [
        {
            "step_idx": idx,
            "assistant_turn": idx,
            "user_turn": idx,
            "phase": "after_action",
            "text": "observation " * 500,
            "text_len": 6000,
            "action_names": ["CLICK"],
            "actions": [{"action_type": "CLICK", "x": idx, "y": idx}],
            "image_start": idx - 1,
            "image_end": idx,
            "terminal": False,
            "termination_reason": None,
        }
        for idx in range(1, 6)
    ]
    output_override.extra_fields["mini_step_image_spans"] = [
        {"step_idx": idx, "image_start": idx - 1, "image_end": idx} for idx in range(1, 6)
    ]
    output_override.extra_fields["web_osgym_assistant_turns"] = [
        {
            "assistant_turn": idx,
            "observation_step_idx": idx,
            "response_start": idx * 2,
            "response_end": idx * 2 + 1,
            "response_text": "assistant response " * 500,
            "actions": [{"action_type": "CLICK", "x": idx, "y": idx}],
        }
        for idx in range(1, 6)
    ]
    output_override.extra_fields["min_global_steps"] = 12
    output_override.extra_fields["max_global_steps"] = 12

    worker = _WindowedWorker()
    worker.output_override = output_override
    generated = asyncio.run(worker.generate_sequences(batch))
    sample = RolloutSample(
        full_batch=generated,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )

    assembled = assemble_batch_from_rollout_samples([sample], _FakeTokenizer(), config=None, balance_batch=None)
    training_tensordict = assembled.to_tensordict()

    heavy_trace_keys = {
        "web_osgym_generation_windows",
        "web_osgym_steps",
        "mini_step_image_spans",
        "web_osgym_assistant_turns",
    }
    assert len(assembled) == 5
    assert heavy_trace_keys.isdisjoint(assembled.non_tensor_batch)
    assert heavy_trace_keys.isdisjoint(training_tensordict.keys())
    assert assembled.batch["response_mask"].sum(dim=1).tolist() == [2, 2, 2, 2, 2]
    assert assembled.non_tensor_batch["uid"].tolist() == ["sample-a"] * 5
    assert assembled.non_tensor_batch["web_osgym_window_row"].tolist() == [True] * 5


def test_agent_loop_worker_expands_three_turn_blocks_before_update_batch():
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
    worker.rollout_config.multi_turn.web_osgym_window_supervision_block_size = 3
    worker.output_override = _web_osgym_output_five_turns()
    output = asyncio.run(worker.generate_sequences(batch))

    assert len(worker.raw_calls) == 1
    assert len(output) == 2
    assert output.batch["response_mask"].sum(dim=1).tolist() == [4, 6]
    assert output.non_tensor_batch["uid"].tolist() == ["sample-a", "sample-a"]
    assert output.non_tensor_batch["index"].tolist() == [7, 7]
    assert output.non_tensor_batch["web_osgym_window_row"].tolist() == [True, True]
    assert output.non_tensor_batch["web_osgym_window_supervision_block_size"].tolist() == [3, 3]
    assert output.meta_info["metrics"][0]["web_osgym/window_update_num_samples"] == 2


def test_agent_loop_worker_passes_block_row_images_into_multi_modal_processor():
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
    worker.rollout_config.multi_turn.web_osgym_window_supervision_block_size = 3
    output_override = _web_osgym_output_five_turns()
    output_override.extra_fields["min_global_steps"] = 12
    output_override.extra_fields["max_global_steps"] = 12
    worker.output_override = output_override
    worker.processor = _RecordingProcessor()

    output = asyncio.run(worker.generate_sequences(batch))

    assert len(output) == 2
    assert len(worker.processor.calls) == 2
    assert worker.processor.calls[0]["images"] == ["obs1", "obs2"]
    assert worker.processor.calls[1]["images"] == ["obs1", "obs2", "obs3", "obs4", "obs5"]
    assert output.non_tensor_batch["multi_modal_inputs"].shape == (2,)


def test_three_block_windowed_rollout_assembly_uses_compacted_training_payload():
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
    worker.rollout_config.multi_turn.web_osgym_window_supervision_block_size = 3
    output_override = _web_osgym_output_five_turns()
    output_override.extra_fields["min_global_steps"] = 12
    output_override.extra_fields["max_global_steps"] = 12
    worker.output_override = output_override
    worker.processor = _RecordingProcessor()

    generated = asyncio.run(worker.generate_sequences(batch))
    sample = RolloutSample(
        full_batch=generated,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )
    assembled = assemble_batch_from_rollout_samples([sample], _FakeTokenizer(), config=None, balance_batch=None)
    training_tensordict = assembled.to_tensordict()

    assert len(generated) == 2
    assert len(assembled) == 2
    assert assembled.non_tensor_batch["web_osgym_window_supervision_block_size"].tolist() == [3, 3]
    assert assembled.batch["response_mask"].sum(dim=1).tolist() == [4, 6]
    assert [item["pixel_values"].shape[0] for item in assembled.non_tensor_batch["multi_modal_inputs"]] == [2, 5]
    assert "web_osgym_steps" not in assembled.non_tensor_batch
    assert "web_osgym_generation_windows" not in assembled.non_tensor_batch
    assert "web_osgym_steps" not in training_tensordict.keys()
    assert "web_osgym_generation_windows" not in training_tensordict.keys()


def test_deferred_windowed_materialization_matches_eager_training_path():
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

    eager_worker = _WindowedWorker()
    eager_worker.processor = _RecordingProcessor()
    eager_worker.output_override = _web_osgym_output_five_turns()
    eager_worker.output_override.extra_fields["min_global_steps"] = 12
    eager_worker.output_override.extra_fields["max_global_steps"] = 12
    eager_generated = asyncio.run(eager_worker.generate_sequences(batch))
    eager_sample = RolloutSample(
        full_batch=eager_generated,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )
    eager_assembled = assemble_batch_from_rollout_samples(
        [eager_sample],
        _FakeTokenizer(),
        config=None,
        balance_batch=None,
    )

    deferred_worker = _WindowedWorker()
    deferred_worker.processor = _RecordingProcessor()
    deferred_worker.output_override = _web_osgym_output_five_turns()
    deferred_worker.output_override.extra_fields["min_global_steps"] = 12
    deferred_worker.output_override.extra_fields["max_global_steps"] = 12
    deferred_payload = asyncio.run(deferred_worker.generate_sequences_compact(batch))

    assert isinstance(deferred_payload, DeferredAgentLoopOutputs)
    assert len(deferred_payload.raw_outputs) == 1
    assert deferred_worker.processor.calls == []

    deferred_sample = RolloutSample(
        full_batch=deferred_payload,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )
    deferred_assembled = assemble_batch_from_rollout_samples(
        [deferred_sample],
        _FakeTokenizer(),
        config=_trainer_config_for_worker(deferred_worker),
        balance_batch=None,
        processor=deferred_worker.processor,
    )

    assert len(deferred_worker.processor.calls) == 5
    assert tuple(deferred_assembled.batch["prompts"].shape) == tuple(eager_assembled.batch["prompts"].shape)
    assert tuple(deferred_assembled.batch["responses"].shape) == tuple(eager_assembled.batch["responses"].shape)
    assert torch.equal(deferred_assembled.batch["prompts"], eager_assembled.batch["prompts"])
    assert torch.equal(deferred_assembled.batch["responses"], eager_assembled.batch["responses"])
    assert torch.equal(deferred_assembled.batch["response_mask"], eager_assembled.batch["response_mask"])
    assert torch.equal(deferred_assembled.batch["attention_mask"], eager_assembled.batch["attention_mask"])
    assert deferred_assembled.non_tensor_batch["uid"].tolist() == eager_assembled.non_tensor_batch["uid"].tolist()
    assert deferred_assembled.non_tensor_batch["web_osgym_window_row"].tolist() == [True] * 5
    assert [item["pixel_values"].shape[0] for item in deferred_assembled.non_tensor_batch["multi_modal_inputs"]] == [1, 2, 3, 3, 3]


def test_deferred_windowed_payload_is_smaller_than_eager_queue_payload():
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

    eager_worker = _WindowedWorker()
    eager_worker.processor = _LargeRecordingProcessor()
    eager_worker.output_override = _web_osgym_output_five_turns()
    eager_worker.output_override.extra_fields["min_global_steps"] = 12
    eager_worker.output_override.extra_fields["max_global_steps"] = 12
    eager_generated = asyncio.run(eager_worker.generate_sequences(batch))
    eager_sample = RolloutSample(
        full_batch=eager_generated,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )

    deferred_worker = _WindowedWorker()
    deferred_worker.processor = _LargeRecordingProcessor()
    deferred_worker.output_override = _web_osgym_output_five_turns()
    deferred_worker.output_override.extra_fields["min_global_steps"] = 12
    deferred_worker.output_override.extra_fields["max_global_steps"] = 12
    deferred_payload = asyncio.run(deferred_worker.generate_sequences_compact(batch))
    deferred_sample = RolloutSample(
        full_batch=deferred_payload,
        sample_id="sample-a",
        epoch=0,
        rollout_status={"count/total_generated_samples": 1},
    )

    eager_size = len(ray.cloudpickle.dumps(eager_sample))
    deferred_size = len(ray.cloudpickle.dumps(deferred_sample))

    assert deferred_size < eager_size
    assert deferred_size * 3 < eager_size
