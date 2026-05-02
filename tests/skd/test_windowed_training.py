from __future__ import annotations

import asyncio

import numpy as np
import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.experimental.async_skd.windowed_training import WindowedSkdConfig, build_windowed_agent_loop_outputs
from tests.experimental.agent_loop.test_agent_loop_extra_fields_schema_on_cpu import _FakeTokenizer


class _PostprocessWorker(AsyncSkdAgentLoopWorker):
    reward_loop_worker_handles = None
    distillation_enabled = False
    stream_teacher_with_rollout = False
    processor = None

    def __init__(self):
        self.rollout_config = OmegaConf.create(
            {
                "prompt_length": 4,
                "response_length": 8,
                "agent": {"default_agent_loop": "skd_agent"},
            }
        )
        self.distillation_config = OmegaConf.create(
            {
                "skd": {
                    "windowed_training_enabled": True,
                    "window_history_n": 5,
                    "window_max_images_per_sample": 6,
                }
            }
        )
        self.tokenizer = _FakeTokenizer()
        self.tokenizer.pad_token_id = 0


def _teacher_rows(count: int, width: int = 2) -> list[list[int]]:
    return [[idx, idx + 100][:width] for idx in range(count)]


def _teacher_logprobs(count: int, width: int = 2) -> list[list[float]]:
    return [[-float(idx), -float(idx + 100)][:width] for idx in range(count)]


def _output() -> AgentLoopOutput:
    response_ids = [11, 12, 90, 91, 13, 14, 15]
    response_mask = [1, 1, 0, 0, 1, 1, 1]
    return AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        response_mask=response_mask,
        multi_modal_data={"images": ["obs1", "obs2"]},
        reward_score=1.0,
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": _teacher_rows(len(response_ids)),
            "teacher_logprobs_list": _teacher_logprobs(len(response_ids)),
            "mini_step_image_spans": [
                {"step_idx": 1, "image_start": 0, "image_end": 1, "terminal": False},
                {"step_idx": 2, "image_start": 1, "image_end": 2, "terminal": False},
            ],
        },
    )


def _overlength_prompt_output() -> AgentLoopOutput:
    output = _output()
    output.prompt_ids = [1, 2, 3, 4, 5, 6]
    return output


def test_windowed_outputs_split_contiguous_response_runs():
    windows, metrics = build_windowed_agent_loop_outputs(
        _output(),
        config=WindowedSkdConfig(enabled=True, history_n=5, max_images_per_sample=6),
    )

    assert len(windows) == 2
    assert metrics["window/num_samples"] == 2

    assert windows[0].response_ids == [11, 12]
    assert windows[0].response_mask == [1, 1]
    assert windows[0].extra_fields["teacher_ids_list"] == [[0, 100], [1, 101]]

    assert windows[1].response_ids == [11, 12, 90, 91, 13, 14, 15]
    assert windows[1].response_mask == [0, 0, 0, 0, 1, 1, 1]
    assert windows[1].extra_fields["teacher_ids_list"][:4] == [[0, 0]] * 4
    assert windows[1].extra_fields["teacher_ids_list"][4:] == [[4, 104], [5, 105], [6, 106]]


def test_windowed_outputs_bound_images_and_keep_current_observation():
    windows, metrics = build_windowed_agent_loop_outputs(
        _output(),
        config=WindowedSkdConfig(enabled=True, history_n=0, max_images_per_sample=1),
    )

    assert len(windows) == 2
    assert windows[0].multi_modal_data["images"] == ["obs1"]
    assert windows[1].multi_modal_data["images"] == ["obs2"]
    assert windows[1].response_ids == [90, 91, 13, 14, 15]
    assert windows[1].response_mask == [0, 0, 1, 1, 1]
    assert metrics["window/max_images"] == 1


def test_windowed_outputs_require_teacher_alignment():
    bad = _output()
    bad.extra_fields["teacher_ids_list"] = bad.extra_fields["teacher_ids_list"][:-1]

    with pytest.raises(ValueError, match="response-relative teacher rows"):
        build_windowed_agent_loop_outputs(
            bad,
            config=WindowedSkdConfig(enabled=True),
        )


def test_worker_postprocess_expands_windowed_outputs_through_real_verl_path():
    worker = _PostprocessWorker()
    raw_prompt = [{"role": "user", "content": "hi"}]
    input_non_tensor_batch = {
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "agent_name": np.array(["skd_agent"], dtype=object),
        "index": np.array([7], dtype=object),
    }

    batch = asyncio.run(
        worker._postprocess_completed_skd_output(
            _output(),
            validate=False,
            input_non_tensor_batch=input_non_tensor_batch,
        )
    )

    assert len(batch) == 2
    assert batch.batch["responses"].shape == (2, 8)
    assert batch.batch["response_mask"][0].sum().item() == 2
    assert batch.batch["response_mask"][1].sum().item() == 3
    assert batch.batch["teacher_ids"].shape[:2] == (2, 12)
    assert batch.non_tensor_batch["raw_prompt"].shape[0] == 2
    assert "window_metrics" not in batch.meta_info
    assert batch.meta_info["metrics"][0]["window/num_samples"] == 2
    assert "window/num_samples" not in batch.meta_info["metrics"][1]


def test_worker_postprocess_uses_completed_prompt_width_when_it_exceeds_config():
    worker = _PostprocessWorker()
    raw_prompt = [{"role": "user", "content": "hi"}]
    input_non_tensor_batch = {
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "agent_name": np.array(["skd_agent"], dtype=object),
        "index": np.array([7], dtype=object),
    }

    batch = asyncio.run(
        worker._postprocess_completed_skd_output(
            _overlength_prompt_output(),
            validate=False,
            input_non_tensor_batch=input_non_tensor_batch,
        )
    )

    assert len(batch) == 2
    assert batch.batch["prompts"].shape == (2, 6)
    assert batch.batch["input_ids"].shape == (2, 14)
    assert batch.batch["teacher_ids"].shape[:2] == (2, 14)


def test_worker_postprocess_preserves_identity_keys_when_reward_loop_owns_metadata():
    worker = _PostprocessWorker()
    worker.reward_loop_worker_handles = object()
    raw_prompt = [{"role": "user", "content": "hi"}]
    input_non_tensor_batch = {
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "agent_name": np.array(["skd_agent"], dtype=object),
        "uid": np.array(["sample-a"], dtype=object),
        "index": np.array([7], dtype=object),
    }

    batch = asyncio.run(
        worker._postprocess_completed_skd_output(
            _output(),
            validate=False,
            input_non_tensor_batch=input_non_tensor_batch,
        )
    )

    assert len(batch) == 2
    assert batch.non_tensor_batch["uid"].tolist() == ["sample-a", "sample-a"]
    assert batch.non_tensor_batch["index"].tolist() == [7, 7]


def test_worker_postprocess_overwrites_stale_identity_keys_from_output_extras():
    worker = _PostprocessWorker()
    output = _output()
    output.extra_fields["uid"] = None
    raw_prompt = [{"role": "user", "content": "hi"}]
    input_non_tensor_batch = {
        "raw_prompt": np.array([raw_prompt], dtype=object),
        "agent_name": np.array(["skd_agent"], dtype=object),
        "uid": np.array(["sample-a"], dtype=object),
        "index": np.array([7], dtype=object),
    }

    batch = asyncio.run(
        worker._postprocess_completed_skd_output(
            output,
            validate=False,
            input_non_tensor_batch=input_non_tensor_batch,
        )
    )

    assert batch.non_tensor_batch["uid"].tolist() == ["sample-a", "sample-a"]
    assert batch.non_tensor_batch["index"].tolist() == [7, 7]
