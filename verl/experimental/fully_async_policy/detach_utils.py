# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Optional
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import (
    WEB_OSGYM_ROLLOUT_TRACE_EXTRA_FIELD_KEYS,
    AgentLoopWorker,
    DeferredAgentLoopOutputs,
    _get_rollout_and_model_config,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None
    val_generations: Optional[list[tuple]] = None


class _DeferredAgentLoopMaterializer(AgentLoopWorker):
    def __init__(self, config, tokenizer, processor):
        rollout_config, _model_config = _get_rollout_and_model_config(config)
        self.rollout_config = omega_conf_to_dataclass(rollout_config, RolloutConfig)
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_loop_worker_handles = None
        self.distillation_enabled = False
        self.teacher_server_manager = None


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def addition_process(output: DataProto):
    """collect metirics"""
    metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
    processing_times_list = [item["generate_sequences"] for item in metrics]
    tool_calls_times_list = [item["tool_calls"] for item in metrics]
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def _materialize_deferred_rollout_sample(
    rollout_sample: RolloutSample,
    *,
    tokenizer,
    config,
    processor,
) -> DataProto:
    deferred = rollout_sample.full_batch
    if not isinstance(deferred, DeferredAgentLoopOutputs):
        return deferred
    if config is None:
        raise ValueError("Deferred fully-async rollout samples require config for trainer-side materialization.")
    materializer = _DeferredAgentLoopMaterializer(config=config, tokenizer=tokenizer, processor=processor)
    batch = materializer.materialize_raw_outputs_sync(
        deferred.raw_outputs,
        deferred.input_non_tensor_batch,
        validate=deferred.validate,
    )
    rollout_sample.full_batch = batch
    return batch


def _validate_rollout_sample_batch(batch: DataProto, *, sample_id: str) -> None:
    if batch.batch is None:
        raise ValueError(f"Rollout sample '{sample_id}' is missing tensor batch data.")
    leaked_trace_keys = sorted(set(batch.non_tensor_batch) & WEB_OSGYM_ROLLOUT_TRACE_EXTRA_FIELD_KEYS)
    if leaked_trace_keys:
        raise ValueError(
            f"Rollout sample '{sample_id}' contains Web/OSGym rollout trace fields before batch assembly: "
            f"{leaked_trace_keys}. These fields must be filtered before training."
        )

    if "response_mask" in batch.batch.keys():
        response_mask = batch.batch["response_mask"]
        if response_mask.dim() != 2:
            raise ValueError(
                f"Rollout sample '{sample_id}' has invalid response_mask rank {response_mask.dim()}; expected rank 2."
            )
        supervised_counts = response_mask.sum(dim=-1)
        zero_rows = (supervised_counts <= 0).nonzero(as_tuple=True)[0].tolist()
        if zero_rows:
            raise ValueError(
                f"Rollout sample '{sample_id}' has no supervised response tokens in rows {zero_rows} before batch padding."
            )


def _validate_rollout_samples_for_concat(rollout_samples_batch: list[DataProto], rollout_samples: list[RolloutSample]) -> None:
    if not rollout_samples_batch:
        return

    reference_keys = set(rollout_samples_batch[0].non_tensor_batch.keys())
    reference_sample_id = rollout_samples[0].sample_id
    for rs, batch in zip(rollout_samples, rollout_samples_batch, strict=True):
        current_keys = set(batch.non_tensor_batch.keys())
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            extra = sorted(current_keys - reference_keys)
            raise ValueError(
                "Rollout samples have non_tensor_batch key mismatch before concat: "
                f"reference_sample='{reference_sample_id}' sample='{rs.sample_id}' "
                f"missing={missing} extra={extra}."
            )


def _tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return sum(_tensor_nbytes(item) for item in value.flat)
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_nbytes(item) for item in value)
    return 0


def _summarize_training_payload(batch: DataProto) -> dict[str, Any]:
    window_rows = 0
    if "web_osgym_window_row" in batch.non_tensor_batch:
        window_rows = sum(bool(value) for value in batch.non_tensor_batch["web_osgym_window_row"].tolist())

    multi_modal_inputs = batch.non_tensor_batch.get("multi_modal_inputs")
    multi_modal_tensor_bytes = _tensor_nbytes(multi_modal_inputs) if multi_modal_inputs is not None else 0
    image_count = 0
    if isinstance(multi_modal_inputs, np.ndarray):
        for item in multi_modal_inputs.tolist():
            if not isinstance(item, dict):
                continue
            image_grid_thw = item.get("image_grid_thw")
            if isinstance(image_grid_thw, torch.Tensor):
                image_count += int(image_grid_thw.shape[0])

    block_sizes = []
    if "web_osgym_window_supervision_block_size" in batch.non_tensor_batch:
        block_sizes = sorted(
            {
                int(value.item() if isinstance(value, np.generic) else value)
                for value in batch.non_tensor_batch["web_osgym_window_supervision_block_size"].tolist()
                if value is not None
            }
        )

    return {
        "rows": len(batch),
        "window_rows": window_rows,
        "web_osgym_window_supervision_block_sizes": block_sizes,
        "multi_modal_image_count": image_count,
        "multi_modal_tensor_mib": round(multi_modal_tensor_bytes / (1024**2), 2),
    }


def _validate_param_version_array(name: str, values: np.ndarray) -> list[int]:
    normalized: list[int] = []
    for idx, value in enumerate(values.tolist()):
        if value is None:
            raise ValueError(f"{name} contains None at row {idx}; fully async batch assembly requires integer values.")
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, Integral):
            raise ValueError(
                f"{name} contains non-integer value {value!r} at row {idx}; fully async batch assembly requires integers."
            )
        normalized.append(int(value))
    return normalized


def _pad_tensor_along_dim(
    tensor: torch.Tensor,
    *,
    dim: int,
    left_pad: int = 0,
    right_pad: int = 0,
    pad_value: int | float = 0,
) -> torch.Tensor:
    if left_pad == 0 and right_pad == 0:
        return tensor
    dim = dim if dim >= 0 else tensor.dim() + dim
    tensor_last = torch.movedim(tensor, dim, -1)
    tensor_last = F.pad(tensor_last, (left_pad, right_pad), value=pad_value)
    return torch.movedim(tensor_last, -1, dim)


def _align_rollout_sample_widths(rollout_samples_batch: list[DataProto], tokenizer) -> None:
    if len(rollout_samples_batch) <= 1:
        return

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    target_prompt_width = max(int(batch.batch["prompts"].shape[1]) for batch in rollout_samples_batch if batch.batch is not None)
    target_response_width = max(
        int(batch.batch["responses"].shape[1]) for batch in rollout_samples_batch if batch.batch is not None
    )

    for batch in rollout_samples_batch:
        if batch.batch is None:
            continue

        prompt_width = int(batch.batch["prompts"].shape[1])
        response_width = int(batch.batch["responses"].shape[1])
        prompt_pad = target_prompt_width - prompt_width
        response_pad = target_response_width - response_width
        if prompt_pad == 0 and response_pad == 0:
            continue

        batch.batch["prompts"] = _pad_tensor_along_dim(
            batch.batch["prompts"],
            dim=-1,
            left_pad=prompt_pad,
            pad_value=pad_token_id,
        )
        batch.batch["responses"] = _pad_tensor_along_dim(
            batch.batch["responses"],
            dim=-1,
            right_pad=response_pad,
            pad_value=pad_token_id,
        )

        for key in ("response_mask", "rollout_log_probs", "rm_scores"):
            if key in batch.batch.keys():
                batch.batch[key] = _pad_tensor_along_dim(
                    batch.batch[key],
                    dim=-1,
                    right_pad=response_pad,
                    pad_value=0,
                )

        for key, pad_value in (("input_ids", pad_token_id), ("attention_mask", 0), ("position_ids", 0)):
            if key in batch.batch.keys():
                batch.batch[key] = _pad_tensor_along_dim(
                    batch.batch[key],
                    dim=-1,
                    left_pad=prompt_pad,
                    right_pad=response_pad,
                    pad_value=pad_value,
                )

        for key, pad_value in (("teacher_ids", pad_token_id), ("teacher_logprobs", 0), ("routed_experts", 0)):
            if key in batch.batch.keys():
                batch.batch[key] = _pad_tensor_along_dim(
                    batch.batch[key],
                    dim=1,
                    left_pad=prompt_pad,
                    right_pad=response_pad,
                    pad_value=pad_value,
                )


def _infer_required_multiple_from_balance_fn(balance_batch) -> int | None:
    if balance_batch is None:
        return None
    trainer = getattr(balance_batch, "__self__", None)
    if trainer is None:
        return None

    get_dp_size = getattr(trainer, "_get_dp_size", None)
    actor_rollout_wg = getattr(trainer, "actor_rollout_wg", None)
    if not callable(get_dp_size) or actor_rollout_wg is None:
        return None

    try:
        multiple = int(get_dp_size(actor_rollout_wg, "actor"))
    except Exception:
        return None
    return multiple if multiple > 1 else None


def _pad_batch_to_required_multiple(batch: DataProto, multiple: int | None) -> DataProto:
    if multiple is None or multiple <= 1 or len(batch) == 0:
        return batch

    remainder = len(batch) % multiple
    if remainder == 0:
        return batch

    pad_size = multiple - remainder
    padding = batch.select_idxs([len(batch) - 1]).repeat(pad_size)
    padding.meta_info = {}

    pad_uids = np.array([f"fully_async_pad_{uuid.uuid4().hex}_{idx}" for idx in range(pad_size)], dtype=object)
    if "uid" in padding.non_tensor_batch:
        padding.non_tensor_batch["uid"] = pad_uids.copy()
    if "index" in padding.non_tensor_batch:
        padding.non_tensor_batch["index"] = np.array([-1] * pad_size, dtype=object)
    if "input_pos" in padding.non_tensor_batch:
        padding.non_tensor_batch["input_pos"] = np.array([-1] * pad_size, dtype=object)

    for key in ("response_mask", "rm_scores", "rollout_log_probs"):
        if key in padding.batch:
            padding.batch[key].zero_()

    return DataProto.concat([batch, padding])


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None, processor=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = _materialize_deferred_rollout_sample(rs, tokenizer=tokenizer, config=config, processor=processor)
        batch = addition_process(batch)
        _validate_rollout_sample_batch(batch, sample_id=rs.sample_id)
        rollout_samples_batch.append(batch)

    _validate_rollout_samples_for_concat(rollout_samples_batch, rollout_samples)
    _align_rollout_sample_widths(rollout_samples_batch, tokenizer)
    final_batch = DataProto.concat(rollout_samples_batch)
    final_batch = _pad_batch_to_required_multiple(final_batch, _infer_required_multiple_from_balance_fn(balance_batch))
    payload_summary = _summarize_training_payload(final_batch)
    print(f"[BatchUtils] Training payload summary {payload_summary}")

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    if "min_global_steps" not in final_batch.non_tensor_batch:
        raise ValueError("min_global_steps is missing from fully async assembled batch.")
    if "max_global_steps" not in final_batch.non_tensor_batch:
        raise ValueError("max_global_steps is missing from fully async assembled batch.")

    param_version_start = _validate_param_version_array(
        "min_global_steps", final_batch.non_tensor_batch["min_global_steps"]
    )
    param_version_end = _validate_param_version_array(
        "max_global_steps", final_batch.non_tensor_batch["max_global_steps"]
    )
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean", "score/sum"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
