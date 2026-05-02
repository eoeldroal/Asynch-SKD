"""Worker primitives for bounded asynchronous SKD rollout."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    RolloutTraceConfig,
    get_trajectory_info,
)
from verl.experimental.async_skd.events import async_skd_event_context
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.experimental.async_skd.windowed_training import WindowedSkdConfig, build_windowed_agent_loop_outputs
from verl.protocol import DataProto

if TYPE_CHECKING:
    from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop


_ASYNC_SKD_INPUT_NON_TENSOR_BATCH = "async_skd_input_non_tensor_batch"
_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))
_DEFAULT_SKD_AGENT_NAME = "skd_agent"


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


class AsyncSkdAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that owns async-SKD-specific execution primitives."""

    @staticmethod
    def _object_array(value: Any) -> np.ndarray:
        array = np.empty(1, dtype=object)
        array[0] = value
        return array

    def _build_sampling_params(self, *, validate: bool) -> dict[str, Any]:
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        return sampling_params

    def _ensure_agent_name(self, batch: DataProto) -> None:
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = self.rollout_config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop], dtype=object)

    def _single_kwargs(self, batch: DataProto) -> dict[str, Any]:
        return {key: value[0] for key, value in batch.non_tensor_batch.items()}

    def _single_input_non_tensor_batch(self, batch: DataProto) -> dict[str, np.ndarray]:
        return {key: value[:1].copy() for key, value in batch.non_tensor_batch.items()}

    def _input_non_tensor_from_partial(self, partial_state: SkdPartialState) -> dict[str, np.ndarray]:
        saved = partial_state.extra_fields.get(_ASYNC_SKD_INPUT_NON_TENSOR_BATCH)
        if saved is None:
            raw_prompt = partial_state.extra_fields.get("raw_prompt", partial_state.messages)
            agent_name = str(partial_state.extra_fields.get("agent_name", _DEFAULT_SKD_AGENT_NAME))
            return {
                "raw_prompt": self._object_array(raw_prompt),
                "agent_name": np.array([agent_name], dtype=object),
            }
        return {key: np.asarray(value, dtype=object) for key, value in saved.items()}

    @staticmethod
    def _agent_name_from_input_non_tensor_batch(
        input_non_tensor_batch: dict[str, np.ndarray],
        *,
        default: str = _DEFAULT_SKD_AGENT_NAME,
    ) -> str:
        values = input_non_tensor_batch.get("agent_name")
        if values is None or len(values) == 0:
            return default
        return str(values[0])

    def _strip_internal_async_skd_extra_fields(self, output: AgentLoopOutput) -> None:
        output.extra_fields.pop(_ASYNC_SKD_INPUT_NON_TENSOR_BATCH, None)

    def _windowed_skd_config(self) -> WindowedSkdConfig:
        skd_config = getattr(getattr(self, "distillation_config", None), "skd", None)
        if skd_config is None:
            return WindowedSkdConfig()
        return WindowedSkdConfig(
            enabled=bool(getattr(skd_config, "windowed_training_enabled", False)),
            history_n=int(getattr(skd_config, "window_history_n", 5)),
            max_images_per_sample=getattr(skd_config, "window_max_images_per_sample", 6),
        )

    @staticmethod
    def _repeat_input_non_tensor_batch(input_non_tensor_batch: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
        if count == 1:
            return {key: value[:1].copy() for key, value in input_non_tensor_batch.items()}
        repeated: dict[str, np.ndarray] = {}
        for key, value in input_non_tensor_batch.items():
            arr = np.empty(count, dtype=object)
            first = value[0] if len(value) else None
            arr[:] = [deepcopy(first) for _ in range(count)]
            repeated[key] = arr
        return repeated

    async def _postprocess_completed_skd_output(
        self,
        output: AgentLoopOutput,
        *,
        validate: bool,
        input_non_tensor_batch: dict[str, np.ndarray],
    ) -> DataProto:
        self._strip_internal_async_skd_extra_fields(output)
        windowed_outputs, window_metrics = build_windowed_agent_loop_outputs(
            output,
            config=self._windowed_skd_config(),
            tokenizer=self.tokenizer,
        )
        prompt_length_override = max(
            int(self.rollout_config.prompt_length),
            max((len(windowed_output.prompt_ids) for windowed_output in windowed_outputs), default=0),
        )
        postprocess_kwargs = self._single_kwargs(DataProto.from_dict(non_tensors=input_non_tensor_batch))
        internal_outputs = [
            await self._agent_loop_postprocess(
                windowed_output,
                validate,
                **postprocess_kwargs,
                _prompt_length_override=prompt_length_override,
            )
            for windowed_output in windowed_outputs
        ]
        repeated_input_batch = self._repeat_input_non_tensor_batch(input_non_tensor_batch, len(internal_outputs))
        completed_batch = self._postprocess(
            internal_outputs,
            input_non_tensor_batch=repeated_input_batch,
            validate=validate,
        )
        for key in ("uid", "index", "input_pos"):
            if key in repeated_input_batch:
                completed_batch.non_tensor_batch[key] = repeated_input_batch[key].copy()
        if window_metrics:
            completed_batch.meta_info["metrics"][0].update(window_metrics)
        return completed_batch

    async def generate_sequence_single(
        self,
        batch: DataProto,
        *,
        async_skd_context: dict[str, Any] | None = None,
    ) -> DataProto:
        """Generate one sequence from agent loop without changing the batched API contract.

        This method is intentionally kept out of the base ``AgentLoopWorker``.
        Async SKD schedulers use it as a sample-level execution primitive while
        existing trainer paths keep calling the original batched method.
        """
        if len(batch) != 1:
            raise ValueError(f"generate_sequence_single expects exactly one sample, got batch size {len(batch)}.")

        validate = batch.meta_info.get("validate", False)
        sampling_params = self._build_sampling_params(validate=validate)

        # by default, we assume it's a single turn agent
        self._ensure_agent_name(batch)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(1)

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        trace_this_sample = max_samples_per_worker is None or max_samples_per_worker >= 1

        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        kwargs = {k: v[0] for k, v in batch.non_tensor_batch.items()}
        with async_skd_event_context(**(async_skd_context or {})):
            internal_output = await self._run_agent_loop(
                sampling_params, trajectory_info[0], trace=trace_this_sample, **kwargs
            )

        output = self._postprocess(
            [internal_output],
            input_non_tensor_batch=batch.non_tensor_batch,
            validate=validate,
        )
        return output

    async def generate_skd_until_boundary(
        self,
        batch: DataProto | None = None,
        *,
        partial_state: SkdPartialState | None = None,
        sample_id: str,
        logical_step: int,
        source_type: str,
        agent_name: str = "skd_agent",
        async_skd_context: dict[str, Any] | None = None,
    ) -> AsyncSkdSample:
        """Run an SKD sample until completion or the next exportable boundary."""
        if (batch is None) == (partial_state is None):
            raise ValueError("generate_skd_until_boundary expects exactly one of batch or partial_state")

        if batch is not None:
            if len(batch) != 1:
                raise ValueError(f"generate_skd_until_boundary expects single-sample batch, got {len(batch)}")
            self._ensure_agent_name(batch)
            kwargs = self._single_kwargs(batch)
            agent_name = kwargs.pop("agent_name")
            validate = batch.meta_info.get("validate", False)
            input_non_tensor_batch = self._single_input_non_tensor_batch(batch)
        else:
            assert partial_state is not None
            kwargs = {}
            validate = False
            input_non_tensor_batch = self._input_non_tensor_from_partial(partial_state)
            agent_name = self._agent_name_from_input_non_tensor_batch(input_non_tensor_batch, default=agent_name)
        _trace_async_skd(
            "worker.generate_until_boundary.entry",
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            has_batch=batch is not None,
            partial_request_id=None if partial_state is None else partial_state.request_id,
            incoming_teacher_replica_id=kwargs.get("teacher_replica_id")
            if batch is not None
            else partial_state.extra_fields.get("teacher_replica_id"),
            incoming_teacher_routing_key=kwargs.get("data_source")
            if batch is not None
            else partial_state.extra_fields.get("teacher_routing_key"),
            incoming_non_tensor_keys=sorted(input_non_tensor_batch.keys()),
        )

        sampling_params = self._build_sampling_params(validate=validate)
        agent_loop = self._get_or_create_agent_loop(agent_name)
        from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop

        if not isinstance(agent_loop, SkdAgentLoop):
            raise TypeError(
                "generate_skd_until_boundary requires skd_agent loop, "
                f"got {type(agent_loop).__name__} for agent_name={agent_name!r}"
            )

        with async_skd_event_context(**(async_skd_context or {})):
            result = await agent_loop.run_until_exportable_boundary(
                sampling_params,
                sample_id=sample_id,
                logical_step=logical_step,
                source_type=source_type,
                partial_state=partial_state,
                **kwargs,
            )

        if isinstance(result, SkdPartialState):
            result.extra_fields.setdefault("agent_name", agent_name)
            if batch is not None:
                result.extra_fields[_ASYNC_SKD_INPUT_NON_TENSOR_BATCH] = deepcopy(input_non_tensor_batch)
            _trace_async_skd(
                "worker.generate_until_boundary.partial",
                sample_id=result.sample_id,
                logical_step=result.logical_step,
                source_type=result.source_type,
                request_id=result.request_id,
                teacher_replica_id=result.extra_fields.get("teacher_replica_id"),
                teacher_routing_key=result.extra_fields.get("teacher_routing_key"),
                response_len=len(result.response_mask),
                committed_gen_chunks=result.committed_gen_chunks,
                committed_prefix_tokens=result.committed_prefix_tokens,
            )
            return AsyncSkdSample.from_partial(partial_state=result)

        if not isinstance(result, AgentLoopOutput):
            raise TypeError(f"Unexpected SKD boundary result type: {type(result).__name__}")

        completed_batch = await self._postprocess_completed_skd_output(
            result,
            validate=validate,
            input_non_tensor_batch=input_non_tensor_batch,
        )
        _trace_async_skd(
            "worker.generate_until_boundary.completed",
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            output_batch_size=len(completed_batch),
        )
        return AsyncSkdSample.from_completed(
            sample_id=sample_id,
            logical_step=logical_step,
            source_type=source_type,
            batch=completed_batch,
        )

    async def generate_skd_from_partial_to_completion(
        self,
        partial_state: SkdPartialState,
        *,
        source_type: str = "resumed_current",
        agent_name: str = "skd_agent",
        async_skd_context: dict[str, Any] | None = None,
    ) -> AsyncSkdSample:
        """Resume an SKD partial as current-step work and run it to completion."""
        print(
            "[ASYNC_SKD] resume "
            f"sample_id={partial_state.sample_id} start_chunks={partial_state.committed_gen_chunks} "
            f"start_resp_len={len(partial_state.response_mask)} "
            f"start_prefix_tokens={partial_state.committed_prefix_tokens}",
            flush=True,
        )
        _trace_async_skd(
            "worker.generate_from_partial.entry",
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=source_type,
            request_id=partial_state.request_id,
            teacher_replica_id=partial_state.extra_fields.get("teacher_replica_id"),
            teacher_routing_key=partial_state.extra_fields.get("teacher_routing_key"),
            response_len=len(partial_state.response_mask),
            committed_gen_chunks=partial_state.committed_gen_chunks,
            committed_prefix_tokens=partial_state.committed_prefix_tokens,
        )
        input_non_tensor_batch = self._input_non_tensor_from_partial(partial_state)
        agent_name = self._agent_name_from_input_non_tensor_batch(input_non_tensor_batch, default=agent_name)
        sampling_params = self._build_sampling_params(validate=False)
        agent_loop = self._get_or_create_agent_loop(agent_name)
        from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop

        if not isinstance(agent_loop, SkdAgentLoop):
            raise TypeError(
                "generate_skd_from_partial_to_completion requires skd_agent loop, "
                f"got {type(agent_loop).__name__} for agent_name={agent_name!r}"
            )

        with async_skd_event_context(**(async_skd_context or {})):
            result = await agent_loop.run_from_partial_to_completion(
                sampling_params,
                partial_state=partial_state,
            )
        completed_batch = await self._postprocess_completed_skd_output(
            result,
            validate=False,
            input_non_tensor_batch=input_non_tensor_batch,
        )
        _trace_async_skd(
            "worker.generate_from_partial.completed",
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=source_type,
            parent_request_id=partial_state.request_id,
            output_batch_size=len(completed_batch),
        )
        return AsyncSkdSample.from_completed(
            sample_id=partial_state.sample_id,
            logical_step=partial_state.logical_step,
            source_type=source_type,
            batch=completed_batch,
        )
