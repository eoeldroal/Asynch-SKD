"""Agent-loop manager variants for bounded asynchronous SKD."""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import ray

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopManager
except ModuleNotFoundError:  # pragma: no cover - local test environments may not have the full rollout stack
    class AgentLoopManager:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def _performance_metrics(self, metrics: list[dict[str, Any]], output: DataProto) -> dict[str, Any]:
            del metrics, output
            return {}

from verl.experimental.async_skd.events import emit_async_skd_event
from verl.experimental.async_skd.metadata import align_non_tensor_keys_for_concat
from verl.experimental.async_skd.state import AsyncSkdSample, SkdPartialState
from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

_ASYNC_SKD_TRACE = int(os.getenv("VERL_ASYNC_SKD_TRACE", os.getenv("VERL_SKD_DEBUG", "0")))


def _trace_async_skd(stage: str, **fields: Any) -> None:
    if _ASYNC_SKD_TRACE <= 0:
        return
    fields = {"pid": os.getpid(), "mono_ns": time.monotonic_ns(), **fields}
    parts = [f"{key}={value!r}" for key, value in fields.items()]
    suffix = f" {' '.join(parts)}" if parts else ""
    print(f"[ASYNC_SKD_TRACE] stage={stage}{suffix}", flush=True)


class AsyncSkdAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with an optional sample-level async execution path.

    The public contract stays identical to ``AgentLoopManager.generate_sequences``:
    callers pass one ``DataProto`` batch and receive one ``DataProto`` batch.
    ``sample_async`` only changes the internal scheduling granularity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(AsyncSkdAgentLoopWorker)
        self._async_skd_data_source = None
        self._teacher_replica_pin_by_sample_id: dict[str, str] = {}
        self._teacher_routing_key_by_sample_id: dict[str, str] = {}
        self._teacher_replica_last_plan_stats: dict[str, Any] = {}
        self._teacher_server_ids_by_routing_key = self._load_teacher_server_ids_by_routing_key()

    def set_async_skd_data_source(self, source: Any | None) -> None:
        self._async_skd_data_source = source

    def _get_async_skd_data_source(self) -> Any | None:
        return getattr(self, "_async_skd_data_source", None)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        mode = self._async_skd_mode()
        if mode in {"sync", "disabled", "none"}:
            return await super().generate_sequences(prompts)
        if mode not in {"sample_async", "lookahead"}:
            raise ValueError(f"Unsupported async SKD rollout mode: {mode!r}")

        rollout_n = self._rollout_n()
        if rollout_n != 1:
            raise ValueError(f"Async SKD {mode} currently requires rollout.n == 1, got {rollout_n}")

        if mode == "lookahead":
            outputs = await self._generate_sequences_lookahead(prompts)
        else:
            outputs = await self._generate_sequences_sample_async(prompts)

        return self._finalize_outputs(outputs)

    def _finalize_outputs(self, outputs: list[DataProto]) -> DataProto:
        output = DataProto.concat(align_non_tensor_keys_for_concat(outputs))
        metrics = [single.meta_info.pop("metrics") for single in outputs]
        timing = self._performance_metrics(metrics, output)
        for key in {
            "window/num_samples",
            "window/avg_target_tokens",
            "window/max_target_tokens",
            "window/avg_images",
            "window/max_images",
            "window/avg_recent_steps",
            "window/skipped_old_steps",
        }:
            values = [metric[key] for chunk in metrics for metric in chunk if key in metric]
            if not values:
                continue
            if key.endswith("/num_samples") or key.endswith("_count") or key.endswith("error_count"):
                timing[key] = float(np.sum(values))
            elif "/max_" in key:
                timing[key] = float(np.max(values))
            else:
                timing[key] = float(np.mean(values))
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        extra_metrics = getattr(self, "_async_skd_last_step_metrics", None)
        if extra_metrics:
            output.meta_info["async_skd_metrics"] = extra_metrics
            self._async_skd_last_step_metrics = None
        return output

    def _async_skd_mode(self) -> str:
        mode = OmegaConf.select(self.config, "actor_rollout_ref.rollout.agent.async_skd_mode", default=None)
        if mode is None:
            mode = OmegaConf.select(self.config, "distillation.async_skd.mode", default="sync")
        return str(mode)

    def _rollout_n(self) -> int:
        value: Any = OmegaConf.select(self.config, "actor_rollout_ref.rollout.n", default=None)
        if value is None:
            value = getattr(self.rollout_config, "n", 1)
        return int(value)

    def _lookahead_prefetch_limit(self, batch_size: int) -> int:
        value = OmegaConf.select(
            self.config,
            "actor_rollout_ref.rollout.agent.async_skd_prefetch_limit",
            default=0,
        )
        return max(0, min(int(value), 2 * batch_size))

    def _lookahead_prefetch_worker_target(self, worker_capacity: int) -> int:
        value = OmegaConf.select(
            self.config,
            "actor_rollout_ref.rollout.agent.async_skd_prefetch_worker_target",
            default=0,
        )
        target = int(value)
        if target <= 0:
            return worker_capacity
        return max(1, min(target, worker_capacity))

    def _teacher_sticky_carryover_enabled(self) -> bool:
        value = OmegaConf.select(
            self.config,
            "actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover",
            default=True,
        )
        return bool(value)

    def _next_lookahead_sample(self, logical_step: int) -> tuple[str, DataProto] | None:
        source = self._get_async_skd_data_source()
        if source is None:
            return None
        return source.reserve_lookahead(logical_step)

    def _next_fresh_quota(self, base_batch_size: int) -> int:
        source = self._get_async_skd_data_source()
        if source is not None:
            return int(source.next_fresh_quota(base_batch_size))
        carryover_count = len(getattr(self, "_async_skd_carryover_partials", []))
        return max(0, base_batch_size - carryover_count)

    @staticmethod
    def _normalize_teacher_server_id_map(server_id_map: Any) -> dict[str, list[str]]:
        if not server_id_map:
            return {}
        if isinstance(server_id_map, dict):
            return {
                str(routing_key): [str(server_id) for server_id in server_ids if server_id is not None]
                for routing_key, server_ids in server_id_map.items()
                if server_ids
            }
        return {"default": [str(server_id) for server_id in server_id_map if server_id is not None]}

    def _load_teacher_server_ids_by_routing_key(self) -> dict[str, list[str]]:
        teacher_model_manager = getattr(self, "teacher_model_manager", None)
        if teacher_model_manager is None:
            return {}
        return self._normalize_teacher_server_id_map(getattr(teacher_model_manager, "server_addresses", None))

    def _refresh_teacher_server_ids_by_routing_key(self, *, reason: str) -> dict[str, list[str]]:
        server_id_map = self._load_teacher_server_ids_by_routing_key()
        self._teacher_server_ids_by_routing_key = server_id_map
        return server_id_map

    def _teacher_server_id_map(self) -> dict[str, list[str]]:
        server_ids_by_routing_key = getattr(self, "_teacher_server_ids_by_routing_key", None)
        if server_ids_by_routing_key:
            return {
                str(routing_key): [str(server_id) for server_id in server_ids]
                for routing_key, server_ids in server_ids_by_routing_key.items()
            }
        server_ids_by_routing_key = self._refresh_teacher_server_ids_by_routing_key(reason="empty_cache")
        if server_ids_by_routing_key:
            return {
                str(routing_key): [str(server_id) for server_id in server_ids]
                for routing_key, server_ids in server_ids_by_routing_key.items()
            }
        return {}

    def _resolve_teacher_routing_key(self, routing_key: Any | None) -> str | None:
        server_id_map = self._teacher_server_id_map()
        if not server_id_map:
            server_id_map = self._refresh_teacher_server_ids_by_routing_key(reason="resolve_routing_key")
        if not server_id_map:
            return None
        if routing_key is not None:
            routing_key = str(routing_key)
            if routing_key in server_id_map:
                return routing_key
        if len(server_id_map) == 1:
            return next(iter(server_id_map))
        return None

    def _teacher_routing_key_from_partial(self, partial: SkdPartialState) -> str | None:
        return self._resolve_teacher_routing_key(partial.extra_fields.get("teacher_routing_key"))

    def _teacher_routing_key_from_payload(self, payload: Any) -> str | None:
        if isinstance(payload, SkdPartialState):
            return self._teacher_routing_key_from_partial(payload)
        if isinstance(payload, DataProto):
            teacher_key = getattr(self, "teacher_key", "data_source")
            if teacher_key in payload.non_tensor_batch:
                routing_key = payload.non_tensor_batch[teacher_key][0]
                if hasattr(routing_key, "item"):
                    routing_key = routing_key.item()
                return self._resolve_teacher_routing_key(routing_key)
        return self._resolve_teacher_routing_key(None)

    def _teacher_replica_ids_for_planning(
        self,
        *,
        routing_key: Any | None = None,
    ) -> list[str]:
        resolved_routing_key = self._resolve_teacher_routing_key(routing_key)
        if resolved_routing_key is None:
            return []
        server_id_map = self._teacher_server_id_map()
        return list(server_id_map.get(resolved_routing_key, []))

    def _teacher_replica_id_from_partial(self, partial: SkdPartialState) -> str | None:
        teacher_replica_id = partial.extra_fields.get("teacher_replica_id")
        if teacher_replica_id is None:
            return None
        return str(teacher_replica_id)

    def _choose_teacher_replica_for_lookahead(self, routing_key: Any | None = None) -> str | None:
        resolved_routing_key = self._resolve_teacher_routing_key(routing_key)
        replica_ids = self._teacher_replica_ids_for_planning(routing_key=resolved_routing_key)
        if not replica_ids:
            return None
        replica_order = {replica_id: idx for idx, replica_id in enumerate(replica_ids)}
        replica_loads: dict[str, int] = {replica_id: 0 for replica_id in replica_ids}
        teacher_routing_key_by_sample_id = getattr(self, "_teacher_routing_key_by_sample_id", {})
        for sample_id, teacher_replica_id in getattr(self, "_teacher_replica_pin_by_sample_id", {}).items():
            if self._resolve_teacher_routing_key(teacher_routing_key_by_sample_id.get(sample_id)) != resolved_routing_key:
                continue
            teacher_replica_id = str(teacher_replica_id)
            if teacher_replica_id not in replica_loads:
                continue
            replica_loads[teacher_replica_id] += 1
        return min(replica_ids, key=lambda replica_id: (replica_loads[replica_id], replica_order[replica_id]))

    @staticmethod
    def _object_array(value: Any) -> np.ndarray:
        array = np.empty(1, dtype=object)
        array[0] = value
        return array

    def _apply_teacher_assignment(
        self,
        *,
        sample_id: str,
        teacher_replica_id: str | None,
        teacher_routing_key: str | None,
        payload: Any,
    ) -> None:
        if teacher_replica_id is None:
            return
        teacher_replica_id = str(teacher_replica_id)
        teacher_replica_pin_by_sample_id = getattr(self, "_teacher_replica_pin_by_sample_id", None)
        if teacher_replica_pin_by_sample_id is None:
            teacher_replica_pin_by_sample_id = {}
            self._teacher_replica_pin_by_sample_id = teacher_replica_pin_by_sample_id
        teacher_replica_pin_by_sample_id[sample_id] = teacher_replica_id
        if teacher_routing_key is not None:
            teacher_routing_key_by_sample_id = getattr(self, "_teacher_routing_key_by_sample_id", None)
            if teacher_routing_key_by_sample_id is None:
                teacher_routing_key_by_sample_id = {}
                self._teacher_routing_key_by_sample_id = teacher_routing_key_by_sample_id
            teacher_routing_key_by_sample_id[sample_id] = str(teacher_routing_key)

        if isinstance(payload, SkdPartialState):
            payload.extra_fields["teacher_replica_id"] = teacher_replica_id
            if teacher_routing_key is not None and payload.extra_fields.get("teacher_routing_key") is None:
                payload.extra_fields["teacher_routing_key"] = str(teacher_routing_key)
            return

        if isinstance(payload, DataProto):
            payload.non_tensor_batch["teacher_replica_id"] = self._object_array(teacher_replica_id)
            teacher_key = getattr(self, "teacher_key", "data_source")
            if teacher_routing_key is not None and teacher_key not in payload.non_tensor_batch:
                payload.non_tensor_batch[teacher_key] = self._object_array(str(teacher_routing_key))

    def _clear_teacher_assignment(self, sample_id: str) -> None:
        teacher_replica_pin_by_sample_id = getattr(self, "_teacher_replica_pin_by_sample_id", None)
        if teacher_replica_pin_by_sample_id is not None:
            teacher_replica_pin_by_sample_id.pop(sample_id, None)
        teacher_routing_key_by_sample_id = getattr(self, "_teacher_routing_key_by_sample_id", None)
        if teacher_routing_key_by_sample_id is not None:
            teacher_routing_key_by_sample_id.pop(sample_id, None)

    def _plan_teacher_replica_assignments(
        self,
        *,
        carryover_sample_ids: list[str],
        fresh_sample_ids: list[str],
        carryover_partials: list[SkdPartialState] | None = None,
        fresh_payloads_by_sample_id: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        assignments: dict[str, str] = {}
        pinned_carryover_count = 0
        fallback_carryover_count = 0
        sticky_carryover_enabled = self._teacher_sticky_carryover_enabled()

        carryover_partial_by_sample_id = (
            {partial.sample_id: partial for partial in carryover_partials} if carryover_partials is not None else {}
        )
        fresh_payloads_by_sample_id = fresh_payloads_by_sample_id or {}
        active_sample_ids = set(carryover_sample_ids) | set(fresh_sample_ids)
        pool_state_by_routing_key: dict[str | None, tuple[list[str], dict[str, int], dict[str, int]]] = {}

        def ensure_pool_state(routing_key: str | None) -> tuple[list[str], dict[str, int], dict[str, int]]:
            resolved_routing_key = self._resolve_teacher_routing_key(routing_key)
            if resolved_routing_key in pool_state_by_routing_key:
                return pool_state_by_routing_key[resolved_routing_key]

            replica_ids = self._teacher_replica_ids_for_planning(routing_key=resolved_routing_key)
            replica_order = {replica_id: idx for idx, replica_id in enumerate(replica_ids)}
            replica_loads: dict[str, int] = {replica_id: 0 for replica_id in replica_ids}

            teacher_routing_key_by_sample_id = getattr(self, "_teacher_routing_key_by_sample_id", {})
            for sample_id, teacher_replica_id in getattr(self, "_teacher_replica_pin_by_sample_id", {}).items():
                if sample_id in active_sample_ids:
                    continue
                if self._resolve_teacher_routing_key(teacher_routing_key_by_sample_id.get(sample_id)) != resolved_routing_key:
                    continue
                teacher_replica_id = str(teacher_replica_id)
                if teacher_replica_id in replica_loads:
                    replica_loads[teacher_replica_id] += 1

            pool_state_by_routing_key[resolved_routing_key] = (replica_ids, replica_order, replica_loads)
            return pool_state_by_routing_key[resolved_routing_key]

        def choose_least_loaded_replica(routing_key: str | None) -> str | None:
            replica_ids, replica_order, replica_loads = ensure_pool_state(routing_key)
            if not replica_ids:
                return None
            return min(replica_ids, key=lambda replica_id: (replica_loads[replica_id], replica_order[replica_id]))

        for sample_id in carryover_sample_ids:
            partial = carryover_partial_by_sample_id.get(sample_id)
            routing_key = self._teacher_routing_key_from_partial(partial) if partial is not None else None
            replica_ids, _, replica_loads = ensure_pool_state(routing_key)
            teacher_replica_id = (
                getattr(self, "_teacher_replica_pin_by_sample_id", {}).get(sample_id)
                if sticky_carryover_enabled
                else None
            )
            hard_pinned = teacher_replica_id is not None and str(teacher_replica_id) in replica_loads
            if sticky_carryover_enabled and not hard_pinned and partial is not None:
                teacher_replica_id = self._teacher_replica_id_from_partial(partial)
                hard_pinned = teacher_replica_id is not None and str(teacher_replica_id) in replica_loads
            if teacher_replica_id is None or str(teacher_replica_id) not in replica_loads:
                teacher_replica_id = choose_least_loaded_replica(routing_key)
                if sticky_carryover_enabled:
                    fallback_carryover_count += 1
            else:
                teacher_replica_id = str(teacher_replica_id)

            if teacher_replica_id is None:
                continue
            assignments[sample_id] = teacher_replica_id
            replica_loads[teacher_replica_id] += 1
            if hard_pinned:
                pinned_carryover_count += 1

        for sample_id in fresh_sample_ids:
            routing_key = self._teacher_routing_key_from_payload(fresh_payloads_by_sample_id.get(sample_id))
            replica_ids, _, replica_loads = ensure_pool_state(routing_key)
            if not replica_ids:
                continue
            teacher_replica_id = choose_least_loaded_replica(routing_key)
            if teacher_replica_id is None:
                continue
            assignments[sample_id] = teacher_replica_id
            replica_loads[teacher_replica_id] += 1

        teacher_replica_pin_by_sample_id = getattr(self, "_teacher_replica_pin_by_sample_id", None)
        if teacher_replica_pin_by_sample_id is None:
            teacher_replica_pin_by_sample_id = {}
            self._teacher_replica_pin_by_sample_id = teacher_replica_pin_by_sample_id
        teacher_routing_key_by_sample_id = getattr(self, "_teacher_routing_key_by_sample_id", None)
        if teacher_routing_key_by_sample_id is None:
            teacher_routing_key_by_sample_id = {}
            self._teacher_routing_key_by_sample_id = teacher_routing_key_by_sample_id
        for sample_id, teacher_replica_id in assignments.items():
            teacher_replica_pin_by_sample_id[sample_id] = teacher_replica_id
            if sample_id in carryover_partial_by_sample_id:
                routing_key = self._teacher_routing_key_from_partial(carryover_partial_by_sample_id[sample_id])
            else:
                routing_key = self._teacher_routing_key_from_payload(fresh_payloads_by_sample_id.get(sample_id))
            if routing_key is not None:
                teacher_routing_key_by_sample_id[sample_id] = routing_key
        self._teacher_replica_last_plan_stats = {
            "async_skd/teacher_pinned_carryover_count": pinned_carryover_count,
            "async_skd/teacher_fallback_carryover_count": fallback_carryover_count,
        }
        self._async_skd_last_step_metrics = dict(self._teacher_replica_last_plan_stats)
        return assignments

    @auto_await
    async def generate_sequences_with_carryover(
        self,
        *,
        fresh_prompts: DataProto | None,
        carryover_partials: list[SkdPartialState],
    ) -> DataProto:
        _trace_async_skd(
            "async_skd_manager.generate_with_carryover_begin",
            fresh_count=len(fresh_prompts) if fresh_prompts is not None else 0,
            carryover_count=len(carryover_partials),
        )
        total_t0 = time.monotonic()
        rollout_n = self._rollout_n()
        if rollout_n != 1:
            raise ValueError(f"Async SKD carryover currently requires rollout.n == 1, got {rollout_n}")

        if fresh_prompts is None and not carryover_partials:
            raise ValueError("generate_sequences_with_carryover requires fresh_prompts or carryover_partials")

        if fresh_prompts is not None and len(fresh_prompts) == 0:
            fresh_prompts = None

        inner_t0 = time.monotonic()
        outputs = await self._generate_sequences_with_carryover(fresh_prompts, carryover_partials)
        inner_ms = (time.monotonic() - inner_t0) * 1000

        finalize_t0 = time.monotonic()
        finalized = self._finalize_outputs(outputs)
        finalize_ms = (time.monotonic() - finalize_t0) * 1000
        _trace_async_skd(
            "async_skd_manager.generate_with_carryover_done",
            total_ms=round((time.monotonic() - total_t0) * 1000, 1),
            inner_ms=round(inner_ms, 1),
            finalize_ms=round(finalize_ms, 1),
            output_count=len(outputs),
            finalized_batch_len=len(finalized),
        )
        return finalized

    async def _generate_sequences_with_carryover(
        self,
        fresh_prompts: DataProto | None,
        carryover_partials: list[SkdPartialState],
    ) -> list[DataProto]:
        _trace_async_skd(
            "async_skd_manager.plan_with_carryover_begin",
            fresh_count=len(fresh_prompts) if fresh_prompts is not None else 0,
            carryover_count=len(carryover_partials),
        )
        fresh_count = len(fresh_prompts) if fresh_prompts is not None else 0
        current_items: list[tuple[str, int, Any]] = []
        carryover_partial_by_sample_id = {partial.sample_id: partial for partial in carryover_partials}

        for pos, partial in enumerate(carryover_partials):
            current_items.append(("carryover", pos, partial))

        if fresh_prompts is not None:
            offset = len(carryover_partials)
            for pos in range(fresh_count):
                current_items.append(("fresh", offset + pos, fresh_prompts[pos : pos + 1]))

        logical_step = 0
        if fresh_prompts is not None:
            logical_step = int(fresh_prompts.meta_info.get("global_steps", 0)) + 1
        elif carryover_partials:
            logical_step = max(partial.logical_step for partial in carryover_partials)
        prefetch_limit = self._lookahead_prefetch_limit(len(current_items))

        def sample_id_for_item(kind: str, order: int, payload: Any) -> str:
            if kind == "carryover":
                return str(payload.sample_id)
            if isinstance(payload, DataProto):
                for key in ("uid", "index", "input_pos"):
                    if key in payload.non_tensor_batch:
                        return str(payload.non_tensor_batch[key][0])
            return f"current-{logical_step}-{order}"

        carryover_sample_ids = [partial.sample_id for partial in carryover_partials]
        fresh_sample_ids = [
            sample_id_for_item(kind, order, payload)
            for kind, order, payload in current_items
            if kind == "fresh"
        ]
        fresh_payloads_by_sample_id = {
            sample_id_for_item(kind, order, payload): payload
            for kind, order, payload in current_items
            if kind == "fresh"
        }
        current_teacher_replica_by_sample_id = self._plan_teacher_replica_assignments(
            carryover_sample_ids=carryover_sample_ids,
            fresh_sample_ids=fresh_sample_ids,
            carryover_partials=carryover_partials,
            fresh_payloads_by_sample_id=fresh_payloads_by_sample_id,
        )
        _trace_async_skd(
            "async_skd_manager.plan_with_carryover_done",
            current_count=len(current_items),
            fresh_count=fresh_count,
            carryover_count=len(carryover_partials),
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
            assigned_teachers=len(current_teacher_replica_by_sample_id),
        )

        def teacher_replica_id_for_item(kind: str, order: int, payload: Any) -> str | None:
            sample_id = sample_id_for_item(kind, order, payload)
            return current_teacher_replica_by_sample_id.get(sample_id)

        return await self._generate_current_work_with_lookahead(
            current_items,
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
            teacher_replica_id_for_item=teacher_replica_id_for_item,
        )

    async def _generate_sequences_sample_async(self, prompts: DataProto) -> list[DataProto]:
        """Run all base samples concurrently and collect by FIRST_COMPLETED.

        Returned outputs are ordered by the original input position, not by
        completion order.  This preserves downstream assumptions about uid,
        index, reward metadata, and repeated rollout layout.
        """
        if len(prompts) == 0:
            return []
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        outputs: list[DataProto | None] = [None] * len(prompts)
        active: dict[asyncio.Task, tuple[int, Any]] = {}

        def worker_for_pos(pos: int) -> Any:
            worker_idx = min(pos * len(self.agent_loop_workers) // len(prompts), len(self.agent_loop_workers) - 1)
            return self.agent_loop_workers[worker_idx]

        def launch(pos: int) -> None:
            worker = worker_for_pos(pos)
            sample = prompts[pos : pos + 1]
            task = asyncio.ensure_future(worker.generate_sequence_single.remote(sample))
            active[task] = (pos, worker)

        # Submit the whole base batch immediately. Ray async actors can execute
        # many async methods concurrently, so this preserves the same request
        # concurrency as AgentLoopWorker.generate_sequences(...), while exposing
        # per-sample completion events to the manager.
        for pos in range(len(prompts)):
            launch(pos)

        while active:
            done, _ = await asyncio.wait(active.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pos, _ = active.pop(task)
                outputs[pos] = await task

        return [output for output in outputs if output is not None]

    async def _generate_sequences_lookahead(self, prompts: DataProto) -> list[DataProto]:
        """Run base samples while opportunistically filling freed slots with bounded lookahead."""
        if len(prompts) == 0:
            return []
        current_items = [("fresh", pos, prompts[pos : pos + 1]) for pos in range(len(prompts))]
        logical_step = int(prompts.meta_info.get("global_steps", 0)) + 1
        prefetch_limit = self._lookahead_prefetch_limit(len(current_items))
        return await self._generate_current_work_with_lookahead(
            current_items,
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
        )

    async def _generate_current_work_with_lookahead(
        self,
        current_items: list[tuple[str, int, Any]],
        *,
        logical_step: int,
        prefetch_limit: int,
        teacher_replica_id_for_item: Any | None = None,
    ) -> list[DataProto]:
        """Run current work and use freed worker slots for bounded lookahead."""
        if not current_items:
            return []
        if not self.agent_loop_workers:
            raise RuntimeError("AsyncSkdAgentLoopManager requires at least one agent loop worker")

        _trace_async_skd(
            "async_skd_manager.lookahead_begin",
            current_count=len(current_items),
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
            worker_count=len(self.agent_loop_workers),
        )
        lookahead_total_t0 = time.monotonic()
        current_count = len(current_items)
        current_completed: list[DataProto | None] = [None] * current_count
        promoted_lookahead: list[tuple[int, AsyncSkdSample]] = []
        carryover_partials: list[tuple[int, SkdPartialState]] = []
        current_active: dict[asyncio.Task, dict[str, Any]] = {}
        lookahead_active: dict[asyncio.Task, dict[str, Any]] = {}
        lookahead_started_count = 0
        drain_requested = False
        num_workers = len(self.agent_loop_workers)
        worker_capacity = max(1, math.ceil(current_count / num_workers))
        prefetch_worker_target = self._lookahead_prefetch_worker_target(worker_capacity)
        worker_active_counts = [0 for _ in range(num_workers)]
        worker_active_max = 0
        lookahead_continued_partial_count = 0

        def worker_idx_for_order(order: int) -> int:
            return min(order * num_workers // current_count, num_workers - 1)

        def worker_for_idx(worker_idx: int) -> Any:
            return self.agent_loop_workers[worker_idx]

        def note_launch(worker_idx: int) -> int:
            nonlocal worker_active_max
            worker_active_counts[worker_idx] += 1
            worker_active_max = max(worker_active_max, max(worker_active_counts))
            return worker_active_counts[worker_idx]

        def note_finish(worker_idx: int) -> int:
            worker_active_counts[worker_idx] -= 1
            if worker_active_counts[worker_idx] < 0:
                raise RuntimeError(f"worker_active_counts[{worker_idx}] became negative")
            return worker_active_counts[worker_idx]

        def sample_id_for_current(kind: str, order: int, payload: Any) -> str:
            if kind == "carryover":
                return str(payload.sample_id)
            if isinstance(payload, DataProto):
                for key in ("uid", "index", "input_pos"):
                    if key in payload.non_tensor_batch:
                        return str(payload.non_tensor_batch[key][0])
            return f"current-{logical_step}-{order}"

        def launch_current(kind: str, order: int, payload: Any) -> None:
            worker_idx = worker_idx_for_order(order)
            worker = worker_for_idx(worker_idx)
            sample_id = sample_id_for_current(kind, order, payload)
            source_type = "resumed_current" if kind == "carryover" else "base_current"
            teacher_replica_id = None
            if teacher_replica_id_for_item is not None:
                teacher_replica_id = teacher_replica_id_for_item(kind, order, payload)
            teacher_routing_key = self._teacher_routing_key_from_payload(payload)
            self._apply_teacher_assignment(
                sample_id=sample_id,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
                payload=payload,
            )
            event_context = {
                "global_step": logical_step,
                "logical_step": logical_step,
                "sample_id": sample_id,
                "scheduler_worker_idx": worker_idx,
                "order": order,
                "source_type": source_type,
                "barrier_role": "current",
                "worker_capacity": worker_capacity,
                "teacher_replica_id": teacher_replica_id,
            }
            if kind == "fresh":
                task = asyncio.ensure_future(
                    worker.generate_sequence_single.remote(payload, async_skd_context=event_context)
                )
            elif kind == "carryover":
                task = asyncio.ensure_future(
                    worker.generate_skd_from_partial_to_completion.remote(payload, async_skd_context=event_context)
                )
            else:
                raise ValueError(f"Unsupported async SKD current work kind: {kind!r}")
            active_after = note_launch(worker_idx)
            _trace_async_skd(
                "async_skd_manager.launch_current",
                logical_step=logical_step,
                sample_id=sample_id,
                kind=kind,
                order=order,
                worker_idx=worker_idx,
                source_type=source_type,
                active_after=active_after,
                worker_capacity=worker_capacity,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
            )
            current_active[task] = {
                "order": order,
                "worker_idx": worker_idx,
                "sample_id": sample_id,
                "source_type": source_type,
                "barrier_role": "current",
                "launch_ts": time.time(),
                "teacher_replica_id": teacher_replica_id,
            }
            emit_async_skd_event(
                "sample_launch",
                global_step=logical_step,
                logical_step=logical_step,
                sample_id=sample_id,
                scheduler_worker_idx=worker_idx,
                order=order,
                source_type=source_type,
                barrier_role="current",
                worker_active_after=active_after,
                worker_capacity=worker_capacity,
                teacher_replica_id=teacher_replica_id,
            )

        def launch_lookahead_batch(sample_id: str, sample: DataProto, admission_order: int, worker_idx: int) -> None:
            worker = worker_for_idx(worker_idx)
            teacher_routing_key = self._teacher_routing_key_from_payload(sample)
            teacher_replica_id = getattr(self, "_teacher_replica_pin_by_sample_id", {}).get(sample_id)
            valid_replica_ids = set(self._teacher_replica_ids_for_planning(routing_key=teacher_routing_key))
            if teacher_replica_id is not None:
                teacher_replica_id = str(teacher_replica_id)
            if teacher_replica_id not in valid_replica_ids:
                teacher_replica_id = self._choose_teacher_replica_for_lookahead(teacher_routing_key)
            self._apply_teacher_assignment(
                sample_id=sample_id,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
                payload=sample,
            )
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    sample,
                    sample_id=sample_id,
                    logical_step=logical_step,
                    source_type="lookahead",
                    async_skd_context={
                        "global_step": logical_step,
                        "logical_step": logical_step,
                        "sample_id": sample_id,
                        "scheduler_worker_idx": worker_idx,
                        "order": admission_order,
                        "source_type": "lookahead",
                        "barrier_role": "lookahead",
                        "worker_capacity": worker_capacity,
                        "prefetch_worker_target": prefetch_worker_target,
                        "teacher_replica_id": teacher_replica_id,
                    },
                )
            )
            active_after = note_launch(worker_idx)
            _trace_async_skd(
                "async_skd_manager.launch_lookahead_batch",
                logical_step=logical_step,
                sample_id=sample_id,
                admission_order=admission_order,
                worker_idx=worker_idx,
                active_after=active_after,
                worker_capacity=worker_capacity,
                prefetch_worker_target=prefetch_worker_target,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
            )
            lookahead_active[task] = {
                "order": admission_order,
                "worker_idx": worker_idx,
                "sample_id": sample_id,
                "source_type": "lookahead",
                "barrier_role": "lookahead",
                "launch_ts": time.time(),
                "teacher_replica_id": teacher_replica_id,
            }
            emit_async_skd_event(
                "lookahead_admit",
                global_step=logical_step,
                logical_step=logical_step,
                sample_id=sample_id,
                scheduler_worker_idx=worker_idx,
                admission_order=admission_order,
                source_type="lookahead",
                barrier_role="lookahead",
                worker_active_after=active_after,
                worker_capacity=worker_capacity,
                prefetch_worker_target=prefetch_worker_target,
                lookahead_started_count=lookahead_started_count,
                prefetch_limit=prefetch_limit,
                reason="slot_available",
                teacher_replica_id=teacher_replica_id,
            )
            emit_async_skd_event(
                "sample_launch",
                global_step=logical_step,
                logical_step=logical_step,
                sample_id=sample_id,
                scheduler_worker_idx=worker_idx,
                order=admission_order,
                source_type="lookahead",
                barrier_role="lookahead",
                worker_active_after=active_after,
                worker_capacity=worker_capacity,
                prefetch_worker_target=prefetch_worker_target,
                teacher_replica_id=teacher_replica_id,
            )
            print(
                "[ASYNC_SKD] prefetch_start "
                f"sample_id={sample_id} admission_order={admission_order} "
                f"worker={worker_idx} active_on_worker={worker_active_counts[worker_idx]} "
                f"worker_capacity={worker_capacity} prefetch_worker_target={prefetch_worker_target}",
                flush=True,
            )

        def launch_lookahead_partial(partial_state: SkdPartialState, admission_order: int, worker_idx: int) -> None:
            worker = worker_for_idx(worker_idx)
            teacher_routing_key = self._teacher_routing_key_from_partial(partial_state)
            teacher_replica_id = partial_state.extra_fields.get("teacher_replica_id")
            valid_replica_ids = set(self._teacher_replica_ids_for_planning(routing_key=teacher_routing_key))
            if teacher_replica_id is not None:
                teacher_replica_id = str(teacher_replica_id)
            if teacher_replica_id not in valid_replica_ids:
                teacher_replica_id = self._choose_teacher_replica_for_lookahead(teacher_routing_key)
            self._apply_teacher_assignment(
                sample_id=partial_state.sample_id,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
                payload=partial_state,
            )
            task = asyncio.ensure_future(
                worker.generate_skd_until_boundary.remote(
                    None,
                    partial_state=partial_state,
                    sample_id=partial_state.sample_id,
                    logical_step=partial_state.logical_step,
                    source_type=partial_state.source_type,
                    async_skd_context={
                        "global_step": logical_step,
                        "logical_step": partial_state.logical_step,
                        "sample_id": partial_state.sample_id,
                        "scheduler_worker_idx": worker_idx,
                        "order": admission_order,
                        "source_type": partial_state.source_type,
                        "barrier_role": "lookahead",
                        "worker_capacity": worker_capacity,
                        "prefetch_worker_target": prefetch_worker_target,
                        "resumed_partial": True,
                        "teacher_replica_id": teacher_replica_id,
                    },
                )
            )
            active_after = note_launch(worker_idx)
            _trace_async_skd(
                "async_skd_manager.launch_lookahead_partial",
                logical_step=logical_step,
                sample_id=partial_state.sample_id,
                admission_order=admission_order,
                worker_idx=worker_idx,
                active_after=active_after,
                worker_capacity=worker_capacity,
                prefetch_worker_target=prefetch_worker_target,
                teacher_replica_id=teacher_replica_id,
                teacher_routing_key=teacher_routing_key,
            )
            lookahead_active[task] = {
                "order": admission_order,
                "worker_idx": worker_idx,
                "sample_id": partial_state.sample_id,
                "source_type": partial_state.source_type,
                "barrier_role": "lookahead",
                "launch_ts": time.time(),
                "teacher_replica_id": teacher_replica_id,
            }
            emit_async_skd_event(
                "sample_launch",
                global_step=logical_step,
                logical_step=partial_state.logical_step,
                sample_id=partial_state.sample_id,
                scheduler_worker_idx=worker_idx,
                order=admission_order,
                source_type=partial_state.source_type,
                barrier_role="lookahead",
                worker_active_after=active_after,
                worker_capacity=worker_capacity,
                prefetch_worker_target=prefetch_worker_target,
                resumed_partial=True,
                teacher_replica_id=teacher_replica_id,
            )

        def try_admit_lookahead(worker_idx: int) -> None:
            nonlocal lookahead_started_count
            if drain_requested or not current_active or lookahead_started_count >= prefetch_limit:
                return
            if worker_active_counts[worker_idx] >= prefetch_worker_target:
                return
            next_item = self._next_lookahead_sample(logical_step)
            if next_item is None:
                return
            sample_id, sample = next_item
            admission_order = lookahead_started_count
            lookahead_started_count += 1
            launch_lookahead_batch(sample_id, sample, admission_order, worker_idx)

        for kind, order, payload in current_items:
            launch_current(kind, order, payload)

        while current_active or lookahead_active:
            _trace_async_skd(
                "async_skd_manager.wait_begin",
                logical_step=logical_step,
                current_active=len(current_active),
                lookahead_active=len(lookahead_active),
                lookahead_started_count=lookahead_started_count,
                drain_requested=drain_requested,
            )
            wait_t0 = time.monotonic()
            done, _ = await asyncio.wait(
                set(current_active.keys()) | set(lookahead_active.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            wait_ms = (time.monotonic() - wait_t0) * 1000
            _trace_async_skd(
                "async_skd_manager.wait_done",
                logical_step=logical_step,
                elapsed_ms=round(wait_ms, 1),
                done_count=len(done),
                current_active=len(current_active),
                lookahead_active=len(lookahead_active),
            )

            for task in done:
                if task in current_active:
                    meta = current_active.pop(task)
                    order = int(meta["order"])
                    worker_idx = int(meta["worker_idx"])
                    active_after = note_finish(worker_idx)
                    _trace_async_skd(
                        "async_skd_manager.current_task_await_begin",
                        logical_step=logical_step,
                        sample_id=meta["sample_id"],
                        order=order,
                        worker_idx=worker_idx,
                        source_type=meta["source_type"],
                    )
                    task_t0 = time.monotonic()
                    result = await task
                    _trace_async_skd(
                        "async_skd_manager.current_task_await_done",
                        logical_step=logical_step,
                        sample_id=meta["sample_id"],
                        order=order,
                        worker_idx=worker_idx,
                        elapsed_ms=round((time.monotonic() - task_t0) * 1000, 1),
                        result_type=type(result).__name__,
                    )
                    if isinstance(result, AsyncSkdSample):
                        result.validate()
                        current_completed[order] = result.require_completed()
                    else:
                        current_completed[order] = result
                    duration_ms = (time.time() - float(meta["launch_ts"])) * 1000
                    emit_async_skd_event(
                        "sample_finish",
                        global_step=logical_step,
                        logical_step=logical_step,
                        sample_id=meta["sample_id"],
                        scheduler_worker_idx=worker_idx,
                        order=order,
                        source_type=meta["source_type"],
                        barrier_role="current",
                        status="completed",
                        duration_ms=duration_ms,
                        worker_active_after=active_after,
                        worker_capacity=worker_capacity,
                    )
                    self._clear_teacher_assignment(str(meta["sample_id"]))
                    if not current_active and not drain_requested:
                        actual_lt_sample_id = str(meta["sample_id"])
                        print(
                            "[ASYNC_SKD] drain_start "
                            f"completed_current={current_count} lookahead_active={len(lookahead_active)} "
                            f"started={lookahead_started_count} promoted={len(promoted_lookahead)} "
                            f"carryover_next={len(carryover_partials)}",
                            flush=True,
                        )
                        emit_async_skd_event(
                            "drain_start",
                            global_step=logical_step,
                            logical_step=logical_step,
                            actual_lt_sample_id=actual_lt_sample_id,
                            actual_lt_worker_idx=worker_idx,
                            actual_lt_duration_ms=duration_ms,
                            current_completed=current_count,
                            lookahead_active=len(lookahead_active),
                            lookahead_started_count=lookahead_started_count,
                            promoted_count=len(promoted_lookahead),
                            carryover_next_count=len(carryover_partials),
                        )
                        drain_requested = True
                    try_admit_lookahead(worker_idx)

            for task in done:
                if task in lookahead_active:
                    meta = lookahead_active.pop(task)
                    admission_order = int(meta["order"])
                    worker_idx = int(meta["worker_idx"])
                    active_after = note_finish(worker_idx)
                    _trace_async_skd(
                        "async_skd_manager.lookahead_task_await_begin",
                        logical_step=logical_step,
                        sample_id=meta["sample_id"],
                        order=admission_order,
                        worker_idx=worker_idx,
                        source_type=meta["source_type"],
                    )
                    task_t0 = time.monotonic()
                    sample: AsyncSkdSample = await task
                    _trace_async_skd(
                        "async_skd_manager.lookahead_task_await_done",
                        logical_step=logical_step,
                        sample_id=meta["sample_id"],
                        order=admission_order,
                        worker_idx=worker_idx,
                        elapsed_ms=round((time.monotonic() - task_t0) * 1000, 1),
                        sample_kind=sample.kind,
                    )
                    sample.validate()
                    duration_ms = (time.time() - float(meta["launch_ts"])) * 1000
                    emit_async_skd_event(
                        "sample_finish",
                        global_step=logical_step,
                        logical_step=sample.logical_step,
                        sample_id=sample.sample_id,
                        scheduler_worker_idx=worker_idx,
                        order=admission_order,
                        source_type=sample.source_type,
                        barrier_role="lookahead",
                        status=sample.kind,
                        duration_ms=duration_ms,
                        committed_gen_chunks=sample.committed_gen_chunks,
                        committed_prefix_tokens=sample.committed_prefix_tokens,
                        worker_active_after=active_after,
                        worker_capacity=worker_capacity,
                    )
                    if sample.kind == "completed":
                        self._clear_teacher_assignment(sample.sample_id)
                        promoted_lookahead.append((admission_order, sample))
                        if not drain_requested:
                            try_admit_lookahead(worker_idx)
                        continue

                    partial = sample.require_partial()
                    teacher_replica_id = meta.get("teacher_replica_id")
                    if teacher_replica_id is not None and partial.extra_fields.get("teacher_replica_id") is None:
                        partial.extra_fields["teacher_replica_id"] = teacher_replica_id
                    if teacher_replica_id is not None:
                        self._teacher_replica_pin_by_sample_id[partial.sample_id] = str(teacher_replica_id)
                    if not drain_requested and bool(current_active):
                        lookahead_continued_partial_count += 1
                        launch_lookahead_partial(partial, admission_order, worker_idx)
                    else:
                        if drain_requested:
                            carryover_reason = "drain"
                        elif not current_active:
                            carryover_reason = "no_current"
                        else:
                            carryover_reason = "unknown"
                        print(
                            "[ASYNC_SKD] carryover "
                            f"sample_id={partial.sample_id} reason={carryover_reason} "
                            f"chunks={partial.committed_gen_chunks} "
                            f"resp_len={len(partial.response_mask)} prefix_tokens={partial.committed_prefix_tokens} "
                            f"worker={worker_idx}",
                            flush=True,
                        )
                        emit_async_skd_event(
                            "carryover_record",
                            global_step=logical_step,
                            logical_step=partial.logical_step,
                            sample_id=partial.sample_id,
                            scheduler_worker_idx=worker_idx,
                            order=admission_order,
                            source_type=partial.source_type,
                            barrier_role="lookahead",
                            reason=carryover_reason,
                            committed_gen_chunks=partial.committed_gen_chunks,
                            response_len=len(partial.response_mask),
                            committed_prefix_tokens=partial.committed_prefix_tokens,
                        )
                        carryover_partials.append((admission_order, partial))

        lookahead_promoted_count = len(promoted_lookahead)
        lookahead_carryover_count = len(carryover_partials)
        lookahead_denominator = max(lookahead_started_count, 1)
        self._async_skd_last_step_metrics = {
            "async_skd/lookahead_started_count": lookahead_started_count,
            "async_skd/lookahead_promoted_count": lookahead_promoted_count,
            "async_skd/lookahead_carryover_count": lookahead_carryover_count,
            "async_skd/lookahead_continued_partial_count": lookahead_continued_partial_count,
            "async_skd/worker_active_max": worker_active_max,
            "async_skd/lookahead_promote_rate": lookahead_promoted_count / lookahead_denominator,
            "async_skd/lookahead_carryover_rate": lookahead_carryover_count / lookahead_denominator,
        }
        self._async_skd_last_step_metrics.update(getattr(self, "_teacher_replica_last_plan_stats", {}))
        _trace_async_skd(
            "async_skd_manager.lookahead_done",
            logical_step=logical_step,
            total_ms=round((time.monotonic() - lookahead_total_t0) * 1000, 1),
            current_count=current_count,
            prefetch_limit=prefetch_limit,
            lookahead_started_count=lookahead_started_count,
            lookahead_promoted_count=lookahead_promoted_count,
            lookahead_carryover_count=lookahead_carryover_count,
            lookahead_continued_partial_count=lookahead_continued_partial_count,
            worker_capacity=worker_capacity,
            prefetch_worker_target=prefetch_worker_target,
            worker_active_max=worker_active_max,
        )
        print(
            "[ASYNC_SKD] rollout "
            f"prefetch_limit={prefetch_limit} started={lookahead_started_count} "
            f"promoted={len(promoted_lookahead)} carryover_next={len(carryover_partials)} "
            f"continued_partial={lookahead_continued_partial_count} "
            f"worker_capacity={worker_capacity} prefetch_worker_target={prefetch_worker_target} "
            f"worker_active_max={worker_active_max}",
            flush=True,
        )
        emit_async_skd_event(
            "rollout_summary",
            global_step=logical_step,
            logical_step=logical_step,
            prefetch_limit=prefetch_limit,
            lookahead_started_count=lookahead_started_count,
            lookahead_promoted_count=len(promoted_lookahead),
            lookahead_carryover_count=len(carryover_partials),
            lookahead_continued_partial_count=lookahead_continued_partial_count,
            worker_capacity=worker_capacity,
            prefetch_worker_target=prefetch_worker_target,
            worker_active_max=worker_active_max,
        )
        self._async_skd_last_promoted_samples = [
            sample for _, sample in sorted(promoted_lookahead, key=lambda item: item[0])
        ]
        self._async_skd_carryover_partials = [
            partial for _, partial in sorted(carryover_partials, key=lambda item: item[0])
        ]
        source = self._get_async_skd_data_source()
        if source is not None:
            source.record_promoted(self._async_skd_last_promoted_samples)
            source.record_carryover(self._async_skd_carryover_partials)

        return [output for output in current_completed if output is not None]
