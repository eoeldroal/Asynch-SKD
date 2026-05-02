#!/usr/bin/env python3
"""Stress WebGym through the same verl WebOsGymTool path used by rollout.

This intentionally does not post raw HTTP action payloads. It loads the current
tool YAML, constructs WebOsGymTool, and exercises create/execute/bundle/reward/
release so schema validation, action normalization, image decode, and response
post-processing stay on the same path as training.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.web_osgym_tool import WebOsGymTool


@dataclass
class DummyAgentData:
    request_id: str
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryResult:
    ok: bool
    round_idx: int
    worker_idx: int
    session_id: int
    create_ms: float = 0.0
    action_ms: list[float] = field(default_factory=list)
    finish_ms: float = 0.0
    reward_ms: float = 0.0
    release_ms: float = 0.0
    total_ms: float = 0.0
    error_type: str | None = None
    error: str | None = None


@dataclass
class ActiveSession:
    worker_idx: int
    session_id: int
    instance_id: str
    agent_data: DummyAgentData
    create_ms: float


@dataclass
class OperationResult:
    ok: bool
    stage: str
    worker_idx: int
    session_id: int
    elapsed_ms: float = 0.0
    error_type: str | None = None
    error: str | None = None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _summarize(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: n=0"
    return (
        f"{label}: n={len(values)} "
        f"avg={statistics.fmean(values):.1f}ms "
        f"p50={statistics.median(values):.1f}ms "
        f"p95={_percentile(values, 95):.1f}ms "
        f"max={max(values):.1f}ms"
    )


def _log_server_req_begin(
    *,
    op: str,
    session_id: int,
    worker_idx: int,
    wave_idx: int | None = None,
    step_idx: int | None = None,
    action_count: int = 0,
) -> None:
    wave = "-" if wave_idx is None else str(wave_idx)
    step = "-" if step_idx is None else str(step_idx)
    print(
        f"[server:req_begin] op={op} worker={worker_idx} session_id={session_id} "
        f"wave={wave} step={step} action_count={action_count} t_ns={time.monotonic_ns()}",
        flush=True,
    )


def _log_server_req_done(
    *,
    op: str,
    session_id: int,
    worker_idx: int,
    elapsed_ms: float,
    ok: bool,
    wave_idx: int | None = None,
    step_idx: int | None = None,
    action_count: int = 0,
    error_type: str | None = None,
    error: str | None = None,
) -> None:
    wave = "-" if wave_idx is None else str(wave_idx)
    step = "-" if step_idx is None else str(step_idx)
    suffix = ""
    if not ok:
        suffix = f" error_type={error_type} error={error}"
    print(
        f"[server:req_done] op={op} worker={worker_idx} session_id={session_id} "
        f"wave={wave} step={step} action_count={action_count} ok={ok} "
        f"elapsed_ms={elapsed_ms:.1f} t_ns={time.monotonic_ns()}{suffix}",
        flush=True,
    )


def _load_click_tool(config_path: Path, *, base_url: str | None, timeout: float | None, include_a11y: bool | None):
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    click_entry = None
    for entry in payload["tools"]:
        function = entry["tool_schema"]["function"]
        if function["name"] == "CLICK":
            click_entry = entry
            break
    if click_entry is None:
        raise ValueError(f"CLICK tool schema not found in {config_path}")

    config = dict(click_entry["config"])
    if base_url is not None:
        config["base_url"] = base_url
    if timeout is not None:
        config["timeout"] = timeout
    if include_a11y is not None:
        config["include_a11y"] = include_a11y

    schema = OpenAIFunctionToolSchema.model_validate(click_entry["tool_schema"])
    return WebOsGymTool(config=config, tool_schema=schema)


async def _create_active_session(
    *,
    tool: WebOsGymTool,
    worker_idx: int,
    task_id: str,
    session_seed: int,
    include_a11y: bool,
) -> ActiveSession:
    session_id = session_seed + worker_idx
    instance_id = f"rolling-{worker_idx}-{session_id}"
    agent_data = DummyAgentData(request_id=f"rolling-{worker_idx}")
    create_t0 = time.monotonic()
    _log_server_req_begin(op="start", session_id=session_id, worker_idx=worker_idx)
    try:
        instance_id, create_response = await tool.create(
            instance_id=instance_id,
            task_id=task_id,
            request_id=session_id,
            include_a11y=include_a11y,
        )
        create_ms = (time.monotonic() - create_t0) * 1000
        _log_server_req_done(op="start", session_id=session_id, worker_idx=worker_idx, elapsed_ms=create_ms, ok=True)
    except Exception as exc:
        create_ms = (time.monotonic() - create_t0) * 1000
        _log_server_req_done(
            op="start",
            session_id=session_id,
            worker_idx=worker_idx,
            elapsed_ms=create_ms,
            ok=False,
            error_type=type(exc).__name__,
            error=repr(exc),
        )
        raise
    agent_data.extra_fields.update(
        {
            "web_osgym_instance_id": instance_id,
            "web_osgym_task_id": task_id,
            "web_osgym_session_id": session_id,
            "web_osgym_include_a11y": include_a11y,
        }
    )
    image_info = ""
    if create_response.image:
        width, height = create_response.image[0].size
        image_info = f" image={width}x{height}"
    print(f"[rolling:create] worker={worker_idx} session_id={session_id} create_ms={create_ms:.1f}{image_info}", flush=True)
    return ActiveSession(
        worker_idx=worker_idx,
        session_id=session_id,
        instance_id=instance_id,
        agent_data=agent_data,
        create_ms=create_ms,
    )


async def _execute_active_click(
    *,
    tool: WebOsGymTool,
    session: ActiveSession,
    wave_idx: int,
    x: int,
    y: int,
    step_idx: int = 1,
) -> OperationResult:
    action_t0 = time.monotonic()
    _log_server_req_begin(
        op="action",
        session_id=session.session_id,
        worker_idx=session.worker_idx,
        wave_idx=wave_idx,
        step_idx=step_idx,
        action_count=1,
    )
    try:
        _, _, metrics = await tool.execute(
            session.instance_id,
            {"x": x, "y": y, "button": "left", "num_clicks": 1},
            agent_data=session.agent_data,
        )
        elapsed_ms = (time.monotonic() - action_t0) * 1000
        _log_server_req_done(
            op="action",
            session_id=session.session_id,
            worker_idx=session.worker_idx,
            wave_idx=wave_idx,
            step_idx=step_idx,
            action_count=1,
            elapsed_ms=elapsed_ms,
            ok=True,
        )
        print(
            f"[rolling:action] wave={wave_idx} worker={session.worker_idx} "
            f"session_id={session.session_id} elapsed_ms={elapsed_ms:.1f} metrics={metrics}",
            flush=True,
        )
        return OperationResult(True, "action", session.worker_idx, session.session_id, elapsed_ms=elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.monotonic() - action_t0) * 1000
        _log_server_req_done(
            op="action",
            session_id=session.session_id,
            worker_idx=session.worker_idx,
            wave_idx=wave_idx,
            step_idx=step_idx,
            action_count=1,
            elapsed_ms=elapsed_ms,
            ok=False,
            error_type=type(exc).__name__,
            error=repr(exc),
        )
        print(
            f"[rolling:action_error] wave={wave_idx} worker={session.worker_idx} "
            f"session_id={session.session_id} elapsed_ms={elapsed_ms:.1f} "
            f"error_type={type(exc).__name__} error={exc!r}",
            flush=True,
        )
        return OperationResult(
            False,
            "action",
            session.worker_idx,
            session.session_id,
            elapsed_ms=elapsed_ms,
            error_type=type(exc).__name__,
            error=repr(exc),
        )


async def _finalize_active_session(*, tool: WebOsGymTool, session: ActiveSession) -> tuple[OperationResult, OperationResult]:
    reward_t0 = time.monotonic()
    _log_server_req_begin(op="reward", session_id=session.session_id, worker_idx=session.worker_idx)
    try:
        reward = await tool.calc_reward(session.instance_id)
        reward_ms = (time.monotonic() - reward_t0) * 1000
        _log_server_req_done(op="reward", session_id=session.session_id, worker_idx=session.worker_idx, elapsed_ms=reward_ms, ok=True)
        print(
            f"[rolling:reward] worker={session.worker_idx} session_id={session.session_id} "
            f"reward={reward:.3f} elapsed_ms={reward_ms:.1f}",
            flush=True,
        )
        reward_result = OperationResult(True, "reward", session.worker_idx, session.session_id, elapsed_ms=reward_ms)
    except Exception as exc:
        reward_ms = (time.monotonic() - reward_t0) * 1000
        _log_server_req_done(
            op="reward",
            session_id=session.session_id,
            worker_idx=session.worker_idx,
            elapsed_ms=reward_ms,
            ok=False,
            error_type=type(exc).__name__,
            error=repr(exc),
        )
        print(
            f"[rolling:reward_error] worker={session.worker_idx} session_id={session.session_id} "
            f"elapsed_ms={reward_ms:.1f} error_type={type(exc).__name__} error={exc!r}",
            flush=True,
        )
        reward_result = OperationResult(
            False,
            "reward",
            session.worker_idx,
            session.session_id,
            elapsed_ms=reward_ms,
            error_type=type(exc).__name__,
            error=repr(exc),
        )

    release_t0 = time.monotonic()
    try:
        await tool.release(session.instance_id)
        release_ms = (time.monotonic() - release_t0) * 1000
        print(
            f"[rolling:local_release] worker={session.worker_idx} session_id={session.session_id} "
            f"elapsed_ms={release_ms:.1f}",
            flush=True,
        )
        release_result = OperationResult(True, "local_release", session.worker_idx, session.session_id, elapsed_ms=release_ms)
    except Exception as exc:
        release_ms = (time.monotonic() - release_t0) * 1000
        print(
            f"[rolling:local_release_error] worker={session.worker_idx} session_id={session.session_id} "
            f"elapsed_ms={release_ms:.1f} error_type={type(exc).__name__} error={exc!r}",
            flush=True,
        )
        release_result = OperationResult(
            False,
            "local_release",
            session.worker_idx,
            session.session_id,
            elapsed_ms=release_ms,
            error_type=type(exc).__name__,
            error=repr(exc),
        )
    return reward_result, release_result


async def _run_rolling_session_stress(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    tool = _load_click_tool(
        Path(args.tool_config),
        base_url=args.base_url,
        timeout=args.timeout,
        include_a11y=args.include_a11y,
    )

    print(
        f"[rolling] begin initial_sessions={args.concurrency} waves={args.waves} "
        f"min_wave_requests={args.min_wave_requests} max_wave_requests={args.max_wave_requests} "
        f"steps_per_wave_request={args.steps} finalizer=reward_then_local_release",
        flush=True,
    )
    create_t0 = time.monotonic()
    create_results = await asyncio.gather(
        *[
            _create_active_session(
                tool=tool,
                worker_idx=worker_idx,
                task_id=args.task_id,
                session_seed=args.session_seed,
                include_a11y=args.include_a11y,
            )
            for worker_idx in range(args.concurrency)
        ],
        return_exceptions=True,
    )

    active_sessions: list[ActiveSession] = []
    errors: list[OperationResult] = []
    for worker_idx, result in enumerate(create_results):
        if isinstance(result, Exception):
            print(
                f"[rolling:create_error] worker={worker_idx} "
                f"error_type={type(result).__name__} error={result!r}",
                flush=True,
            )
            errors.append(
                OperationResult(False, "create", worker_idx, args.session_seed + worker_idx, error_type=type(result).__name__, error=repr(result))
            )
        else:
            active_sessions.append(result)
    print(
        f"[rolling] initial_create_done active={len(active_sessions)}/{args.concurrency} "
        f"elapsed_ms={(time.monotonic() - create_t0) * 1000:.1f}",
        flush=True,
    )

    action_results: list[OperationResult] = []
    if not active_sessions:
        print("[rolling] no active sessions; skipping waves", flush=True)
    else:
        max_wave_requests = min(args.max_wave_requests, len(active_sessions))
        min_wave_requests = min(args.min_wave_requests, max_wave_requests)
        for wave_idx in range(args.waves):
            request_count = rng.randint(min_wave_requests, max_wave_requests)
            selected_sessions = rng.sample(active_sessions, request_count)
            print(
                f"[rolling:wave] begin wave={wave_idx + 1}/{args.waves} request_count={request_count}",
                flush=True,
            )
            wave_t0 = time.monotonic()
            wave_results: list[OperationResult] = []
            for step_idx in range(args.steps):
                step_results = await asyncio.gather(
                    *[
                        _execute_active_click(
                            tool=tool,
                            session=session,
                            wave_idx=wave_idx + 1,
                            x=args.x,
                            y=args.y,
                            step_idx=step_idx + 1,
                        )
                        for session in selected_sessions
                    ]
                )
                wave_results.extend(step_results)
                if args.step_sleep > 0 and step_idx + 1 < args.steps:
                    await asyncio.sleep(args.step_sleep)
            action_results.extend(wave_results)
            ok_count = sum(result.ok for result in wave_results)
            print(
                f"[rolling:wave] done wave={wave_idx + 1}/{args.waves} "
                f"ok={ok_count}/{len(wave_results)} elapsed_ms={(time.monotonic() - wave_t0) * 1000:.1f}",
                flush=True,
            )
            if args.round_sleep > 0 and wave_idx + 1 < args.waves:
                await asyncio.sleep(args.round_sleep)

    print("[rolling] final_reward_begin", flush=True)
    final_results = await asyncio.gather(
        *[_finalize_active_session(tool=tool, session=session) for session in active_sessions],
        return_exceptions=True,
    )
    reward_results: list[OperationResult] = []
    release_results: list[OperationResult] = []
    for session, result in zip(active_sessions, final_results, strict=False):
        if isinstance(result, Exception):
            errors.append(
                OperationResult(
                    False,
                    "finalize",
                    session.worker_idx,
                    session.session_id,
                    error_type=type(result).__name__,
                    error=repr(result),
                )
            )
        else:
            reward_result, release_result = result
            reward_results.append(reward_result)
            release_results.append(release_result)

    errors.extend([result for result in action_results + reward_results + release_results if not result.ok])
    create_ms = [session.create_ms for session in active_sessions]
    action_ms = [result.elapsed_ms for result in action_results if result.ok]
    reward_ms = [result.elapsed_ms for result in reward_results if result.ok]
    release_ms = [result.elapsed_ms for result in release_results if result.ok]

    print("[summary] " + _summarize("rolling_create", create_ms), flush=True)
    print("[summary] " + _summarize("rolling_action", action_ms), flush=True)
    print("[summary] " + _summarize("rolling_reward", reward_ms), flush=True)
    print("[summary] " + _summarize("rolling_local_release", release_ms), flush=True)
    print(
        f"[summary] rolling active_sessions={len(active_sessions)} "
        f"action_ops={len(action_results)} errors={len(errors)}",
        flush=True,
    )
    for error in errors[:30]:
        print(
            f"[summary:error] stage={error.stage} worker={error.worker_idx} "
            f"session_id={error.session_id} error_type={error.error_type} error={error.error}",
            flush=True,
        )
    return 0 if not errors else 1


async def _one_trajectory(
    *,
    tool: WebOsGymTool,
    round_idx: int,
    worker_idx: int,
    task_id: str,
    session_seed: int,
    include_a11y: bool,
    steps: int,
    x: int,
    y: int,
    finish_mode: str,
    terminal_action: str,
) -> TrajectoryResult:
    session_id = session_seed + round_idx * 100_000 + worker_idx
    instance_id = f"stress-{round_idx}-{worker_idx}-{session_id}"
    agent_data = DummyAgentData(request_id=f"stress-{round_idx}-{worker_idx}")
    result = TrajectoryResult(ok=False, round_idx=round_idx, worker_idx=worker_idx, session_id=session_id)
    total_t0 = time.monotonic()

    try:
        create_t0 = time.monotonic()
        _log_server_req_begin(op="start", session_id=session_id, worker_idx=worker_idx)
        try:
            instance_id, create_response = await tool.create(
                instance_id=instance_id,
                task_id=task_id,
                request_id=session_id,
                include_a11y=include_a11y,
            )
            result.create_ms = (time.monotonic() - create_t0) * 1000
            _log_server_req_done(
                op="start",
                session_id=session_id,
                worker_idx=worker_idx,
                elapsed_ms=result.create_ms,
                ok=True,
            )
        except Exception as exc:
            result.create_ms = (time.monotonic() - create_t0) * 1000
            _log_server_req_done(
                op="start",
                session_id=session_id,
                worker_idx=worker_idx,
                elapsed_ms=result.create_ms,
                ok=False,
                error_type=type(exc).__name__,
                error=repr(exc),
            )
            raise
        agent_data.extra_fields.update(
            {
                "web_osgym_instance_id": instance_id,
                "web_osgym_task_id": task_id,
                "web_osgym_session_id": session_id,
                "web_osgym_include_a11y": include_a11y,
            }
        )
        if create_response.image:
            width, height = create_response.image[0].size
            print(
                f"[trajectory] round={round_idx} worker={worker_idx} "
                f"create_ms={result.create_ms:.1f} image={width}x{height}",
                flush=True,
            )

        for step_idx in range(steps):
            action_t0 = time.monotonic()
            _log_server_req_begin(
                op="action",
                session_id=session_id,
                worker_idx=worker_idx,
                step_idx=step_idx + 1,
                action_count=1,
            )
            try:
                _, _, metrics = await tool.execute(
                    instance_id,
                    {"x": x, "y": y, "button": "left", "num_clicks": 1},
                    agent_data=agent_data,
                )
                action_ms = (time.monotonic() - action_t0) * 1000
                _log_server_req_done(
                    op="action",
                    session_id=session_id,
                    worker_idx=worker_idx,
                    step_idx=step_idx + 1,
                    action_count=1,
                    elapsed_ms=action_ms,
                    ok=True,
                )
            except Exception as exc:
                action_ms = (time.monotonic() - action_t0) * 1000
                _log_server_req_done(
                    op="action",
                    session_id=session_id,
                    worker_idx=worker_idx,
                    step_idx=step_idx + 1,
                    action_count=1,
                    elapsed_ms=action_ms,
                    ok=False,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                raise
            result.action_ms.append(action_ms)
            print(
                f"[trajectory] round={round_idx} worker={worker_idx} "
                f"step={step_idx + 1}/{steps} action_ms={action_ms:.1f} metrics={metrics}",
                flush=True,
            )

        if finish_mode == "terminal":
            finish_t0 = time.monotonic()
            _log_server_req_begin(
                op="action",
                session_id=session_id,
                worker_idx=worker_idx,
                step_idx=steps + 1,
                action_count=1,
            )
            try:
                _, _, metrics = await tool.execute_action_bundle(
                    instance_id,
                    [{"action_type": terminal_action}],
                    agent_data=agent_data,
                )
                result.finish_ms = (time.monotonic() - finish_t0) * 1000
                _log_server_req_done(
                    op="action",
                    session_id=session_id,
                    worker_idx=worker_idx,
                    step_idx=steps + 1,
                    action_count=1,
                    elapsed_ms=result.finish_ms,
                    ok=True,
                )
            except Exception as exc:
                result.finish_ms = (time.monotonic() - finish_t0) * 1000
                _log_server_req_done(
                    op="action",
                    session_id=session_id,
                    worker_idx=worker_idx,
                    step_idx=steps + 1,
                    action_count=1,
                    elapsed_ms=result.finish_ms,
                    ok=False,
                    error_type=type(exc).__name__,
                    error=repr(exc),
                )
                raise
            print(
                f"[trajectory] round={round_idx} worker={worker_idx} "
                f"terminal={terminal_action} finish_ms={result.finish_ms:.1f} metrics={metrics}",
                flush=True,
            )

        reward_t0 = time.monotonic()
        _log_server_req_begin(op="reward", session_id=session_id, worker_idx=worker_idx)
        try:
            reward = await tool.calc_reward(instance_id)
            result.reward_ms = (time.monotonic() - reward_t0) * 1000
            _log_server_req_done(
                op="reward",
                session_id=session_id,
                worker_idx=worker_idx,
                elapsed_ms=result.reward_ms,
                ok=True,
            )
        except Exception as exc:
            result.reward_ms = (time.monotonic() - reward_t0) * 1000
            _log_server_req_done(
                op="reward",
                session_id=session_id,
                worker_idx=worker_idx,
                elapsed_ms=result.reward_ms,
                ok=False,
                error_type=type(exc).__name__,
                error=repr(exc),
            )
            raise
        print(
            f"[trajectory] round={round_idx} worker={worker_idx} "
            f"reward={reward:.3f} reward_ms={result.reward_ms:.1f}",
            flush=True,
        )

        release_t0 = time.monotonic()
        await tool.release(instance_id)
        result.release_ms = (time.monotonic() - release_t0) * 1000
        result.ok = True
        return result
    except Exception as exc:
        result.error_type = type(exc).__name__
        result.error = repr(exc)
        print(
            f"[trajectory:error] round={round_idx} worker={worker_idx} "
            f"session_id={session_id} error_type={result.error_type} error={result.error}",
            flush=True,
        )
        try:
            await tool.release(instance_id)
        except Exception as release_exc:
            print(
                f"[trajectory:release_error] round={round_idx} worker={worker_idx} "
                f"release_error_type={type(release_exc).__name__} release_error={release_exc!r}",
                flush=True,
            )
        return result
    finally:
        result.total_ms = (time.monotonic() - total_t0) * 1000


async def _run(args: argparse.Namespace) -> int:
    if args.scenario == "rolling":
        return await _run_rolling_session_stress(args)

    tool = _load_click_tool(
        Path(args.tool_config),
        base_url=args.base_url,
        timeout=args.timeout,
        include_a11y=args.include_a11y,
    )
    all_results: list[TrajectoryResult] = []

    for round_idx in range(args.rounds):
        print(
            f"[round] begin round={round_idx} concurrency={args.concurrency} "
            f"steps={args.steps} finish_mode={args.finish_mode}",
            flush=True,
        )
        round_t0 = time.monotonic()
        tasks = [
            _one_trajectory(
                tool=tool,
                round_idx=round_idx,
                worker_idx=worker_idx,
                task_id=args.task_id,
                session_seed=args.session_seed,
                include_a11y=args.include_a11y,
                steps=args.steps,
                x=args.x,
                y=args.y,
                finish_mode=args.finish_mode,
                terminal_action=args.terminal_action,
            )
            for worker_idx in range(args.concurrency)
        ]
        round_results = await asyncio.gather(*tasks)
        all_results.extend(round_results)
        ok_count = sum(result.ok for result in round_results)
        print(
            f"[round] done round={round_idx} ok={ok_count}/{len(round_results)} "
            f"elapsed_ms={(time.monotonic() - round_t0) * 1000:.1f}",
            flush=True,
        )
        if args.round_sleep > 0:
            await asyncio.sleep(args.round_sleep)

    ok_results = [result for result in all_results if result.ok]
    errors = [result for result in all_results if not result.ok]
    create_ms = [result.create_ms for result in ok_results]
    action_ms = [value for result in ok_results for value in result.action_ms]
    finish_ms = [result.finish_ms for result in ok_results if result.finish_ms > 0]
    reward_ms = [result.reward_ms for result in ok_results]
    release_ms = [result.release_ms for result in ok_results]
    total_ms = [result.total_ms for result in ok_results]

    print("[summary] " + _summarize("create", create_ms), flush=True)
    print("[summary] " + _summarize("action", action_ms), flush=True)
    print("[summary] " + _summarize("finish", finish_ms), flush=True)
    print("[summary] " + _summarize("reward", reward_ms), flush=True)
    print("[summary] " + _summarize("local_release", release_ms), flush=True)
    print("[summary] " + _summarize("trajectory_total", total_ms), flush=True)
    print(f"[summary] ok={len(ok_results)} errors={len(errors)}", flush=True)
    for error in errors[:20]:
        print(
            f"[summary:error] round={error.round_idx} worker={error.worker_idx} "
            f"session_id={error.session_id} error_type={error.error_type} error={error.error}",
            flush=True,
        )
    return 0 if not errors else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["trajectory", "rolling"], default="trajectory")
    parser.add_argument("--tool-config", default="WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml")
    parser.add_argument("--base-url", default="http://127.0.0.1:18001")
    parser.add_argument("--task-id", default="counter")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--waves", type=int, default=20)
    parser.add_argument("--min-wave-requests", type=int, default=1)
    parser.add_argument("--max-wave-requests", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--include-a11y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x", type=int, default=696)
    parser.add_argument("--y", type=int, default=475)
    parser.add_argument("--session-seed", type=int, default=50_000_000)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--round-sleep", type=float, default=0.0)
    parser.add_argument("--step-sleep", type=float, default=0.0)
    parser.add_argument("--finish-mode", choices=["terminal", "nonterminal"], default="terminal")
    parser.add_argument("--terminal-action", choices=["DONE", "FAIL"], default="FAIL")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
