"""Tests for adapting SKD response-aligned teacher rows in AgentLoopWorker."""

from __future__ import annotations

import asyncio

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker


def _make_output(*, teacher_ids_list: list[list[int]], teacher_logprobs_list: list[list[float]]) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12, 13],
        response_mask=[1, 1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": teacher_ids_list,
            "teacher_logprobs_list": teacher_logprobs_list,
        },
    )


def test_compute_teacher_logprobs_rebuilds_skd_rows_to_full_sequence_layout():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True

    output = _make_output(
        teacher_ids_list=[[11, 111], [12, 222], [13, 333]],
        teacher_logprobs_list=[[-1.1, -1.11], [-1.2, -1.22], [-1.3, -1.33]],
    )

    asyncio.run(
        worker._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=False,
        )
    )

    assert "teacher_ids_list" not in output.extra_fields
    assert "teacher_logprobs_list" not in output.extra_fields
    assert torch.equal(
        output.extra_fields["teacher_ids"],
        torch.tensor(
            [
                [0, 0],
                [0, 0],
                [11, 111],
                [12, 222],
                [13, 333],
                [0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        output.extra_fields["teacher_logprobs"],
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-1.1, -1.11],
                [-1.2, -1.22],
                [-1.3, -1.33],
                [0.0, 0.0],
            ]
        ),
    )


def test_compute_teacher_logprobs_rejects_short_skd_rows():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True

    output = _make_output(
        teacher_ids_list=[[11, 111]],
        teacher_logprobs_list=[[-1.1, -1.11]],
    )

    with pytest.raises(ValueError, match="must exactly match response length"):
        asyncio.run(
            worker._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=False,
            )
        )


def test_compute_teacher_logprobs_rejects_empty_skd_rows_for_nonempty_response():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True

    output = _make_output(teacher_ids_list=[], teacher_logprobs_list=[])

    with pytest.raises(ValueError, match="must exactly match response length"):
        asyncio.run(
            worker._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=False,
            )
        )


def test_compute_teacher_logprobs_rejects_overlong_skd_rows():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True

    output = _make_output(
        teacher_ids_list=[[11, 111], [12, 222], [13, 333], [14, 444]],
        teacher_logprobs_list=[[-1.1, -1.11], [-1.2, -1.22], [-1.3, -1.33], [-1.4, -1.44]],
    )

    with pytest.raises(ValueError, match="must exactly match response length"):
        asyncio.run(
            worker._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=False,
            )
        )


def test_compute_teacher_logprobs_rejects_partial_skd_rows():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True

    output = AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12, 13],
        response_mask=[1, 1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={"teacher_ids_list": [[11, 111]]},
    )

    with pytest.raises(ValueError, match="both teacher_ids_list and teacher_logprobs_list"):
        asyncio.run(
            worker._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=False,
            )
        )


def test_async_skd_worker_imports_and_subclasses_agent_loop_worker():
    from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker

    assert issubclass(AsyncSkdAgentLoopWorker, AgentLoopWorker)
