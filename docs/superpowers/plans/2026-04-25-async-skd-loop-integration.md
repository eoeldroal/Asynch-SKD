# Async SKD Loop Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port APSKD Async SKD loop/teacher integration into the target `verl` repo while preserving upstream multi-teacher OPD, SGLang prompt-logprob support, and current tool-call handling.

**Architecture:** Keep APSKD's runtime protocol: SKD uses SGLang only, teacher verification requests `prompt_logprobs_start_len`, and SGLang returns chunk-suffix rows only. Adapt that protocol to target `verl` by adding narrow compatibility points in agent-loop and teacher-loop code; do not add vLLM fallback, full-row slicing fallback, new SKD enabled flags, or teacher wake/sleep wrappers. Treat an SKD generation chunk and a tool/environment append as separate committed units, but preserve the boundary-driver rule that no partial state is exported while a tool result is still pending.

**Tech Stack:** Python, Ray async actors, OmegaConf/Hydra config, PyTorch tensors, SGLang async server, pytest.

---

## File Structure

- Modify `verl/experimental/agent_loop/agent_loop.py`
  - Add Async SKD config guard using existing rollout agent config signals.
  - Add worker-local agent-loop cache for Async SKD boundary calls.
  - Pass `teacher_server_manager` into instantiated agent loops.
  - Add SKD teacher row adapter: `teacher_ids_list` / `teacher_logprobs_list` to OPD-compatible full-sequence layout.
  - Add `rollout_server_id` to `TokenOutput.extra_fields`.
  - Add small timing helper shims required by imported `verl/experimental/async_skd/worker.py`.
- Modify `verl/experimental/teacher_loop/teacher_manager.py`
  - Keep target multi-teacher routing.
  - Add `request_id` and `logprob_start_len` to `compute_teacher_logprobs_single`.
  - Require SGLang when `logprob_start_len > 0`.
  - Pass `prompt_logprobs_start_len` to SGLang sampling params.
  - Assert returned rows equal chunk suffix length.
- Modify `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
  - Port APSKD `prompt_logprobs_start_len` delta behavior.
  - Full OPD mode stays unchanged: full sequence rows plus final dummy.
  - Delta SKD mode returns suffix rows only, with no final dummy.
- Modify `verl/experimental/async_skd/manager.py`
  - Remove teacher `wake_up()` / `sleep()` calls from Async SKD generation paths.
  - Teacher servers remain always alive in the dedicated teacher inference pool.
- Do not modify `verl/experimental/agent_loop/tool_agent_loop.py`
  - Preserve target upstream tool-call and malformed tool-call handling.
- Do not change teacher-top1 replacement semantics in `verl/experimental/agent_loop/skd_agent_loop.py`
  - If the teacher replacement token is EOS, current APSKD logic commits it and ends the assistant turn.
  - Empty/short output behavior caused by teacher replacement EOS is a post-port experiment/debugging item, not part of this integration plan.
- Test files:
  - Create `tests/skd/test_async_skd_backend_config.py`
  - Create `tests/skd/test_teacher_manager_delta_contract.py`
  - Create `tests/skd/test_agent_loop_skd_teacher_rows.py`
  - Create or extend `tests/skd/test_sglang_prompt_logprobs_delta.py`

Before editing code, run:

```bash
git -C /home/sogang_nlpy/verl status --short
```

Do not revert unrelated modified files. If a target file already contains user edits, read the relevant hunks and layer the SKD changes on top.

---

### Task 1: Add Async SKD Backend Config Guard

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Test: `tests/skd/test_async_skd_backend_config.py`

- [ ] **Step 1: Write failing config guard tests**

Create `tests/skd/test_async_skd_backend_config.py`:

```python
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import _validate_async_skd_backend_config


ASYNC_SKD_MANAGER = "verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager"


def _base_config(*, rollout_name: str = "sglang", teacher_name: str = "sglang", mode: str = "lookahead"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": rollout_name,
                    "n": 1,
                    "agent": {
                        "default_agent_loop": "skd_agent",
                        "agent_loop_manager_class": ASYNC_SKD_MANAGER,
                        "async_skd_mode": mode,
                    },
                }
            },
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "teacher_key": "data_source",
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "topk": 32,
                    "use_task_rewards": False,
                    "use_policy_gradient": False,
                },
                "teacher_models": {
                    "teacher_model": {
                        "model_path": "dummy-teacher",
                        "inference": {
                            "name": teacher_name,
                            "prompt_length": 16,
                            "response_length": 16,
                            "tensor_model_parallel_size": 1,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": 64,
                            "max_num_batched_tokens": 64,
                            "engine_kwargs": {"sglang": {}, "vllm": {}},
                        },
                    }
                },
            },
        }
    )


def test_async_skd_backend_guard_accepts_sglang_student_and_teacher():
    _validate_async_skd_backend_config(_base_config())


def test_async_skd_backend_guard_rejects_non_sglang_student_rollout():
    with pytest.raises(ValueError, match="actor_rollout_ref.rollout.name='vllm'"):
        _validate_async_skd_backend_config(_base_config(rollout_name="vllm"))


def test_async_skd_backend_guard_rejects_non_sglang_teacher():
    with pytest.raises(ValueError, match="teacher 'default'.*inference.name='vllm'"):
        _validate_async_skd_backend_config(_base_config(teacher_name="vllm"))


def test_async_skd_backend_guard_ignores_sync_non_skd_rollout():
    cfg = _base_config(rollout_name="vllm", teacher_name="vllm", mode="sync")
    cfg.actor_rollout_ref.rollout.agent.default_agent_loop = "tool_agent"
    cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class = None
    _validate_async_skd_backend_config(cfg)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_async_skd_backend_config.py
```

Expected:

```text
ImportError: cannot import name '_validate_async_skd_backend_config'
```

- [ ] **Step 3: Implement config guard**

In `verl/experimental/agent_loop/agent_loop.py`, add near `DEFAULT_ROUTING_CACHE_SIZE`:

```python
ASYNC_SKD_MANAGER_CLASS = "verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager"
ASYNC_SKD_MODES = {"sample_async", "lookahead"}


def _select_config(config: DictConfig, path: str, default: Any = None) -> Any:
    return OmegaConf.select(config, path, default=default)


def _uses_async_skd(config: DictConfig) -> bool:
    default_agent_loop = _select_config(config, "actor_rollout_ref.rollout.agent.default_agent_loop")
    manager_class = _select_config(config, "actor_rollout_ref.rollout.agent.agent_loop_manager_class")
    async_skd_mode = str(_select_config(config, "actor_rollout_ref.rollout.agent.async_skd_mode", default="sync"))
    return (
        default_agent_loop == "skd_agent"
        or manager_class == ASYNC_SKD_MANAGER_CLASS
        or async_skd_mode in ASYNC_SKD_MODES
    )


def _validate_async_skd_backend_config(config: DictConfig) -> None:
    if not _uses_async_skd(config):
        return

    rollout_name = _select_config(config, "actor_rollout_ref.rollout.name")
    if rollout_name != "sglang":
        raise ValueError(
            "Async SKD requires SGLang for student rollout; "
            f"got actor_rollout_ref.rollout.name={rollout_name!r}."
        )

    if not _select_config(config, "distillation.enabled", default=False):
        raise ValueError("Async SKD requires distillation.enabled=True.")

    distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
    for teacher_key, teacher_model in distillation_config.teacher_models.items():
        teacher_backend = teacher_model.inference.name
        if teacher_backend != "sglang":
            raise ValueError(
                "Async SKD requires SGLang teacher inference; "
                f"teacher {teacher_key!r} has inference.name={teacher_backend!r}."
            )
```

Then call it at the start of `AgentLoopManager.__init__`, immediately after `self.config = config`:

```python
self.config = config
_validate_async_skd_backend_config(self.config)
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_async_skd_backend_config.py
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Checkpoint**

Do not commit unless explicitly requested by the user. Record the diff:

```bash
git -C /home/sogang_nlpy/verl diff -- verl/experimental/agent_loop/agent_loop.py tests/skd/test_async_skd_backend_config.py
```

---

### Task 2: Port SGLang `prompt_logprobs_start_len` Delta Contract

**Files:**
- Modify: `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
- Test: `tests/skd/test_sglang_prompt_logprobs_delta.py`

- [ ] **Step 1: Write failing SGLang extractor tests**

Create `tests/skd/test_sglang_prompt_logprobs_delta.py`:

```python
from __future__ import annotations

import pytest

from verl.workers.rollout.sglang_rollout.async_sglang_server import _extract_prompt_logprobs_sglang


def _meta_info(rows):
    return {
        "input_token_logprobs": [(None, 10, ""), (-0.1, 11, ""), (-0.2, 12, ""), (-0.3, 13, "")],
        "input_top_logprobs": rows,
    }


def test_sglang_prompt_logprobs_full_mode_keeps_full_sequence_contract():
    result = {}
    _extract_prompt_logprobs_sglang(
        meta_info=_meta_info(
            [
                None,
                [(-0.1, 11, "a"), (-1.1, 111, "b")],
                [(-0.2, 12, "c"), (-1.2, 112, "d")],
                [(-0.3, 13, "e"), (-1.3, 113, "f")],
            ]
        ),
        num_prompt_logprobs=2,
        sequence_length=4,
        result_dict=result,
    )

    assert result["prompt_ids"] == [[11, 111], [12, 112], [13, 113], [0, 0]]
    assert result["prompt_logprobs"][-1] == [0.0, 0.0]


def test_sglang_prompt_logprobs_delta_mode_returns_suffix_rows_only():
    result = {}
    _extract_prompt_logprobs_sglang(
        meta_info=_meta_info(
            [
                None,
                [(-0.1, 101, "p")],
                [(-0.2, 201, "s0")],
                [(-0.3, 202, "s1")],
            ]
        ),
        num_prompt_logprobs=1,
        sequence_length=4,
        result_dict=result,
        prompt_logprobs_start_len=1,
    )

    assert result["prompt_ids"] == [[201], [202]]
    assert result["prompt_logprobs"] == [[-0.2], [-0.3]]


def test_sglang_prompt_logprobs_delta_mode_rejects_bad_suffix_length():
    result = {}
    with pytest.raises(ValueError, match="SGLang delta prompt_logprobs length"):
        _extract_prompt_logprobs_sglang(
            meta_info=_meta_info([None, [(-0.1, 101, "p")]]),
            num_prompt_logprobs=1,
            sequence_length=4,
            result_dict=result,
            prompt_logprobs_start_len=1,
        )
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_sglang_prompt_logprobs_delta.py
```

Expected:

```text
TypeError: _extract_prompt_logprobs_sglang() got an unexpected keyword argument 'prompt_logprobs_start_len'
```

- [ ] **Step 3: Implement SGLang delta extraction**

In `verl/workers/rollout/sglang_rollout/async_sglang_server.py`, change `_extract_prompt_logprobs_sglang` signature to:

```python
def _extract_prompt_logprobs_sglang(
    meta_info: dict,
    num_prompt_logprobs: int,
    sequence_length: int,
    result_dict: dict[str, list],
    prompt_logprobs_start_len: int | None = None,
) -> None:
```

Inside the function, after reading `input_top_logprobs`, use this implementation shape:

```python
    delta_mode = prompt_logprobs_start_len is not None and prompt_logprobs_start_len > 0
    start_position = (prompt_logprobs_start_len + 1) if delta_mode else 1
    prompt_ids_ls: list[list[int]] = []
    prompt_logprobs_ls: list[list[float]] = []

    for position in range(start_position, len(input_token_logprobs)):
        if num_prompt_logprobs == 0:
            logprob, token_id, _ = input_token_logprobs[position]
            prompt_ids_ls.append([int(token_id)])
            prompt_logprobs_ls.append([float(logprob)])
        else:
            top_entries = input_top_logprobs[position]
            if top_entries is None:
                continue
            ids = [int(tok_id) for _, tok_id, _ in top_entries]
            logprobs = [float(logprob) for logprob, _, _ in top_entries]
            assert len(ids) == num_prompt_logprobs, (
                f"SGLang returned {len(ids)} top logprobs at position {position}, expected {num_prompt_logprobs}."
            )
            prompt_ids_ls.append(ids)
            prompt_logprobs_ls.append(logprobs)

    pad_width = max(num_prompt_logprobs, 1)
    if delta_mode:
        expected_len = sequence_length - prompt_logprobs_start_len - 1
        if len(prompt_ids_ls) != expected_len:
            raise ValueError(
                f"SGLang delta prompt_logprobs length ({len(prompt_ids_ls)}) does not match "
                f"expected suffix length ({expected_len}); "
                f"sequence_length={sequence_length}, prompt_logprobs_start_len={prompt_logprobs_start_len}."
            )
    else:
        prompt_ids_ls.append([0] * pad_width)
        prompt_logprobs_ls.append([0.0] * pad_width)
        assert len(prompt_ids_ls) == sequence_length, (
            f"SGLang prompt_logprobs length ({len(prompt_ids_ls)}) does not match "
            f"sequence length ({sequence_length}); check logprob_start_len=0 invariant."
        )

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
```

In `_generate`, pop and forward `prompt_logprobs_start_len`:

```python
prompt_logprobs = sampling_params.pop("prompt_logprobs", None)
prompt_logprobs_start_len = sampling_params.pop("prompt_logprobs_start_len", None)
```

Change request construction:

```python
if prompt_logprobs is not None:
    request["logprob_start_len"] = prompt_logprobs_start_len or 0
    if prompt_logprobs > 0:
        request["top_logprobs_num"] = prompt_logprobs
```

Change extraction call:

```python
_extract_prompt_logprobs_sglang(
    meta_info=meta_info,
    num_prompt_logprobs=prompt_logprobs,
    sequence_length=len(prompt_ids),
    result_dict=extra_fields,
    prompt_logprobs_start_len=prompt_logprobs_start_len,
)
```

- [ ] **Step 4: Run SGLang delta tests**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_sglang_prompt_logprobs_delta.py
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Checkpoint**

Do not commit unless explicitly requested. Record:

```bash
git -C /home/sogang_nlpy/verl diff -- verl/workers/rollout/sglang_rollout/async_sglang_server.py tests/skd/test_sglang_prompt_logprobs_delta.py
```

---

### Task 3: Update Teacher Manager for SKD Delta Verification

**Files:**
- Modify: `verl/experimental/teacher_loop/teacher_manager.py`
- Test: `tests/skd/test_teacher_manager_delta_contract.py`

- [ ] **Step 1: Write failing teacher manager tests**

Create `tests/skd/test_teacher_manager_delta_contract.py`:

```python
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.workers.rollout.replica import TokenOutput


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeLoadBalancer:
    def __init__(self):
        self.acquire_server = _RemoteMethod(self._acquire_server)
        self.release_server = _RemoteMethod(self._release_server)

    async def _acquire_server(self, request_id: str) -> str:
        return "teacher-replica-0"

    def _release_server(self, server_id: str) -> None:
        assert server_id == "teacher-replica-0"


class _FakeTeacherServer:
    def __init__(self, rows):
        self.rows = rows
        self.generate = _RemoteMethod(self._generate)
        self.last_kwargs = None

    async def _generate(self, **kwargs):
        self.last_kwargs = kwargs
        return TokenOutput(
            token_ids=[],
            log_probs=None,
            stop_reason="length",
            extra_fields={
                "prompt_ids": [row[0] for row in self.rows],
                "prompt_logprobs": [row[1] for row in self.rows],
            },
        )


def _config(*, teacher_backend: str = "sglang"):
    return OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "teacher_key": "data_source",
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "topk": 2,
                    "use_task_rewards": False,
                    "use_policy_gradient": False,
                },
                "teacher_models": {
                    "teacher_model": {
                        "model_path": "dummy",
                        "inference": {
                            "name": teacher_backend,
                            "prompt_length": 16,
                            "response_length": 16,
                            "temperature": 1.0,
                            "tensor_model_parallel_size": 1,
                            "data_parallel_size": 1,
                            "pipeline_model_parallel_size": 1,
                            "max_model_len": 64,
                            "max_num_batched_tokens": 64,
                            "engine_kwargs": {"sglang": {}, "vllm": {}},
                        },
                    }
                },
            }
        }
    )


@pytest.mark.asyncio
async def test_teacher_manager_forwards_prompt_logprobs_start_len_and_checks_suffix_length():
    server = _FakeTeacherServer(
        rows=[
            ([201, 999], [-0.1, -9.9]),
            ([202, 998], [-0.2, -9.8]),
        ]
    )
    manager = AsyncTeacherLLMServerManager(
        config=_config(),
        servers={"default": [("teacher-replica-0", server)]},
        load_balancer_handle={"default": _FakeLoadBalancer()},
    )

    teacher_ids, teacher_logprobs = await manager.compute_teacher_logprobs_single(
        request_id="sample-request",
        sequence_ids=[101, 102, 103, 201, 202],
        logprob_start_len=2,
    )

    assert server.last_kwargs["request_id"] != "sample-request"
    assert server.last_kwargs["sampling_params"]["prompt_logprobs_start_len"] == 2
    assert teacher_ids.tolist() == [[201, 999], [202, 998]]
    assert torch.allclose(teacher_logprobs, torch.tensor([[-0.1, -9.9], [-0.2, -9.8]]))


@pytest.mark.asyncio
async def test_teacher_manager_rejects_non_sglang_for_skd_delta_mode():
    server = _FakeTeacherServer(rows=[])
    manager = AsyncTeacherLLMServerManager(
        config=_config(teacher_backend="vllm"),
        servers={"default": [("teacher-replica-0", server)]},
        load_balancer_handle={"default": _FakeLoadBalancer()},
    )

    with pytest.raises(ValueError, match="requires SGLang teacher inference"):
        await manager.compute_teacher_logprobs_single(
            sequence_ids=[101, 102, 201],
            logprob_start_len=1,
        )


@pytest.mark.asyncio
async def test_teacher_manager_rejects_wrong_delta_length():
    server = _FakeTeacherServer(rows=[([201, 999], [-0.1, -9.9])])
    manager = AsyncTeacherLLMServerManager(
        config=_config(),
        servers={"default": [("teacher-replica-0", server)]},
        load_balancer_handle={"default": _FakeLoadBalancer()},
    )

    with pytest.raises(ValueError, match="Unexpected teacher logprob length"):
        await manager.compute_teacher_logprobs_single(
            sequence_ids=[101, 102, 103, 201, 202],
            logprob_start_len=2,
        )
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_teacher_manager_delta_contract.py
```

Expected:

```text
TypeError: AsyncTeacherLLMServerManager.compute_teacher_logprobs_single() got an unexpected keyword argument 'request_id'
```

- [ ] **Step 3: Implement teacher manager delta protocol**

In `verl/experimental/teacher_loop/teacher_manager.py`, change `_get_teacher_sampling_params` error text:

```python
if teacher_model_config.inference.temperature != 1.0:
    raise NotImplementedError("Temperature != 1.0 is not supported for teacher prompt_logprobs.")
```

Change `compute_teacher_logprobs_single` signature:

```python
async def compute_teacher_logprobs_single(
    self,
    sequence_ids: list[int],
    multi_modal_data: Optional[dict[str, Any]] = None,
    routing_key: Optional[str] = None,
    request_id: Optional[str] = None,
    logprob_start_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
```

Inside it, after resolving `teacher_model_config`:

```python
if logprob_start_len > 0 and teacher_model_config.inference.name != "sglang":
    raise ValueError(
        "Async SKD teacher verification requires SGLang teacher inference; "
        f"teacher {teacher_key!r} has inference.name={teacher_model_config.inference.name!r}."
    )

sampling_params = _get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config)
if logprob_start_len > 0:
    sampling_params["prompt_logprobs_start_len"] = logprob_start_len
```

Pass `request_id` into the server manager:

```python
teacher_output = await server_manager.generate(
    request_id=request_id or uuid4().hex,
    prompt_ids=sequence_ids,
    sampling_params=sampling_params,
    image_data=multi_modal_data.get("images"),
    video_data=multi_modal_data.get("videos"),
)
```

Replace the old length assert with:

```python
expected_len = len(sequence_ids)
if logprob_start_len > 0:
    expected_len = len(sequence_ids) - logprob_start_len - 1
if teacher_ids.shape[0] != expected_len or teacher_logprobs.shape[0] != expected_len:
    raise ValueError(
        f"Unexpected teacher logprob length: ids={teacher_ids.shape[0]}, "
        f"logprobs={teacher_logprobs.shape[0]}, expected={expected_len}, "
        f"seq_len={len(sequence_ids)}, start={logprob_start_len}."
    )
```

- [ ] **Step 4: Run teacher manager tests**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_teacher_manager_delta_contract.py
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Checkpoint**

Do not commit unless explicitly requested. Record:

```bash
git -C /home/sogang_nlpy/verl diff -- verl/experimental/teacher_loop/teacher_manager.py tests/skd/test_teacher_manager_delta_contract.py
```

---

### Task 4: Add Agent Loop SKD Call Boundary and Teacher Row Adapter

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Test: `tests/skd/test_agent_loop_skd_teacher_rows.py`

**Boundary contract:**
- `stop_after_skd_chunk=True` may pause after a committed SKD generation chunk only when the next state is `GENERATING`.
- If a generated assistant turn reaches `PROCESSING_TOOLS`, `_run_until_exportable_boundary()` must continue through `_handle_processing_tools_state()` first.
- Tool/user/interact appended spans are prompt/context spans with `response_mask=0` and dummy teacher rows; they are not teacher-verified.
- Do not redefine one SKD chunk as "generation plus tool result append." The safe export contract is "no pending tool result and teacher rows aligned," not "tool append is part of the generation chunk."

- [ ] **Step 1: Write failing teacher row adapter tests**

Create `tests/skd/test_agent_loop_skd_teacher_rows.py`:

```python
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker


def _worker_without_init():
    worker = object.__new__(AgentLoopWorker)
    worker.distillation_enabled = True
    worker.stream_teacher_with_rollout = True
    worker.teacher_server_manager = None
    return worker


@pytest.mark.asyncio
async def test_compute_teacher_logprobs_rebuilds_skd_rows_as_full_sequence_layout():
    worker = _worker_without_init()
    output = AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[201, 202],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": [[201, 999], [202, 998]],
            "teacher_logprobs_list": [[-0.1, -9.9], [-0.2, -9.8]],
        },
    )

    await worker._compute_teacher_logprobs(
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={},
    )

    assert output.extra_fields["teacher_ids"].tolist() == [
        [0, 0],
        [0, 0],
        [201, 999],
        [202, 998],
        [0, 0],
    ]
    assert torch.allclose(
        output.extra_fields["teacher_logprobs"],
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [-0.1, -9.9],
                [-0.2, -9.8],
                [0.0, 0.0],
            ]
        ),
    )


@pytest.mark.asyncio
async def test_compute_teacher_logprobs_pads_short_skd_rows_to_response_length():
    worker = _worker_without_init()
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[201, 202, 203],
        response_mask=[1, 1, 1],
        metrics=AgentLoopMetrics(),
        extra_fields={
            "teacher_ids_list": [[201, 999]],
            "teacher_logprobs_list": [[-0.1, -9.9]],
        },
    )

    await worker._compute_teacher_logprobs(
        output,
        prompt_ids=output.prompt_ids,
        response_ids=output.response_ids,
        validate=False,
        sample_kwargs={},
    )

    assert output.extra_fields["teacher_ids"].tolist() == [
        [0, 0],
        [201, 999],
        [0, 0],
        [0, 0],
        [0, 0],
    ]


def test_async_skd_worker_imports_agent_loop_compatibility_helpers():
    from verl.experimental.async_skd.worker import AsyncSkdAgentLoopWorker

    assert issubclass(AsyncSkdAgentLoopWorker, AgentLoopWorker)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_agent_loop_skd_teacher_rows.py
```

Expected before implementation may be one of:

```text
ImportError: cannot import name '_monkey_patch_log_timing'
```

or:

```text
KeyError: 'teacher_ids'
```

- [ ] **Step 3: Add compatibility helpers and `rollout_server_id`**

In `verl/experimental/agent_loop/agent_loop.py`, add imports:

```python
import time
```

Add near logger setup:

```python
def monkey_patch_timing_begin(capture_gpu: bool = False) -> float:
    del capture_gpu
    return time.perf_counter()


def _monkey_patch_log_timing(name: str, start_time: float, **extra: Any) -> None:
    del name, start_time, extra
```

In `AsyncLLMServerManager.generate`, after receiving `output` and before `return output`, add:

```python
if hasattr(output, "extra_fields") and output.extra_fields is not None:
    output.extra_fields["rollout_server_id"] = server_id
```

- [ ] **Step 4: Add agent loop cache and teacher manager injection**

In `AgentLoopWorker.__init__`, ensure non-distillation path initializes teacher fields:

```python
else:
    self.teacher_key = "data_source"
    self.teacher_server_manager = None
```

After `RolloutTraceConfig.init(...)`, add:

```python
self._agent_loop_instances: dict[str, AgentLoopBase] = {}
```

Add method in `AgentLoopWorker`:

```python
def _get_or_create_agent_loop(self, agent_name: str) -> AgentLoopBase:
    if agent_name in self._agent_loop_instances:
        return self._agent_loop_instances[agent_name]

    assert agent_name in _agent_loop_registry, (
        f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
    )
    agent_loop_config = _agent_loop_registry[agent_name]
    agent_loop = hydra.utils.instantiate(
        config=agent_loop_config,
        trainer_config=DictConfigWrap(config=self.config),
        server_manager=self.server_manager,
        teacher_server_manager=self.teacher_server_manager,
        tokenizer=self.tokenizer,
        processor=self.processor,
        dataset_cls=self.dataset_cls,
        data_config=DictConfigWrap(self.config.data),
    )
    self._agent_loop_instances[agent_name] = agent_loop
    return agent_loop
```

In `_run_agent_loop`, keep current per-sample instantiate behavior, but add `teacher_server_manager=self.teacher_server_manager` to the existing instantiate call:

```python
agent_loop = hydra.utils.instantiate(
    config=agent_loop_config,
    trainer_config=DictConfigWrap(config=self.config),
    server_manager=self.server_manager,
    teacher_server_manager=self.teacher_server_manager,
    tokenizer=self.tokenizer,
    processor=self.processor,
    dataset_cls=self.dataset_cls,
    data_config=DictConfigWrap(self.config.data),
)
```

This preserves normal rollout instantiate behavior while allowing Async SKD worker boundary calls to use `_get_or_create_agent_loop`.

- [ ] **Step 5: Add SKD teacher row adapter**

At the top of `_compute_teacher_logprobs`, before standard distillation teacher calls, add:

```python
if "teacher_ids_list" in output.extra_fields:
    teacher_ids_list = output.extra_fields.pop("teacher_ids_list")
    teacher_logprobs_list = output.extra_fields.pop("teacher_logprobs_list")
    if teacher_ids_list:
        topk_width = len(teacher_ids_list[0])
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        skd_len = len(teacher_ids_list)
        if prompt_len <= 0:
            raise ValueError("SKD teacher reconstruction requires prompt_len > 0.")
        teacher_ids_list = teacher_ids_list[:response_len]
        teacher_logprobs_list = teacher_logprobs_list[:response_len]
        while len(teacher_ids_list) < response_len:
            teacher_ids_list.append([0] * topk_width)
            teacher_logprobs_list.append([0.0] * topk_width)
        full_ids = [[0] * topk_width for _ in range(prompt_len - 1)]
        full_logprobs = [[0.0] * topk_width for _ in range(prompt_len - 1)]
        full_ids.extend(teacher_ids_list)
        full_logprobs.extend(teacher_logprobs_list)
        full_ids.append([0] * topk_width)
        full_logprobs.append([0.0] * topk_width)
        expected_len = prompt_len + response_len
        if len(full_ids) != expected_len:
            raise ValueError(
                f"[SKD] teacher_ids length mismatch: {len(full_ids)} != {expected_len} "
                f"(prompt={prompt_len}, response={response_len}, skd_accumulated={skd_len})"
            )
        output.extra_fields["teacher_ids"] = torch.tensor(full_ids, dtype=torch.int32)
        output.extra_fields["teacher_logprobs"] = torch.tensor(full_logprobs)
    return
```

- [ ] **Step 6: Run agent loop adapter tests**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_agent_loop_skd_teacher_rows.py
```

Expected:

```text
3 passed
```

- [ ] **Step 7: Checkpoint**

Do not commit unless explicitly requested. Record:

```bash
git -C /home/sogang_nlpy/verl diff -- verl/experimental/agent_loop/agent_loop.py tests/skd/test_agent_loop_skd_teacher_rows.py
```

---

### Task 5: Remove Async SKD Teacher Wake/Sleep Calls

**Files:**
- Modify: `verl/experimental/async_skd/manager.py`
- Test: `tests/skd/test_async_skd_manager_no_teacher_lifecycle.py`

- [ ] **Step 1: Write failing lifecycle test**

Create `tests/skd/test_async_skd_manager_no_teacher_lifecycle.py`:

```python
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from verl.experimental.async_skd.manager import AsyncSkdAgentLoopManager
from verl.protocol import DataProto


class _TeacherManagerThatMustNotBeCalled:
    async def wake_up(self):
        raise AssertionError("Async SKD manager must not wake teacher per rollout.")

    async def sleep(self):
        raise AssertionError("Async SKD manager must not sleep teacher per rollout.")


class _ManagerForLifecycleTest(AsyncSkdAgentLoopManager):
    def __init__(self):
        self.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "n": 1,
                        "agent": {
                            "async_skd_mode": "sample_async",
                            "async_skd_prefetch_limit": 0,
                            "async_skd_prefetch_worker_target": 0,
                        },
                    }
                }
            }
        )
        self.rollout_config = OmegaConf.create({"n": 1})
        self.stream_teacher_with_rollout = True
        self.teacher_model_manager = _TeacherManagerThatMustNotBeCalled()

    async def _generate_sequences_sample_async(self, prompts):
        return [prompts]

    def _finalize_outputs(self, outputs):
        return outputs[0]


@pytest.mark.asyncio
async def test_async_skd_manager_does_not_wake_or_sleep_teacher_per_rollout():
    manager = _ManagerForLifecycleTest()
    prompts = DataProto.from_dict(non_tensors={"raw_prompt": [["hello"]]})
    output = await manager.generate_sequences(prompts)
    assert output is prompts
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_async_skd_manager_no_teacher_lifecycle.py
```

Expected:

```text
AssertionError: Async SKD manager must not wake teacher per rollout.
```

- [ ] **Step 3: Remove wake/sleep from Async SKD manager**

In `verl/experimental/async_skd/manager.py`, replace `generate_sequences` body section:

```python
if self.stream_teacher_with_rollout:
    await self.teacher_model_manager.wake_up()
try:
    if mode == "lookahead":
        outputs = await self._generate_sequences_lookahead(prompts)
    else:
        outputs = await self._generate_sequences_sample_async(prompts)
finally:
    if self.stream_teacher_with_rollout:
        await self.teacher_model_manager.sleep()
```

with:

```python
if mode == "lookahead":
    outputs = await self._generate_sequences_lookahead(prompts)
else:
    outputs = await self._generate_sequences_sample_async(prompts)
```

In `generate_sequences_with_carryover`, replace:

```python
if self.stream_teacher_with_rollout:
    await self.teacher_model_manager.wake_up()
try:
    outputs = await self._generate_sequences_with_carryover(fresh_prompts, carryover_partials)
finally:
    if self.stream_teacher_with_rollout:
        await self.teacher_model_manager.sleep()
```

with:

```python
outputs = await self._generate_sequences_with_carryover(fresh_prompts, carryover_partials)
```

- [ ] **Step 4: Run lifecycle test**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q tests/skd/test_async_skd_manager_no_teacher_lifecycle.py
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Checkpoint**

Do not commit unless explicitly requested. Record:

```bash
git -C /home/sogang_nlpy/verl diff -- verl/experimental/async_skd/manager.py tests/skd/test_async_skd_manager_no_teacher_lifecycle.py
```

---

### Task 6: Run Focused Regression Suite and Compile Check

**Files:**
- Verify only; no edits expected.

- [ ] **Step 1: Run focused SKD and config tests**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m pytest -q \
  tests/skd/test_async_skd_backend_config.py \
  tests/skd/test_sglang_prompt_logprobs_delta.py \
  tests/skd/test_teacher_manager_delta_contract.py \
  tests/skd/test_agent_loop_skd_teacher_rows.py \
  tests/skd/test_async_skd_manager_no_teacher_lifecycle.py \
  tests/skd/test_skd_logic.py \
  tests/skd/test_async_skd_manager.py \
  tests/skd/test_async_skd_manager_lookahead.py \
  tests/skd/test_async_skd_worker_boundary.py \
  tests/workers/config/test_distillation_config_on_cpu.py \
  tests/workers/config/test_rollout_config_on_cpu.py
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Compile touched modules**

Run:

```bash
cd /home/sogang_nlpy/verl
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m compileall -q \
  verl/experimental/agent_loop/agent_loop.py \
  verl/experimental/agent_loop/skd_agent_loop.py \
  verl/experimental/async_skd \
  verl/experimental/teacher_loop/teacher_manager.py \
  verl/workers/rollout/sglang_rollout/async_sglang_server.py \
  tests/skd
```

Expected:

```text
command exits with status 0 and prints no errors
```

- [ ] **Step 3: Review final diff**

Run:

```bash
git -C /home/sogang_nlpy/verl diff --stat
git -C /home/sogang_nlpy/verl diff -- \
  verl/experimental/agent_loop/agent_loop.py \
  verl/experimental/teacher_loop/teacher_manager.py \
  verl/workers/rollout/sglang_rollout/async_sglang_server.py \
  verl/experimental/async_skd/manager.py \
  tests/skd/test_async_skd_backend_config.py \
  tests/skd/test_sglang_prompt_logprobs_delta.py \
  tests/skd/test_teacher_manager_delta_contract.py \
  tests/skd/test_agent_loop_skd_teacher_rows.py \
  tests/skd/test_async_skd_manager_no_teacher_lifecycle.py
```

Expected review points:

```text
- No vLLM fallback for SKD delta verification.
- No full-row suffix slicing fallback.
- No new distillation.skd.enabled flag.
- No MultiTeacherModelManager wake_up/sleep wrapper.
- No changes to tool_agent_loop.py.
- No change to teacher replacement EOS semantics; investigate empty/short outputs only after the port runs.
- Boundary export does not occur in PROCESSING_TOOLS; tool result append must complete before exporting a partial.
- Config guard fails early when Async SKD is combined with non-SGLang rollout or teacher backend.
- SGLang delta mode returns exactly chunk suffix rows.
- Teacher manager preserves multi-teacher routing and enforces SGLang delta contract.
- Agent loop converts SKD response-aligned teacher rows into OPD-compatible full-sequence layout.
```

---

## Self-Review

- Spec coverage:
  - Async SKD core already added in Step 1 of the port; this plan wires it into agent/teacher loops.
  - Config guard covers existing script signals: `default_agent_loop=skd_agent`, `agent_loop_manager_class=AsyncSkdAgentLoopManager`, `async_skd_mode=lookahead/sample_async`.
  - SGLang delta mode is ported instead of implementing teacher-manager full-row slicing.
  - Multi-teacher routing is preserved in `AsyncTeacherLLMServerManager`.
  - Tool loop is intentionally unchanged.
  - Teacher wake/sleep wrapper is intentionally not added.
  - Teacher replacement EOS behavior is intentionally unchanged and deferred to post-port experiment debugging.
  - Boundary semantics are clarified: generation chunks and tool/environment appends are separate units, and export is allowed only after pending tool work has been closed.
- Placeholder scan:
  - No `TBD`, `TODO`, or "implement later" steps are used as implementation instructions.
  - Every code-changing task includes concrete snippets and exact commands.
- Type consistency:
  - `request_id`, `logprob_start_len`, and `routing_key` are consistently used in teacher manager tests and implementation.
  - `_validate_async_skd_backend_config` is defined in `agent_loop.py` and imported by its tests.
  - `prompt_logprobs_start_len` is consistently used as the SGLang sampling parameter key.
