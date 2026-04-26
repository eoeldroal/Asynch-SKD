# Teacher Sticky Carryover Hard-Pin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend teacher-side sticky routing so carryover samples keep using the same real teacher server across steps, while new base samples are rebalanced only after pinned carryover load is accounted for.

**Architecture:** Keep the existing same-step teacher sticky model, but stop releasing sticky mappings for unfinished carryover trajectories. Add manager-owned ledgers for `sample_id -> real teacher server_id` and `sample_id -> teacher_routing_key`, use those ledgers to hard-pin carryover samples inside the correct teacher pool, and rebalance fresh base samples against the remaining pool-local load. The student path stays unchanged because rollout sleep/update releases its KV state every step.

**Tech Stack:** Python, Ray async actors, OmegaConf, existing async SKD manager/worker stack, existing agent-loop sticky load balancer tests, pytest

**Note:** Some code snippets below still use names like `teacher-replica-0` as compact test-fixture labels. They are illustrative fixture values only, not the runtime source of truth. Actual runtime binding must use the real teacher `server_id` exposed by the teacher manager, scoped by `teacher_routing_key`.

---

## File Structure

- Modify: `verl/experimental/agent_loop/agent_loop.py`
  - Extend the global sticky load balancer so a caller can explicitly bind and release `request_id -> server_id` mappings instead of relying only on least-loaded selection.
- Modify: `verl/experimental/teacher_loop/teacher_manager.py`
  - Thread an optional pinned teacher replica through teacher logprob requests and expose explicit sticky retain/release helpers for carryover lifecycle.
- Modify: `verl/experimental/agent_loop/skd_agent_loop.py`
  - Stop releasing teacher sticky mapping at export-boundary time for unfinished carryover trajectories; only release on terminal completion or explicit drop.
- Modify: `verl/experimental/async_skd/manager.py`
  - Add the manager-owned carryover pin ledger, use it to hard-pin resumed carryover, and rebalance fresh base samples after pinned load is counted.
- Modify: `verl/experimental/async_skd/state.py`
  - Mirror teacher replica pin metadata into `SkdPartialState.extra_fields` so checkpoints/restarts can reconstruct manager state cleanly.
- Modify: `verl/experimental/async_skd/data_source.py`
  - Persist and restore the carryover pin metadata along with carryover partials.
- Test: `tests/experimental/agent_loop/test_basic_agent_loop.py`
  - Add load-balancer tests for explicit binding/release of sticky mappings.
- Test: `tests/skd/test_teacher_manager_delta_contract.py`
  - Add teacher-manager tests for pinned replica routing and carryover sticky retention.
- Test: `tests/skd/test_async_skd_manager_lookahead.py`
  - Add scheduler tests proving carryover hard-pin and base-only rebalance.

### Task 1: Add Explicit Sticky Binding Primitives

**Files:**
- Modify: `verl/experimental/agent_loop/agent_loop.py`
- Test: `tests/experimental/agent_loop/test_basic_agent_loop.py`

- [ ] **Step 1: Write the failing load-balancer tests**

```python
def test_bind_request_to_server_forces_future_acquire(ray_for_lb):
    lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1", "s2"])
    ray.get(lb.bind_request_to_server.remote(request_id="carry-1", server_id="s2"))
    server = ray.get(lb.acquire_server.remote(request_id="carry-1"))
    assert server == "s2"


def test_release_request_binding_removes_sticky_mapping(ray_for_lb):
    lb = GlobalRequestLoadBalancer.remote(server_actor_ids=["s0", "s1"])
    ray.get(lb.bind_request_to_server.remote(request_id="carry-1", server_id="s1"))
    ray.get(lb.release_request_binding.remote(request_id="carry-1"))
    server = ray.get(lb.acquire_server.remote(request_id="carry-1"))
    assert server in {"s0", "s1"}
```

- [ ] **Step 2: Run the tests to confirm the API does not exist yet**

Run: `pytest tests/experimental/agent_loop/test_basic_agent_loop.py -k "bind_request_to_server or release_request_binding" -v`
Expected: FAIL with missing remote method or attribute errors on `GlobalRequestLoadBalancer`.

- [ ] **Step 3: Implement the smallest load-balancer API**

```python
class GlobalRequestLoadBalancer:
    def bind_request_to_server(self, request_id: str, server_id: str) -> None:
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for bind: {server_id}")
        self._request_id_to_server[request_id] = server_id

    def release_request_binding(self, request_id: str) -> None:
        self._request_id_to_server.pop(request_id, None)
```

- [ ] **Step 4: Re-run the focused tests**

Run: `pytest tests/experimental/agent_loop/test_basic_agent_loop.py -k "bind_request_to_server or release_request_binding" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/experimental/agent_loop/test_basic_agent_loop.py verl/experimental/agent_loop/agent_loop.py
git commit -m "feat(agent-loop): add explicit sticky request binding"
```

### Task 2: Extend Teacher Manager for Hard-Pinned Carryover Routing

**Files:**
- Modify: `verl/experimental/teacher_loop/teacher_manager.py`
- Test: `tests/skd/test_teacher_manager_delta_contract.py`

- [ ] **Step 1: Write the failing teacher-manager tests**

```python
@pytest.mark.asyncio
async def test_teacher_manager_uses_pinned_server_for_carryover():
    manager = make_teacher_manager()
    await manager.bind_sticky_request(
        routing_key="teacher_model",
        request_id="carry-1",
        server_id="teacher-replica-2",
    )
    await manager.compute_teacher_logprobs_single(
        sequence_ids=[1, 2, 3],
        routing_key="teacher_model",
        request_id="carry-1",
    )
    assert manager.server_managers["teacher_model"]._load_balancer.bound["carry-1"] == "teacher-replica-2"


@pytest.mark.asyncio
async def test_teacher_manager_release_sticky_session_clears_bound_request():
    manager = make_teacher_manager()
    await manager.bind_sticky_request(
        routing_key="teacher_model",
        request_id="carry-1",
        server_id="teacher-replica-1",
    )
    await manager.release_sticky_session("carry-1", routing_key="teacher_model")
    assert "carry-1" not in manager.server_managers["teacher_model"]._load_balancer.bound
```

- [ ] **Step 2: Run the focused teacher-manager tests**

Run: `pytest tests/skd/test_teacher_manager_delta_contract.py -k "pinned_server or release_sticky_session" -v`
Expected: FAIL because `bind_sticky_request()` / real `release_sticky_session()` behavior is missing.

- [ ] **Step 3: Implement teacher-manager sticky lifecycle helpers**

```python
class AsyncTeacherLLMServerManager:
    async def bind_sticky_request(self, *, routing_key: str, request_id: str, server_id: str) -> None:
        teacher_key = self._resolve_teacher_key(routing_key)
        await self.server_managers[teacher_key]._load_balancer.bind_request_to_server.remote(
            request_id=request_id,
            server_id=server_id,
        )

    async def release_sticky_session(self, request_id: str, routing_key: Optional[str] = None) -> None:
        teacher_key = self._resolve_teacher_key(routing_key)
        await self.server_managers[teacher_key]._load_balancer.release_request_binding.remote(request_id=request_id)
```

- [ ] **Step 4: Re-run the focused tests**

Run: `pytest tests/skd/test_teacher_manager_delta_contract.py -k "pinned_server or release_sticky_session" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/skd/test_teacher_manager_delta_contract.py verl/experimental/teacher_loop/teacher_manager.py
git commit -m "feat(teacher): add sticky binding lifecycle helpers"
```

### Task 3: Preserve Teacher Sticky Mapping Across Carryover Export/Resume

**Files:**
- Modify: `verl/experimental/agent_loop/skd_agent_loop.py`
- Modify: `verl/experimental/async_skd/state.py`
- Modify: `verl/experimental/async_skd/data_source.py`
- Test: `tests/skd/test_skd_logic.py`
- Test: `tests/skd/test_async_skd_state.py`

- [ ] **Step 1: Write the failing boundary-lifecycle tests**

```python
@pytest.mark.asyncio
async def test_export_boundary_does_not_release_teacher_sticky_for_partial():
    loop = make_skd_loop()
    partial = await loop.run_until_exportable_boundary(...)
    assert partial.extra_fields["teacher_replica_id"] == "teacher-replica-1"
    assert loop.teacher_server_manager.released_request_ids == []


def test_partial_state_round_trips_teacher_replica_pin():
    partial = _make_partial("carry-1")
    partial.extra_fields["teacher_replica_id"] = "teacher-replica-1"
    restored = AsyncSkdSample.from_partial(partial_state=partial).require_partial()
    assert restored.extra_fields["teacher_replica_id"] == "teacher-replica-1"
```

- [ ] **Step 2: Run the targeted tests**

Run: `pytest tests/skd/test_skd_logic.py -k export_boundary -v`
Run: `pytest tests/skd/test_async_skd_state.py -k teacher_replica_pin -v`
Expected: FAIL because partial export currently releases teacher sticky and does not preserve replica pin metadata.

- [ ] **Step 3: Implement the lifecycle change**

```python
class SkdAgentLoop(ToolAgentLoop):
    async def run_until_exportable_boundary(...):
        ...
        if next_state == AgentState.TERMINATED:
            await self._release_teacher_sticky_session(agent_data.request_id)
            return self._finalize_boundary_agent_output(agent_data)
        return self._export_partial_state(...)

    async def run_from_partial_to_completion(...):
        ...
        try:
            await self._run_until_terminated(...)
            return self._finalize_boundary_agent_output(agent_data)
        finally:
            await self._release_teacher_sticky_session(parent_request_id)
            await self._release_teacher_sticky_session(agent_data.request_id)
```

- [ ] **Step 4: Mirror the pinned teacher replica into carryover state**

```python
partial.extra_fields["teacher_replica_id"] = teacher_replica_id
state_dict["carryover_teacher_replica_ids"] = {
    partial.sample_id: partial.extra_fields.get("teacher_replica_id")
    for partial in self._carryover_partials
}
```

- [ ] **Step 5: Re-run the focused tests**

Run: `pytest tests/skd/test_skd_logic.py -k export_boundary -v`
Run: `pytest tests/skd/test_async_skd_state.py -k teacher_replica_pin -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/skd/test_skd_logic.py tests/skd/test_async_skd_state.py verl/experimental/agent_loop/skd_agent_loop.py verl/experimental/async_skd/state.py verl/experimental/async_skd/data_source.py
git commit -m "feat(async-skd): retain teacher sticky mapping across carryover"
```

### Task 4: Add Manager-Owned Carryover Pin Ledger and Base-Only Rebalance

**Files:**
- Modify: `verl/experimental/async_skd/manager.py`
- Test: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Write the failing scheduler tests**

```python
@pytest.mark.asyncio
async def test_carryover_current_work_uses_pinned_teacher_replica_first():
    manager, calls, _ = _make_manager(...)
    manager._teacher_replica_pin_by_sample_id = {
        "carry-200": "teacher-replica-2",
        "carry-201": "teacher-replica-0",
    }
    await manager.generate_sequences_with_carryover(
        fresh_prompts=_make_prompts(4),
        carryover_partials=[_make_partial("carry-200"), _make_partial("carry-201")],
    )
    assert manager._teacher_replica_load["teacher-replica-2"] >= 1
    assert manager._teacher_replica_load["teacher-replica-0"] >= 1


@pytest.mark.asyncio
async def test_fresh_base_rebalances_after_pinned_carryover_load():
    manager, calls, _ = _make_manager(...)
    manager._teacher_replica_pin_by_sample_id = {
        "carry-a": "teacher-replica-0",
        "carry-b": "teacher-replica-0",
        "carry-c": "teacher-replica-2",
    }
    assignments = manager._plan_teacher_replica_assignments(
        carryover_sample_ids=["carry-a", "carry-b", "carry-c"],
        fresh_sample_ids=["base-0", "base-1", "base-2", "base-3", "base-4"],
    )
    assert assignments["carry-a"] == "teacher-replica-0"
    assert assignments["carry-b"] == "teacher-replica-0"
    assert assignments["carry-c"] == "teacher-replica-2"
    assert list(assignments.values()).count("teacher-replica-1") >= 1
```

- [ ] **Step 2: Run the focused scheduler tests**

Run: `pytest tests/skd/test_async_skd_manager_lookahead.py -k "pinned_teacher_replica or rebalances_after_pinned" -v`
Expected: FAIL because there is no teacher pin ledger or assignment planner.

- [ ] **Step 3: Implement the manager-owned pin ledger and assignment planner**

```python
class AsyncSkdAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        ...
        self._teacher_replica_pin_by_sample_id: dict[str, str] = {}

    def _plan_teacher_replica_assignments(
        self,
        *,
        carryover_sample_ids: list[str],
        fresh_sample_ids: list[str],
    ) -> dict[str, str]:
        assignments = {}
        loads = self._initial_teacher_replica_load_from_pins(carryover_sample_ids)
        for sample_id in carryover_sample_ids:
            assignments[sample_id] = self._teacher_replica_pin_by_sample_id[sample_id]
        for sample_id in fresh_sample_ids:
            server_id = min(loads, key=loads.get)
            assignments[sample_id] = server_id
            loads[server_id] += 1
        return assignments
```

- [ ] **Step 4: Thread the assigned teacher replica through current/carryover launch metadata**

```python
event_context["teacher_replica_id"] = assignments[sample_id]
partial.extra_fields["teacher_replica_id"] = assignments[partial.sample_id]
self._teacher_replica_pin_by_sample_id[partial.sample_id] = assignments[partial.sample_id]
```

- [ ] **Step 5: Re-run the focused tests**

Run: `pytest tests/skd/test_async_skd_manager_lookahead.py -k "pinned_teacher_replica or rebalances_after_pinned" -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/skd/test_async_skd_manager_lookahead.py verl/experimental/async_skd/manager.py
git commit -m "feat(async-skd): hard-pin carryover teacher replicas"
```

### Task 5: Wire Teacher Replica Pin Into Real Teacher Requests

**Files:**
- Modify: `verl/experimental/teacher_loop/teacher_manager.py`
- Modify: `verl/experimental/agent_loop/skd_agent_loop.py`
- Test: `tests/skd/test_teacher_manager_delta_contract.py`
- Test: `tests/skd/test_async_skd_replica_events.py`

- [ ] **Step 1: Write the failing integration-level tests**

```python
@pytest.mark.asyncio
async def test_teacher_verification_respects_replica_pin_from_agent_state():
    loop = make_skd_loop()
    agent_data = make_agent_data()
    agent_data.extra_fields["teacher_replica_id"] = "teacher-replica-3"
    await loop._verify_chunk_with_teacher(agent_data, ...)
    assert loop.teacher_server_manager.last_bound_server_id == "teacher-replica-3"
```

- [ ] **Step 2: Run the focused tests**

Run: `pytest tests/skd/test_teacher_manager_delta_contract.py tests/skd/test_async_skd_replica_events.py -k "replica_pin" -v`
Expected: FAIL because the pin metadata is not consumed by the teacher verification path yet.

- [ ] **Step 3: Thread the replica pin into teacher verification**

```python
teacher_replica_id = agent_data.extra_fields.get("teacher_replica_id")
if teacher_replica_id is not None:
    await self.teacher_server_manager.bind_sticky_request(
        routing_key=routing_key,
        request_id=agent_data.request_id,
        server_id=teacher_replica_id,
    )
teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
    ...,
    request_id=agent_data.request_id,
)
```

- [ ] **Step 4: Re-run the focused tests**

Run: `pytest tests/skd/test_teacher_manager_delta_contract.py tests/skd/test_async_skd_replica_events.py -k "replica_pin" -v`
Expected: PASS

- [ ] **Step 5: Run the broader async-SKD scheduler suite**

Run: `pytest tests/skd/test_async_skd_manager_lookahead.py tests/skd/test_skd_logic.py tests/skd/test_teacher_manager_delta_contract.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/skd/test_teacher_manager_delta_contract.py tests/skd/test_async_skd_replica_events.py tests/skd/test_async_skd_manager_lookahead.py tests/skd/test_skd_logic.py verl/experimental/teacher_loop/teacher_manager.py verl/experimental/agent_loop/skd_agent_loop.py
git commit -m "feat(skd): route carryover teacher verification by pinned replica"
```

### Task 6: Add Observability and End-to-End Verification

**Files:**
- Modify: `verl/experimental/async_skd/manager.py`
- Modify: `verl/experimental/async_skd/events.py`
- Test: `tests/skd/test_async_skd_manager_lookahead.py`

- [ ] **Step 1: Write the failing observability test**

```python
@pytest.mark.asyncio
async def test_async_skd_metrics_report_pinned_carryover_counts():
    manager, _, source = _make_manager(...)
    source.record_carryover([_make_partial("carry-a"), _make_partial("carry-b")])
    manager._teacher_replica_pin_by_sample_id = {
        "carry-a": "teacher-replica-0",
        "carry-b": "teacher-replica-2",
    }
    await manager.generate_sequences_with_carryover(
        fresh_prompts=_make_prompts(2),
        carryover_partials=[_make_partial("carry-a"), _make_partial("carry-b")],
    )
    metrics = manager._async_skd_last_step_metrics
    assert metrics["async_skd/teacher_pinned_carryover_count"] == 2
```

- [ ] **Step 2: Run the focused observability test**

Run: `pytest tests/skd/test_async_skd_manager_lookahead.py -k teacher_pinned_carryover_count -v`
Expected: FAIL because the metric does not exist yet.

- [ ] **Step 3: Add minimal metrics and event fields**

```python
step_metrics.update(
    {
        "async_skd/teacher_pinned_carryover_count": pinned_carryover_count,
        "async_skd/teacher_pin_fallback_count": teacher_pin_fallback_count,
    }
)
emit_async_skd_event(
    "teacher_pin_assignment",
    sample_id=sample_id,
    teacher_replica_id=server_id,
    pinned=is_carryover,
)
```

- [ ] **Step 4: Re-run the focused and broad tests**

Run: `pytest tests/skd/test_async_skd_manager_lookahead.py -k teacher_pinned_carryover_count -v`
Run: `pytest tests/experimental/agent_loop/test_basic_agent_loop.py tests/skd/test_teacher_manager_delta_contract.py tests/skd/test_async_skd_manager_lookahead.py tests/skd/test_skd_logic.py -v`
Expected: PASS

- [ ] **Step 5: Do one smoke run command without long training**

Run:

```bash
python -m verl.trainer.main_ppo \
  model_engine=veomni \
  actor_rollout_ref.rollout.mode=async \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager \
  +actor_rollout_ref.rollout.agent.async_skd_mode=lookahead \
  trainer.total_epochs=1 \
  data.train_batch_size=4
```

Expected: startup succeeds, async SKD manager logs show pinned carryover metrics/events, and there is no regression in teacher request routing startup.

- [ ] **Step 6: Commit**

```bash
git add tests/experimental/agent_loop/test_basic_agent_loop.py tests/skd/test_teacher_manager_delta_contract.py tests/skd/test_async_skd_manager_lookahead.py tests/skd/test_skd_logic.py verl/experimental/async_skd/events.py verl/experimental/async_skd/manager.py
git commit -m "chore(async-skd): add teacher carryover pin metrics"
```

## Self-Review

- Spec coverage:
  - teacher sticky extended across carryover: Task 2, Task 3
  - manager-owned `sample_id -> teacher_replica` ledger: Task 4
  - carryover hard pin: Task 4, Task 5
  - base-only rebalance after pinned load: Task 4
  - fallback/release lifecycle: Task 2, Task 3
  - observability for validation: Task 6
- Placeholder scan:
  - no `TODO` / `TBD` placeholders left in task steps
  - each code-changing step includes exact code or command shape
- Type consistency:
  - uses one consistent vocabulary: `teacher_replica_id`, `bind_sticky_request()`, `release_request_binding()`, `_teacher_replica_pin_by_sample_id`
