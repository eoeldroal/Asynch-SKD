# WebGym Unified Trajectory Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the split WebGym RL rollout dump and sidecar trace with a single session-directory trajectory log that includes readable model output, tool calls, tool results, and images.

**Architecture:** Keep logging ownership in the Web/OSGym agent-loop layer, where full response text, normalized actions, tool results, and image observations already coexist. Introduce a small trajectory logger that writes one directory per `{task_id}___{sample_uid}___{session_id}`, appends turn events to `trajectory.jsonl`, writes `summary.json` at finalize time, and disables the legacy trainer rollout dump for WebGym RL batches.

**Tech Stack:** Python, existing `verl.experimental.agent_loop` WebGym path, pytest.

---

### Task 1: Lock down logger behavior with tests

**Files:**
- Modify: `verl/tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`
- Modify: `verl/tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

- [ ] Add a test that runs a minimal WebGym trace write and asserts the logger creates a session directory named `{task_id}___{sample_uid}___{session_id}` with `trajectory.jsonl`, `summary.json`, and `images/`.
- [ ] Add a test that asserts `model_output_text` is readable and does not keep raw special-token padding tails.
- [ ] Add a test that asserts image files are written inside the session-local `images/` directory and referenced by relative path from `trajectory.jsonl`.
- [ ] Add a test that asserts the old per-process `events_<pid>.jsonl` layout is no longer produced by the WebGym logger path.

### Task 2: Implement unified WebGym trajectory logger

**Files:**
- Create: `verl/experimental/agent_loop/web_osgym_trajectory_logger.py`
- Modify: `verl/experimental/agent_loop/web_tool_agent_loop.py`

- [ ] Add a focused logger module that:
  - computes session directory paths from `task_id`, `sample_uid`, and `session_id`
  - writes turn events to `trajectory.jsonl`
  - writes images under `images/`
  - writes `summary.json` at session completion
- [ ] Keep the logger WebGym-specific and trajectory-oriented; do not mix trainer batch metrics into `trajectory.jsonl`.
- [ ] Record only readable model text. Do not persist raw padded decode strings.
- [ ] Use append-safe JSONL writes for turns so partial trajectories remain inspectable if a run dies mid-session.

### Task 3: Replace the current split WebGym sidecar trace

**Files:**
- Modify: `verl/experimental/agent_loop/web_tool_agent_loop.py`

- [ ] Replace `_dump_web_osgym_tool_trace(...)` with unified session-directory logging.
- [ ] Ensure each turn row includes:
  - `assistant_turn`
  - `user_turn`
  - `request_id`
  - `task_id`
  - `sample_uid`
  - `session_id`
  - `model_output_text`
  - `tool_calls_raw`
  - `tool_calls_parsed`
  - `actions`
  - `result`
  - `observation_text`
  - `image_paths`
- [ ] Ensure finalize-time summary includes:
  - `task_id`
  - `sample_uid`
  - `session_id`
  - `reward_score`
  - `termination_reason`
  - `num_turns`
  - `invalid_action_count`
  - `parse_error_count`
  - `completed`

### Task 4: Disable legacy rollout dump for WebGym RL

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py`

- [ ] Add a narrow guard so the old row-oriented `_log_rollout_data()` path does not emit `1.jsonl`, `2.jsonl`, ... for WebGym RL batches.
- [ ] Keep trainer metrics, validation, and non-WebGym dump behavior unchanged.
- [ ] Use existing batch metadata or reward-info signals to detect the WebGym RL path instead of adding broad new trainer modes.

### Task 5: Verify end-to-end behavior

**Files:**
- Test: `verl/tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

- [ ] Run the new focused tests.
- [ ] Run the existing affected WebGym agent-loop test file.
- [ ] Sanity-check that the implementation does not regress the earlier `server_prompt_ids` non-window test coverage.
