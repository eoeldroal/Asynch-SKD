# Fully Async RL Static Analysis Logbook

## 2026-05-02 - WebGym fully async RL launch script

- Added `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`.
- The script is based on `WebOSWorld/run_qwen35_math_fully_async_rl_tool_fsdp.sh`.
- Training and rollout dynamics are intentionally unchanged from the math fully async RL script:
  - prompt/response lengths
  - train/gen batch settings
  - rollout `n`, temperature/top-p/top-k
  - actor/ref/log-prob token budgets
  - async training staleness, sync, partial rollout settings
- Task-facing changes only:
  - WebGym train/val parquet paths
  - WebGym reward function
  - WebOSGym tool config
  - `web_tool_agent` instead of the generic math `tool_agent`
  - WebGym-specific logging/checkpoint names
- `WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml` owns the WebOSGym server endpoint, currently `http://127.0.0.1:18001`.
- The run script intentionally does not add Ray/runtime-env overrides; backend/training skeleton should stay aligned with the math fully async RL script.

## 2026-05-02 - Protocol/implementation connection review

Reviewed:

- `WebOSWorld/document/web_osgym/protocol.md`
- `WebOSWorld/document/web_osgym/design.md`
- `verl/experimental/agent_loop/web_osgym_protocol.py`
- `verl/tools/web_osgym_tool.py`
- `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
- `verl/experimental/agent_loop/web_tool_agent_loop.py`
- `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`

Protocol connections that match:

- HTTP protocol shape matches the document: `POST /`, `op=start|action|reward`, `session_id`, `task_id`, and `include_a11y`.
- `image.data` and `image.mimeType` are parsed into a PIL RGB image; `image: null` or omitted image becomes text-only observation.
- `web_tool_agent` starts one WebOSGym session before generation and reuses the same session for later tool calls.
- WebGym tool config is connected to `WebOsGymTool`, and the new fully async RL script points at that config.
- The new script points at `web_tool_agent`, which is the intended pure fully async RL loop rather than the SKD loop.
- `conda activate skd-cudnn` config composition check succeeded for the new script with `--cfg job`; the composed config contains the WebGym tool config, reward function, `web_tool_agent`, and Ray runtime `WEBOSGYM_BASE_URL`.

Protocol/implementation mismatch to watch:

- The protocol/design says `DONE`/`FAIL` action response text/image must not imply a new post-terminal training step.
- Current `web_tool_agent` commits terminal action response text/image if the server returns it, then fetches reward. This does not create loss target because the appended observation receives `response_mask=0`, but the terminal observation can still enter actor forward as extra context/image.
- This is a real contract mismatch, especially if the real WebGym server returns a screenshot on `DONE` or `FAIL`. The clean behavior is to fetch reward immediately on terminal action and ignore terminal response observation for training state.

Follow-up fix:

- Confirmed the real WebGym gateway returns an `ActionResponse` with screenshot image even for terminal `DONE`/`FAIL` actions, then exposes the scalar through a separate `reward` op.
- Updated `web_tool_agent` so `result["terminated"]` is handled before parsing or committing the action response observation.
- This means terminal action response text/image is ignored for training state, and only the subsequent reward op is used for `web_osgym_reward_score`.
- Added CPU tests covering terminal response image/text not being committed and preserving the non-terminal response-budget guard.

SKD-path check:

- `web_skd_agent_loop` currently still has the terminal-response commit pattern.
- It executes the WebOSGym action, then parses `tool_response.text/image`, builds student/teacher observation messages, extends image data, writes `mini_step_image_spans` with `"terminal": true`, commits prompt/server/teacher streams, and only afterwards finalizes reward when `result["terminated"]` is true.
- Therefore the same protocol mismatch exists in the SKD loop unless SKD intentionally wants to keep terminal screenshot metadata for windowing. This should be treated as a separate SKD-path fix because it touches teacher streams and windowed-training metadata, not only pure async RL.

Data/config connection note:

- Current `data/webgym_rl_counter/train.parquet` still has `agent_name=web_skd_agent` for all checked rows.
- Because `AgentLoopWorker` only uses `default_agent_loop` when `agent_name` is absent, the new pure RL script needs parquet regenerated with `--agent-name web_tool_agent` or with the column removed. This is a dataset preparation issue, not a code-path bug.

Follow-up:

- Created `data/webgym_rl_counter_fully_async_rl/{train,val}.parquet` as a copy of `data/webgym_rl_counter`, changing only `agent_name` from `web_skd_agent` to `web_tool_agent`.
- Updated `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh` to use this copied dataset.
- This keeps the original SKD dataset intact and avoids changing learning dynamics.

## 2026-05-03 - Runtime log check: WebGym server interaction

Checked latest run logs:

- `logs/webgym_fully_async_rl/qwen35_webgym_fully_async_rl_20260503_152619.log`
- `logs/webgym_fully_async_rl/qwen35_webgym_fully_async_rl_20260503_153742.log`
- `goonco/webgym-rl/logs/webgym_logs.log`

Observed:

- The RL loop is reaching the WebGym server. The training logs contain many `WebOsGymTool][ServerPayload] op=action` lines with `request_id`, `task_id='counter'`, and actions such as `CLICK`, `WAIT`, and `DONE`.
- The configured endpoint matches the running server: tool config uses `http://127.0.0.1:18001`, and `ss` shows a Python process listening on `127.0.0.1:18001`.
- The interaction is not healthy yet. The RL logs show repeated failures at reward fetch time: `assert response.reward is not None` inside `web_osgym_protocol.py`.
- The latest run also shows `httpx.ReadTimeout` during `client.start(...)`, so some new session creation requests did not receive a timely response.
- Server logs show repeated `OmniboxBusyError: Error: 503 - {"detail":"No available nodes with capacity to create new instance"}`. This points to server-side capacity saturation under the current fully async concurrency, not a missing endpoint or wrong URL.

Current conclusion:

- Server connectivity is established.
- End-to-end training is blocked by WebGym service capacity/reward-response behavior under load.
- This is distinct from the previous script-routing issue; the current path is using `web_tool_agent` and is actually issuing WebOSGym actions.

## 2026-05-03 - Fully async concurrency config correction

Problem:

- `actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu` looked like a per-GPU trajectory cap, but the fully async rollouter used it as a per-GPU `RolloutSample` cap.
- In this path, one `RolloutSample` is expanded by `actor_rollout_ref.rollout.n` inside generation. With `rollout.n=8`, a cap of `16` was effectively `16 * 8 = 128` trajectories per GPU.
- With 4 rollout GPUs, the current script therefore allowed up to 512 concurrent trajectories, which matches the observed WebGym server saturation symptoms.

Change:

- In fully async mode, an explicit `max_concurrent_samples_per_gpu` value is now interpreted as maximum concurrent trajectories per GPU.
- The value must be divisible by `actor_rollout_ref.rollout.n`; otherwise TaskRunner fails early after `OmegaConf.resolve(config)`, before model/tokenizer and worker initialization.
- Internally, the rollouter converts the trajectory cap back to its active-task unit:
  - `rollout_sample_cap_per_gpu = max_concurrent_samples_per_gpu / rollout.n`
  - total active `RolloutSample` cap = `num_rollout_server_handles * rollout_sample_cap_per_gpu`

Current script effect:

- `rollout.n=8`
- `max_concurrent_samples_per_gpu=16`
- Internal cap becomes 2 `RolloutSample` objects per GPU.
- With 4 rollout GPUs, max active `RolloutSample` count becomes 8.
- Effective max concurrent trajectories becomes 64.

Verification:

- Added a CPU-only unit test for the conversion and divisibility guard.
- `pytest -q tests/experimental/fully_async_policy/test_concurrency_config.py` passes in `skd-cudnn`.
- `python -m py_compile verl/experimental/fully_async_policy/fully_async_main.py verl/experimental/fully_async_policy/fully_async_rollouter.py` passes.

## 2026-05-03 - WebGym gateway error response handling

Runtime observation:

- The WebGym gateway can log an internal command failure such as unsupported browser command handling, while still returning outer HTTP 200 to the client.
- In that case the response body carries `status="error"`, plus fields such as `error_type` and `message`.
- This is not specific to one action name like `SCROLL`; it is a body-level gateway error pattern.

Problem:

- `httpx.raise_for_status()` does not catch this because the outer HTTP status is 200.
- The previous tool path treated the response like a normal action response and only looked for text/image afterward.
- If the error response has no image and no useful `text`, the agent loop can effectively receive no tool observation, so the model does not get clear feedback that its action failed.

Change:

- `WebOsGymResponse` now preserves `error_type` and `message` from the protocol payload.
- `WebOsGymTool._send_actions` now checks `response.status` generically after the HTTP call.
- Any non-`ok` status is converted into a text `ToolResponse`, marked with `invalid_action=True`, and carries `web_osgym_error_type` in metrics.
- Normal fake/test responses without a `status` field are treated as `ok` for test compatibility.

Verification:

- Added a CPU test covering HTTP-200 gateway error bodies becoming model-visible tool feedback.
- `pytest -q tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py` passes.
- `python -m ruff check verl/experimental/agent_loop/web_osgym_protocol.py verl/tools/web_osgym_tool.py tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py` passes.
- `python -m py_compile verl/experimental/agent_loop/web_osgym_protocol.py verl/tools/web_osgym_tool.py tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py` passes.

Async SKD impact:

- `web_skd_agent` reaches WebOSGym through the same `WebOsGymLoopMixin._execute_web_osgym_tool_calls` and shared `WebOsGymTool`.
- Therefore the HTTP-200/body-`status:error` gateway pattern is fixed for async SKD as well, not only for pure async RL.
- In `WebSkdAgentLoop`, image-less tool response text is sent to both student and teacher observations, so the converted error text remains visible to the next model turn.
- This does not cover a true outer HTTP 5xx, because that still raises through `httpx.raise_for_status()` and is a separate retry/fail-soft policy question.
- `pytest -q tests/skd/test_web_skd_agent_loop_on_cpu.py tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py::TestWebOsGymTool::test_tool_execute_returns_observation_for_gateway_error_response` passes.

## 2026-05-03 - WebGym tool-call burst contract

Runtime observation:

- Latest RL log under `logs/webgym_fully_async_rl` shows a WebGym action request with 34 browser actions in one `actions=[...]` payload.
- The fully async trajectory cap was working as configured: `max_concurrent_trajectories=128`.
- The failure mode is different: one assistant turn can emit many action-named tool calls, and `WebOsGymLoopMixin` bundles those calls into one WebGym request.
- Server-side `WAIT` maps to a one-second sleep and actions are executed sequentially, so long bundles containing `CLICK`, `WAIT`, and failing `SCROLL` calls can exceed the client/server timeout budget.

Prompt-only mitigation:

- Kept `WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml` tool descriptions focused on each tool's own behavior.
- Added the 10-call contract once at the higher-level WebGym system prompt in `WebOSWorld/webgym_rl/create_webgym_rl_dataset.py`.
- Patched the current `data/webgym_rl_counter_fully_async_rl/{train,val}.parquet` files so the next run sees the contract without regenerating the dataset.
- This is only a model-visible contract. It does not structurally enforce the limit because the current tool surface exposes action-named tools (`CLICK`, `WAIT`, `SCROLL`, etc.) rather than a single `computer(actions=[...])` array schema.
- If bursts continue, the next stronger fix is a runtime guard in `WebOsGymLoopMixin` that enforces `max_parallel_calls` or a WebGym-specific `max_actions_per_request`.

Verification:

- YAML load check confirms the tool descriptions no longer repeat the 10-call contract.
- Parquet check confirms all 256 train rows and 256 val rows include `Use at most 10 browser action tool calls in one assistant turn` in the system prompt.
