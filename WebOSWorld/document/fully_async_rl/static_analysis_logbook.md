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

Superseded note:

- This prompt-only 10-action mitigation was later rolled back.
- The later server-side evidence pointed to session capacity/release semantics rather than action burst alone as the direct capacity failure.
- Current parquet files were patched to remove the 10-action sentence, and no runtime hard cap is active.
- Action bursts remain a real model-behavior issue, but they should be treated separately from the server capacity/release bug.

## 2026-05-03 - WebGym tool-call/result dump gap

Current state:

- `trainer.rollout_data_dir` uses the generic trainer generation dump after a training batch is assembled.
- That dump decodes prompt/response tokens into text and writes score/metadata JSONL, but it does not preserve per-turn WebGym tool-call to tool-result pairs.
- It also does not carry image artifacts in an inspectable way.
- The WebOSGym path already has the right runtime information at tool execution time: parsed tool calls, bundled server actions, result metrics, text observation, and decoded screenshots.

Preferred debug design:

- Add a WebGym sidecar trace written from the agent/tool execution path, not from the trainer batch dump.
- Write one JSONL event per WebGym tool step, and save screenshots as PNG files referenced by relative path from the JSONL.
- Include model tool calls, normalized/bundled actions, server/session ids, result status, error metadata, termination metadata, observation text, image size, image hash, and image path.
- Use one event file per process or per session to avoid concurrent append ambiguity.
- Keep image bytes out of the main JSONL; inline base64 would make the dump hard to inspect and too large for normal rollout logging.

Reason:

- A sidecar trace survives crashes/timeouts before a batch reaches `trainer.rollout_data_dir`.
- It records the exact boundary that matters for current debugging: assistant tool calls -> WebGym request actions -> tool result text/image.

Implementation:

- `WebOsGymToolAgentLoop` now writes a sidecar event when `WEB_OSGYM_TOOL_TRACE_DIR` is set.
- Each event is appended to `events_<pid>.jsonl`.
- Screenshots are saved as PNG files under `images/`, with relative paths, dimensions, and SHA256 hashes recorded in the event.
- The event records model tool calls, parsed arguments, bundled/normalized actions, session/task/instance ids, result status, error metadata, observation text, and image metadata.
- Real `WebOsGymTool` executions now include the final server action payload in result metadata as `web_osgym_actions`, so dumps prefer the actual normalized payload over reconstructing from raw model calls.
- Trace dump failures are logged as warnings and do not fail the rollout.
- `run_qwen35_webgym_fully_async_rl_tool_veomni.sh` enables the trace under `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni/webgym_tool_trace`.

Verification:

- Added a CPU test that enables `WEB_OSGYM_TOOL_TRACE_DIR`, executes a WebGym tool step with an image result, and verifies both the JSONL event and PNG artifact.
- `pytest -q tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py tests/experimental/agent_loop/test_web_osgym_tool_on_cpu.py tests/experimental/agent_loop/test_web_osgym_protocol_on_cpu.py` passes in `skd-cudnn`.
- `ruff check` passes for the touched Python files.
- `bash -n WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh` passes.

## 2026-05-03 - WebGym capacity exhaustion and release contract

Runtime evidence:

- The live gateway config is `gateway.max_workers=256`, `gateway.max_in_flight=256`, and `omnibox.instances=256`.
- The current master `/info` response shows the single node at `capacity=256`, `available=0`.
- This means the server is not merely close to capacity; all 256 browser instances are currently leased.
- The latest fully async script has `actor_rollout_ref.rollout.n=8`, `n_gpus_per_node=4`, and `max_concurrent_samples_per_gpu=32`.
- In the fully async interpretation, this is `4 * 32 = 128` max concurrent trajectories, so 256 server instances should be enough only if completed or aborted trajectories reliably release their browser instances.

Protocol/code check:

- `WebOsGymClient` only exposes `start`, `action`, and `reward`. There is no explicit client-side release operation.
- `WebOsGymTool.release()` only removes the local verl tool instance from `_instance_dict`; it does not call the WebGym gateway or omnibox master.
- The WebGym gateway schedules server release in `_handle_action()` only when it sees a terminal `DONE` or `FAIL` action.
- The gateway `_handle_reward()` only pops `reward_cache` and returns the scalar reward. It does not release the server session or reset the browser instance.
- Therefore, the design assumption "final reward fetch also frees the session" is not true for the current server implementation.

Likely failure mechanism:

- Normal `DONE` / `FAIL` trajectories release correctly because terminal action handling pops `session_map` and schedules reset.
- Trajectories that end by `system_stop`, response budget exhaustion, timeout, parser failure, unknown tool, or rollout shutdown still call reward from verl when possible.
- Those non-terminal terminations do not send a terminal action to the server, so the gateway keeps the session in `session_map` and the omnibox instance remains leased.
- Over repeated runs, these leaked sessions can fill all 256 instances even when the configured live trajectory concurrency is only 128.
- This explains the server-side `No available nodes with capacity to create new instance` errors more directly than the earlier tool-call burst hypothesis.

Supporting clue:

- The WebGym OpenAI e2e runner has its own cleanup path: if a session started but no terminal action was sent, it sends a `FAIL` action in `finally`.
- The fully async `web_tool_agent` path currently lacks an equivalent server-visible cleanup action.

Open fix direction:

- Do not reintroduce the 10-action prompt or runtime hard cap as the primary fix for this capacity error.
- Add a server-visible cleanup path for non-terminal trajectory termination.
- The simplest compatible cleanup is to send a standalone `FAIL` action when a session started but no terminal action was sent, while avoiding a duplicate terminal action after normal `DONE` / `FAIL`.
- A cleaner protocol-level fix would add an explicit `release` operation, but the current gateway protocol does not expose one.
- Separately, start/action/reward timeouts should be fail-soft so one WebGym timeout does not kill the whole Ray worker.

Superseded note:

- The client-side cleanup-`FAIL` idea is not benchmark-compatible if it happens before reward evaluation.
- OSWorld evaluates the current state at cap/max-step termination; it does not automatically force score `0`.
- Therefore non-terminal finalization should be fixed on the server/protocol side by evaluating current state and releasing the instance, not by injecting semantic `FAIL` before reward.
- The current verl-side reward path should stay strict: if the reward response is missing the required reward field, crashing is acceptable because that indicates a protocol violation.

## 2026-05-03 - OSWorld cap termination reward semantics

OSWorld benchmark check:

- In `/home/sogang_nlpy/OSWorld/lib_run_single.py`, the standard runner loops while `not done and step_idx < max_steps`, then calls `env.evaluate()` unconditionally after the loop.
- `DesktopEnv.step()` sets `done=True` for `DONE` and `FAIL`, but the per-step reward is still a placeholder `0`.
- `DesktopEnv.evaluate()` is the real final scoring path.
- For normal feasible tasks, a final `FAIL` action forces score `0`.
- For infeasible tasks, a final `FAIL` action forces score `1`.
- A `DONE` action does not force success; the final state is still evaluated by task-specific metrics.
- A max-step/cap termination without `DONE` or `FAIL` is not automatically forced to `0` in the standard runner; it still calls `env.evaluate()` on the current environment state.

Implication for WebGym/Async RL:

- Treating every non-terminal termination as cleanup `FAIL` before reward would change benchmark semantics.
- It would force normal tasks to score `0` even if the current state is actually correct but the model did not emit `DONE` before the cap.
- The more OSWorld-compatible behavior is: on cap/system termination, evaluate current state, return that reward, and release the server instance.
- Therefore the preferred fix is server/protocol-side finalization on `reward` or a separate explicit `release/finalize` op, not semantic `FAIL` injection before reward.
- If a temporary client cleanup is needed, it should happen after reward evaluation and should not overwrite the reward with `FAIL=0`.

## 2026-05-03 - Latest fully async run trace dump check

Runtime evidence:

- Latest log under `logs/webgym_fully_async_rl` is `qwen35_webgym_fully_async_rl_20260503_153742.log`.
- The run exports `WEB_OSGYM_TOOL_TRACE_DIR=/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni/webgym_tool_trace`.
- Sidecar trace files are being written for four agent workers: `events_3783360.jsonl`, `events_3783361.jsonl`, `events_3783362.jsonl`, and `events_3783363.jsonl`.
- At the check point, the sidecar contained 484 JSONL events, 0 JSON parse failures, 173 events with image observations, and 173 PNG screenshots.
- The event schema contains the expected tool-call boundary data: model tool calls, parsed arguments, normalized WebGym actions, result status, observation text, and image metadata.

Conclusion:

- The latest run is correctly producing the new WebGym tool-call/result sidecar dump.
- Console logs still show only partial tool interaction evidence; they are useful for `ServerPayload` heartbeat checks, but the sidecar JSONL/PNG files are the reliable source for tool call -> tool result inspection.
- The current noisy warnings are mainly model/tool-format issues such as invalid coordinates, undefined tool names, and non-standalone `DONE`/`FAIL`, not a failure of the dump mechanism.

## 2026-05-03 - Sidecar-based rollout behavior check

Scope:

- Latest run: `logs/webgym_fully_async_rl/qwen35_webgym_fully_async_rl_20260503_153742.log`.
- Primary evidence: `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni/webgym_tool_trace/events_*.jsonl` and `images/*.png`.
- The run was still appending while inspected, so counts are snapshot-level evidence.

Image-grounded behavior:

- The model appears to understand the high-level task text/image semantics only partially.
- Rollout text often says the current value is `0`, the target is `5`, and the plus button should be clicked.
- However, the actual pointer coordinates are usually not grounded to the visible plus button.
- In sampled screenshots, the plus button is roughly around `x=648..744, y=428..524`, while many emitted clicks cluster around `x=540..550, y=605..620`, below the button area.
- Most screenshot hashes remained the unchanged counter-`0` screen, and sampled rollout scores were all `0.0`.
- Representative problematic sessions include `2072426605`, `479950076`, and `1694562302`.

Session/environment consistency:

- Sidecar inspection found no evidence that trajectories are seeing mixed environments.
- In the snapshot, `request_id`, `session_id`, and `instance_id` were effectively 1:1.
- Per-trajectory invariants had zero observed violations: stable `session_id`, stable `instance_id`, stable `task_id`, stable worker pid, and no events appended after terminal events.
- Image metadata also matched the event metadata: image path prefix, dimensions, and SHA256 references were consistent.
- Sidecar events do not include wall-clock timestamps, so exact concurrency overlap cannot be reconstructed from sidecar alone, but no shared `instance_id` across different sessions was observed.

Abnormal model behavior:

- Invalid action rate is high in the sidecar snapshot.
- Main patterns are non-standalone `DONE`/`FAIL`, repeated same-coordinate clicks, very large action bursts, non-integer/list-string coordinates, lowercase or unknown tool names, and missing click coordinates.
- Largest observed bursts reached hundreds of actions in a single tool step.
- This looks mostly like early model/tool-format behavior rather than a session routing bug.
- Protocol-side improvements worth considering later are action burst handling and clearer standalone terminal-action policy, but normalization should be used carefully because over-tolerating malformed calls can weaken the learning signal.

## 2026-05-03 - Cursor visibility in WebGym screenshots

Question:

- The model often emits cursor-relative actions such as `CLICK` without reliable coordinates, or repeatedly clicks around the wrong region.
- We checked whether the current mouse cursor is visible in the screenshot observations.

Evidence:

- A subagent inspected 55 unique sidecar PNG screenshots across all four worker pids, 52 sessions, and turns `a001_u000`, `a002_u001`, and `a003_u002`.
- It also checked crops around the event action coordinates by joining `events_*.jsonl` to the corresponding image paths.
- No visible cursor was found in the screenshots: no arrow pointer, hand pointer, black pointer, white-outline pointer, or cursor overlay near the action coordinates.
- Visible `+` and `-` marks in the screenshots are UI button glyphs, not cursor artifacts.
- Representative checked images include:
  - `images/3783360_180759837_a002_u001_00.png` for `MOVE_TO @ 542,625`.
  - `images/3783360_320847749_a002_u001_00.png` for `CLICK @ 546,616`.
  - `images/3783362_616128630_a003_u002_00.png` for `CLICK @ 546,616`.

Implication:

- The model cannot infer the current cursor position from the image observations.
- Tool schemas that allow current-position clicks without explicit `x/y` are therefore risky in this visual setting.
- Missing-coordinate `CLICK {}` calls should be treated as malformed or at least low-quality behavior unless we add an explicit cursor overlay or text state.
- This also explains why coordinate-free or cursor-relative interaction may fail even when the environment session itself is consistent.

## 2026-05-03 - `max_tool_response_length=1024` check

Question:

- The run uses `actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024`.
- We checked whether this value could be truncating image tool results or hiding important observations.

Code path:

- The generic `ToolAgentLoop._call_tool()` applies `max_tool_response_length` to tool response text.
- The active WebGym path is `agent.default_agent_loop=web_tool_agent`.
- `WebOsGymToolAgentLoop` overrides the WebGym processing path and does not use the generic text truncation at that boundary.
- For image observations, `WebOsGymToolAgentLoop._split_env_observation()` drops the text and sends the screenshot as the actual student observation.
- The effective guard in this path is the full response budget: `len(agent_data.response_mask) + len(response_ids) >= self.response_length`.

Runtime evidence:

- In the latest sidecar snapshot, observation text length was at most 526 characters.
- There were 0 events with observation text length over 1024.
- Events with image observations had text length 0 because the visual observation is carried by the screenshot.

Conclusion:

- `max_tool_response_length=1024` is not the cause of the current visual grounding failure.
- If `include_a11y=True` or long textual DOM/accessibility observations are enabled later, this value may become relevant.
- For the current image-only WebGym counter run, the more relevant limits are `data.max_response_length`, `rollout.max_model_len`, `max_num_batched_tokens`, turn count, and accumulated images.

## 2026-05-03 - Multi-turn cap increase to 50/50

Question:

- The script was changed from a small turn cap to:
  - `actor_rollout_ref.rollout.multi_turn.max_user_turns=50`
  - `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50`
- We checked whether this path is active and what it changes.

Code path:

- `WebOsGymToolAgentLoop` inherits `ToolAgentLoop`.
- `WebOsGymToolAgentLoop._handle_generating_state()` calls `super()._handle_generating_state(...)`.
- Therefore it uses the parent turn guards:
  - terminate if `agent_data.assistant_turns >= max_assistant_turns`;
  - terminate if `agent_data.user_turns >= max_user_turns`.
- So newly launched runs with the modified script will use the 50/50 cap.

Important runtime clarification:

- The analyzed latest log `qwen35_webgym_fully_async_rl_20260503_153742.log` was launched before this script change.
- Its command line still shows `max_user_turns=4` and `max_assistant_turns=4`.
- In that analyzed run, sidecar turns reached only `(assistant_turn, user_turn)` values up to about `(3,2)`.

Expected effect:

- Raising the cap to 50 allows a trajectory to keep interacting much longer if it does not hit `DONE`, `FAIL`, response budget, context budget, server timeout, or other system termination first.
- It can give the model more recovery chances after invalid actions.
- It can also amplify current failure patterns: repeated wrong clicks, action bursts, accumulated images, longer server-session occupation, staleness, and context/memory pressure.

Current judgment:

- The observed failures are not primarily caused by too few turns.
- They are mainly visual-coordinate grounding and tool-format issues.
- For debugging, 50/50 can be useful to observe long trajectories.
- For stable training, a smaller intermediate cap such as 8/8 or 12/12 is more conservative unless there is evidence that successful trajectories need many more steps.

## 2026-05-03 - Consolidated current problem list

What looks healthy:

- The WebGym fully async path is using `web_tool_agent`.
- Tool-call/result sidecar dumping works and preserves screenshots as PNGs.
- Within the sidecar snapshot, `request_id`, `session_id`, and `instance_id` remain consistent; no cross-session environment mixing was observed.
- Terminal `DONE`/`FAIL` tool result observations are not committed into the training state before reward finalization in the current `web_tool_agent` path.
- Reward values fetched by the WebGym tool are propagated into `AgentLoopOutput.reward_score` and then into RL reward/advantage computation.

What is currently problematic:

- Screenshots do not show the mouse cursor, so current-position cursor actions are not visually grounded.
- The model partially recognizes the UI/task semantics but does not ground clicks to the visible plus button.
- Many clicks cluster below the buttons instead of on the plus button.
- Invalid action rate is high: mixed terminal actions, malformed coordinates, unknown/lowercase tools, missing `x/y`, and huge action bursts.
- The generic rollout dump is not enough for diagnosing image tool results; sidecar JSONL/PNG is the reliable evidence source.
- Previous server capacity errors are best explained by server/protocol finalization and release semantics, not by local session mixing.

Current preferred next checks:

- Add optional visual debug overlays to sidecar images: action point, previous cursor position if available, and maybe target element bounding boxes.
- Add model-input image ids/paths to the sidecar event so we can prove exactly which screenshot was used for each generation call.
- If possible, include explicit cursor state as text or render cursor overlay into screenshots before relying on cursor-relative actions.
- Keep normalization conservative; malformed tool calls should remain visible as learning signal unless a recovery rule is clearly protocol-compatible.

## 2026-05-03 - Prompt/tool contract alignment with WebSKD

Goal:

- Keep Qwen3.5's native tool-calling advantage: expose action-named tools such as `CLICK`, `WAIT`, `DONE`, and `FAIL`.
- Match the OSWorld/Qwen3-VL harness everywhere else that affects browser policy behavior: screenshot grounding instructions, previous-action framing, terminal-action semantics, and message ordering.
- Avoid splitting SKD and RL into subtly different prompt distributions.

Current comparison:

- Both active launchers now point at the same canonical tool config:
  - `WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml`
- The old duplicate config `web_osgym_tool_config_webgym_rl.yaml` differed only by endpoint and was removed to avoid divergent contracts.
- The active tool schema is the Computer 13 action set exposed as named Qwen3.5 functions, not the OSWorld `computer_use(action=...)` wrapper.
- Current SKD and RL parquet prompts are identical except for routing:
  - SKD rows use `agent_name=web_skd_agent`.
  - RL rows use `agent_name=web_tool_agent`.
- That routing difference is intended. Prompt text, tool schema, and task-facing browser contract should remain shared.

Prompt gap:

- The current dataset prompt is valid but too terse compared with the OSWorld Qwen3-VL harness.
- It lists tools and terminal rules, but it weakly states the screenshot-grounded policy behavior:
  - use the current screenshot to choose the next browser action;
  - click near the center of visible target elements;
  - use `WAIT` when the page may still be loading;
  - adjust failed clicks based on the latest screenshot;
  - provide `Instruction:` and `Previous actions:` in the user message.

Preferred prompt direction:

- Put browser-GUI behavior rules in the shared dataset system prompt.
- Put the OSWorld-style instruction block in the user message:

```text
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {task_instruction}

Previous actions:
None
```

- For later bounded-window training, replace `None` with a compact action-only summary of old steps outside the recent multimodal window.
- Keep `qwen3_coder` / named-tool serialization delegated to the chat template and active tool schema.

Non-goals for this prompt pass:

- Do not switch to the OSWorld `computer_use` wrapper. That would throw away the current Qwen3.5 named-tool contract.
- Do not reintroduce the temporary 10-action cap prompt.
- Do not promise `1000x1000` coordinates until the server, screenshots, and stored trajectory metadata are consistently normalized.
- Do not make RL-only prompt edits. SKD and RL should consume the same task-facing prompt unless there is a deliberate experiment.

## 2026-05-03 - WebOSGym RL windowing implementation boundary

Implemented:

- Extracted WebOSGym step/window primitives into a shared module so AsyncSKD and AsyncRL do not import each other for basic parsing.
- AsyncSKD windowed training now imports the shared `contiguous_one_spans` and image-span normalization helpers, while its teacher-row window builder remains SKD-specific.
- WebOSGym fully async RL now records committed observation metadata in `web_osgym_steps`.
- `mini_step_image_spans` is derived from `web_osgym_steps`, so the SKD-compatible image projection and RL observation ledger share one source of truth.
- Added a pure prompt-window builder that can produce an OSWorld-style model-facing view from base task messages, image data, and `web_osgym_steps`.
- Added RL runtime controls under `actor_rollout_ref.rollout.multi_turn`:
  - `web_osgym_window_enable`
  - `web_osgym_window_history_n`
  - `web_osgym_window_max_images_per_sample`
- `WebOsGymToolAgentLoop` now uses the prompt-window builder during generation when `web_osgym_window_enable=True`.
- The active WebGym fully async RL launcher enables this path with `history_n=5` and `max_images_per_sample=6`.
- The rollout trace now reports whether the model-facing prompt was `windowed_prompt` or `full_accumulated_prompt`, plus the window settings, fallback count, and recorded generation-window count.
- WebOSGym reward finalization now also writes `web_osgym_reward_score` and `web_osgym_termination_reason` into `reward_extra_info`, so trainer rollout dumps can show whether a sample ended by `model_done`, `model_fail`, `system_stop`, or `tool_response_budget_exhausted`.

Updated boundary:

- Runtime rollout generation is windowed when the config flag is enabled.
- Update/backprop now uses the same generation-window boundary. Each assistant generation records its exact prompt ids and selected image indices, then the completed trajectory is expanded into one `AgentLoopOutput` row per assistant generation before postprocess. This is visible in `web_osgym_unit_trace` as:
  - `rollout_context=windowed_prompt`
  - `backprop_context=windowed_generation_rows`
- `data.max_response_length` remains the full trajectory output cap. With windowed rollout/update it no longer directly controls the per-step model context or mini-row padding width; those are handled by `max_model_len`, `max_num_batched_tokens`, `web_osgym_window_history_n`, `web_osgym_window_max_images_per_sample`, and the maximum target response length among emitted window rows.

Remaining follow-up:

- Keep termination-reason fields in rollout dumps while tuning `max_response_length`, turn caps, and server/session release behavior.

## 2026-05-03 - RL harness-style prompt window alignment

Implemented:

- The earlier RL helper shape was too shallow: it attached only the current screenshot and collapsed all previous steps into `Previous actions`.
- The RL-only prompt helper now mirrors the OSWorld Qwen harness topology:
  - old actions outside the recent window are summarized in `Previous actions`
  - recent `history_n` steps remain live as `user(observation)` -> `assistant(response)` chat messages
  - the current observation is the final `user` message
  - text-only failure observations remain as text-only `user` messages and do not force fake image placeholders
- `normalize_web_osgym_steps()` now preserves raw observation text in addition to `text_len`, so text-only RL windows can be reconstructed without relying on decoded dumps.
- WebOSGym RL rollout now records assistant-turn metadata keyed by observation step, including response text and parsed actions.
- `web_osgym_generation_windows` now records:
  - `prompt_image_indices`
  - `old_summary_turn_indices`
  - `recent_observation_step_indices`
  - `recent_assistant_turn_indices`
  - `text_only_recent_step_count`
- Tool sidecar events now include a compact `prompt_window` object so tool-call/result traces can be audited against the exact rollout prompt contract.
- RL update-row reconstruction now consumes recorded prompt-image indices and recent-history metadata while still keeping the target restricted to the current assistant span only.

Result:

- Rollout prompt semantics and update prompt semantics are now aligned to the same OSWorld/Qwen-style window contract.
- The remaining known gap versus the benchmark harness is not prompt window shape but environment/protocol behavior such as session release/finalization and any future coordinate normalization work.
