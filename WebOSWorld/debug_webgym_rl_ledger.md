# WebGym RL Debug Ledger

Last updated: 2026-05-13

## Scope

Track only the facts that still matter for debugging or operating the current WebGym fully async RL path. Old exploratory detail has been compressed. Git history remains the archive.

---

## Current State

### 1. Rollout quality

- The earliest severe rollout collapse was **not** explained by one thing alone.
- Two factors were confirmed:
  1. `window=false` originally had a real multimodal SGLang bug: non-window image-bearing requests did not build `server_prompt_ids`.
  2. RL rollout sampling was much noisier than the SKD run that produced the checkpoint.

- The non-window multimodal bug was fixed in:
  - [web_tool_agent_loop.py](/home/sogang_nlpy/verl/verl/experimental/agent_loop/web_tool_agent_loop.py)

- Sampling mismatch was a major confirmed factor:
  - original RL rollout: `temperature=1.0`, `top_p=1.0`, `top_k=-1`
  - SKD rollout: `temperature=0.6`, `top_p=0.95`, `top_k=20`
  - matching RL rollout to SKD values materially improved both `window=true` and `window=false`.

- Current interpretation:
  - `window=true` can still shift the prompt distribution,
  - but the earlier “immediate empty termination everywhere” pattern was mostly amplified by the old RL sampling.

### 2. Dump interpretation

- Long `<|endoftext|>` tails in the old JSONL rollout dump were **not** reliable evidence of model collapse.
- Reason:
  - the old trainer dump decoded padded `responses` with `skip_special_tokens=False`
  - Qwen 3.5 uses:
    - `eos_token = <|im_end|>`
    - `pad_token = <|endoftext|>`

- Therefore:
  - early `<|im_end|>` is meaningful,
  - long repeated `<|endoftext|>` tails were often padding artifact.

### 3. Prompt / system prompt conclusions

- The WebGym RL path does inject the runtime system prompt.
- The first RL prompt was checked and did contain:
  - the system prompt
  - the serialized `computer` tool schema

- Therefore:
  - “RL forgot the system prompt” was disproved.

- Also:
  - math500 and WebGym RL do **not** share the same agent-loop path
  - math500 staying healthy is not evidence that WebGym RL prompt assembly is healthy

### 4. Prozilla observations

- The Prozilla slice is broadly visually broken.
- Subagent analysis over:
  - `/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni/webgym_tool_trace/events_*.jsonl`
  - and corresponding `images/*.png`
  found:
  - `2528` Prozilla trace rows
  - `953` unique sessions
  - a dominant near-black image hash repeated `477` times
  - representative Prozilla screenshots were essentially black canvases with only cursor movement

- This is not isolated to `prozilla_explorer_11`.
- It appears across Prozilla explorer / terminal / calc / scripts subfamilies.

- Current working explanation:
  - the model is often acting against a visually uninformative dark screen,
  - so it falls back to blind navigation, malformed actions, or unsupported key guesses.

### 5. `prozilla_explorer_11`

- `prozilla_explorer_11` is not uniquely broken on the server side.
- It is a **good stress task** for current RL weaknesses because it combines:
  - dark / low-information starting screen
  - file navigation / deletion objective
  - launcher / keyboard-heavy action attempts

- Observed failure pattern:
  - model emits unsupported key names such as `win`
  - older runtime normalized `cmd`, `command`, `meta` to `Meta`
  - current Linux-focused runtime normalizes shortcut intent `ctrl`, `control`, `cmd`, `command`, and `controlormeta` to `Control`
  - current Linux-focused runtime normalizes literal OS modifier `meta`, `super`, `win`, and `windows` to `Meta`
  - current runtime also splits HOTKEY combo strings such as `ctrl+a` into canonical key lists such as `["Control", "A"]`
  - Playwright rejects `Keyboard.down("win")`

- This is a contract gap:
  - tool schema allows arbitrary key strings
  - runtime alias map is incomplete

### 6. Server-side failure classes

- There are two distinct server failure classes and they must not be mixed:

1. **`start` failure**
   - root cause: `localhost:3100` unreachable
   - symptom:
     - `CREATE_FAILED`
     - `ERR_CONNECTION_REFUSED at http://localhost:3100/`
   - affected tasks:
     - `prozilla_terminal_03`
     - `prozilla_terminal_04`
     - `prozilla_calc_02`

2. **`action` failure**
   - root cause: unsupported key name such as `win`
   - symptom:
     - gateway retries
     - `_execute_browser_command` deadline exceeded
     - wavepool instance error:
       - `Keyboard.down: Unknown key: "win"`

- `3100` is operationally fragile because:
  - Prozilla depends on it,
  - but `launch_all.bash` does not launch it,
  - and `stop_all.bash` does not manage it either.

### 7. Reward failure handling

- A `Web/OSGym reward failed` exception is not merely “one bad sample”.
- Current fully async control flow means:
  - agent-loop reward fetch exception escapes,
  - rollouter eventually emits queue termination,
  - trainer stops after receiving the queue termination signal.

- So reward fetch failure is a run-stopping class of error under the current logic.

---

## Implemented Changes

### 1. Non-window multimodal fix

- Non-window image-bearing WebGym generation now builds `server_prompt_ids`.
- Covered by tests in:
  - [test_web_tool_agent_loop_on_cpu.py](/home/sogang_nlpy/verl/tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py)

### 2. WebGym unified trajectory logging

- WebGym RL logging was redesigned to stop relying on split:
  - row dump
  - sidecar trace
  - separate image pool

- New WebGym trajectory logger:
  - [web_osgym_trajectory_logger.py](/home/sogang_nlpy/verl/verl/experimental/agent_loop/web_osgym_trajectory_logger.py)

- Current session layout:
```text
{task_id}___{sample_uid}___{session_id}/
  summary.json
  trajectory.jsonl
  images/
```

- `trajectory.jsonl` now records:
  - readable `model_output_text`
  - `tool_calls_raw`
  - `tool_calls_parsed`
  - normalized `actions`
  - `result`
  - `observation_text`
  - image paths

- `summary.json` records:
  - `reward_score`
  - `termination_reason`
  - `invalid_action_count`
  - `parse_error_count`
  - `completed`

- WebGym RL now skips the old trainer rollout dump path:
  - [ray_trainer.py](/home/sogang_nlpy/verl/verl/trainer/ppo/ray_trainer.py)

- RL launcher root points the trajectory logger at:
  - `ROLLOUT_DATA_DIR`
  instead of a separate `webgym_tool_trace` subdirectory:
  - [run_qwen35_webgym_fully_async_rl_tool_veomni.sh](/home/sogang_nlpy/verl/WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh)

### 3. Sampling alignment

- RL launcher now uses SKD-aligned rollout sampling:
  - `temperature=0.6`
  - `top_p=0.95`
  - `top_k=20`

---

## What Was Disproved

- “The old `<|endoftext|>` tails prove the model generated garbage for the whole sequence.”
  - false; padding artifact was a major confounder.

- “The main problem was simply that RL forgot the system prompt.”
  - false.

- “The server failures were all one issue.”
  - false; `3100 down` and `win` key failure are different classes.

- “`prozilla_explorer_11` is uniquely broken.”
  - false; it is mainly a highly revealing task inside a broader Prozilla dark-screen problem.

---

## What Still Matters

1. **Need end-to-end validation of the new unified trajectory logger in a fresh RL run.**
   - Code and tests are in place.
   - A fresh run should be used to verify that only session directories are produced and legacy row dumps are absent for WebGym RL.

2. **Need end-to-end confirmation that the new Linux-focused key normalization reduces invalid keyboard actions.**
   - Current runtime policy:
     - shortcut intent `ctrl`, `control`, `cmd`, `command`, `controlormeta` -> `Control`
     - literal OS modifier `meta`, `super`, `win`, `windows` -> `Meta`
     - HOTKEY combo strings are split and canonicalized in `verl`
     - single-key actions reject combo strings early
   - Remaining work is empirical validation in a fresh RL run, not a contract decision.

3. **Need an operational answer for `3100`.**
   - Either:
     - manage it in `launch_all` / `stop_all`
     - or add an explicit preflight health check before RL

4. **Need a real answer for the Prozilla dark-screen issue.**
   - The current evidence strongly suggests a shared rendering / observation problem across Prozilla tasks.
   - This should now be treated as a first-class environment issue, not a minor side effect.

---

## Practical Rule For Future Incidents

When the next failure happens, classify it first:

1. `start` failure with `ERR_CONNECTION_REFUSED at 3100`
   - environment / Prozilla server availability

2. `action` failure with `Unknown key: "win"`
   - action contract / key normalization

3. early `<|im_end|>` or malformed tool-call behavior
   - actor / prompt / sampling side

Do not mix these three classes during debugging.

---

## 2026-05-13 Fully Async WebGym RL Crash Ledger

This section records the facts confirmed while debugging the current Qwen 3.5 WebGym fully async RL path.

### 1. Current crash is not a rollout timeout or reward-server 500

- The earlier `httpx.ReadTimeout` class was a tool-server request timeout, not an HTTP 500 response.
- In the latest actor crash run, timeout dropping did not appear to be the immediate failure mode:
  - `count/timeout_dropped_samples=0` was observed in the run context.
- The latest failure happened after enough rollout samples had already been collected for one trainer step.

### 2. One trainer step entered the training path, but actor update did not complete

- Latest main log:
  - [logs/qwen35_webgym_fully_async_rl.out](/home/sogang_nlpy/verl/logs/qwen35_webgym_fully_async_rl.out)

- Confirmed timeline:
  - `Loop collection completed: 16/16 samples`
  - `Batch assembly completed`
  - trainer entered `_fit_update_actor`
  - failure occurred at `self.actor_rollout_wg.update_actor(batch_td)`
  - no `actor/loss`, `actor/grad_norm`, or completed `step:1` was observed

- Evidence:
```text
[FullyAsyncTrainer] Loop collection completed: 16/16 samples
[BatchUtils] Training payload summary {'rows': 258, 'window_rows': 258, 'web_osgym_window_supervision_block_sizes': [3], 'multi_modal_image_count': 1805, 'multi_modal_tensor_mib': 86301.62}
[BatchUtils] Batch assembly completed
...
File ".../verl/trainer/ppo/ray_trainer.py", line 1599, in _update_actor
    actor_output = self.actor_rollout_wg.update_actor(batch_td)
ray.exceptions.ActorUnavailableError: ... RpcError: RPC error: Socket closed rpc_code: 14.
```

- Current interpretation:
  - Backprop/update path was reached.
  - Actor update did not finish.
  - The failure boundary is the actor update RPC / actor-side receive or very early actor update path.
  - There is not enough evidence to claim a normal model-forward CUDA OOM, because actor-side loss/backward logs were not reached.

### 3. The failing payload is already huge before actor mini-batching starts

- Latest assembled training payload:
  - rows: `258`
  - Web/OSGym window rows: `258`
  - block size: `[3]`
  - multi-modal image count: `1805`
  - multi-modal tensor size: `86301.62 MiB`

- This means the apparent trainer collection size of `16` samples is misleading.
- The actual actor update payload is:
```text
16 queue samples
-> many Web/OSGym window rows
-> 258 training rows
-> 1805 processed image tensors
-> about 86 GiB of multi_modal_inputs
```

- The crash should be treated as a payload-boundary problem before normal actor micro-batch training, not as a simple per-forward micro-batch sizing issue.

### 4. The 1805 images were not from block size 1 in the latest failed run

- The latest failed payload explicitly reports:
```text
'web_osgym_window_supervision_block_sizes': [3]
```

- Therefore, for this run, `1805` images were produced under block size `3`, not block size `1`.
- Average image count per training row was about:
```text
1805 / 258 ~= 7.0 images per row
```

- This matters because reverting to block size `1` would likely increase row count and can make total payload pressure worse unless image materialization is fixed.

### 5. Image downscaling is not the preferred fix

- Reducing screenshot resolution would reduce `pixel_values` size, but it is risky for WebGym/WebOSWorld tasks because:
  - small text,
  - input fields,
  - button labels,
  - layout details,
  - and cursor-relative UI state
  can be task-critical.

- The preferred direction is to avoid materializing all full-resolution `pixel_values` at once, not to destroy visual information.

### 6. One processed full-resolution screenshot is about 47.8 MiB

- From the latest payload:
```text
86301.62 MiB / 1805 images ~= 47.8 MiB per image
```

- This matches the expected scale of full-resolution Qwen-VL processor output for 1920x1080 screenshots.
- The heavy object is not just a PNG/JPEG screenshot; it is the processor-produced torch tensor payload, mainly `pixel_values`.

### 7. Where `multi_modal_inputs` are currently created

- Windowed output first creates `AgentLoopOutput` rows that still carry sliced raw `multi_modal_data`.
- The critical conversion happens later in agent-loop postprocess:
  - [agent_loop.py](/home/sogang_nlpy/verl/verl/experimental/agent_loop/agent_loop.py)

- Current path:
```text
_run_raw_agent_loop()
-> _maybe_window_web_osgym_outputs(raw_output)
-> _agent_loop_postprocess(output_item)
-> _compute_multi_modal_inputs()
-> processor(..., return_tensors="pt")
-> multi_modal_inputs in DataProto.non_tensor_batch
```

- Relevant code facts:
  - `build_web_osgym_windowed_agent_loop_outputs()` slices `multi_modal_data` by image indices.
  - `_agent_loop_postprocess()` calls `_compute_multi_modal_inputs()`.
  - `_compute_multi_modal_inputs()` runs the processor and converts the result to torch tensors.
  - `_postprocess()` stores each row's `multi_modal_inputs` under `non_tensor_batch["multi_modal_inputs"]`.

### 8. Queue sample vs training row are different units

- `RolloutSample` is the queue item used by fully async rollouter/trainer.
- The rollouter initially creates one `RolloutSample` per original dataset sample.
- After `generate_sequences_single()`, `RolloutSample.full_batch` may already contain many Web/OSGym window rows.
- The trainer collects `required_samples` queue samples, currently `16`, then concatenates their `full_batch` values.

- Current shape:
```text
RolloutSample 1
  full_batch = DataProto with many window rows

Trainer collection:
  16 RolloutSample objects
  -> DataProto.concat(...)
  -> one large training batch
```

- Therefore, the phrase "16 samples" does not bound actor update memory for Web/OSGym windowed training.

### 9. Actor mini-batching is too late to protect Ray transfer / actor receive

- Trainer `_update_actor()` converts the whole `DataProto` into one `TensorDict`:
  - `batch.to_tensordict()`
  - `left_right_2_no_padding(batch_td)`
  - `self.actor_rollout_wg.update_actor(batch_td)`

- `DataProto.to_tensordict()` wraps non-tensor batch entries, including `multi_modal_inputs`, into TensorDict non-tensor structures.
- It does not split the multi-modal payload.

- Actor-side `train_mini_batch()` and engine micro-batching happen after the full `batch_td` has reached the actor worker.
- Therefore:
  - `ppo_mini_batch_size`,
  - `use_dynamic_bsz`,
  - `ppo_max_token_len_per_gpu`,
  - and internal micro-batching
  do not prevent the full `multi_modal_inputs` payload from being sent to actor update.

### 10. Calling `update_actor()` multiple times is not equivalent to one PPO update

- `update_actor()` calls `self.actor.train_mini_batch(data=data)`.
- The training engine performs:
  - `optimizer_zero_grad()`
  - forward/backward
  - `optimizer_step()`

- Therefore, simply splitting the batch in trainer and calling `update_actor()` repeatedly would produce multiple optimizer steps.
- That is not equivalent to one PPO update with gradient accumulation.
- If actor update RPC is chunked, it needs explicit gradient accumulation semantics, with optimizer step only after the final chunk.

### 11. Raw window rows are the right long-term transfer unit

- The best cut point is after window rows are built and before `_agent_loop_postprocess()` materializes `multi_modal_inputs`.

- Desired direction:
```text
trajectory
-> Web/OSGym window rows
-> queue / trainer collection over raw window rows or raw window row groups
-> processor runs only inside model worker micro-batch
```

- Raw window row should carry:
  - `prompt_ids`
  - `response_ids`
  - `response_mask`
  - `response_logprobs` if present
  - reward / reward metadata
  - raw images or image references
  - image indices / window metadata
  - identity keys such as `uid`, `index`, `input_pos`
  - `web_osgym_window_*` metadata

- It should not carry full processed `pixel_values` for all rows in the trainer batch.

### 12. Moving processor later requires handling position ids

- Current postprocess computes `multi_modal_inputs` first and then computes `position_ids`.
- Qwen-VL position ids depend on image grid metadata such as `image_grid_thw`.
- Therefore, delaying processor execution also means delaying or relocating position-id construction.

- This affects not just actor update but any model-forward path that uses the batch:
  - old log prob
  - ref log prob
  - critic
  - actor update

### 13. GRPO grouping must not be accidentally broken

- Current GRPO advantage computation uses `data.non_tensor_batch["uid"]` as the group id.
- Fully async rollouter currently writes the same uid across all rows in a `RolloutSample.full_batch`.

- If queue or trainer collection changes from trajectory-level samples to window-row-level units, the implementation must define:
  - which rows share `uid`,
  - whether a GRPO group can be split across trainer steps,
  - and how to avoid partial-group advantage distortion.

- Safe default:
  - keep complete GRPO groups together for advantage computation,
  - do not silently train on partial uid groups unless that change is explicitly intended.

### 14. Immediate safe mitigations before the raw-window redesign

- Lower actor update collection size so each trainer step carries fewer window rows and fewer images:
```text
actor_rollout_ref.actor.ppo_mini_batch_size=4 or 8
async_training.require_batches=1
```

- Keep values compatible with `rollout.n` so GRPO group structure is not accidentally disturbed.

- Add a fail-fast guard before actor update:
  - compute row count,
  - image count,
  - estimated `multi_modal_tensor_mib`,
  - block size,
  - and relevant sample ids,
  then raise a clear exception before `actor_rollout_wg.update_actor(batch_td)` if the payload is over a configured limit.

- The guard does not solve memory by itself, but it prevents opaque Ray actor death and makes the next failure diagnosable.

### 15. ReadTimeout isolation design decision

- ReadTimeout should be isolated at the sample group boundary, not turned into fake training data.
- If `generate_sequences_single()` fails with ReadTimeout:
  - drop the entire rollout group before queue put,
  - do not enqueue partial successes,
  - do not inject tool error text as model training data,
  - increment a visible failure counter,
  - print a clear console log.

- Server cleanup requirement:
  - even when dropping the group, the client should still attempt the reward/close request needed by the tool server to release the session.
  - the response does not need to be used for training.

### 16. Timeout value must avoid action/reward overlap on the same server lease

- If client-side HTTP timeout is shorter than the server's action deadline, the client can time out and send reward/close while the server is still processing an action.
- That can create concurrent action/reward execution on the same lease.
- Client timeout should be at least as long as the server-side action deadline used by the environment contract.

### 17. Confirmed non-goals

- Do not bypass existing safety code.
- Do not silently drop rows inside a GRPO group without explicit accounting.
- Do not fix this by reducing screenshot resolution unless there is no viable memory-safe path.
- Do not treat actor internal mini-batching as sufficient; it does not protect actor update RPC payload size.

### 18. Current root-cause hypothesis

- Direct cause:
  - actor update receives a single huge TensorDict containing about `86 GiB` of processed multi-modal tensors.

- Repeating cause:
  - `multi_modal_inputs` are materialized in agent-loop postprocess for every Web/OSGym window row before trainer collection and actor micro-batching.

- Structural fix:
  - keep window rows raw until model-worker micro-batch time,
  - or introduce actor update chunking with true gradient accumulation and a single optimizer step.

### 19. Effect of lowering `actor.ppo_mini_batch_size` from 16 to 4

- Confirmed code path:
  - FullyAsyncTrainer sets `required_samples = actor_rollout_ref.actor.ppo_mini_batch_size * async_training.require_batches`.
  - FullyAsyncRollouter also sets the same `required_samples`.
  - Rollouter computes:
```text
max_required_samples =
  required_samples
  * (staleness_threshold + 1)
  * trigger_parameter_sync_step
```
  - Rollouter active task cap is:
```text
max_concurrent_samples =
  min(
    len(server_handles) * resolve_max_concurrent_rollout_samples_per_gpu(config),
    max_required_samples,
  )
```

- Current launcher values:
```text
actor_rollout_ref.actor.ppo_mini_batch_size=16
async_training.require_batches=1
async_training.staleness_threshold=1.0
async_training.trigger_parameter_sync_step=2
actor_rollout_ref.rollout.n=4
actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16
rollout.n_gpus_per_node=4
rollout.nnodes=1
```

- Current observed rollouter stats:
```text
required_samples: 16
max_required_samples: 64
max_concurrent_samples: 16
max_concurrent_trajectories: 64
```

- If only `actor.ppo_mini_batch_size` changes from `16` to `4`:
```text
required_samples = 4
max_required_samples = 4 * (1 + 1.0) * 2 = 16
rollout-side capacity = 4 rollout servers * (16 / rollout.n=4) = 16 sample groups
max_concurrent_samples = min(16, 16) = 16
max_concurrent_trajectories = 16 * rollout.n=4 = 64
```

- Conclusion:
  - With the current launcher, lowering `ppo_mini_batch_size` from `16` to `4` does not reduce the active rollout concurrency cap.
  - It does reduce trainer collection size from 16 queue samples to 4 queue samples per local update.
  - It also reduces `max_queue_size` / stale-sample budget from 64 to 16, so rollouter can pause sooner if trainer cannot consume quickly enough.
  - It increases trainer local update frequency and can increase parameter-sync frequency per consumed sample unless `trigger_parameter_sync_step` is adjusted.

- Practical implication:
  - `ppo_mini_batch_size=4`, `rollout.n=4`, `async_training.require_batches=1` is a reasonable immediate memory mitigation.
  - Rollout active concurrency should remain at 16 sample groups / 64 trajectories under the current 4 rollout replicas and `agent.max_concurrent_samples_per_gpu=16`.
  - If higher buffering is needed without increasing actor update payload, adjust `staleness_threshold` or `trigger_parameter_sync_step`, not `ppo_mini_batch_size`.

### 20. Rollout-speed risk when lowering `ppo_mini_batch_size`

- Active rollout concurrency and rollout throughput are related but not identical.
- With current values, lowering `ppo_mini_batch_size` from `16` to `4` keeps the active rollout cap unchanged:
```text
current:
  ppo_mini_batch_size=16
  required_samples=16
  trigger_parameter_sync_step=2
  staleness_threshold=1.0
  max_required_samples=64
  rollout-side active cap=16 sample groups
  max_concurrent_samples=min(16, 64)=16
  max_concurrent_trajectories=64

candidate:
  ppo_mini_batch_size=4
  required_samples=4
  trigger_parameter_sync_step=2
  staleness_threshold=1.0
  max_required_samples=16
  rollout-side active cap=16 sample groups
  max_concurrent_samples=min(16, 16)=16
  max_concurrent_trajectories=64
```

- However, if `trigger_parameter_sync_step` stays at `2`, trainer consumes fewer samples per parameter sync:
```text
current samples per sync:
  required_samples * trigger_parameter_sync_step = 16 * 2 = 32

candidate samples per sync:
  required_samples * trigger_parameter_sync_step = 4 * 2 = 8
```

- That means parameter sync can happen 4x more often per consumed sample.
- More frequent sync can reduce effective rollout speed because:
  - rollout weight update has non-trivial cost,
  - `reset_staleness()` is called after every sync,
  - with `partial_rollout=True`, generation can be interrupted/resumed around sync boundaries,
  - smaller `max_queue_size` / stale budget can make rollouter pause sooner if trainer-side work stalls.

- If preserving rollout speed is the top priority while using `ppo_mini_batch_size=4`, keep the per-sync sample budget similar by increasing:
```text
async_training.trigger_parameter_sync_step=8
```

- Then:
```text
required_samples=4
trigger_parameter_sync_step=8
samples per sync=4 * 8 = 32
max_required_samples=4 * (1 + 1.0) * 8 = 64
max_concurrent_samples=min(16, 64)=16
max_concurrent_trajectories=64
```

- Tradeoff:
  - `trigger_parameter_sync_step=8` preserves rollout buffering/sync cadence better.
  - But it performs more local actor updates per rollout parameter version than the current `trigger_parameter_sync_step=2` setup.
  - This changes training dynamics more than only lowering `ppo_mini_batch_size` with `trigger_parameter_sync_step=2`.

### 21. `rollout.n=6` payload and concurrency risk

- Current launcher values as of 2026-05-13:
```text
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.rollout.n=4
actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40
actor_rollout_ref.rollout.multi_turn.web_osgym_window_supervision_block_size=3
actor_rollout_ref.rollout.multi_turn.web_osgym_window_history_n=5
actor_rollout_ref.rollout.multi_turn.web_osgym_window_max_images_per_sample=6
async_training.require_batches=1
async_training.trigger_parameter_sync_step=4
async_training.staleness_threshold=1.0
```

- Important config constraint:
  - `max_concurrent_samples_per_gpu` is treated as a trajectory budget and must be divisible by `rollout.n`.
  - Therefore `rollout.n=6` with the current `max_concurrent_samples_per_gpu=16` is invalid and will raise before running.
  - Valid nearby values are `12`, `18`, or `24`.

- Payload estimate using latest measured tensor density:
```text
latest measured density:
  1805 images -> 86301.62 MiB
  ~= 47.8 MiB per processed image

block_size=3 worst per row:
  prompt images up to 6 + two later step images ~= 8 images

max_assistant_turns=40:
  rows per trajectory = ceil(40 / 3) = 14
```

- Current `n=4`, `ppo_mini_batch_size=4` strict worst-case:
```text
rows = 4 queue groups * 4 trajectories/group * 14 rows/trajectory = 224 rows
images = 224 rows * 8 images/row = 1792 images
tensor payload ~= 1792 * 47.8 MiB = 85.7 GiB
```

- Candidate `n=6`, `ppo_mini_batch_size=4` strict worst-case:
```text
rows = 4 queue groups * 6 trajectories/group * 14 rows/trajectory = 336 rows
images = 336 rows * 8 images/row = 2688 images
tensor payload ~= 2688 * 47.8 MiB = 128.5 GiB
```

- Average-case estimate from the previous failed run:
```text
previous failed run:
  ppo_mini_batch_size=16
  rollout.n=4
  rows=258
  images=1805
  tensor payload=86.3 GiB

estimated ppo=4, n=4:
  rows ~= 64.5
  images ~= 451
  tensor payload ~= 21.6 GiB

estimated ppo=4, n=6:
  rows ~= 96.8
  images ~= 677
  tensor payload ~= 32.4 GiB
```

- Safety conclusion:
  - `n=6` may be acceptable on average if trajectories remain similar to the failed run's average length.
  - It is not safe under strict worst-case; the estimated payload is larger than the already-failed 86 GiB actor-update payload.
  - Do not run `n=6` without an actor-update payload guard.

- Rollout concurrency implication:
```text
if max_concurrent_samples_per_gpu=12:
  per-GPU group cap = 12 / 6 = 2
  total group cap with 4 rollout replicas = 8
  max trajectories = 48
  -> rollout concurrency likely lower

if max_concurrent_samples_per_gpu=18:
  per-GPU group cap = 18 / 6 = 3
  total group cap with 4 rollout replicas = 12
  max trajectories = 72
  -> group concurrency lower, trajectory concurrency slightly higher than current 64

if max_concurrent_samples_per_gpu=24:
  per-GPU group cap = 24 / 6 = 4
  total group cap with 4 rollout replicas = 16
  max trajectories = 96
  -> group concurrency preserved, trajectory concurrency 50% higher and resource risk rises
```

### 22. Meaning of `web_osgym_window_supervision_block_size=1`

- `block_size=1` does not mean "train only the final assistant turn."
- It means:
```text
for each assistant generation span:
  create one training row
  use the exact `prompt_ids` recorded for that generation window
  use only that assistant generation as the supervised/RL response target
  use only that generation window's prompt image indices
```

- Code behavior:
  - `groups = [(idx, idx + 1) for idx in range(len(assistant_spans))]`
  - `warmup_count = 0`
  - `prompt_ids = generation_windows[idx]["prompt_ids"]`
  - `response_ids = output.response_ids[assistant_span_start:assistant_span_end]`
  - `response_mask` supervises that generation span only
  - image indices come from `generation_window["prompt_image_indices"]`

- Therefore `block_size=1` is the most faithful backprop path for windowed rollout:
  - each trained row matches the conditional prompt context that was used during rollout,
  - no later assistant turn is trained under an earlier prompt window,
  - no zero-loss warmup prefix is needed.

- Tradeoff:
  - it creates one row per assistant generation,
  - so long trajectories produce many rows and can increase total processed-image payload.

- `block_size>1` is a memory/row-count compromise:
  - multiple assistant generations are packed into one row,
  - some earlier response tokens may be masked as zero-loss warmup,
  - but later supervised generations can be trained under a prompt context that is not exactly the one used when that later generation was produced.

### 23. Incomplete tool calls ending after `"x": <number>,`

- Current run inspected:
```text
logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260513_050040
```

- Initial scan found 18 incomplete tool-call outputs; while the run continued, one more appeared under `step_22/web_bmi_03`.
- All incomplete outputs end immediately after an `x` coordinate followed by a comma, for example:
```text
<parameter=actions>
[{"action_type": "MOVE_TO", "x": 463,
```

- Artifact symptom:
  - `result.termination_reason` is `system_stop`
  - `parse_error` is `null`
  - some later-turn events show `tool_call_count=1`, but that can be stale because pre-parse termination writes the event without clearing `agent_data.tool_calls`.

- Current-source parser check:
  - Re-feeding the same rendered text to `Qwen3XMLToolParser.extract_tool_calls()` produces `actions_json_malformed`.
  - `tokenizer.decode()` default keeps `<tool_call>` markers, so this is not explained by `skip_special_tokens`.

- Constraint-decoding implication:
  - If `structural_tag` and `ignore_eos=True` are active and honored, the decoder should not be able to finish a JSON object after a trailing comma.
  - Completed tool-call outputs in this run often include `<|endoftext|><|im_start|>` after `</tool_call>`, which is suspicious for the expected structured-output termination behavior.
  - The current logs do not include per-request `sampling_params`, `finish_reason`, `prompt_len`, or `max_new_tokens` because `VERL_ASYNC_SKD_TRACE` was not enabled in this RL launcher.

- Working hypothesis:
  - This is not proven to be an xgrammar schema limit.
  - The more likely failure is that structured output was not applied to those generation requests, or SGLang ended the request by `length`/stop while the local trajectory artifact discarded the exact finish reason.

- Required next diagnostic:
  - log per generation whether `structural_tag` is present,
  - log `ignore_eos`,
  - log SGLang `finish_reason`,
  - log `prompt_len`, `max_new_tokens`, and output token count into the WebOSGym trajectory event or an equivalent sidecar.

### 24. Minimal fix for incomplete structured tool call handling

- Root-cause facts confirmed locally with the real Qwen 3.5 tokenizer, bundled tool schema, and xgrammar matcher:
  - The prefix ending at `"x": 463,` is a valid partial prefix, but it is not a terminated structured output.
  - A completed `</tool_call>` is still not terminal for the current structural tag; the grammar terminates only after `<|im_end|>`.
  - Therefore relying only on grammar termination is a bad protocol boundary for this tool-call rollout path.

- Minimal code change retained:
  - When `rollout.name=sglang`, `multi_turn.format=qwen3_coder`, and `enable_qwen3_coder_structured_output=True`, the SGLang sampling params now include:
```text
stop=["</tool_call>"]
no_stop_trim=True
```

- Reason:
  - `</tool_call>` is the protocol boundary the parser needs.
  - `no_stop_trim=True` keeps the closing tag in the decoded output so the existing parser path can parse it.

- Safety fix retained:
  - `_handle_generating_state()` now runs the parser before response-budget / max-turn `system_stop`.
  - If the model output contains Qwen tool-call markers but the parser returns no call and no parse error, the agent loop creates a `tool_call_incomplete` parse error instead of silently recording `system_stop`.

- Over-extended diagnostic change removed:
  - No extra SGLang metadata is propagated through `extra_fields`.
  - No new trajectory event field is added.
  - This keeps debugging surface small while preventing the silent bad-data path.

### 25. Root cause of apparent `"x": <number>,` truncation

- The apparent truncation point in `trajectory.jsonl` is misleading.
- `WebOsGymToolAgentLoop._decode_response_text()` decodes with `skip_special_tokens=False` and then calls `.strip()`.
- Therefore trailing whitespace generated by the model is removed before `model_output_text` is written to the trajectory event.

- Training dump rows preserve the raw `output` string. In the dumped rows that end at `"x": <number>,` after `rstrip()`, the raw output actually contains about 54k characters, mostly trailing spaces:

```text
1.jsonl row 94: len_raw=54724, len_rstrip=946, trailing_ws=53778
1.jsonl row 95: len_raw=54609, len_rstrip=777, trailing_ws=53832
1.jsonl row 96: len_raw=54306, len_rstrip=400, trailing_ws=53906
2.jsonl row 80: len_raw=54811, len_rstrip=1074, trailing_ws=53737
2.jsonl row 81: len_raw=54456, len_rstrip=616, trailing_ws=53840
2.jsonl row 82: len_raw=54304, len_rstrip=392, trailing_ws=53912
```

- Padding was checked explicitly:
  - rollout dump decodes `batch.batch["responses"]` with `skip_special_tokens=False`,
  - this tokenizer's `pad_token_id` is `<|endoftext|>`,
  - therefore response padding appears as repeated `<|endoftext|>`, not as spaces.

- The same x-comma rows have no `<|endoftext|>` suffix:

```text
1.jsonl row 94: endoftext_count=0, space_suffix_len=53778
1.jsonl row 95: endoftext_count=0, space_suffix_len=53832
1.jsonl row 96: endoftext_count=0, space_suffix_len=53906
2.jsonl row 80: endoftext_count=0, space_suffix_len=53737
2.jsonl row 81: endoftext_count=0, space_suffix_len=53840
2.jsonl row 82: endoftext_count=0, space_suffix_len=53912
```

- Therefore these suffixes are not ordinary right-padding in the dump.
- They are decoded whitespace tokens that occupied the response tensor instead of pad tokens.

- This means the model did not stop immediately after the comma.
- It entered a whitespace loop after a valid JSON prefix:

```text
{"action_type": "MOVE_TO", "x": 463, [many spaces...]
```

- Why this prefix is allowed:
  - after a JSON comma, whitespace is valid before the next property name,
  - for coordinate actions the next property is normally `"y"`,
  - so the prefix ending after `"x": <number>,` plus whitespace is still a valid partial structured output.

- SGLang confirms why this can terminate:
  - `schedule_batch.py` checks `len(output_ids) >= sampling_params.max_new_tokens` before checking `grammar.is_terminated()`.
  - Therefore a request can finish by length while still sitting on a grammar-valid partial prefix.

- Actual root cause:
  - not an immediate hard cut after `x`,
  - not directly an OS/tool-server issue,
  - but a whitespace loop inside structured JSON after a comma, followed by max-token length termination.

- Consequence:
  - `</tool_call>` stop helps only for already completed tool calls.
  - It does not prevent this whitespace loop because the output never reaches `</tool_call>`.
- The required safety guard is parser-before-termination so this length-finished partial tool call cannot be recorded as a clean `system_stop`.

### 26. Fully-async WebOSGym queue inflation root cause and structural fix

- The fully-async `MessageQueue` does **not** hold pre-rollout prompts.
- It holds rollout results after `generate_sequences_single()` returns.

- Before this change, the WebOSGym path eagerly did:

```text
raw trajectory
-> window row expansion
-> row-level multimodal processor() call
-> DataProto assembly
-> queue
```

- This means the queue/object-store payload already contained:
  - assistant-turn-level window fanout,
  - repeated recent-history image materialization across rows,
  - `multi_modal_inputs` tensors for every row.

- The observed large training payloads are therefore explained by repeated row materialization, not by a single trajectory exceeding the turn cap.

- Structural fix implemented:

```text
raw trajectory
-> compact deferred payload
-> queue
-> trainer-side window row expansion
-> trainer-side multimodal processor() call
-> trainer batch assembly
```

- Concretely:
  - `AgentLoopWorker.generate_sequences_compact()` now returns raw `AgentLoopOutput` objects plus the original non-tensor batch.
  - Fully-async WebOSGym queueing uses this compact path.
  - Trainer batch assembly materializes deferred raw outputs with the same windowing and postprocess logic immediately before concat/update.

- Important implementation note:
  - The deferred trainer-side materializer must reuse the full `AgentLoopWorker` postprocess stack, not only a partial mixin.
  - A mixin-only helper was insufficient because windowing/postprocess depend on methods such as `_maybe_window_web_osgym_outputs`, `_compute_multi_modal_inputs`, `_compute_position_ids`, and `_postprocess`.

- Realistic tests added and passing:
  - eager worker path vs deferred queue path produce the same final training batch,
  - deferred payload performs no processor calls before queueing,
  - deferred queue payload is smaller than eager payload on a large synthetic multimodal processor,
  - adjacent agent-loop and fully-async assembly tests remain green.

### 27. Current primary throughput blocker after `window off`: malformed tool calls under structured decoding

- Current run:
  - `trainer.rollout_data_dir=/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260513_155341`
  - `actor_rollout_ref.rollout.multi_turn.web_osgym_window_enable=False`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.80`
  - `actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16`

- After lowering `gpu_memory_utilization` to `0.80`, the previous image-preprocess OOM around long full-history trajectories no longer appeared immediately.
- However, the run still looked "stuck" because sample-group completion slowed down sharply even though workers stayed busy and trajectory files kept being written.

- Evidence from the live run:
  - trainer stayed at `Requesting 4 samples from queue`
  - rollouter monitor repeatedly showed `count/total_generated_samples=2`, `monitor/active_tasks_size=14`, `mq_queue_size=0`
  - rollout dumps continued to grow, proving this was not a dead process

- Sample-group completion state at inspection time:
  - fully completed groups: `3`
  - partial groups: `11`
  - therefore trainer starvation was caused by slow completion of `rollout.n=4` groups, not by queue congestion

- The dominant new blocker is malformed structured tool output:
  - completed summaries at inspection time: `41`
  - summaries with `parse_error_count > 0`: `18`
  - termination reasons:
    - `tool_response_budget_exhausted`: `20`
    - `model_done`: `14`
    - `model_fail`: `5`
    - `system_stop`: `2`

- Direct classification of assistant turns with parse errors in this run showed `30` malformed tool-call rows:
  - `11`: `MOVE_TO` with `"x"` present, `"y"` missing, no closing tags
  - `8`: `DOUBLE_CLICK` with `"x"` present, `"y"` missing, no closing tags
  - `6`: `CLICK` with `"x"` present, `"y"` missing, no closing tags
  - `4`: `TYPING` with chat special-token leakage inside `text`
  - `1`: mixed coordinate action plus `TYPING` special-token leakage

- The dominant malformed pattern is therefore:

```text
</think>
<tool_call>
<function=computer>
<parameter=actions>
[{"action_type": "...", "x": 203,
```

- In these rows:
  - `<tool_call>` opened,
  - `<parameter=actions>` opened,
  - `action_type` was emitted,
  - `"x"` and its numeric value were emitted,
  - but `"y"` never appeared,
  - `</parameter>`, `</function>`, and `</tool_call>` were also missing.

- The second malformed family is:

```text
[{"action_type": "TYPING", "text": "<|im_end|><|im_start|>assistant..."
```

- This is not a parser-only problem.
- The parser is doing its job:
  - it converts the malformed `actions` payload to a string when list/object parsing fails,
  - then raises `actions_json_malformed` because `computer.actions` is not a list.

- The real root cause is earlier:
  - structured decoding is still allowing partial `actions` JSON that stalls after `"x": <number>,`
  - and in some `TYPING` cases, special chat tokens leak into parameter text
  - therefore the model never reaches a valid closed tool call

- Why this burns budget:
  - the malformed assistant output is appended with `response_mask=1`
  - then `_handle_tool_parse_error()` injects parse-error feedback with `response_mask=0`
  - generation retries continue
  - many trajectories then terminate as `tool_response_budget_exhausted`

- Operational consequence:
  - parser-side safety prevents silent bad actions from reaching the environment
  - but it does not prevent budget waste
  - therefore the current run's apparent "stuck" state is primarily a constraint-decoding quality problem, not a queue deadlock and not a trainer-side blockage

- Updated diagnosis:
  - parser recovery remains necessary as a safety layer
  - but the main target for the next root-cause fix is now the structured-decoding / grammar path itself
  - especially:
    - forcing completion of coordinate actions after `"x"`
    - preventing chat special-token leakage inside `TYPING.text`
