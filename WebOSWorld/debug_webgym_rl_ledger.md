# WebGym RL Debug Ledger

Last updated: 2026-05-10

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
  - runtime normalizes `cmd`, `command`, `meta` to `Meta`
  - runtime does **not** normalize `win`
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

2. **Need a decision on `win` handling.**
   - Possible fixes:
     - alias `win -> Meta`
     - alias `win -> ControlOrMeta`
     - reject `win` earlier with clearer feedback
   - Current evidence only proves that raw `win` is invalid in the active Playwright path.

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
