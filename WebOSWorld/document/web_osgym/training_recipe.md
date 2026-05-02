# WebGym Training Recipe

## 0. Purpose

이 문서는 WebGym / OSWorld 계열 multi-turn visual agent trajectory를 학습 batch로 변환하는 방식을 정리한다.

현재 WebGym Async SKD 경로에서는 generation/inference 단계는 정상적으로 full trajectory를 생성할 수 있지만, actor update 단계에서 전체 trajectory를 그대로 backprop에 넣으면 이미지가 과도하게 누적된다. 실제 exact replay 조사에서는 한 training sample 안에 10-16장의 screenshot이 들어가고, Qwen3.5 / Qwen3-VL 계열 vision forward가 `get_image_features() -> self.visual(...)` 구간에서 매우 오래 걸리는 현상이 확인되었다.

따라서 학습 recipe의 목표는 다음과 같다.

1. generation은 full trajectory로 유지한다.
2. backprop용 training sample은 bounded window로 재구성한다.
3. 각 mini-step의 action은 정확히 한 번만 loss를 받게 한다.
4. teacher top-k는 generation 당시 full-context teacher signal을 재사용한다.
5. actor update에 들어가는 이미지 수를 구조적으로 제한한다.
6. tool-call format은 Qwen3.5 native `qwen3_coder` XML format을 유지한다.

이 recipe는 임시 truncation이 아니라, WebGym policy를 OSWorld Qwen3-VL benchmark harness와 가까운 형태로 재정의한다.

```text
policy(task anchor, old action summary, bounded recent multimodal history, current observation)
  -> current reasoning/action
```

## 1. Current Failure Mode

기존 full-trajectory training view는 late step으로 갈수록 모든 이전 observation image를 계속 들고 간다.

예를 들어 16 mini-step trajectory의 마지막 action을 학습할 때, training input은 다음과 비슷해진다.

```text
instruction
obs_1 image
assistant_1 reasoning/action
obs_2 image
assistant_2 reasoning/action
...
obs_16 image
assistant_16 reasoning/action
```

이 구조에서는 actor update forward가 모든 image를 다시 vision encode 해야 한다.

실제 capture에서 관측된 representative input:

```text
input_ids:      ~18k tokens
image_count:    16
image_grid_thw: [16, 3]
pixel_values:   [61440, 1536]
```

이 정도 입력은 Qwen3.5 / Qwen3-VL vision path에 매우 부담이 크다. 문제는 rollout이나 tool server가 아니라, actor update의 model forward 안에서 image feature extraction이 반복되는 데 있다.

## 2. Core Principle

Full trajectory는 behavior source다.

Training sample은 current mini-step prediction problem이다.

즉 full rollout은 그대로 수행하되, backprop에는 다음 형태의 sample만 넣는다.

```text
bounded_context_i -> assistant_i reasoning/action
```

여기서 bounded context는 permanent anchor, 오래된 action summary, 최근 multimodal mini-step, 그리고 현재 observation으로 구성된다.

중요한 원칙:

- window는 context를 만들기 위한 것이다.
- gradient는 current mini-step의 assistant reasoning/action token에만 준다.
- 이전 mini-step의 assistant text/action은 context일 뿐 loss target이 아니다.
- 각 original mini-step은 정확히 한 번만 loss target이 된다.
- 오래된 history는 전체 대화나 reasoning을 보존하지 않고, 실행된 action만 짧은 text progress log로 남긴다.
- 최근 history만 screenshot과 assistant response를 함께 가진 multimodal context다.

## 3. Mini-Step Definition

Mini-step은 모델이 한 번 observation을 보고 reasoning/action을 낸 단위다.

```text
mini_step_i =
  observation_i
  assistant_i reasoning/action
  teacher_topk_i for assistant_i tokens
  optional tool_result_i / next_observation_{i+1}
```

학습 sample for target step `i`:

```text
context_i =
  task_anchor
  old_action_summary
  recent previous mini-steps
  observation_i

target_i =
  assistant_i reasoning/action

loss_i =
  teacher top-k distillation on target_i tokens only
```

`observation_i`는 assistant가 `action_i`를 내기 직전에 본 화면/텍스트 상태다. 따라서 policy 관점에서는 "현재 observation/history를 보고 다음 reasoning/action을 낸다"가 된다.

중요한 점은 `observation_i`가 항상 screenshot일 필요는 없다는 것이다. 정상 경로에서는 image-bearing observation이 기본이지만, action failure / screenshot failure 같은 복구 경로에서는 text-only failure observation이 합법이다. 따라서 mini-step parser는 `image 유무`를 step 경계의 기준으로 삼지 말고, **student trajectory 안에서 commit된 observation bundle과 assistant target span의 순서**를 기준으로 step을 복원해야 한다.

## 4. Example

Task:

```text
Find the price of the cheapest red running shoes and add it to cart.
```

Full generation trajectory:

```text
task_anchor:
  system prompt
  user instruction

mini_step_1:
  observation_1: homepage screenshot
  assistant_1: search for "red running shoes"
  teacher_topk_1: rows for assistant_1 tokens

mini_step_2:
  observation_2: search results screenshot
  assistant_2: sort by price low to high
  teacher_topk_2: rows for assistant_2 tokens

mini_step_3:
  observation_3: sorted product grid screenshot
  assistant_3: open first product
  teacher_topk_3: rows for assistant_3 tokens

mini_step_4:
  observation_4: product page screenshot
  assistant_4: select size
  teacher_topk_4: rows for assistant_4 tokens

mini_step_5:
  observation_5: size selected screenshot
  assistant_5: click add to cart
  teacher_topk_5: rows for assistant_5 tokens

mini_step_6:
  observation_6: cart confirmation screenshot
  assistant_6: final answer with price
  teacher_topk_6: rows for assistant_6 tokens
```

For compact illustration, with `history_n=3`, training samples become:

```text
sample 1:
  context = task_anchor + observation_1
  target  = assistant_1
  loss    = assistant_1 only

sample 2:
  context = task_anchor + mini_step_1 + observation_2
  target  = assistant_2
  loss    = assistant_2 only

sample 3:
  context = task_anchor + mini_step_1 + mini_step_2 + observation_3
  target  = assistant_3
  loss    = assistant_3 only

sample 4:
  context = task_anchor + mini_step_1 + mini_step_2 + mini_step_3 + observation_4
  target  = assistant_4
  loss    = assistant_4 only

sample 5:
  old_action_summary = Step 1: search for "red running shoes"
  context = task_anchor + old_action_summary + mini_step_2 + mini_step_3 + mini_step_4 + observation_5
  target  = assistant_5
  loss    = assistant_5 only

sample 6:
  old_action_summary =
    Step 1: search for "red running shoes"
    Step 2: sort by price low to high
  context = task_anchor + old_action_summary + mini_step_3 + mini_step_4 + mini_step_5 + observation_6
  target  = assistant_6
  loss    = assistant_6 only
```

In token/image form, sample 5 looks like:

```text
input sequence:
  [system]
  [instruction]

  Previous actions:
    Step 1: search for "red running shoes"

  <image> obs_2
  assistant_2 reasoning/action

  <image> obs_3
  assistant_3 reasoning/action

  <image> obs_4
  assistant_4 reasoning/action

  <image> obs_5
  assistant_5 reasoning/action

loss_mask:
  0 for system/instruction
  0 for previous action summary
  0 for obs_2 and assistant_2
  0 for obs_3 and assistant_3
  0 for obs_4 and assistant_4
  0 for obs_5 prompt/image portion
  1 only for assistant_5 reasoning/action tokens

teacher_topk:
  rows only for assistant_5 tokens
```

The recent assistant turns remain useful context. They do not receive gradient in this sample. Older turns outside the recent window are reduced to action-only text summary; their images and full reasoning are dropped.

## 5. Window Policy

Recommended default:

```yaml
training_window:
  enabled: true
  sample_policy: all_mini_steps
  history_n: 5
  max_images_per_sample: 6
  history_n_source: runtime_arg
  anchor_mode: task_prefix_only
  old_history_mode: action_summary
  old_action_invalid_policy: omit
  prompt_source: osworld_qwen3vl_harness
  coordinate_mode: deferred
  coordinate_target_mode: osworld_relative_1000
  keep_initial_observation_image: false
  target_loss: current_step_only
  include_terminal_actions: true
  teacher_target_mode: reuse_original_full_context_topk
  step_weighting: uniform_by_mini_step
  drop_old_images: true
  drop_old_reasoning: true
  validate_alignment: strict
```

Suggested semantics:

```text
For target mini-step i:

recent_start = max(1, i - history_n)

task_anchor = system prompt + tool/action schema + original user instruction
old_action_summary = action-only summary for mini_steps[1 .. recent_start-1]
recent_context = mini_steps[recent_start .. i-1] with screenshots and assistant responses

context = task_anchor + old_action_summary + recent_context + observation_i

target = assistant_i
loss = assistant_i only
```

`sample_policy: all_mini_steps` means every original assistant action becomes exactly one windowed training sample. A trajectory with seven assistant actions produces seven training samples. `DONE` and `FAIL` are actions too, so terminal action steps remain trainable targets.

`history_n` bounds the number of previous mini-steps retained in recent context. The recommended default is `history_n=5`, so a late training sample carries at most five previous mini-steps plus the current observation bundle. Some of those steps may be text-only failure observations with no image metadata. This should be exposed as a runtime argument rather than hard-coded, because ablations may need smaller or larger windows. The key invariant is that the window is chosen by student-topology step order, while visual inputs remain bounded by the subset of selected steps that actually carry screenshots.

`step_weighting: uniform_by_mini_step` means longer trajectories naturally contribute more training examples because they contain more policy decisions. Task-level reweighting is a later experiment, not part of the first implementation.

## 6. History Policy

History has three layers.

```text
permanent_anchor:
  system prompt
  Qwen3.5 qwen3_coder tool/action schema
  original user instruction

old_history_summary:
  action-only text for turns outside the recent window
  no screenshots
  no full assistant reasoning

recent_multimodal_history:
  recent history_n mini-steps
  each previous step keeps its observation screenshot and assistant response
  current step always keeps the current observation screenshot
```

The permanent anchor preserves task identity, not visual state. The initial observation image should not be kept forever by default. Once it falls outside the recent window, it is dropped like any other old screenshot.

Old history is a progress log, not a memory dump. It should be short and canonical, for example:

```text
Previous actions:
Step 1: CLICK(x=412, y=280)
Step 2: TYPING(text="red running shoes")
Step 3: PRESS(key="ENTER")
```

This is intentionally stronger than "drop everything outside the window" and intentionally weaker than "keep the full old conversation as text." It preserves task progress while preventing both image explosion and unbounded text trajectory growth.

Only valid parsed actions should enter `Previous actions`. If an old assistant turn has no parseable tool call, omit it from the old action summary rather than writing `INVALID_ACTION`. The assistant span itself may still be used as a target in its own training sample; this rule only controls the compact history text shown to later steps.

## 7. Student Prompt Policy

The student prompt should follow the OSWorld Qwen3-VL harness as closely as possible in content and message ordering, while keeping Qwen3.5 / verl-native tool syntax.

Recommended default:

```yaml
student_prompt_policy:
  prompt_semantics_source: osworld_qwen3vl_harness
  tool_syntax_source: qwen3_5_qwen3_coder
  observation_type: screenshot_only
  privileged_a11y_for_student: false
  coordinate_mode: deferred
  coordinate_target_mode: osworld_relative_1000
  add_new_current_observation_label: false
```

Carry over the benchmark harness prompt content:

```text
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {task_instruction}

Previous actions:
{old_action_summary_or_None}
```

The `Previous actions` block is real benchmark-harness behavior. It should contain only old, window-external Computer 13 actions in our named-tool form:

```text
Previous actions:
Step 1: CLICK(x=412, y=280)
Step 2: TYPING(text="red running shoes")
Step 3: PRESS(key="ENTER")
```

Do not introduce a new literal label such as `Current observation:` unless the evaluation harness also uses it. In the OSWorld Qwen3-VL harness, the current screenshot is represented by message order: it is the final image-only user message before the model predicts the next action.

For a target mini-step `i`, the windowed student message order should be:

```text
system:
  OSWorld-style GUI rules
  Qwen3.5 qwen3_coder tool schema and format rules

user:
  <image observation_recent_start>
  instruction_prompt with Instruction and Previous actions

assistant:
  assistant_recent_start reasoning/action
  loss = 0

user:
  <image observation_recent_start+1>

assistant:
  assistant_recent_start+1 reasoning/action
  loss = 0

...

user:
  <image current_observation_i>

assistant:
  assistant_i reasoning/action
  loss = 1
```

When there is no recent history, the harness attaches the instruction prompt to the current screenshot. Mirror that behavior for early mini-steps.

Coordinate handling is deferred for the first implementation. The target benchmark-aligned mode is OSWorld-style `1000x1000` relative coordinates, and the tool server is expected to provide a11y/tree coordinates in that same frame. However, existing local rollout data and screenshots are still in the original image coordinate frame. Therefore the first windowed-loss implementation should avoid changing coordinate semantics and should not rewrite coordinate prompts yet. Once the tool server and stored trajectory metadata are consistently `1000x1000`, coordinate preprocessing can be added as a separate step.

## 8. Action and Tool-Call Policy

The benchmark environment should be OSWorld/WebGym-like, but the tool-call format should remain native to Qwen3.5.

Recommended default:

```yaml
action_policy:
  action_space: computer_13_semantics
  tool_call_format: qwen3_coder_xml
  parser: qwen3_coder
  use_computer_use_wrapper: false
```

This means the model should emit named function calls:

```text
<tool_call>
<function=CLICK>
<parameter=x>
412
</parameter>
<parameter=y>
280
</parameter>
</function>
</tool_call>
```

not the OSWorld Qwen3-VL wrapper style:

```text
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [412, 280]}}
</tool_call>
```

The difference is serialization, not environment semantics. `computer_use` is a single wrapper function used by the OSWorld Qwen3-VL harness; Qwen3.5's native tool template and the verl `qwen3_coder` parser expect `<function=...>` and `<parameter=...>` XML blocks. Therefore the training and evaluation harness should keep named tools such as `CLICK`, `TYPING`, `PRESS`, `HOTKEY`, `SCROLL`, `WAIT`, `DONE`, and `FAIL`.

In one sentence:

```text
Match OSWorld on environment, observation history, and action semantics; match Qwen3.5 on tool-call syntax and parser.
```

## 9. Teacher Target Policy

The first implementation should reuse the teacher top-k rows collected during full-context generation.

This means the objective is not strict same-prefix KL:

```text
not:
  KL(teacher(. | same window context) || student(. | same window context))
```

Instead, it is behavior distillation:

```text
student(. | windowed context_i) imitates teacher_full_context action_i distribution
```

This is intentional. The teacher observed the full rollout context during generation; training compresses that long-context behavior into a bounded-context policy.

Teacher top-k should not be recomputed for the first implementation. Recomputing teacher probabilities on each window would add a second inference role with ambiguous boundaries and substantial extra cost.

## 10. Reward and Advantage Policy

This recipe assumes the WebGym SKD path is distillation-centered.

For the current path:

```text
distillation.distillation_loss.use_policy_gradient = False
distillation.distillation_loss.use_task_rewards = False
```

Therefore reward/advantage assignment is not part of the primary training objective. Full-trajectory reward may still be used for logging, filtering, curriculum, or later experiments, but it should not affect the first windowed distillation implementation.

If policy-gradient training is introduced later, reward assignment must be redesigned separately.

## 11. Required Metadata

Window reconstruction should reuse the original token-id sequence as much as possible. The primary boundaries are already present in the rollout tensors:

```text
responses:      full response-side token ids
response_mask:  1 for assistant-generated tokens, 0 for tool/observation tokens
teacher_topk:   response-aligned teacher rows, with dummy rows on non-loss spans
```

The intended token policy is:

```yaml
token_reuse:
  target: original_token_slice
  recent_history: original_token_slices
  observations: original_token_slices
  new_old_action_summary: tokenize_new_text
  tensor_packaging:
    recompute_padding: true
    recompute_attention_mask: true
    recompute_position_ids: true
    rebuild_image_inputs: true
```

This means the content tokens should be preserved from rollout whenever possible. The new text we introduce is the compact `Previous actions` summary; that part must be tokenized because it did not exist in the original full trajectory.

The target assistant spans do not need a separate heavy metadata table. They can be recovered from contiguous `response_mask == 1` runs:

```python
assistant_spans = contiguous_runs(response_mask == 1)

for step_idx, (start, end) in enumerate(assistant_spans, start=1):
    target_token_ids = responses[start:end]
    target_teacher_ids = teacher_ids[start:end]
    target_teacher_logprobs = teacher_logprobs[start:end]
```

This is better than decoding text and re-tokenizing it. Decoding may still be used to build readable previous-action summaries, but the loss target must remain the original token-id slice:

```python
assistant_text = tokenizer.decode(target_token_ids, skip_special_tokens=False)
old_action_summary = parse_actions_for_history(assistant_text)

# Do not use tokenizer(assistant_text) as the supervised target.
# The supervised target is target_token_ids from the original rollout.
```

Teacher rows should follow the same response-relative span as the target tokens. In the current SKD path, real teacher top-k rows are appended for assistant tokens and dummy rows are appended for tool/observation tokens, so `teacher_row_start/end` is redundant with `target_start/end`.

Mini-step parsing itself should be student-topology-driven:

```text
step 1 observation  = prompt-side initial observation
step i>=2 observation = response-side committed zero-run bundle between assistant_{i-1} and assistant_i
step target          = the original contiguous response_mask == 1 run for assistant_i
```

This means:

- assistant target spans come from contiguous `response_mask == 1` runs
- response-side committed observation bundles come from contiguous `response_mask == 0` runs between assistant spans
- image metadata is an optional attribute of a step:
  - visual step => one attached image
  - text-only failure step => no attached image

Do not require `assistant span count == image span count`. A valid trajectory may contain text-only failure observations, so image metadata is intentionally sparse.

If a reconstructed training row contains no visual steps in its selected window, do not materialize `multi_modal_data["images"] = []`. Treat that row as text-only and omit the image field entirely so downstream multimodal processors do not interpret it as a malformed visual batch.

The Qwen3.5 chat template opens generation with an assistant prefix such as:

```text
<|im_start|>assistant
<think>
```

That prefix belongs to the prompt side. The supervised target should follow the existing rollout/loss convention: whatever the model actually generated and `response_mask` marks as assistant output is the target. Do not infer a second target boundary from decoded text unless a validation dump proves the mask is wrong.

The only metadata that should be kept explicitly in the first implementation is lightweight visual-step image indexing, because image placeholder order and actual image object order are the most fragile part of reconstruction.

Minimum useful sparse visual metadata per trajectory:

```python
mini_step_image_spans = [
    {
        "step_idx": 1,
        "image_start": int,
        "image_end": int,
        "terminal": bool,
    },
]
```

This table is **not** the full mini-step table. It records only steps that actually carried images. A text-only failure observation step may have no corresponding `mini_step_image_spans` entry and is still a valid mini-step.

Optional debug metadata can include `target_start` and `target_end`, but those should be treated as cached derivations from `response_mask`, not as a separate source of truth.

```python
mini_step_debug = [
    {
        "step_idx": 1,
        "target_start": int,  # contiguous response_mask == 1 run start
        "target_end": int,    # contiguous response_mask == 1 run end
        "image_start": int,
        "image_end": int,
    },
]
```

The source of truth is:

```text
target tokens       = original responses[target_start:target_end]
target loss mask    = 1 only on that target slice in the reconstructed sample
teacher supervision = original teacher rows for that same target slice
observations        = prompt initial observation + response-side committed zero-run bundles
images              = sparse explicit metadata for visual steps only
```

EOS, stop, and tool-call closing tokens should be included if they are inside the original `response_mask == 1` run. This preserves the current training semantics exactly. If a future dump shows non-generated prompt/header tokens inside a target run, treat that as an alignment bug.

If a completed trajectory ends with a committed observation bundle that has no following assistant-generated tokens yet, that trailing observation is not a trainable mini-step and should be excluded from window rows.

## 12. Alignment Invariants

The implementation should hard-fail on alignment errors during development.

Target token invariant:

```text
window target tokens == original assistant_i tokens
```

Teacher row invariant:

```text
teacher_rows[target_start:target_end] aligns 1:1 with responses[target_start:target_end]
target_start/end are derived from a contiguous response_mask == 1 run
```

Loss mask invariant:

```text
loss_mask.sum() == number of target assistant tokens
loss_mask is nonzero only on current assistant_i
```

Generation-boundary invariant:

```text
assistant role/header tokens are prompt/context tokens, not supervised target tokens
target span follows the original response_mask == 1 run
```

Image invariant:

```text
for every selected visual step, image placeholder order matches the selected image object order
num_image_placeholders == image_grid_thw.shape[0]
sum(image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]) == pixel_values.shape[0]
```

Uniqueness invariant:

```text
each original mini_step_i contributes loss exactly once
```

This is what keeps the dataset balanced. Earlier context actions can appear many times as context, but their loss is applied only in their own target sample.

## 13. Implementation Boundary

The transformation should happen after full rollout generation and before actor update.

Actual flow:

```text
completed agent-loop output
-> worker _postprocess_completed_skd_output()
-> windowed AgentLoopOutput rows
-> _agent_loop_postprocess() for each window row
-> AsyncSkdSample.completed(batch=DataProto(len=N))
-> manager finalize / promotion queue
-> trainer _assemble_async_skd_training_batch()
-> batch.union(gen_batch_output)
-> reward / advantage bookkeeping
-> actor update
```

This boundary is preferable because it has access to:

- full generated tokens
- response/loss masks
- multimodal inputs
- teacher top-k rows
- lightweight image span metadata

The actor engine should not be responsible for dropping old images or slicing windows. By the time data reaches `prepare_model_inputs()`, text/image alignment is already delicate and engine-level truncation would be too late. `partial` async-SKD samples must bypass this path because they are resumable generation state, not training data.

## 14. Async SKD Scheduler Contract

Windowing changes the number of training rows, but it must not change the unit of async generation scheduling.

The clean contract is:

```text
AsyncSkdSample
  = trajectory scheduling unit

DataProto row
  = actor-update training unit
```

Before windowing these happened to be identical:

```text
1 completed trajectory -> 1 AsyncSkdSample.completed -> DataProto(len=1)
```

After windowing they are intentionally different:

```text
1 completed trajectory -> 1 AsyncSkdSample.completed -> DataProto(len=N window rows)
```

This distinction is important for async SKD prefetch.

`partial` samples are resumable generation states. They are not training data yet and must not be windowed:

```text
AsyncSkdSample.partial
  partial_state = unfinished trajectory
  batch = None
  windowing = forbidden
```

Only completed trajectories may be converted into training windows:

```text
AsyncSkdSample.completed
  batch = windowed training DataProto
  len(batch) >= 1
  every row comes from the same completed trajectory
```

Prefetch / promotion should remain trajectory-based:

```text
reserved sample_id A
  -> partial(A) while generation is unfinished
  -> completed(A) if generation finishes before the current-step barrier
  -> promoted(A) for a future training batch
  -> carryover partial(A) if generation does not finish in time
```

Do not create one `AsyncSkdSample` per window row, such as `A#1`, `A#2`, `A#3`. That would mix scheduler accounting with training-row accounting, make promotion counts ambiguous, and complicate reserved input matching. A promoted trajectory may contribute multiple training rows, but it is still one promoted trajectory.

The completed-batch envelope should therefore validate trajectory-level metadata across rows instead of requiring a single row. Values such as `uid`, `rollout_birth_version`, `rollout_min_version`, `rollout_max_version`, `skd_committed_gen_chunks`, `skd_committed_env_units`, and `skd_committed_prefix_tokens` must be identical across all rows from the same trajectory. Window-specific fields such as `response_ids`, `response_mask`, `teacher_ids`, `teacher_logprobs`, `multi_modal_inputs`, `window_step_idx`, and `window_image_start/end` may differ per row.

This keeps the two accounting systems separate:

```text
prefetch / promotion metrics
  count trajectories

backprop / window metrics
  count training rows
```

## 15. Backprop Entry Contract

The actor update path starts only after `batch.union(gen_batch_output)`.

The required contract at this point is:

```text
input batch rows == output batch rows
each output window row can be matched to its original input row by uid or index
general meta_info contains only batch-invariant values
per-window/per-trajectory numbers live in metrics or row fields
```

This is why a completed trajectory that becomes `N` window rows must also expand the original input prompt to `N` rows before union:

```text
input:
  row 0 = trajectory A prompt

output:
  row 0 = trajectory A, window 1
  row 1 = trajectory A, window 2
  row 2 = trajectory A, window 3

trainer assembly:
  duplicate input row 0 three times by uid/index
  union expanded input with windowed output
```

Do not put window-specific dictionaries under a new top-level `DataProto.meta_info` key. `DataProto.concat()` requires ordinary `meta_info` values to be identical across concatenated batches. Window metrics should therefore be recorded through the existing `metrics` channel, and row-specific state should live in `batch` or `non_tensor_batch`.

If windowing drops old image blocks, the output-side padded `prompts` may differ from the original input prompt. In that case the windowed output prompt is canonical for actor update, so trainer assembly must align the input-side `prompts` to the output-side `prompts` before `DataProto.union()`.

Windowing also changes the actor-update row count. Before dispatching to data-parallel actor ranks, the assembled training batch must be divisible by actor DP size. If needed, append synthetic padding rows with a fresh padding `uid` and zero `response_mask` so they carry no distillation gradient. Keep copied teacher top-k rows intact on padding rows; the zero `response_mask` is the loss gate. Do not create artificial all-zero teacher distributions just for padding. Loss normalization should count only rows with `response_mask.sum() > 0`.

The actor-update path then becomes:

```text
trainer _update_actor(batch)
-> batch.to_tensordict()
-> left_right_2_no_padding()
-> WorkerDict.actor_rollout_update_actor()
-> engine.train_batch()
-> prepare_micro_batches()
-> forward_step()
-> prepare_model_inputs()
-> self.module(**model_inputs, use_cache=False)
-> prepare_model_outputs()
-> distillation loss
-> loss.backward()
```

`left_right_2_no_padding()` is a hard boundary. It converts padded `input_ids`, `position_ids`, and `response_mask` into no-padding/nested tensors and sets:

```text
loss_mask = response_mask
```

It also unpads `teacher_ids` and `teacher_logprobs` with the same token indices. Therefore window construction must preserve these invariants before actor update:

```text
attention_mask marks all real context + target tokens
response_mask is 1 only on the current mini-step assistant target
teacher_ids/logprobs have the same padded sequence length as input_ids before unpadding
multi_modal_inputs contains only the bounded image window for that row
```

`self.module(...)` is the expensive model forward stage. For Qwen-VL this includes text forward, vision input handling, image placeholder alignment, attention kernels, gradient checkpointing, and FSDP/VeOmni collectives. The windowing code must therefore finish all text/image pruning before this point; once execution reaches `self.module(...)`, the engine should only run the already-formed training row.

Even in supervised async-SKD mode, the trainer may still compute reward and advantage tensors for framework compatibility. With `distillation_loss.use_policy_gradient=False` and `distillation_loss.use_task_rewards=False`, those tensors are not the source of the distillation gradient. The gradient is applied through the distillation loss on the current window target tokens.

## 16. Logging Policy

Windowing should log compact metrics to console and WandB without dumping large decoded trajectories by default.

Recommended metrics:

```text
window/num_samples
window/avg_target_tokens
window/max_target_tokens
window/avg_images
window/max_images
window/avg_recent_steps
window/skipped_old_steps
```

Decoded trajectory dumps are intentionally not part of the first implementation. If needed later, add them as an explicit debug-only feature rather than as default config.

## 17. Expected Effect

Without windowing:

```text
late target step:
  images: 10-16
  visual forward cost: very high
```

With windowing:

```text
late target step:
  images: bounded by history_n + current observation
  old progress: retained as action-only text summary
  target action: unchanged
  teacher top-k: reused from full trajectory
  gradient: current assistant action only
```

This should directly reduce the actor update vision forward cost while preserving full-trajectory generation behavior.

## 18. Non-Goals

This recipe does not:

- change the inference/generation trajectory.
- recompute teacher probabilities under windowed contexts.
- train on all assistant tokens inside each window.
- solve reward assignment for policy-gradient RL.
- cache vision features or freeze the vision tower.

Those can be separate experiments. The first goal is to bound training-time image horizon while keeping the current SKD behavior signal.

## 19. Summary

The intended training recipe is:

```text
Generate a full WebGym trajectory.
Split it into mini-steps from original token ids and response_mask.
Treat step boundaries as student-topology-driven observation bundles plus assistant target spans, not as image-count boundaries.
For each mini-step, build one bounded-context training sample.
Keep task identity in the permanent anchor.
Keep old progress as action-only text summary, omitting invalid/no-tool old actions.
Follow OSWorld Qwen3-VL prompt semantics and message ordering.
Defer coordinate conversion until tool-server and stored trajectory coordinates are consistently OSWorld-style 1000x1000.
Keep only recent history_n steps as multimodal screenshot/action context.
Reuse original token-id slices for recent context and target whenever possible.
Use the current mini-step assistant reasoning/action as the only loss target, following the original response_mask == 1 span.
Reuse the original full-context teacher top-k rows for those target tokens.
Drop old images outside the window while allowing text-only failure observation steps to remain in-context without images.
Validate token, teacher, image, and loss-mask alignment strictly.
Log compact window metrics, with debug text dumps sharply limited.
```

In one sentence:

```text
Full trajectory rollout provides behavior; bounded mini-step windows define the trainable policy view.
```
