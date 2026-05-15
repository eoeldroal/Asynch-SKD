# KILL_IT

## Goal

WebGym Async SKD에서 초기에 발생하는 `no-tool <|im_end|>` 붕괴가

1. 실제로 초기 step부터 심한지,
2. teacher-side privileged guidance가 그 붕괴를 키우는지,
3. teacher replacement가 어떤 경로에서 plain ending을 만들어 내는지,
4. bad target을 먼저 차단한 뒤 teacher candidate quality를 어떻게 개선할지

를 단계적으로 검증한다.

---

## Current Status

- `verify_top_k=15`와 strict masking까지 포함한 기존 Async SKD 정리는 일단 `roughly KILL` 단계까지 갔다.
  - usable actor checkpoint를 merge했고
  - downstream fully async RL probe용 launcher wiring도 끝냈다.
- 하지만 그 뒤 다시 좁힌 `same-prompt + verify_top_k=1` 실험에서,
  - teacher privileged prompt mismatch 가설은 무너졌고
  - 여전히 `teacher replacement -> early <|im_end|>`가 발생했다.
- 현재 최신 결론은
  - prompt mismatch,
  - `prompt_logprobs` extractor-only bug,
  - mRoPE position construction,
  - CUDA graph,
  - Qwen3.5 fused QKV projection fast path
  가 아니라,
  - **`sglang 0.5.10`의 dense Qwen3.5 Triton GDN prefill/extend 경로**
  - 그중에서도 **`fused_gdn_gating` 출력 경로**
  가 exact sample의 EOS 왜곡을 직접 만든다는 것이다.
- 그리고 local patch를 넣은 뒤의 새 run에서는
  - **teacher replacement가 early plain `<|im_end|>`를 주입하는 현상은 사라졌다**
  - 대신 남은 실패는
    - parse error
    - invalid action
    - malformed tool payload
  로 바뀌었고,
  - 이 양상은 base Qwen3.5 fully async rollout의 원래 failure mode와 더 가깝다.
- 그 다음 단계에서 새로 드러난 건
  - backend EOS bug가 아니라
  - **strict assistant-turn masking이 빈 actor batch를 만들어 trainer를 죽이는 문제**였다.
- 현재 최신 결론은
  - `invalid_action`에서는
    - **termination은 유지**
    - **loss masking만 끄는 것**
  이 가장 좁고 효율적인 대응이라는 것이다.
- 이를 위해 `distillation.skd`에 아래 인자를 추가했다.
  - `mask_invalid_action`
  - `mask_tool_parse_error`
  - `mask_no_tool_call`
  - 현재 권장 실험값은:
    - `mask_invalid_action=false`
    - `mask_tool_parse_error=true`
    - `mask_no_tool_call=true`
- 따라서 지금 문서의 실제 읽는 순서는:
  1. earlier SKD masking / few-shot / `verify_top_k` 실험 기록
  2. usable checkpoint merge 및 RL probe 준비
  3. latest same-prompt top1 isolation
  4. exact local repro로 teacher verify divergence를 SGLang GDN fused gating까지 좁힌 결과
  5. patched run으로 EOS-cutoff bug가 실제 사라졌는지 검증하고, 남은 문제가 base-model-like tool-call quality인지 대조한 결과
  6. empty actor batch의 직접 원인이 invalid-action masking이었음을 확인하고, 마스킹 정책을 인자화한 결과
  7. merged Async SKD actor를 fully async RL에 물린 뒤, reward path / hard cap / actor LR / late-stage format reward hacking까지 확인한 결과
- 현재 fully async RL 기준 최신 결론은:
  - reward는 **실제로 학습 경로에 들어간다**
  - `max_assistant_turns=10`은 WebOSGym path에서 **attempted tool-call hard cap**으로 읽는 것이 맞다
  - actor LR 0 문제는 `trainer.total_training_steps`를 너무 늦게 넣어서 생긴 fully async initialization-order bug였다
  - launcher에 `actor_rollout_ref.actor.optim.total_training_steps`를 직접 넣은 뒤 `actor/lr ≈ 1e-6`로 정상화되었다
  - 그 다음 late-stage 병목은 **`WAIT` only valid action으로 format reward를 먹는 reward hacking**이었고, hard gate는 이 경로를 막는 대신 **SKD 직후에는 reward를 너무 sparse하게 만들었다**
  - 따라서 현재 operational reward는 hard gate가 아니라:
    - `env + 0.1 * effective_format`
    - 여기서 `effective_format`은 **기존 format reward를 유지하되, `WAIT/HOTKEY/TYPING` 인접 반복 비율로 positive format bonus만 감쇠**한 값이다
  - `score/format`은 이제 raw format이 아니라 **실제로 학습에 들어간 effective format**이다
  - `score/format_raw`와 `score/non_grounding_adjacency_ratio`를 같이 봐야 현재 reward가 어떻게 깎였는지 읽을 수 있다
  - one-turn plain-text babbling collapse를 막기 위해:
    - `actor_rollout_ref.rollout.multi_turn.max_assistant_response_tokens=2048`
    - runtime에서 `max_new_tokens = min(2048, remaining_budget)`
    를 assistant generation 직전에 주입하도록 바꾸었다
  - 이 패치 이후 latest live run `qwen35_webgym_fully_async_tool_veomni_20260515_112641`에서는:
    - assistant-turn text length
      - `p50 ≈ 946`
      - `p95 ≈ 4909`
      - `p99 ≈ 5164`
      - `max ≈ 10415`
    로, 이전의 수십만~백만 문자 단일-turn collapse는 사실상 사라졌다
  - 그러나 latest live bottleneck은 이제 **tool format / schema adherence**다:
    - summary 기준
      - `reward_zero_ratio ≈ 97.18%`
      - `parse_error_summary_ratio ≈ 83.65%`
      - `invalid_action_summary_ratio ≈ 54.38%`
    - assistant-turn 기준
      - `parse_error_turn_ratio ≈ 72.75%`
      - `valid_exec_turn_ratio ≈ 10.64%`
    - parse kinds top:
      - `tool_call_incomplete`
      - `actions_json_malformed`
      - `tool_tag_incomplete`
      - `parameter_block_malformed`
    - invalid causes top:
      - invalid `WebOsGymAction` payload
      - unknown functions such as `computer.press_key`, `wait`, `computer.press`
  - phase-by-phase reading on the latest live run is:
    - early `step <= 20`
      - semantic intent often looks plausible, but nearly every sample still breaks at the tool boundary
    - mid `100 <= step <= 140`
      - the dominant failure is still tool formatting / schema mismatch, not broadly weak reasoning
    - late `220 <= step <= 248`
      - still mostly tool-format failure on top of plausible plans
    - very late `249 <= step <= 260`
      - some trajectories stop being merely format-broken and also show weaker, more repetitive reasoning
  - 다만 **env false positive**는 여전히 별도 문제다
  - checkpoint 해석에서는
    - `training/global_step`
    - `current_param_version`
    - checkpoint folder `global_step_N`
    를 반드시 분리해서 읽어야 한다
- unconstrained decode path isolation 완료 (2026-05-15):
  - 27B val run WITHOUT `enable_qwen3_coder_structured_output`
    - `logs/rollout_data/qwen35_27b_webgym_fully_async_tool_veomni_only_val_20260515_173102/step_1`
  - CLICK x_as_list: `187 / 203` (92.1%)
  - CLICK x_as_scalar: `16 / 203` (7.9%)
  - dominant error: `"Input should be a valid integer [input_value=[x, y]]"` — Pydantic validation
  - 27B with `enable_qwen3_coder_structured_output=True` 기준 (이전 probe):
    - CLICK scalar: `85 / 85` (100%)
    - CLICK list: `0 / 85` (0%)
  - 즉 decode constraint 유무가 CLICK payload mode를 결정적으로 분리함이 확인됨
- SKD turn log 같은 패턴 확인 (2026-05-15):
  - source: `logs/async_skd_turns_webgym_20260515_165049.jsonl`
  - total turn events: `405`
  - `invalid_action = 192` (47%)
  - `tool_parse_error = 144` (36%)
  - CLICK x_as_list (invalid_action 중): `187 / 192` (97%)
  - val run과 SKD student 모두 동일한 list-shaped CLICK 패턴
- **root cause 최종 확정**:
  - 툴 스키마와 프롬프트는 올바르게 전달되고 있음
    - prompt_len이 student/teacher 동일, 스키마에 "x must be single integer" 명시
  - 그럼에도 unconstrained 생성에서 x 위치 top-1 = `' ['`, mass ≈ 0.9946
  - 텍스트 instruction만으로는 모델의 `[x, y]` coordinate pair 선호를 이길 수 없음
- **constrained decoding 방향 폐기**:
  - `structural_tag + xgrammar` 조합은 surface 증상을 제거하지만
  - SKD student는 constraint된 토큰을 생성하고, teacher는 unconstrained verify
  - teacher 자연 분포(list 선호)와 student 강제 출력(scalar) 간 distillation 신호 불일치 발생
  - inference 시에도 constraint 의존성이 남아 정책이 brittle해짐
- **새로운 방향: 스키마 정렬 (schema alignment)**:
  - constraint로 모델을 교정하는 대신, 스키마를 모델의 자연 분포에 맞춤
  - `x: integer, y: integer` → `coordinate: [x, y]` 배열 표현으로 전환
  - 이렇게 하면 constrained decoding 없이도 student/teacher 분포가 정렬되고
  - RL 이후에도 constraint 없이 동일한 형태를 재현 가능
  - 변경 범위: tool schema YAML, `web_osgym_tool.py` 파싱, few-shot asset, system prompt 문구
- **새로운 발견 (2026-05-15): RIGHT_CLICK/DRAG_TO/MOUSE_DOWN/MOUSE_UP 구현 누락**:
  - YAML 스키마와 `COMPUTER_13_ACTIONS`에는 등록되어 있으나
  - `_normalize_actions()` (`verl/tools/web_osgym_tool.py:414`)에 핸들러가 없음
  - 스칼라 `{x: 500, y: 500}`을 보내도 → `"Unsupported action_type: RIGHT_CLICK"` (ValueError)
  - 리스트 `{x: [500, 500]}`을 보내면 → Pydantic ValidationError
  - 어떤 포맷으로 보내든 무조건 실패하는 유령 액션 상태
- **전체 action type별 unconstrained 분포 (val run 173102 기준)**:
  - CLICK: scalar=14%, list=**85%**
  - DOUBLE_CLICK: scalar=34%, list=**66%**
  - MOVE_TO: scalar=11%, list=**88%**
  - RIGHT_CLICK: scalar=62%, list=37% → 스칼라도 핸들러 없어 실패
  - DRAG_TO: 출력 없음 → 핸들러 없어 실패
  - SCROLL/TYPING/HOTKEY/PRESS/KEY_DOWN/KEY_UP/WAIT/DONE/FAIL: 정상 (변경 불필요)
- **채택 구현 방향: tolerate-at-boundary normalizer**:
  - 스키마: `x, y → coordinate: [x, y]` (CLICK/DOUBLE_CLICK/RIGHT_CLICK/MOVE_TO/DRAG_TO)
  - normalizer (`_normalize_web_osgym_action_payload`): Pydantic 이전 단계에서 세 가지 변형 통일
    - `{x: [35, 175]}` → `{coordinate: [35, 175]}` (기존 오류 패턴)
    - `{x: 35, y: 175}` → `{coordinate: [35, 175]}` (레거시 스칼라)
    - `{coordinate: [35, 175]}` → 그대로 (신규 정상 포맷)
  - `_normalize_actions()`: `action.coordinate[0/1]`로 읽고 서버에는 기존 `{x, y}` scalar 전달
  - `_require_coordinates()`: coordinate 필드 우선 읽기, cursor fallback 로직 유지
  - RIGHT_CLICK/DRAG_TO/MOUSE_DOWN/MOUSE_UP 핸들러 추가 (스키마는 벤치마크 구성 호환을 위해 유지)
  - 서버 페이로드(`WebOsGymClient` → `/action`) 불변
- **반복 val 수렴 계획**:
  - Round 1: 위 변경 후 val → parse_error_turn_ratio / invalid_action 비율 측정
  - Round 2: 남은 실패 분류(DRAG_TO 의미론? 알 수 없는 action_type?) → 추가 normalizer rule 또는 스키마 조정 → val 재실행
  - Round 3: `parse_error_turn_ratio < 5%` 달성 시 RL 학습 진입

---

## Next Plan

### Goal

- 다음 단계의 1차 목표는:
  - 일반 reasoning 향상
  - reward 재설계
  가 아니라
  - **Async SKD 단계에서 current bundled WebGym tool contract compliance를 직접 끌어올리는 것**
- latest live diagnosis는 다음을 지지한다:
  - early / mid phase에서는 semantic intent가 종종 plausible하다
  - 하지만 그 intent가 tool formatting / schema mismatch 때문에 환경에 거의 반영되지 못한다
  - late tail의 weaker reasoning은 존재하지만, 이는 1차 병목이라기보다 tool-failure 누적 뒤의 2차 왜곡으로 읽는 것이 맞다

### Why the previous teacher-only conditioning did not help this actor

- Async SKD는 이미:
  - teacher-only system prompt
  - optional teacher-only structured few-shot
  를 지원한다
- 그리고 과거 실험에서:
  - few-shot multimodal wiring
  - teacher-only prompt stream stability
  는 이미 여러 번 검증되었다
- 그런데 latest live RL에서도 still dominant failure가:
  - `tool_call_incomplete`
  - `actions_json_malformed`
  - `tool_tag_incomplete`
  - `parameter_block_malformed`
  - wrong action vocabulary such as `TYPE`
  - legacy wrapper such as `computer.press_key`
  로 남아 있다
- therefore the immediate conclusion is:
  - teacher-only conditioning이라는 메커니즘이 없는 것은 아니다
  - but the SKD path that produced the current actor did **not** use that conditioning
  - so the next probe should be read as:
    - **first real teacher-only compliance-conditioning test**

### 27B teacher-capability probe before rebuilding the few-shot

- before rewriting the teacher asset, first measure whether the current `27B` model itself is capable of producing
  usable WebGym trajectories under the current bundled contract
- probe script:
  - `WebOSWorld/val_run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
- important reading rule:
  - this script runs the `27B` model as the rollout actor for validation-style trajectory collection
  - this is **not** the exact SKD teacher verification path
  - but it is still the right direct capability probe for:
    - current model-side tool compliance
    - whether the `27B` model can supply a clean teacher-only few-shot candidate at all
- latest probe root:
  - `logs/rollout_data/qwen35_27b_webgym_fully_async_tool_veomni_only_val_20260515_161150/step_1`
- summary schema note:
  - these `summary.json` files do not carry the RL-style `reward` breakdown block
  - here `reward_score` should be read directly as the task success signal for the probe
  - current observed values are binary:
    - `0.0`
    - `1.0`

#### Observed 27B probe result

- total samples:
  - `93`
- positive `reward_score`:
  - `18 / 93`
  - `≈ 19.35%`
- parse-error presence:
  - `4 / 93`
  - `≈ 4.30%`
- invalid-action presence:
  - `30 / 93`
  - `≈ 32.26%`
- any valid tool call:
  - `88 / 93`
  - `≈ 94.62%`
- means:
  - `valid_tool_call_count mean ≈ 24.10`
  - `attempted_tool_call_count mean ≈ 25.11`
- termination distribution:
  - `tool_response_budget_exhausted: 41`
  - `model_done: 32`
  - `system_stop: 16`
  - `model_fail: 4`

#### Interpretation

- the `27B` model is **not** a `0%` hopeless teacher candidate under the current contract
- more importantly:
  - tool formatting is much healthier than the problematic `9B` RL runs
  - valid tool calls are common
  - parse collapse is comparatively rare in this probe
- the main remaining weakness in this `27B` probe is:
  - not “cannot speak the contract at all”
  - but “often fails to finish the task within the budget”
- therefore this probe says:
  - the teacher model itself likely has enough compliance headroom to serve as a useful teacher-only few-shot source
  - the next asset-building step is justified

#### Representative examples

- clean success:
  - `warn_ex_toll_basic_exact`
  - `reward_score=1.0`
  - `valid=5`
  - `parse=0`
  - `invalid=0`
  - `termination=model_done`
- valid but incomplete:
  - `web_bmi_01`
  - `reward_score=0.0`
  - `valid=38`
  - `parse=0`
  - `invalid=0`
  - `termination=tool_response_budget_exhausted`
- implication:
  - some trajectories are already compliant enough to mine
  - but candidate selection must explicitly prefer:
    - successful completion
    - clean contract adherence
    - low retry / low loop behavior
  - and reject:
    - budget-exhausted long trajectories even if they are syntactically valid

### New finding. The `CLICK` collapse in current SKD is not explained by prompt mismatch

- latest SKD fast-test evidence:
  - `logs/async_skd_turns_webgym_20260515_165049.jsonl`
  - `logs/async_skd_events_webgym_20260515_165049.jsonl`
  - `logs/skd_fast_test_gdn_fix_20260515_165049.out`
- prompt/tool-contract parity check:
  - current fast-test launcher does **not** pass
    - `distillation.skd.teacher_system_prompt_path`
    - `distillation.skd.teacher_fewshot_path`
  - `WebSkdAgentLoop` applies the same active tool schemas to both student and teacher prompt rendering
  - live trace also showed:
    - `pending_student_template_done ... prompt_len=4241`
    - `pending_teacher_template_done ... teacher_prompt_len=4241`
    - `pending_commit_done ... prompt_len=4241 teacher_prompt_len=4241`
  - therefore the current `CLICK` failure is **not** explained by teacher/student prompt text divergence or a different bundled tool schema

#### Actual `CLICK` contrast

- `27B` actor-only validation probe (`step_1`) under the current bundled contract:
  - `CLICK` scalar payload samples: `85`
  - `CLICK` list-shaped payload samples: `0`
- current SKD fast-test snapshot:
  - turn rows: `73`
  - `tool_exec_status=invalid_action`: `35`
  - invalid action types:
    - `CLICK: 33`
    - `DOUBLE_CLICK: 1`
    - `RIGHT_CLICK: 1`
  - invalid shape cluster:
    - `x_as_list: 35 / 35`
- implication:
  - the same `27B` family can produce scalar `CLICK` payloads in actor-only validation
  - but the current SKD top-1 path collapses specifically on `CLICK` payload shape

#### Root-cause narrowing

- the decisive difference is **decode/runtime path**, not prompt text:
  1. actor-only validation path:
     - `WebOSWorld/val_run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
     - includes:
       - `+actor_rollout_ref.rollout.custom.enable_qwen3_coder_structured_output=True`
     - therefore the WebOSWorld Qwen coder branch injects:
       - `structural_tag`
       - `ignore_eos=True`
       - `stop=["</tool_call>"]`
  2. current SKD fast-test path:
     - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni_fast_test_GDN_fix.sh`
     - does **not** set `enable_qwen3_coder_structured_output=True`
     - teacher verification uses:
       - `TeacherManager.compute_teacher_logprobs_single(...)`
       - prompt-logprob scoring only
       - no `structural_tag`
       - no constrained generation branch
- therefore the current comparison is:
  - actor-only `27B` with the structured-output generation path
  - versus SKD teacher top-1 from an unconstrained prompt-logprob verify path
- this explains why the `CLICK` payload mode can diverge even when prompt text and bundled tool schemas match

#### What the teacher top-1 rows actually showed

- on the current SKD fast-test invalid `CLICK` requests, the failure was **not** merely low-probability tail drift under `top-10`
- representative teacher top-1 rows around `x` payload construction showed:
  - after `x:`:
    - top-1 = `' ['`
    - top-5 normalized mass ≈ `0.9946`
  - list opening:
    - top-1 = `'['`
    - top-5 normalized mass ≈ `0.9983`
- interpretation:
  - under the current unconstrained teacher verify path, `CLICK` payload corruption is already present in the teacher local top-1 mode itself
  - this is why simply tightening `verify_top_k` cannot solve the current `CLICK` failure

### Proposed SKD plan

#### Exact next intervention

- current decision changed:
  - the **next** SKD probe should not start with teacher-only few-shot
  - the next probe should first isolate **decode/runtime-path parity**
- reason:
  - current evidence does **not** support a prompt-mismatch story
  - current evidence instead supports:
    - actor-only `27B` and SKD teacher verify are using materially different decoding contracts
    - this difference is already enough to explain the `CLICK` scalar-vs-list divergence

#### ~~Step A. Align the SKD teacher path to the actor-only path before rebuilding teacher assets~~ (Closed)

- 이 step은 2026-05-15 실험으로 answer가 나와 closed 처리함
- 27B val run WITHOUT `enable_qwen3_coder_structured_output`
  - `logs/rollout_data/qwen35_27b_webgym_fully_async_tool_veomni_only_val_20260515_173102`
  - CLICK x_as_list: 92.1%
- SKD turn log (20260515_165049) 동일 패턴: 97%
- **결론**: unconstrained greedy도 list-shaped CLICK을 낸다. 이 이상의 probe는 불필요.

#### Step A (revised). Schema alignment — coordinate 표현을 모델 자연 분포에 맞게 전환

##### 배경 및 근거

- 모델의 unconstrained x token top-1 = `' ['`, mass ≈ 0.9946
- 텍스트 instruction으로 이 prior를 이길 수 없음이 확인됨 (YAML에 “Do not pass a list” 명시에도 92.1% list 출력)
- `structural_tag + xgrammar` constraint는 surface 증상만 제거:
  - student: constraint된 scalar 토큰 생성
  - teacher: unconstrained verify (list 선호 분포)
  - → distillation 신호 불일치 → SKD learning signal 품질 저하
- RIGHT_CLICK/DRAG_TO/MOUSE_DOWN/MOUSE_UP: 스키마 선언은 있으나 `_normalize_actions()` 핸들러가 없었음 → 이미 별도 패치로 수정 완료 (2026-05-15)

##### Anthropic Computer Use API 비교 결과

- Anthropic의 `computer_20241022` / `computer_20251124` API와 스키마 전면 비교 수행
- 공유 가능한 부분: `coordinate: [x, y]` 배열 표현 — 이것만 채택
- 채택하지 않는 부분:
  - action type 이름 (`left_click` vs `CLICK` 등): 모델이 이미 우리 이름으로 출력하므로 변경 불필요
  - scroll 형식 (`scroll_direction/scroll_amount` vs `dx/dy`): 모델이 이미 dx/dy로 출력 (100% 일치)
  - 좌표계 (픽셀 절댓값 vs 1000×1000 상대 좌표): 서버 프로토콜 변경 필요하므로 유지
- **결론**: `coordinate: [x, y]` 필드명만 Anthropic 방식으로 채택. 나머지는 현행 유지.

##### 변경 확정 — 스키마 레이어 (YAML)

- 대상 action: CLICK, DOUBLE_CLICK, RIGHT_CLICK, MOVE_TO, DRAG_TO (5개)
- `x: integer, y: integer` → `coordinate: array[2 integers, 0–999]`로 교체
- required 변경:
  - MOVE_TO, DRAG_TO: `coordinate` 필수
  - CLICK, DOUBLE_CLICK, RIGHT_CLICK: `coordinate` 선택 (cursor fallback 있음)
- 모델 출력 예시 (변경 후):
  ```json
  {“action_type”: “CLICK”, “coordinate”: [531, 582]}
  ```
- MOUSE_DOWN/MOUSE_UP: 스키마 유지 (벤치마크 호환), 핸들러는 이미 추가 완료

##### 변경 확정 — Description 정리

- 기존 “Do not pass a list / bracketed value” 계열 경고 **전부 제거**
  - 실증적으로 효과 없었음 (경고 있어도 92.1% list 출력)
  - 이제 list 형태가 정상 포맷이므로 경고가 모순이 됨
- `x and y` 언급을 `coordinate` 로 교체
- 확정된 description 변경:

  | Action | 변경 후 description |
  |--------|---------------------|
  | MOVE_TO | `”Move the cursor to coordinate [x, y] in the 1000×1000 grid.”` |
  | CLICK | `”Click at coordinate [x, y] if provided, otherwise at the current cursor position.”` |
  | DOUBLE_CLICK | `”Double click at coordinate [x, y] if provided, otherwise at the current cursor position.”` |
  | RIGHT_CLICK | `”Right click at coordinate [x, y] if provided, otherwise at the current cursor position.”` |
  | DRAG_TO | `”Drag to coordinate [x, y] with the left button pressed.”` |
  | HOTKEY | `”Press the specified key combination. keys is an array of key names, for example ['ctrl', 'c'].”` |

- 변경 없음: SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, MOUSE_DOWN, MOUSE_UP, WAIT, DONE, FAIL

##### 변경 확정 — 파서 레이어 (tolerate-at-boundary)

**원칙**: 모델 출력 포맷 ↔ 서버 페이로드를 파서가 중간에서 분리. 서버는 기존 `{x: int, y: int}` 그대로 수신.

**계층 1 — `_normalize_web_osgym_action_payload()` (Pydantic 이전)**

세 가지 모델 출력 변형을 `coordinate`로 통일:

```
{x: [35, 175]}        → {coordinate: [35, 175]}   (기존 주요 실패 패턴)
{x: 35, y: 175}       → {coordinate: [35, 175]}   (레거시 scalar)
{coordinate: [35, 175]} → 그대로                   (신규 정상 포맷)
```

변환 후 `x`, `y` 키 제거.

**계층 2 — `WebOsGymAction` Pydantic 모델 (`web_osgym_protocol.py`)**

- `coordinate: list[int] | None = None` 추가
- `x, y` 필드는 내부 legacy fallback용으로만 유지 (스키마 미노출)

**계층 3 — `_normalize_actions()` 내부 (`web_osgym_tool.py`)**

- `_require_coordinates()`: `action.coordinate` 우선 → legacy `x,y` → cursor fallback 순서
- 스케일링 조건: `if action.coordinate is not None or (action.x is not None and action.y is not None)`
  - cursor fallback은 이미 screen 좌표 → scale 불필요
  - coordinate 또는 x,y 명시 시에만 1000×1000 → pixel 변환
- 각 핸들러: `payload.pop(“coordinate”, None)` 후 `payload.update({x: scaled_x, y: scaled_y})`
  - 서버 payload에 coordinate 필드가 포함되지 않도록 보장

**전체 데이터 흐름**:
```
모델: {“action_type”: “CLICK”, “x”: [35, 175]}
  ↓ _normalize_web_osgym_action_payload()
{“action_type”: “CLICK”, “coordinate”: [35, 175]}
  ↓ WebOsGymAction(**payload)  — Pydantic 통과
action.coordinate = [35, 175], action.x = None
  ↓ _normalize_actions() → _require_coordinates() → scale → pop coordinate
서버 수신: {“action_type”: “CLICK”, “x”: 672, “y”: 189, “button”: “left”, “num_clicks”: 1}
```

##### 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml` | `x,y` → `coordinate` 교체, description 정리 |
| `verl/experimental/agent_loop/web_osgym_protocol.py` | `WebOsGymAction`에 `coordinate` 필드 추가 |
| `verl/tools/web_osgym_tool.py` | normalizer, `_require_coordinates()`, 각 핸들러 스케일링 조건 |
| few-shot JSON 및 system prompt | `x,y` → `coordinate` 예시 교체 |

##### 예상 효과

- CLICK parse error 85% 즉시 제거 (list → coordinate 자동 변환)
- DOUBLE_CLICK 66%, MOVE_TO 88% 동일 해소
- RIGHT_CLICK/DRAG_TO 처음으로 실제 서버 도달 가능 (핸들러 이미 추가)
- constrained decoding 불필요 → student/teacher 분포 정렬
- RL 이후에도 constraint 없이 동일 포맷 재현 가능

#### Step B. Schema 변경 후 val 반복 수렴

- tolerate-at-boundary normalizer 덕분에 `x_as_list` 자체는 normalizer가 흡수 → 더 이상 invalid_action 원인이 되지 않아야 함
- Round 1 val 주요 readout:
  - coordinate-type parse error 소멸 여부 (normalizer 정상 작동 확인)
  - `parse_error_turn_ratio` 전체 감소 폭
  - `valid_exec_turn_ratio` 상승 여부
  - 남은 실패 패턴 재분류 (DRAG_TO 의미론? 알 수 없는 action_type 문자열?)
- Round 2: 남은 패턴 → 추가 normalizer rule 또는 description 조정 → val 재실행
- Round 3 수렴 기준: `parse_error_turn_ratio < 5%` → RL 진입 판단

#### Step C. Compliance 개선 확인 후 RL 진입 여부 판단

- RL 진입 조건:
  - `invalid_action` 중 CLICK x_as_list 비율이 의미 있게 감소
  - `tool_exec_status=ok` turn 비율 상승
  - early phase에서 executable action이 실제로 환경에 반영되는 것 확인
- 조건 미충족 시:
  - RL 진입 postpone
  - schema / few-shot 추가 조정 우선

### Go / No-Go metrics for the next SKD probe (schema alignment 이후)

- must improve (schema alignment 이후 가장 먼저 볼 것):
  - CLICK x_as_list invalid_action 비율
  - `invalid_action` 전체 비율
  - `tool_exec_status=ok` turn 비율
- should improve:
  - `tool_call_incomplete`
  - `actions_json_malformed`
  - valid tool call rate
  - no-valid-call sample ratio
- only secondary:
  - downstream reward
  - long-horizon task completion

### Practical interpretation rule

- if compliance improves but task completion is still mediocre:
  - continue iterating on SKD / prompt-side compliance
  - do not jump to “model reasoning is weak” yet
- if compliance improves and env reward still stays flat:
  - then the remaining bottleneck moves to:
    - grounding
    - state tracking
    - or evaluator correctness
- if compliance does **not** improve:
  - then the next intervention should still stay on the tool boundary
  - not on reward shaping

---

## Experiment Plan

### Step 0. Baseline log inspection

- 대상 로그
  - `logs/async_skd_events_webgym_20260512_030331.jsonl`
  - `logs/async_skd_chunk_live_webgym_20260512_030331.jsonl`
- 목적
  - `logical_step 1~3`에서 이미 `tool_call` 없이 `<|im_end|>`로 끝나는 현상이 심한지 확인
  - `student direct`인지, `teacher replacement 이후 plain ending`인지 분리
- 산출물
  - step별 `plain_end`, `toolish_end`
  - `rejected=0` / `rejected>0` 분해
  - 대표 예시 tail

### Step 1. Remove teacher system prompt

- 변경
  - `distillation.skd.teacher_system_prompt_path` 제거 또는 빈 값으로 실행
- 목적
  - teacher privileged guidance 없이도 동일 붕괴가 나는지 확인
  - 초기 step에서
    - `툴 없이 바로 종료`
    - `툴 시도는 하지만 malformed`
    중 어느 쪽으로 분포가 이동하는지 확인
- 산출물
  - Step 0와 동일 지표 비교

### Step 2. Judge Step 1

- 목적
  - teacher system prompt가 plain ending을 유도하는지 판단
- 산출물
  - baseline 대비 변화 요약
  - 다음 실험 진행 여부 판단

### Step 3. Block bad targets before improving teacher candidates

- 변경
  - parse error turn: last assistant turn mask + terminate
  - invalid action turn: last assistant turn mask + terminate
  - no-tool plain text turn: last assistant turn mask + terminate
  - turn-level decision log 추가
- 목적
  - replacement가 만들어 낸 bad plain-ending target이 실제 학습에 남지 않게 막기
  - WebGym / WebOSGym task contract에서 `유효한 tool call turn만 keep` 되게 만들기
  - 이후 teacher-side 실험이 meaningfully read되도록 learning target을 먼저 정리
- 산출물
  - `assistant_turn_decision` 로그
  - `turn_mask_before/after`
  - parse / invalid / no-tool 분기별 동작 검증

### Step 4. Improve teacher-side candidate quality in controlled order

- 순서
  1. current mask+terminate patch 유지
  2. teacher-only clean few-shot 1개 추가
  3. 필요하면 teacher-only system guidance를 다시 분리 실험
- 목적
  - bad target을 차단한 뒤
  - teacher가 action-needed turn에서 더 좋은 candidate를 내게 만들기
  - few-shot이 실제로 teacher verification candidate quality를 올리는지 판단
- 산출물
  - early-step collapse 지표 비교
  - `assistant_turn_decision` 분포 비교

### Step 5. Judge Step 4

- 목적
  - 어떤 teacher-side guidance가
    - `mask_and_terminate`를 줄이고
    - `tool_exec_status=ok + keep`를 늘리는지
  - few-shot이 실제로 teacher candidate quality를 개선하는지 판단
- 산출물
  - 최종 결론
  - 유지할 구성 / 버릴 구성

---

## Findings Log

### Step 1. Pre-run preparation for teacher-system-prompt ablation

#### Status

- Completed

#### Goal

- `teacher-only` system suffix prompt만 제거하고,
- 공통 prompt와 나머지 runtime contract는 그대로 유지해서
- 변화 원인을 최대한 한 축으로 제한한다.

#### Fairness / Change Control

- 유지한 것
  - `WEBGYM_SYSTEM_PROMPT_PATH`
    - 학생/교사 공통 base system prompt로 그대로 유지
  - tool config
    - `webgym_rl_tool_config_bundled.yaml` 그대로 유지
  - sampling / verification / loss 관련 주요 인자
    - `chunk_size=256`
    - `verify_top_k=10`
    - `loss_mode=forward_kl_topk`
    - `use_task_rewards=False`
    - `use_policy_gradient=False`
  - rollout format
    - `multi_turn.format=qwen3_coder`
  - a11y 설정
    - `web_skd_include_a11y=false`

- 제거한 것
  - `distillation.skd.teacher_system_prompt_path=...`
  - 의미:
    - teacher에게만 뒤에 덧붙던 suffix system guidance 제거
    - 공통 system prompt는 유지

#### Why this is a fair ablation

- 공통 prompt 파일 경로와 Hydra key를 건드리지 않았다.
- 학생이 보는 prompt와 teacher가 공통으로 보는 base prompt는 그대로다.
- teacher-only suffix guidance만 빠지므로,
  이번 실험의 차이는 최대한 `teacher privileged system suffix 유무`에 집중된다.

#### Whitespace / formatting risk check

- 공통 prompt 문자열 자체는 수정하지 않았다.
- shell 변수 치환이나 prompt 파일 내용도 수정하지 않았다.
- 실행 스크립트에서는 Hydra 실행 인자 한 줄만 제거했다.
- 따라서 공통 prompt의 공백/줄바꿈 차이로 인한 drift는 만들지 않았다.

#### Logging isolation

- event log
  - 원래부터 `async_skd_events_webgym_${RUN_TS}.jsonl`
- chunk live log
  - 원래부터 `async_skd_chunk_live_webgym_${RUN_TS}.jsonl`
- rollout data dir
  - 고정 경로 `webgym_async_skd_current` 대신
  - `webgym_async_skd_${RUN_TS}` 로 변경

#### Result

- 현재 Step 1 run은
  - `teacher-only system suffix removed`
  - `shared base system prompt unchanged`
  - `logs isolated per run`
  상태에서 실행되도록 준비됨

### Step 1. Live result snapshot (teacher-only system suffix removed)

#### Status

- In progress

#### Sources

- `logs/async_skd_events_webgym_20260514_011423.jsonl`
- `logs/async_skd_chunk_live_webgym_20260514_011423.jsonl`

#### Terminology correction

- `final_plain_tail`
  - `(request_id, logical_step)`의 마지막 chosen chunk tail이 plain `<|im_end|>`로 끝남
- `final_toolish_tail`
  - 마지막 chosen chunk tail에 tool-like syntax가 남아 있음
- `pure_no_tool_end`
  - 같은 request-step의 어떤 verified chunk에도 tool-like syntax가 없음
- `tool_attempt_then_plain_end`
  - earlier verified chunk에는 tool-like syntax가 있었지만 final tail은 plain ending

#### Current observation

- 현재 run에서는 `logical_step 1`이 아니라 `logical_step 2`부터 기록됨
  - manager가 `logical_step = global_steps + 1`로 잡기 때문
- 현재 시점에는 `logical_step 2`만 존재
  - `logical_step 3`은 아직 없음

#### Current step 2 snapshot

- current step 2
  - `n = 32`
  - `final_plain_tail = 24`
  - `tool_attempt_then_plain_end = 23`
  - `pure_no_tool_end = 1`
  - `plain_teacher_replaced = 24`
  - `final_toolish_tail = 8`
  - `toolish_student_direct = 8`

- baseline step 2
  - `n = 32`
  - `final_plain_tail = 22`
  - `tool_attempt_then_plain_end = 22`
  - `pure_no_tool_end = 0`
  - `plain_teacher_replaced = 22`
  - `final_toolish_tail = 9`
  - `toolish_student_direct = 9`

#### Interpretation

- 현재까지는 step 2 기준으로 baseline 대비 개선이 보이지 않음
- 오히려 약간 악화된 쪽에 가까움
  - `final_plain_tail: 22 -> 24`
  - `plain_teacher_replaced: 22 -> 24`
  - `final_toolish_tail: 9 -> 8`
- 문제의 중심은 여전히
  - `pure_no_tool_end`
  보다는
  - `tool_attempt_then_plain_end`
  - 그리고 `teacher replacement 이후 final plain tail`
  쪽임

#### Replacement dynamics under `verify_top_k=10`

- 범위
  - 이 분석은 `verify_top_k=10`이었던 직전 run만 대상으로 함
  - chunk live:
    - `logs/async_skd_chunk_live_webgym_20260514_011423.jsonl`
  - token-level trace:
    - `tmp/ray/session_2026-05-14_01-14-33_450449_677019/logs/*.err`

- replacement frequency
  - total chunk rows: `833`
  - replaced chunk rows: `571`
  - replacement rate: `68.55%`
  - request-step (`logical_step 2`) count: `32`
  - `32 / 32` request-step 모두 최소 1회 replacement 경험
  - request-step당 replaced chunk 수
    - min: `1`
    - median: `12`
    - mean: `17.84`
    - max: `58`

- accepted tokens before first mismatch in replaced chunks
  - min: `0`
  - p25: `7`
  - median: `23`
  - p75: `49`
  - max: `237`
  - mean: `35.33`

- token-level replacement examples
  - request `e4f309cca4eb4972bc203547bd1abc5c`
    - `student='{"'`
    - `final='parameter'`
    - JSON-style continuation을 XML/tool scaffold token으로 교체
  - request `0334d4d2e69147dd9a53215aa9a77a43`
    - `student='mock'`
    - `final='모'`
    - 영어 lexical continuation을 한국어 page-context token으로 교체
  - request `490a2824bb6a455c97d27d9a7b7d4063`
    - `student='mock'`
    - `final='모'`
    - 위와 같은 유형의 lexical recentering

- what happens right after the first replacement
  - first replacement 이후 바로 다음 chunk 분류
    - `next_chunk_toolish = 9`
    - `next_chunk_plain = 0`
    - `next_chunk_non_eos_partial = 21`
    - `no_next_chunk = 2`
  - 즉, first replacement 직후 곧바로 plain `<|im_end|>` ending으로 죽는 경우는 없었음
  - 오히려 다음 chunk는
    - partial continuation
    - 또는 toolish continuation
    으로 이어지는 경우가 대부분

- interpretation
  - replacement는 local token correction 수준에서는 생각보다 제대로 작동함
  - 그러나 그것이 sequence-level repair로 이어지지는 않음
  - 즉
    - `교체 자체`
    - `교체 직후 다음 chunk`
    는 크게 망가지지 않지만,
    - 여러 번의 replacement와 drift가 누적된 뒤
    - 최종적으로 `final_plain_tail`로 수렴하는 패턴이 남음

### Step 1b. Tighten `verify_top_k` from 10 to 3

#### Status

- Completed for `logical_step 2`
- `logical_step 3`은 아직 없음

#### Sources

- `logs/async_skd_events_webgym_20260514_012918.jsonl`
- `logs/async_skd_chunk_live_webgym_20260514_012918.jsonl`
- `tmp/ray/session_2026-05-14_01-29-28_258029_705487/logs/*.err`

#### Method

- `logical_step 2`만 대상으로 분석
- `(request_id, logical_step)`별로 final row를 구성
  - `eos_in_new_tokens=True`인 마지막 row가 있으면 그것을 final row로 사용
  - 없으면 마지막 chunk를 사용
- 다음을 분리해서 집계
  - replacement 빈도
  - token-level replacement 예시
  - first replacement 직후 next chunk가 어떤 형태로 이어지는지
  - final plain tail이
    - `teacher pass`
    - `teacher replacement involved`
    중 무엇인지

#### Result

- step 2 aggregate
  - total chunk rows: `927`
  - replaced rows: `859`
  - replacement rate: `92.66%`
  - request-step count: `32`
  - `32 / 32` request-step 모두 최소 1회 replacement 경험
  - request-step당 replaced chunk 수
    - min: `1`
    - median: `16.5`
    - mean: `26.84`
    - max: `132`

- accepted tokens before first replacement
  - min: `0`
  - median: `18.5`
  - mean: `20.63`
  - max: `67`

- final step 2 distribution
  - `final_plain_tail = 31`
  - `tool_attempt_then_plain_end = 27`
  - `pure_no_tool_end = 4`
  - `plain_teacher_replaced = 31`
  - `plain_student_direct = 0`
  - `final_toolish_tail = 1`
  - `toolish_student_direct = 1`

- comparison against baseline and top-10
  - baseline
    - `final_plain_tail = 22`
    - `final_toolish_tail = 9`
  - top-10
    - `final_plain_tail = 24`
    - `final_toolish_tail = 8`
  - top-3
    - `final_plain_tail = 31`
    - `final_toolish_tail = 1`

- token-level replacement examples
  - request `9ff58be5f1f94971bc6321c60beea36f`
    - `student='I'`
    - `final='The'`
  - request `008162c389d948b48a67c2ebf46e0add`
    - `student='0'`
    - `final='1'`
  - request `5eb2b285b5d44b2c8068f0f91f1d447c`
    - `student=\"'m\"`
    - `final=' can'`
  - request `28c8826ab191490d863a4de9bc6600bf`
    - `student='),'`
    - `final=').'`

- what happens right after the first replacement
  - `next_chunk_toolish = 1`
  - `next_chunk_plain = 2`
  - `next_chunk_non_eos_partial = 28`
  - `no_next_chunk = 1`
  - 즉, first replacement 직후에는
    - 곧바로 plain `<|im_end|>`로 닫히기보다
    - partial continuation으로 이어지는 경우가 대부분

- how final plain tails are actually formed
  - `final_plain_tail = 31` 전부 `rejected=1` in final chunk
  - 즉, final plain tail은 `teacher가 그냥 통과시킨 plain ending`이 아님
  - raw final tail category
    - `raw_toolish_tail = 28`
    - `raw_partial_plain = 2`
    - `raw_partial_toolish = 1`
    - `raw_plain_tail = 0`
  - 의미
    - 대부분의 final plain tail은
      - 학생 raw final chunk에서는 tool-like continuation이 있었고
      - final chunk replacement + suffix discard 이후
      - plain assistant ending으로 끝남

- representative examples
  - request `c5a8ee2df1e243ad852246ac91844c03`
    - raw final tail:
      - malformed tool call
      - `[{\"action_type\": \"CLICK\", \"x\": [33, 724], \"y\": [721]}`
    - verified final tail:
      - `I need to open the terminal first to navigate to the Scripts directory and open the helloworld script using vi editor.<|im_end|>`
    - final chunk:
      - `accepted=23`
      - `rejected=1`
  - request `e6bfbe6ece6f4c50b1f4de10faac6bb6`
    - raw final tail:
      - tool-like `DOUBLE_CLICK`
    - verified final tail:
      - `the Pictures/crumbling city directory.<|im_end|>`
    - final chunk:
      - `accepted=8`
      - `rejected=1`
  - request `a9911cfe2e494eeeab64455ae0132bc6`
    - chunk 1:
      - plain partial, `eos=false`
    - chunk 2:
      - raw final tail은 tool-ish payload partial
      - verified final tail은 `25.<|im_end|>`
    - 의미:
      - 중간 chunk 하나만 보면 오해할 수 있지만
      - 이 request-step도 최종적으로는 plain tail로 끝남

#### Interpretation

- `verify_top_k=3`은 현재 방향이 아님
- teacher replacement가 더 엄격해지면서
  - student continuation을 훨씬 더 자주 자르고
  - 결과적으로 tool-like continuation이 살아남지 못함
- 중요한 점은
  - `teacher가 final plain tail을 그냥 통과시킨다`가 아니라
  - `teacher replacement가 final plain tail 형성에 직접 개입한다`는 것
- 즉 top-3에서는
  - `teacher에 더 비슷해진다`기보다
  - `replacement가 너무 자주 걸리고`
  - `first-mismatch replacement + suffix discard` 구조 때문에
  - final output이 plain assistant ending으로 더 쉽게 수렴함

### Step 1c. Mitigation chosen after replacement analysis

#### Status

- Implemented
- TDD completed

#### Sources

- runtime path
  - `verl/experimental/agent_loop/skd_agent_loop.py`
  - `verl/experimental/agent_loop/web_skd_agent_loop.py`
- tests
  - `tests/skd/test_web_skd_agent_loop_on_cpu.py`
  - `tests/skd/test_skd_logic.py`
- launcher
  - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
  - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni_fast_test.sh`

#### Reasoning flow

- replacement 분석으로 확인한 것
  - 문제의 중심은 `student가 아무것도 안 한다`가 아니라
  - `tool-like raw continuation -> teacher replacement -> final plain tail` 경로임
  - `verify_top_k`를 더 줄여도 개선되지 않고 오히려 악화됨

- 그래서 먼저 내린 판단
  - `teacher 분포를 바꾸는 것`보다
  - `나쁜 final target을 학습시키지 않게 막는 것`이 우선
  - 즉 few-shot은 보조 수단일 수 있지만 첫 대응은 아님

- 구현 아이디어를 좁힌 이유
  - 현재 SKD는 chunk를 바로 commit하지 않고
  - assistant turn이 `EOS`로 끝날 때만 `_commit_pending_turn_state()`에서 `response_mask=1`로 commit함
  - 그래서 실제 supervision 경계는 chunk가 아니라 `assistant turn span`
  - 이 span만 마스킹하면 bad target을 깔끔하게 제거할 수 있음

- 최종 선택
  - `parse_error`
    - 마지막 committed assistant turn을 mask
    - recovery observation append 생략
    - 즉시 terminate
  - `invalid_action`
    - 마지막 committed assistant turn을 mask
    - invalid observation append 생략
    - 즉시 terminate
  - valid turn은 그대로 keep

#### Method

- TDD 원칙
  - 먼저 failing test를 작성
  - 현재 동작이
    - `parse_error -> GENERATING 계속`
    - `invalid_action -> GENERATING 계속`
    임을 실제 WebSKD 경로에서 확인
  - 그 뒤 최소 구현만 추가

- test scope
  - direct branch test
    - parse error branch
    - invalid action branch
  - end-to-end WebSKD path test
    - `WebSkdAgentLoop._handle_generating_state()`
    - `EOS commit`
    - parser
    - tool execution
    - final `response_mask`
    까지 실제 경로를 모사

- logging
  - 기존 `chunk_live`와 별도로
  - `VERL_ASYNC_SKD_TURN_LOG`
  - `async_skd_turns_webgym_${RUN_TS}.jsonl`
  - event name: `assistant_turn_decision`
  - 주요 필드
    - `assistant_turn`
    - `response_start`
    - `response_end`
    - `parse_status`
    - `tool_exec_status`
    - `mask_decision`
    - `turn_mask_before`
    - `turn_mask_after`
    - `verified_text_tail`

#### Result

- 구현된 동작
  - parse error가 나면
    - 마지막 assistant turn span의 `response_mask`가 `1 -> 0`
    - 같은 span의 teacher rows/logprobs도 dummy row로 zeroing
    - `termination_reason = tool_parse_error`
  - invalid action이 나면
    - 마지막 assistant turn span의 `response_mask`가 `1 -> 0`
    - 같은 span의 teacher rows/logprobs도 dummy row로 zeroing
    - `termination_reason = invalid_action`
  - valid turn 또는 no-tool terminal turn은 `mask_decision=keep`으로 로깅

- verification
  - targeted RED tests
    - parse error path fail 확인
    - invalid action path fail 확인
  - GREEN after implementation
    - new tests pass
  - broader regression
    - `tests/skd/test_web_skd_agent_loop_on_cpu.py`
    - `tests/skd/test_skd_logic.py`
    - total `89 passed`

#### Interpretation

- 이번 변경은 구조를 갈아엎는 작업이 아니라
  - 기존 `assistant turn commit` 경계를 활용한 국소 패치
  - 즉 `패치형 작업`
- 하지만 학습 의미는 큼
  - 이전에는 `bad assistant turn`이 그대로 supervised target으로 남았고
  - 이제는 `invalid assistant turn`을 target에서 제거함
- 즉 이번 단계의 목적은
  - teacher replacement를 더 똑똑하게 만드는 것이 아니라
  - replacement가 만든 bad target을 더 이상 학습하지 않게 막는 것
- 이 단계가 끝나고 나서야
  - `teacher가 더 좋은 candidate를 내게 만들기`
  - 즉 clean teacher-only few-shot 같은 teacher-side quality 실험이
  실제로 읽을 수 있는 상태가 됨
- 따라서 다음 단계의 초점은
  - `안전한 subset만 학습`
  에서
  - `애초에 teacher candidate 자체를 더 좋게 만들기`
  로 이동함

### Step 1d. Validate masking run and close the `no_tool_call` gap

#### Status

- Completed

#### Sources

- `logs/async_skd_events_webgym_20260514_022134.jsonl`
- `logs/async_skd_chunk_live_webgym_20260514_022134.jsonl`
- `logs/async_skd_turns_webgym_20260514_022134.jsonl`
- runtime path
  - `verl/experimental/agent_loop/skd_agent_loop.py`
  - `verl/experimental/agent_loop/web_skd_agent_loop.py`
- tests
  - `tests/skd/test_web_skd_agent_loop_on_cpu.py`
  - `tests/skd/test_skd_logic.py`

#### Method

- 먼저 turn decision log를 기준으로
  - `mask_and_terminate`
  - `keep`
  분포를 확인
- 특히 `parse_status`, `tool_exec_status`, `termination_reason`을 step별로 분리
- 그 다음 `chunk_live` final tail을 함께 보되,
  - 학습에 실제로 남았는지는 `turns` 로그의 `mask_decision`을 source of truth로 사용
- 마지막으로 `no_tool_call`가 왜 keep되는지 production code와 테스트 커버리지를 역추적

#### Result

- masking run aggregate (`async_skd_turns_webgym_20260514_022134.jsonl`)
  - total `assistant_turn_decision` rows: `294`
  - `mask_and_terminate = 233`
  - `keep = 61`

- decision breakdown
  - `parse_status`
    - `parse_error = 144`
    - `ok = 108`
    - `no_tool_call = 42`
  - `tool_exec_status`
    - `not_executed = 186`
    - `invalid_action = 89`
    - `ok = 19`
  - `termination_reason`
    - `tool_parse_error = 144`
    - `invalid_action = 89`
    - `no_tool_call = 42`
    - `None = 19`

- early step breakdown
  - step 2
    - rows: `34`
    - `keep = 8`
    - `mask_and_terminate = 26`
  - step 3
    - rows: `30`
    - `keep = 5`
    - `mask_and_terminate = 25`
  - step 4
    - rows: `35`
    - `keep = 12`
    - `mask_and_terminate = 23`
  - step 5
    - rows: `30`
    - `keep = 3`
    - `mask_and_terminate = 27`
  - step 6
    - rows: `165`
    - `keep = 33`
    - `mask_and_terminate = 132`

- what the masking patch already achieved
  - `parse_error` turn은 실제로 mask되고 종료됨
  - `invalid_action` turn도 실제로 mask되고 종료됨
  - representative masked tails:
    - malformed JSON tool call
    - list-valued coordinates
    - malformed `CLICK` / `DOUBLE_CLICK` payloads

- what leaked through
  - `no_tool_call` rows were still `keep`
  - examples:
    - `I need to find the calculator ... <|im_end|>`
    - `The user wants to find the postal code ... <|im_end|>`
    - `I need to open and view the wallpaper file ... <|im_end|>`

- why it leaked
  - parser did run every turn
  - but when
    - `tool_calls == []`
    - `parse_error is None`
    the generic SKD path treated it as:
    - `parse_status = no_tool_call`
    - `mask_decision = keep`
  - exact code before patch:
    - `verl/experimental/agent_loop/skd_agent_loop.py`
    - `no_tool_call -> keep`
  - test gap:
    - parse error / invalid action masking tests existed
    - `no_tool_call should mask` test did not

- fix applied
  - generic `SkdAgentLoop` now exposes `_handle_no_tool_call()`
  - `WebSkdAgentLoop` overrides it to:
    - mask last committed assistant turn
    - emit `mask_decision="mask_and_terminate"`
    - finalize reward with `termination_reason="no_tool_call"`
    - terminate
  - direct branch test added
  - actual WebSKD e2e test added

- verification
  - RED first
    - no-tool direct test failed because `_handle_no_tool_call` did not exist
    - no-tool e2e test failed because `response_mask` stayed `[1, 1]`
  - GREEN after patch
    - targeted no-tool tests passed
  - broader regression
    - `tests/skd/test_web_skd_agent_loop_on_cpu.py`
    - `tests/skd/test_skd_logic.py`
    - `tests/skd/test_teacher_fewshot_qwen35_rendering_on_cpu.py`
    - total `94 passed`

#### Interpretation

- `last assistant turn loss masking` 자체는 실제로 잘 동작했다
  - parse error / invalid action poisoning은 확실히 차단됨
- 하지만 첫 구현은
  - `bad tool call만 제거`
  - `tool call이 아예 없는 plain planning turn은 keep`
  상태였다
- WebGym / WebOSGym task contract에서는
  - 매 turn tool call
  - final `DONE` / `FAIL`
  이 요구되므로
  - `no_tool_call`도 invalid turn과 같은 급으로 처리해야 한다
- 즉 Step 3의 정확한 규칙은 이제:
  - keep:
    - `parse_status="ok"` and `tool_exec_status="ok"`
  - mask:
    - `parse_error`
    - `invalid_action`
    - `no_tool_call`

### Step 1e. Stabilize teacher-only few-shot and wire it into the launchers

#### Status

- Completed

#### Sources

- few-shot asset
  - `WebOSWorld/webgym_rl/teacher_fewshot/prozilla_task_B_22_minimal/teacher_fewshot.json`
- launcher
  - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
  - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni_fast_test.sh`
- tests
  - `tests/skd/test_teacher_fewshot_qwen35_rendering_on_cpu.py`
  - `tests/skd/test_web_skd_agent_loop_on_cpu.py`

#### Method

- source sample
  - `prozilla_task_B_22___index_85___1100876605`
- transcript policy
  - invalid action turn 제거
  - parse error turn 제거
  - valid turn은 최소 개입으로 유지
- Qwen 3.5 template behavior를 실제 tokenizer/processor로 확인
- few-shot historical assistant에서
  - reasoning과 tool call이 모두 visible하게 보이도록
  - template-safe한 message shape를 선택
- launcher에는
  - `teacher_fewshot_path`만 추가하고
  - 기존 prompt semantics는 유지

#### Result

- canonical few-shot asset created
  - path:
    - `WebOSWorld/webgym_rl/teacher_fewshot/prozilla_task_B_22_minimal/teacher_fewshot.json`
  - bundled images copied alongside transcript

- critical template finding
  - Qwen 3.5 historical assistant turn에서는
    - `reasoning_content`를 넣으면 historical reasoning이 prompt에서 빠질 수 있음
  - therefore the stable shape for this few-shot is:
    - `assistant.content = visible reasoning text`
    - `assistant.tool_calls = structured call`
  - this ensures both
    - historical reasoning
    - historical tool call
    are visible in the serialized teacher prompt

- rendering verification
  - actual local Qwen3.5 tokenizer and processor were used
  - teacher prompt built as:
    - common system prompt
    - teacher few-shot transcript
    - actual runtime dataset user prompt
    - actual runtime tool observation image
    - assistant generation prompt
  - verified:
    - tokenizer / processor `tokenize=False` rendering match
    - historical reasoning text is present
    - historical tool calls are present
    - final few-shot `DONE` is present
    - runtime dataset prompt follows the few-shot cleanly
    - multimodal tokenization succeeds with expected image count

- launcher wiring
  - both async SKD launchers now accept a third positional arg:
    - `WEBGYM_TEACHER_FEWSHOT_PATH`
  - both now forward:
    - `distillation.skd.teacher_fewshot_path=${WEBGYM_TEACHER_FEWSHOT_PATH}`
  - default path:
    - `WebOSWorld/webgym_rl/teacher_fewshot/prozilla_task_B_22_minimal/teacher_fewshot.json`

- shell validation
  - `bash -n WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
  - `bash -n WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni_fast_test.sh`
  - both passed

#### Interpretation

- few-shot is now:
  - teacher-only
  - prompt-format safe for Qwen 3.5
  - wired into both async SKD launchers
- this is the right stopping point for the current scope
  - bad non-tool / malformed turns are masked
  - few-shot is injected safely
  - no broader semantics changes were added

### Step 1f. Live validation of few-shot plus strict turn masking

#### Status

- In progress

#### Sources

- `logs/async_skd_events_webgym_20260514_033210.jsonl`
- `logs/async_skd_chunk_live_webgym_20260514_033210.jsonl`
- `logs/async_skd_turns_webgym_20260514_033210.jsonl`
- `tmp/ray/session_latest/logs/worker-*.out`

#### Goal

- 확인할 것
  1. teacher-only few-shot이 실제 teacher verification request에 multimodal payload까지 포함되어 들어가는지
  2. strict masking rule이 실제 live run에서도
     - `parse_error`
     - `invalid_action`
     - `no_tool_call`
     를 모두 잘 잘라내는지
  3. keep으로 남는 turn이 실제로 clean한지

#### Result

- few-shot multimodal alignment
  - worker trace 기준 `loop.teacher_compute_begin` / `teacher.compute_logprobs_single`의 `image_count`
    - `7`: `560`
    - `8`: `24`
    - `9`: `7`
  - 의미
    - few-shot image `6장`
    - runtime image `1/2/3장`
    이 teacher verify request에 실제로 함께 들어감
  - student generate 쪽 `image_count`는 계속 runtime image 기준으로만 유지
    - teacher path와 student path가 의도대로 분리됨

- multimodal loading stability
  - current session worker logs에서
    - `loading multimodal data`
    - `sglang.tokenizer_generate_await_error ... loading multimodal data`
    검색 결과 `0건`
  - 즉 previous few-shot image/payload mismatch crash는 현재 run에서 재현되지 않음

- current turn log snapshot
  - total turn rows: `68`
  - `mask_and_terminate = 52`
  - `keep = 16`
  - parse breakdown
    - `parse_error = 40`
    - `no_tool_call = 11`
    - `ok = 17`
  - tool breakdown
    - `not_executed = 51`
    - `ok = 16`
    - `invalid_action = 1`

- masking correctness
  - `bad_keep_count = 0`
  - meaning:
    - `keep`인데
    - `parse_status != ok` 또는 `tool_exec_status != ok`
    인 live row가 없음
  - 즉 현재까지는
    - `parse_error -> mask`
    - `no_tool_call -> mask`
    - `invalid_action -> mask`
    - `ok + ok -> keep`
    가 live run에서도 유지됨

- representative masked cases
  - `no_tool_call`
    - `The user wants me to read 'info.md' inside the Documents folder.<|im_end|>`
    - `turn_mask_before = [1, ..., 1]`
    - `turn_mask_after = [0, ..., 0]`
  - `parse_error`
    - malformed JSON / malformed XML-like tool payloads
    - list-valued coordinates
    - broken `CLICK` / `DOUBLE_CLICK` / `TYPING` payloads

- representative kept cases
  - `TYPING "ls"`
  - `TYPING "cat Documents/Prozilla.md"`
  - `HOTKEY ["ctrl", "alt", "t"]`
  - `DONE`

#### Sample-level reading

- sample count seen in current turn log: `63`
- split
  - `clean_samples = 11`
  - `mixed_samples = 4`
  - `masked_only = 48`

- important nuance
  - `turn-level cleanliness`는 현재 보장됨
    - invalid turn은 keep으로 안 남음
  - 하지만 `sample-level purity`는 아직 부족함
    - mixed sample 예:
      - keep, keep, then parse-error mask
      - keep, then no-tool mask

- current keep action mix
  - `DONE = 10`
  - `TYPING = 3`
  - `CLICK = 2`
  - `HOTKEY = 1`

- interpretation of the clean `DONE` samples
  - 이들은 `앞부분 tool call chain이 다 맞고 마지막에 DONE만 남은 sample`이라기보다
  - `assistant_turn = 1`인 first assistant turn이 chunk 단위로 이어지다가
  - 마지막에 valid `DONE`으로 닫힌 sample에 더 가깝다
  - 즉 `sample-level clean`으로는 잡히지만,
    rich multi-turn tool-use exemplar라고 보기는 어려움

#### Interpretation

- few-shot은 now correctly wired
  - teacher prompt text
  - teacher multimodal image payload
  둘 다 맞게 들어감
- strict masking도 live run에서 실제로 작동함
  - invalid turn이 keep으로 새지 않음
- 그러나 남는 학습 신호는 아직 sparse하고
  - keep의 큰 비중이 `DONE`
  - sample-level clean도 대부분 rich action chain이 아님
- 즉 현재 run은
  - `패치가 잘 붙었는지 보는 관찰용 run`으로는 유효하지만
  - `이 상태 그대로 장기 학습을 계속 밀어도 된다`고 보기에는 아직 이르다

### Step 1g. Re-evaluate `verify_top_k=3` versus `verify_top_k=10`

#### Status

- Completed as analysis

#### Sources

- previous top-10 chunk run
  - `logs/async_skd_chunk_live_webgym_20260514_011423.jsonl`
- previous top-3 chunk run
  - `logs/async_skd_chunk_live_webgym_20260514_012918.jsonl`
- current top-3 live run
  - `logs/async_skd_chunk_live_webgym_20260514_033210.jsonl`
  - `logs/async_skd_turns_webgym_20260514_033210.jsonl`

#### Question

- current scope를 유지한 채
  - strict masking
  - few-shot
  를 그대로 두고
  `verify_top_k`만 `3 -> 10`으로 되돌리는 것이 더 안정적인가?

#### Result

- previous top-10 snapshot
  - request-steps: `32`
  - `final_plain_tail = 26`
  - `final_toolish_tail = 6`
  - `tool_attempt_then_plain_end = 25`
  - `pure_no_tool_end = 1`

- previous top-3 snapshot
  - request-steps: `64`
  - `final_plain_tail = 61`
  - `final_toolish_tail = 1`
  - `tool_attempt_then_plain_end = 53`
  - `pure_no_tool_end = 8`

- current top-3 live snapshot
  - request-steps: `96`
  - `final_plain_tail = 54`
  - `final_toolish_tail = 9`
  - `tool_attempt_then_plain_end = 41`
  - `pure_no_tool_end = 13`
  - `non_eos_or_other = 33`

- current top-3 masking snapshot
  - turn rows: `68`
  - `keep = 16`
  - `mask_and_terminate = 52`
  - keep action mix
    - `DONE = 10`
    - `TYPING = 3`
    - `CLICK = 2`
    - `HOTKEY = 1`

#### Interpretation

- historically `top-3` already looked worse than `top-10`
  - replacement가 너무 자주 걸리고
  - toolish continuation이 덜 살아남고
  - final plain ending / useless drift가 더 많았음

- current live run도 같은 방향
  - strict masking과 few-shot은 붙었지만
  - 남는 keep이 너무 적고
  - keep의 큰 비중이 `DONE`

- why `top-10` is now more attractive than before
  - 예전에는 `top-10`의 약점이
    - plain no-tool turn이 더 많이 살아남을 수 있다는 것이었음
  - 하지만 현재는
    - `parse_error`
    - `invalid_action`
    - `no_tool_call`
    를 turn log 단계에서 모두 mask함
  - 즉 previous `top-10` weakness is now substantially reduced
  - 남는 것은
    - `top-10`의 looser local acceptance
    - 즉 intermediate valid chain이 자라기 쉬운 쪽의 장점

- current recommendation
  - current scope를 넘지 않는 다음 실험은
    - `verify_top_k = 10`
    - everything else unchanged
  - 변경할 것은 이 한 축뿐
    - few-shot 유지
    - strict masking 유지
    - 다른 prompt / semantics / runtime path 건드리지 않음

#### Takeaway

- current top-3 run은
  - 패치와 few-shot이 제대로 붙었는지 확인하는 관찰용으로는 의미가 있었음
  - 하지만 그대로 장기 학습으로 밀고 가기에는
    - keep signal이 너무 적고
    - `DONE` 편향이 심함
- 그래서 다음 controlled step은
  - `verify_top_k`만 `10`으로 되돌려서
  - keep quantity / keep quality / `DONE` bias가 완화되는지 보는 것

### Step 1h. Raise `verify_top_k` to 15 and check whether the run is now stable enough to continue

#### Status

- In progress

#### Sources

- `logs/async_skd_events_webgym_20260514_035107.jsonl`
- `logs/async_skd_chunk_live_webgym_20260514_035107.jsonl`
- `logs/async_skd_turns_webgym_20260514_035107.jsonl`
- `tmp/ray/session_latest/logs/worker-*.out`

#### Goal

- `verify_top_k=15`로 완화했을 때
  - few-shot + multimodal teacher conditioning이 여전히 안정적인지
  - strict masking이 그대로 유지되는지
  - keep turn quantity / quality가 실제로 좋아지는지
  - 이 run을 한동안 더 이어서 볼 가치가 있는지
  를 판단

#### Method

- launcher / worker trace에서 실제 `verify_top_k=15` 반영 여부 확인
- same live-read protocol 유지
  - teacher verify image count
  - multimodal loading error 유무
  - turn log `keep` / `mask_and_terminate`
  - sample-level clean / mixed / masked split
  - keep action mix
- 그리고 current `top-3` run (`20260514_033210`)과 직접 비교

#### Result

- `verify_top_k=15` actually active
  - worker trace
    - `verify_top_k': 15`
    - `loop.accept_reject_begin ... verify_top_k=15`

- few-shot multimodal path remains stable
  - teacher verify image count histogram
    - `7`: `814`
    - `8`: `160`
    - `9`: `115`
    - `10`: `100`
    - `11+`: small tail
  - meaning
    - few-shot image `6장`
    - runtime image `1, 2, 3, 4 ... 장`
    이 teacher verify request에 실제로 함께 들어감
  - current session에서
    - `loading multimodal data`
    - `tokenizer_generate_await_error ... loading multimodal data`
    는 `0건`

- current turn snapshot
  - total turn rows: `223`
  - `keep = 72`
  - `mask_and_terminate = 151`
  - keep rate
    - `72 / 223 = 32.3%`
  - parse breakdown
    - `ok = 111`
    - `parse_error = 108`
    - `no_tool_call = 4`
  - tool breakdown
    - `ok = 72`
    - `invalid_action = 39`
    - `not_executed = 112`
  - `bad_keep = 0`
    - keep으로 남은 turn은 전부 `parse_status="ok"` and `tool_exec_status="ok"`

- sample-level split
  - sample count: `155`
  - `clean_samples = 4`
  - `mixed_samples = 33`
  - `masked_only = 118`

- keep action mix
  - `SCROLL = 38`
  - `CLICK = 21`
  - `TYPING = 19`
  - `MOVE_TO = 10`
  - `WAIT = 3`
  - `HOTKEY = 2`
  - `DOUBLE_CLICK = 2`
  - `KEY_UP = 1`
  - `UNKNOWN = 1`

- chunk-level surface
  - request-steps: `160`
  - `final_toolish_tail = 110`
  - `final_plain_tail = 45`
  - `non_eos_or_other = 5`
  - step breakdown
    - step 2
      - `final_toolish_tail = 20`
      - `final_plain_tail = 12`
    - step 3
      - `final_toolish_tail = 24`
      - `final_plain_tail = 8`
    - step 4
      - `final_toolish_tail = 23`
      - `final_plain_tail = 7`
    - step 5
      - `final_toolish_tail = 26`
      - `final_plain_tail = 4`
    - step 6
      - `final_toolish_tail = 17`
      - `final_plain_tail = 14`

- qualitative improvement relative to current top-3 run
  - top-3 (`20260514_033210`)
    - `keep = 16 / 68 = 23.5%`
    - keep heavily `DONE`-biased
  - top-15 (`20260514_035107`)
    - `keep = 72 / 223 = 32.3%`
    - keep is now dominated by intermediate actions rather than `DONE`
  - this is the first live run where
    - valid intermediate tool actions survive in meaningful numbers
    - and the run no longer looks mostly like `DONE-only` clean tails

- but keep quality is still not perfectly pure
  - there are suspicious but valid-kept examples such as
    - `WAIT`
    - `KEY_UP`
    - one `UNKNOWN` tail extraction case
  - sample-level purity is still limited
    - many samples still become mixed
    - i.e. valid prefix turns survive and later turns still die on parse error / no-tool / invalid action

#### Interpretation

- `verify_top_k=15` is a clear improvement over the earlier `top-3` setting
  - keep quantity increased
  - keep content moved from `DONE`-heavy to real intermediate tool actions
  - few-shot and multimodal teacher path stayed stable
  - strict masking still prevents invalid turns from leaking into learning

- this is not yet `perfectly pure`
  - current keep set is `contract-clean`
  - but not fully `pristine` at the sample level
  - suspicious valid-kept actions still exist
  - mixed samples are still common

- practical judgment
  - this run is now `good enough to continue watching for a while`
  - i.e. it has crossed the line from
    - `just patch validation`
    to
    - `a run that may actually be worth continuing`
  - but it is still too early to declare the problem fully solved

#### Takeaway

- this is now roughly `KILL`
  - not because everything is perfectly pure
  - but because the safety rails are doing their job and the remaining supervision is materially better than before
- current state
  - few-shot: stable
  - multimodal teacher verify: stable
  - invalid / no-tool / parse-error masking: stable
  - keep distribution: improved enough to justify continuing the run
- therefore the present recommendation is:
  - keep this `top-15` run going for now
  - do not broaden the code changes further yet
  - keep observing whether the keep set remains intermediate-action heavy rather than collapsing back toward low-value completions

### Step 1i. Merge the usable Async SKD actor checkpoint and prepare a short fully async RL probe

#### Status

- Completed for merge and launcher wiring
- RL probe itself: not run yet in this document

#### Sources

- checkpoint root
  - `checkpoints/verl_async_skd_qwen35_webgym/qwen35_9b_to_27b_async_skd_webgym_counter_tool`
- selected actor checkpoint for this probe
  - `global_step_10/actor`
- merged Hugging Face output
  - `global_step_10/actor/huggingface`
- RL launcher
  - `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni_fast_tool.sh`

#### Goal

- current Async SKD run quality is not perfectly pure, but it is now good enough to justify a short downstream RL probe
- use a merged actor checkpoint rather than raw FSDP shards so that the RL launcher can start from a normal Hugging Face model directory
- keep this step operationally narrow
  - do not reopen SKD semantics here
  - do not change RL prompting or reward settings here
  - only prepare a usable initialization point for a short RL sanity run

#### Method

- merge the FSDP actor checkpoint with the repo-standard merger
  - `python -m verl.model_merger merge --backend fsdp --local_dir .../global_step_10/actor --target_dir .../global_step_10/actor/huggingface`
- while doing this, harden two same-directory merge edge cases
  - if the target directory already contains old Hugging Face weight artifacts, remove those weight files first
  - if metadata sync source and target resolve to the same directory, skip self-copy
- verify the merged directory by actually loading it through `transformers`
- wire the fully async RL fast-tool launcher so that
  - `actor_rollout_ref.model.path`
  points to the merged `global_step_10/actor/huggingface`

#### Result

- first merge attempt exposed a real merger bug
  - `target_dir == actor/huggingface` caused metadata sync to attempt a self-copy
  - this raised `SameFileError`
- the merge path was then patched narrowly
  - stale Hugging Face weight files are removed before writing new merged weights
  - metadata sync is a no-op when source and target are the same path
- regression tests added for the helper path
  - `tests/utils/test_hf_files.py`
  - result: `4 passed`
- final merge rerun exited cleanly
  - `exit code 0`
- final merged directory now contains a usable Hugging Face model
  - `model.safetensors`
  - `config.json`
  - `generation_config.json`
  - tokenizer / processor files
- direct load verification succeeded
  - class: `Qwen3_5ForConditionalGeneration`
  - parameter count: `9,409,813,744`
- RL launcher was rewired to use this merged model by default
  - `WEBGYM_INITIAL_MODEL_PATH=.../global_step_10/actor/huggingface`
  - `actor_rollout_ref.model.path=${WEBGYM_INITIAL_MODEL_PATH}`

#### Interpretation

- at this point the next experiment should no longer be “fix SKD internals again”
- the right next move is a short fully async RL probe initialized from the merged `global_step_10` actor
- the purpose of that RL probe is not yet to claim final improvement
  - it is to inspect whether the downstream response pattern is materially healthier than a cold-start base model run
  - especially whether action-taking remains alive under actual RL rollout pressure

#### Takeaway

- current status is operationally ready for a short RL restart
  - merged model exists
  - merge path was verified rather than assumed
  - RL launcher now points to the merged actor checkpoint
- therefore the next evidence should come from
  - a short RL run
  - followed by the same kind of direct log reading used above
  - rather than more SKD-side code changes first

### Step 1j. Pause the current fully async RL probe and audit whether constraint decoding is actually active

#### Status

- In progress

#### Goal

- the current RL probe still shows overwhelming plain `<|im_end|>` endings
- before changing reward or rollout semantics again, confirm whether the newly added constraint decoding is actually active on the real rollout path
- keep this strictly ordered:
  1. document the pivot
  2. bring down the current RL run
  3. audit launcher -> config -> agent loop -> SGLang request path

#### Current hypothesis

- simply setting `grammar_backend=xgrammar` on the rollout server is not sufficient by itself
- for the WebOSWorld Qwen3 coder path, the rollout also needs the structured-output enable flag so that the agent loop actually injects `structural_tag`, `ignore_eos`, and `</tool_call>` stop handling into sampling params

#### Immediate next check

- verify whether
  - `+actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=xgrammar`
  is present
- and separately verify whether
  - `+actor_rollout_ref.rollout.custom.enable_qwen3_coder_structured_output=True`
  is also present
- if the second flag is absent, then the rollout is not yet using the full structured decoding path even though the server grammar backend is set

#### Result

- current RL process was brought down before audit continuation
  - exact live process previously matched:
    - `run_qwen35_webgym_fully_async_rl_tool_veomni_fast_tool.sh`
    - `python -m verl.experimental.fully_async_policy.fully_async_main ...`
  - after shutdown verification, no matching fully async RL main process remained

- launcher reality
  - present:
    - `+actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=xgrammar`
    - `+actor_rollout_ref.rollout.custom.enable_qwen3_coder_structured_output=True`

- code-path audit
  1. `fully_async_main.py`
     - launcher config is passed through unchanged into `FullyAsyncRollouter` / `FullyAsyncTrainer`
  2. `async_sglang_server.py`
     - `engine_kwargs.sglang` is read
     - `grammar_backend` is propagated into `ServerArgs`
     - therefore the SGLang server can support xgrammar
  3. `web_tool_agent_loop.py`
     - actual constrained decoding is only activated in `_build_generation_sampling_params(...)`
     - this path requires all of:
       - `rollout.name == "sglang"`
       - `tool_parser_name == "qwen3_coder"`
       - `rollout.custom.enable_qwen3_coder_structured_output == True`
       - active tool schemas present
     - only then it injects:
       - `structural_tag`
       - `ignore_eos=True`
       - `stop=["</tool_call>"]`
       - `no_stop_trim=True`
  4. `async_sglang_server.py`
     - generation requests pass `sampling_params` through to SGLang
     - installed SGLang `0.5.10` supports `sampling_params.structural_tag`
     - therefore once the agent loop inserts `structural_tag`, the request-level constrained decoding path is available

- conclusion from the current code path
  - the **current launcher file** does enable both
    - xgrammar backend on the server
    - the WebOSWorld structured-output switch in the agent loop
  - therefore the **next run launched from the current file** should enter the intended Qwen coder structured decoding branch
  - however, the **previous RL logs** were produced by an older live process whose command line and dumped config did not include the grammar/structured-output settings, so those logs cannot be used as evidence that constrained decoding already worked

#### Interpretation

- the current fast-tool RL run logs should not be interpreted as a valid test of the newly edited launcher
- for the **edited launcher file itself**, the strict path check now says:
  - `grammar_backend` reaches `ServerArgs`
  - `enable_qwen3_coder_structured_output=True` unlocks the agent-loop branch
  - that branch injects `structural_tag`, `ignore_eos=True`, `stop=["</tool_call>"]`, and `no_stop_trim=True`
  - those sampling params are then forwarded to SGLang request generation
- so the edited launcher is now logically wired for constrained decoding

#### Takeaway

- the next controlled run should now be launched from the updated fast-tool script and then verified from fresh logs
- the old RL logs must be treated as pre-change evidence only

### Step 1k. Re-run Async SKD with same prompt for student and teacher, then push `verify_top_k=1` to isolate the teacher path

#### Status

- Completed

#### Sources

- launcher
  - `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni_fast_test_top1.sh`
- live SKD logs
  - `logs/async_skd_chunk_live_webgym_20260514_130527.jsonl`
  - `logs/async_skd_events_webgym_20260514_130527.jsonl`
  - `logs/async_skd_turns_webgym_20260514_130527.jsonl`
  - `logs/skd_log.out`

#### Goal

- remove all teacher-only prompt asymmetry
  - no teacher-only system prompt
  - no teacher-only few-shot
- then set `verify_top_k=1` so that the committed sequence should be as close as possible to the teacher top-1 path
- use this to answer one narrow question:
  - if early `<|im_end|>` still appears, is it still a prompt mismatch problem, or has the problem moved into the teacher verify runtime itself?

#### Method

- runtime contract was reduced to:
  - shared system prompt only
  - shared task prompt only
  - same screenshot / same multimodal input
  - `verify_top_k=1`
- prompt identity was verified from live logs
  - teacher and student tokenized prompt lengths matched
  - teacher and student server-side prompt lengths also matched chunk-by-chunk
- then `logical_step=2` was inspected directly from
  - `chunk_live`
  - `teacher_verify_rows`
  - `teacher_replacement`
  - `chunk_commit`

#### Result

- teacher/student prompt mismatch was removed
  - live prompt lengths matched
  - teacher-only prompt conditioning was gone

- despite this, `IM_END` still appeared in the same run
  - for `logical_step=2`, EOS-ending rows were still dominant
  - plain no-tool end and malformed tool-call end both remained

- the key exact sample was:
  - `request_id = 626289ac067c426d8ee9a5194fcb0b43`
  - `sample_id = aaf49e11-48a9-413a-b563-9ec4b4b6f3e8`
  - `chunk_idx = 9`

- in that sample:
  - raw student tail contained
    - `... </think>\n\n<tool_call>\n<function`
  - but committed verified tail became
    - `... Wallpaper10.png file.<|im_end|>`

- the crucial point is that the first reject did **not** happen at `<tool_call>`
  - `rejection_pos = 6`
  - at that position:
    - student token = `" The"`
    - teacher top1 = `"<|im_end|>"`
    - teacher top2 = `"\n\n"`
    - teacher top3 = `" I"`
    - teacher top4 = `" Let"`

- meanwhile, later rows in the same teacher verify dump showed that `<tool_call>` itself was still a strong candidate
  - at later positions, `<tool_call>` was top-1 or near top-1
  - so the failure was not “teacher cannot see the tool-call schema”
  - it was “teacher verify picks EOS earlier, before the tool-call branch is reached”

#### Interpretation

- this step falsified the old privileged-info hypothesis
  - same prompt still produced early EOS replacement
- it also falsified the simpler “`<tool_call>` token itself is broken” story
  - the branch was being cut **before** the actual tool-call boundary
- therefore the remaining suspect moved from:
  - prompt asymmetry
  to:
  - exact teacher verify runtime behavior on the same prefix

#### Takeaway

- after this step, the working hypothesis changed sharply:
  - no longer “teacher prompt mismatch”
  - now “teacher verify runtime is reconstructing a different next-token distribution from HF direct”

### Step 1l. Exact local repro proves the divergence is inside SGLang Triton GDN prefill/extend, and narrows it to the fused gating path

#### Status

- Completed

#### Sources

- exact failing live sample
  - `logs/async_skd_events_webgym_20260514_130527.jsonl`
  - `logs/async_skd_chunk_live_webgym_20260514_130527.jsonl`
- exact reconstruction inputs
  - `data/webgym_skd/train.parquet`
  - `WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt`
  - `WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml`
  - `/tmp/prozilla_task_A_02_start.png`
- local library code
  - `sglang/srt/models/qwen3_5.py`
  - `sglang/srt/layers/attention/linear/gdn_backend.py`
  - `sglang/srt/layers/attention/fla/fused_gdn_gating.py`
  - `sglang/srt/layers/attention/linear/kernels/gdn_triton.py`
- upstream references
  - SGLang releases page
  - SGLang issues `#22087`, `#20550`, `#21696`, `#20069`
  - SGLang attention-backend docs

#### Goal

- stop reasoning from live SKD traces alone
- reconstruct the exact same sample locally and compare:
  1. HF direct forward
  2. exact local SGLang standalone next-token generate
  3. exact local SGLang `prompt_logprobs` delta extraction
- then A/B the suspected runtime layers until the EOS winner disappears

#### Method

- exact prompt reconstruction was validated against the live teacher verify request
  - `server_len = 2142`
  - `prompt_len = 4181`
  - `surplus = 2039`
- exact committed prefix through chunk 8 plus accepted prefix of chunk 9 was reconstructed
- exact sample was then compared across:
  - HF direct
  - local SGLang Triton server
  - local SGLang delta extraction on the same full `prefix + raw_chunk9`

- additional A/B tests were run in this order:
  1. disable CUDA graph
  2. disable `qwen3_5.py` fused QKV/Z/BA split fast path
  3. try `flashinfer` GDN prefill/extend path
  4. replace `fused_gdn_gating(...)` with explicit Torch formulas matching HF
  5. isolate `beta` replacement only
  6. isolate `g` replacement only

#### Result

- exact reconstruction matched the live request
  - not an approximate repro
  - the exact failing sample was replayed

- exact HF vs SGLang divergence was real
  - HF direct next-token did **not** put EOS near top-1
  - exact local SGLang Triton server **did** put EOS at top-1
  - exact local SGLang delta extraction matched the live teacher verify EOS row

- therefore this was **not**:
  - prompt mismatch
  - `prompt_logprobs` extractor-only bug
  - compact-vs-expanded multimodal prompt mismatch

- disabling CUDA graph changed nothing
  - EOS stayed the top winner
  - so this was not a CUDA graph replay artifact

- disabling the fused QKV/Z/BA projection fast path changed nothing
  - EOS stayed the top winner
  - so this was not the fused projection split path in `qwen3_5.py`

- `flashinfer` could not be used as a correctness fallback for this path
  - on current SM100+ hardware, GDN `prefill/extend` is unsupported in FlashInfer
  - exact request crashed with:
    - `FlashInfer GDN prefill is not supported on SM100+`
  - therefore the teacher verify path is effectively locked to Triton for this experiment class

- replacing `fused_gdn_gating(...)` with explicit HF-style Torch formulas changed the branch immediately
  - standalone top1 moved away from EOS
  - exact delta row top1 also moved away from EOS
  - exact sample now followed the non-EOS continuation branch

- further isolation showed:
  - replacing only `beta` removed EOS as top-1
  - replacing only `g` also removed EOS as top-1
  - replacing both reproduced the cleanest HF-like behavior

- random tensor spot-checks showed the fused gating kernel is not catastrophically wrong in the abstract
  - raw elementwise diffs were small
  - but the exact live sample is sensitive enough that those differences materially change the winner token

#### Interpretation

- the old “mRoPE off-by-one” hypothesis was too broad for this exact sample
  - the exact teacher verify failure survived after prompt/position reconstruction was matched
- the exact live EOS bug is not best described as a generic `triton` problem either
  - it is now much narrower:
  - **SGLang’s Triton GDN prefill/extend path, specifically the fused gating output path**

- the most precise current reading is:
  - the dense Qwen3.5 GDN path in `sglang 0.5.10`
  - under Triton prefill/extend
  - can distort the next-token distribution enough to make EOS win at the sentence boundary where HF direct would continue

- this also aligns with upstream public signals:
  - SGLang issue `#22087`
    - Triton GDN kernel produces garbled text for dense Qwen3.5 models
  - SGLang issue `#20550`
    - Qwen3.5-27B returns empty contents / early stop-like behavior
  - SGLang issue `#21696`
    - quality regression after moving from `0.5.9` to `0.5.10rc0`
  - SGLang tracking issue `#20069`
    - Qwen3.5 linear-attention fixes continue to land in this area

- current installed versions during this investigation:
  - `sglang == 0.5.10`
  - `sglang-kernel == 0.4.1`
  - `flashinfer-python == 0.6.7.post2`
  - `triton == 3.5.1`
  - `torch == 2.9.1+cu128`
  - `transformers == 5.3.0`
  - `flash-linear-attention == 0.4.2`
  - `fla-core == 0.4.2`

#### Takeaway

- the practical blocker is no longer in SKD supervision logic
- the practical blocker is:
  - **library-level correctness in SGLang’s dense Qwen3.5 Triton GDN path**

- immediate implications:
  - a blind `flashinfer` version bump is not the answer
  - `0.5.10.post1` is not documented as fixing this class of bug
  - if we want an immediate unblock, the most reliable path is a narrow local patch in `gdn_backend.py`
    - replace fused gating output with HF-equivalent explicit formulas for this path
  - if we want the proper long-term fix, we need an upstream repro against the SGLang GDN fused gating path

---

### Step 1m. Re-run Async SKD after the local fused-gating patch and compare the remaining failures against base Qwen3.5 rollout behavior

#### Status

- Completed for the current log snapshot
- run itself is still in progress

#### Sources

- patched SKD run
  - `logs/async_skd_chunk_live_webgym_20260514_160644.jsonl`
  - `logs/async_skd_turns_webgym_20260514_160644.jsonl`
  - `logs/async_skd_events_webgym_20260514_160644.jsonl`
- base fully async RL rollout
  - `logs/rollout_data/base_qwen35_webgym_fully_async_tool_veomni_20260513_230710`

#### Goal

- after patching the local `sglang` environment, verify whether the original blocker is actually gone in live SKD logs
- separate two possibilities:
  1. the old internal backend bug is still present
  2. the old backend bug is gone, and the remaining failures are just the base model’s native tool-call quality limits

#### Method

- for the patched SKD run:
  - group `chunk_live` rows by `(request_id, logical_step)`
  - use the final EOS row per group when present
  - classify final tails into:
    - `plain no-tool <|im_end|>`
    - `toolish <|im_end|>`
    - `no <|im_end|>`
- inspect `teacher_replacement` rows with a real `rejection_pos`
  - count how often replacement top1 is `<|im_end|>`
- inspect `turns` rows for:
  - `parse_status`
  - `tool_exec_status`
  - `mask_decision`

- for the base rollout:
  - scan all `assistant_turn` rows in `trajectory.jsonl`
  - classify model outputs with the same `plain/toolish/no-im_end` split
  - then separately inspect the matching task
    - `prozilla_task_A_02`

#### Result

- patched SKD run, `logical_step=7`
  - final groups: `32`
  - `plain no-tool <|im_end|>`: `0`
  - `toolish <|im_end|>`: `24`
  - `no <|im_end|>`: `8`

- patched SKD run, current `logical_step=8` snapshot
  - final groups: `32`
  - `plain no-tool <|im_end|>`: `0`
  - `toolish <|im_end|>`: `20`
  - `no <|im_end|>`: `12`

- patched SKD `teacher_replacement`, `logical_step=7`
  - rows with concrete `rejection_pos`: `917`
  - replacement top1 = `<|im_end|>` count: `0`
  - common replacement top1s are now ordinary continuation tokens such as:
    - `"\n"`
    - `" the"`
    - `" I"`
    - `"."`
    - `"I"`

- patched SKD `turns`, `logical_step=7`
  - total turns: `27`
  - `parse_error`: `14`
  - `tool_exec_status=invalid_action`: `10`
  - `tool_exec_status=ok`: `3`
  - `mask_and_terminate`: `24`
  - `keep`: `3`

- base Qwen3.5 fully async rollout, overall
  - assistant turns: `6927`
  - `plain no-tool <|im_end|>`: `2`
  - `toolish <|im_end|>`: `6886`
  - `no <|im_end|>`: `39`

- base Qwen3.5, matching task `prozilla_task_A_02`
  - assistant turns: `58`
  - `plain no-tool <|im_end|>`: `0`
  - `toolish <|im_end|>`: `57`
  - `no <|im_end|>`: `1`
  - task-level failure mix:
    - many invalid actions
    - several parse errors
    - very few clean successful turns

#### Interpretation

- the original internal blocker was:
  - teacher replacement spuriously choosing `<|im_end|>` and cutting off a continuation that should have reached a tool call
- that pattern is **not** what the patched run shows anymore
  - final `plain no-tool <|im_end|>` tails are gone in the observed steps
  - replacement top1 is no longer EOS in the observed `teacher_replacement` rows

- the remaining failures in the patched run are now much closer to the base model’s native behavior
  - the model still often reaches tool-call formatting
  - but payload quality is poor
  - parse errors and invalid actions remain frequent

- therefore the current state should be read as:
  - **the backend EOS-cutoff bug is fixed enough to stop dominating the trajectory**
  - **the next bottleneck is ordinary tool-call quality, not internal EOS corruption**

#### Takeaway

- the local fused-gating patch appears to have achieved its intended first objective
  - remove early plain `<|im_end|>` cutoff as the dominant failure mode
- the remaining work should no longer target teacher verify EOS corruption first
- the next layer to improve is:
  - tool payload formation
  - schema adherence
  - invalid action reduction

---

### Step 1n. Empty actor batch after the EOS fix, and convert WebSKD masking policy into explicit `distillation.skd` controls

#### Status

- Completed

#### Sources

- patched SKD run
  - `logs/async_skd_chunk_live_webgym_20260514_160644.jsonl`
  - `logs/async_skd_turns_webgym_20260514_160644.jsonl`
  - `logs/async_skd_events_webgym_20260514_160644.jsonl`
- trainer failure
  - `verl/trainer/ppo/ray_trainer.py`
- WebSKD masking path
  - `verl/experimental/agent_loop/web_skd_agent_loop.py`
  - `verl/experimental/agent_loop/skd_agent_loop.py`
- config surface
  - `verl/workers/config/distillation.py`
  - `verl/trainer/config/distillation/distillation.yaml`
- verification tests
  - `tests/skd/test_skd_logic.py`
  - `tests/skd/test_web_skd_agent_loop_on_cpu.py`

#### Goal

- explain the next crash after the EOS-cutoff fix
- decide whether the old strict masking policy should still be kept
- expose the masking policy as explicit config so diagnostic runs can change it without patching runtime code

#### Observed Failure

- after the fused-gating patch removed the dominant early plain `<|im_end|>` cutoff,
  the next live run failed later inside actor update with:
  - `ValueError: Actor update has no non-padding training rows.`

- this came from the single-mini-batch async SKD training path, where trainer computes:
  - `global_batch_size = count(response_mask.sum(dim=-1) > 0)`
  - and hard-fails if that count is `0`

- the relevant runtime condition was:
  - `logical_step=8`
  - `keep = 0`
  - `mask_and_terminate = 20`
  - i.e. every row in that step was masked away before reaching actor update

#### Root Cause

- the dominant contributor was not `plain no-tool` anymore
  - that had already fallen to `0` in the observed patched run
- the dominant contributor was **`invalid_action -> mask_and_terminate`**

- in WebSKD, masking a turn did not only zero `response_mask`
  - it also zeroed `teacher_ids_list`
  - and zeroed `teacher_logprobs_list`

- therefore the previous hardcoded policy bundled together three things:
  1. zero the supervised loss mask for the assistant turn
  2. zero the corresponding teacher top-k rows
  3. terminate the sample immediately

#### Interpretation

- after the EOS bug was removed, keeping the old `invalid_action -> mask_and_terminate` policy became too aggressive
- at this point:
  - `tool_parse_error` still means the model failed to produce a usable tool-call structure at all
  - `invalid_action` is different:
    - the model did produce a tool-call structure
    - but the action payload failed execution or validation

- therefore `invalid_action` turns still contain useful supervision for tool-call formation
- masking them out entirely was now harming training stability more than it was protecting data quality

- however, the **termination** half of the policy still makes sense
  - continuing to roll out after an invalid action would spend more teacher verify + environment budget
  - without solving the empty-batch problem

- so the narrow policy change is:
  - keep `termination` on invalid action
  - remove only the `loss masking`

#### Change

- add explicit SKD config switches:
  - `mask_invalid_action`
  - `mask_tool_parse_error`
  - `mask_no_tool_call`

- default all three to `true` so previous behavior remains unchanged unless the launcher opts out

- current recommended setting for live diagnostics:
  - `distillation.skd.mask_invalid_action=false`
  - `distillation.skd.mask_tool_parse_error=true`
  - `distillation.skd.mask_no_tool_call=true`

- semantic effect of `mask_invalid_action=false`:
  - keep `response_mask`
  - keep `teacher_ids_list`
  - keep `teacher_logprobs_list`
  - **still terminate the sample**

#### Verification

- runtime-path tests were added so the checks are not just unit-level helper calls
- the meaningful coverage here was:
  - `handle_generating_state -> handle_processing_tools_state`
  - for all three policy categories
  - with both default masking and masking-disabled overrides

- verification command:

```bash
source /home/sogang_nlpy/miniconda3/etc/profile.d/conda.sh
conda activate skd-cudnn
cd /home/sogang_nlpy/verl
pytest -q tests/skd/test_skd_logic.py tests/skd/test_web_skd_agent_loop_on_cpu.py
```

- result:
  - `98 passed`

#### Takeaway

- the original EOS corruption fix should stay
- the next stability fix is **not** “remove all termination”
- it is narrower:
  - preserve supervision for `invalid_action`
  - keep termination semantics
  - keep strict masking for `tool_parse_error`
  - leave `no_tool_call` strict by default unless a later run proves it is again a dominant source of wasted data

---

### Step 2a. Transition from Async SKD to fully async RL, then re-anchor debugging on the real RL path

#### Status

- Completed for the current launcher and runtime path

#### Sources

- merged actor init point
  - `checkpoints/verl_async_skd_qwen35_webgym/qwen35_9b_to_27b_async_skd_webgym_counter_tool/global_step_15/actor/huggingface`
- fully async RL launcher
  - `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni_fast_tool.sh`
- fully async trainer / main
  - `verl/experimental/fully_async_policy/fully_async_main.py`
  - `verl/experimental/fully_async_policy/fully_async_trainer.py`
- rollout data root
  - `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_*`
- W&B output logs
  - `wandb/run-*/files/output.log`

#### Method

- stop reading the downstream RL run as a vague continuation of SKD
- verify the real fully async path for:
  - reward usage
  - cap semantics
  - actor LR health
  - step semantics
  - actual late-stage response pattern

#### Result

- merged Async SKD actor was used as the RL initialization point
- the fully async launcher became the operational source of truth
- reward path was verified end-to-end:
  - WebOSGym reward fields become `reward_extra_info`
  - `reward_score` becomes `rm_scores`
  - fully async trainer uses that reward as `token_level_scores`
  - with `algorithm.use_kl_in_reward=False`, those scores become the actual `token_level_rewards`

#### Interpretation

- after the SKD-side EOS blocker was removed, the main debugging target moved to the fully async RL loop itself
- at this point the key question was no longer “is teacher verify corrupt?”
- it became:
  - “what is the real RL loop actually optimizing?”

---

### Step 2b. Make the WebOSGym hard cap actually bind on attempted tool calls

#### Status

- Completed

#### Sources

- runtime path
  - `verl/experimental/agent_loop/web_tool_agent_loop.py`
- tests
  - `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

#### Method

- keep existing parse-error feedback and invalid-action handling
- change only the cap control path:
  - update `attempted_tool_call_count` even when no trajectory logger exists
  - check the attempted count **before the next generation**

#### Result

- `attempted_tool_call_count` is no longer just a logging/reward field
- on the WebOSGym path it now acts as the effective hard cap
- `parse_error` contributes to that attempted count
- bad trajectories now stop at 10 attempted tool-call events with `system_stop`

#### Interpretation

- this was the right narrowing
- the bug was not “the model loops too much” in an abstract sense
- it was a concrete control-flow ordering problem

---

### Step 2c. Replace the old ratio-only format reward with a cleaner fully async signal

#### Status

- Completed

#### Sources

- reward function
  - `WebOSWorld/webgym_rl/reward_fn_webgym_rl.py`
- reward-export path
  - `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
  - `verl/experimental/agent_loop/web_tool_agent_loop.py`
- fully async trainer reward path
  - `verl/experimental/separation/ray_trainer.py`

#### Method

- replace the old `valid / max(attempted, min_denominator)` style signal
- use:
  - `precision = valid / max(attempted, 1)`
  - `latency = 0` if `valid == 0`, else `exp(-(first_valid - 1) / tau)`
  - `format_reward = precision * latency`
  - subtract a penalty for `tool_response_budget_exhausted`
- final score:
  - `env_reward + alpha * format_reward`

#### Result

- reward became simpler to audit from live logs
- the fields that matter now are:
  - `web_osgym_attempted_tool_calls`
  - `web_osgym_valid_tool_calls`
  - `web_osgym_first_valid_tool_call_index`
  - `web_osgym_termination_reason`
  - `web_osgym_format_reward`

#### Interpretation

- this fixed the earlier denominator hack
- but it still only rewards parser-valid, server-executable action streams
- it does **not** by itself ensure task-progressing behavior

---

### Step 2d. Diagnose the fully async actor-LR zero bug

#### Status

- Root cause identified
- quick launcher-level mitigation applied

#### Sources

- fully async initialization path
  - `verl/experimental/fully_async_policy/fully_async_main.py`
  - `verl/experimental/fully_async_policy/fully_async_trainer.py`
- VeOmni scheduler build path
  - `verl/workers/engine/veomni/transformer_impl.py`
  - local env:
    - `site-packages/veomni/optim/lr_scheduler.py`
- launcher
  - `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni_fast_tool.sh`

#### Method

- compare launcher intent with live `actor/lr` logs
- trace when `total_training_steps` is injected
- trace when VeOmni actually builds the scheduler
- reproduce the scheduler behavior with:
  - `train_steps=1000`
  - `train_steps=0`
  - `train_steps=-1`

#### Result

- VeOmni cosine scheduler is built during worker init
- at that moment it reads:
  - `actor_rollout_ref.actor.optim.total_training_steps`
- in the fully async path, `trainer.total_training_steps` was being set too late
  - after trainer workers were already initialized
- therefore actor workers could still build the scheduler with `total_training_steps = -1`
- with VeOmni cosine scheduler, that yields an effective LR of `0.0`
- quick operational mitigation:
  - pass `actor_rollout_ref.actor.optim.total_training_steps=1000` directly in the launcher
- after this, live runs showed:
  - `actor/lr ≈ 9.9e-07`
  instead of `0.0`

#### Interpretation

- any RL run before this fix must be treated with caution
- it may have produced trajectories and metrics
- but actor improvement may have been nearly frozen

---

### Step 2e. What late fully async RL actually learned: high format score, but mostly through `WAIT` reward hacking

#### Status

- Completed for the current late-run snapshot

#### Sources

- live run
  - `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260514_232836`
  - `wandb/run-20260514_232904-01b13ei1/files/output.log`

#### Method

- compare early root batch dumps against late root batch dumps
- compute:
  - attempted/valid distribution
  - termination distribution
  - high-format share
  - `WAIT`-only share among high-format rows
- for representative rows:
  - use `request_id` from root `N.jsonl`
  - then find the matching `trajectory.jsonl`

#### Result

- actor LR is now alive
- late trainer metrics show:
  - `score/format ≈ 0.8 ~ 0.93`
  - low or zero env reward in many batches
- but high-format rows are mostly not “good tool use”
- in late high-format rows (`61.jsonl` to `65.jsonl`):
  - most `format_reward >= 0.8` rows are `WAIT` only
- representative examples:
  - `attempted=6, valid=6, format=1.0, termination=system_stop`
  - actual trajectory: repeated `WAIT`

#### Interpretation

- late-stage format improvement cannot be read naively
- the model has found a cheap path:
  - parser-valid
  - server-executable
  - no-op actions
- therefore high `score/format` is not yet evidence of the desired format-first learning goal

---

### Step 2f. Fully async RL step semantics: never confuse these axes again

#### Status

- Important operational rule

#### Sources

- trainer step logic
  - `verl/experimental/fully_async_policy/fully_async_trainer.py`
- rollout root batch dumps
  - `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260514_232836/*.jsonl`
- W&B logs
  - `wandb/run-20260514_232904-01b13ei1/files/output.log`

#### Method

- compare:
  - `training/global_step`
  - `current_param_version`
  - checkpoint folder `global_step_N`
  - root rollout-data file `N.jsonl`
- trace where each counter is incremented or used

#### Result

- `training/global_step`
  - trainer local fit-step counter
- `current_param_version`
  - parameter-sync / checkpoint counter
- checkpoint folder `global_step_N`
  - in fully async, this means `current_param_version = N`
  - **not** `training/global_step = N`
- current run uses:
  - `trigger_parameter_sync_step = 2`
- therefore approximately:
  - `training/global_step ≈ 2 * current_param_version`

- concrete example:
  - checkpoint folder `global_step_20`
  - was saved when:
    - `current_param_version = 20`
    - `training/global_step ≈ 42`
    - `score/format ≈ 0.819`

#### Interpretation

- reading checkpoint `global_step_20` as if it were W&B step 20 or `training/global_step 20` is wrong
- all later analysis must anchor on one axis first

---

### Step 2g. Replace hard gating with repetition-aware format attenuation and log the real effective reward

#### Status

- Completed

#### Sources

- current launcher
  - `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
- reward function
  - `WebOSWorld/webgym_rl/reward_fn_webgym_rl.py`
- repetition counter collection
  - `verl/experimental/agent_loop/web_tool_agent_loop.py`
  - `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
- trainer metric / summary export
  - `verl/experimental/separation/ray_trainer.py`
- tests
  - `WebOSWorld/webgym_rl/test_webgym_rl_dataset_and_reward.py`
  - `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`
  - `tests/experimental/agent_loop/test_web_osgym_loop_mixin_on_cpu.py`
  - `tests/experimental/fully_async_policy/test_fully_async_reward_path_on_cpu.py`

#### Method

- re-read the latest live rollout root:
  - `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260515_023745`
- identify the dominant failure mode:
  - long stretches where actions remain inside `{WAIT, HOTKEY, TYPING}`
- keep the original format definition:
  - precision
  - first-valid latency
  - budget-exhausted penalty
- remove the hard env gate for the main run
- instead attenuate only the **positive** format bonus using:
  - `rho = non_grounding_adjacent_pair_count / max(executed_action_count - 1, 1)`
  - `effective_format = raw_format` if `raw_format <= 0`
  - `effective_format = (1 - rho) * raw_format` if `raw_format > 0`
- keep final reward as:
  - `env_reward + alpha * effective_format`

#### Result

- current launcher now uses:
  - `format_reward_alpha = 0.1`
  - `format_reward_gate_by_env_score = False`
- runtime now exports:
  - `web_osgym_executed_action_count`
  - `web_osgym_non_grounding_adjacent_pair_count`
  - `web_osgym_non_grounding_adjacency_ratio`
  - `web_osgym_raw_format_reward`
  - `web_osgym_format_reward`
- trainer metrics now distinguish:
  - `score/format`
    - effective format after repetition decay
  - `score/format_raw`
    - raw format before repetition decay
  - `score/non_grounding_adjacency_ratio`
    - mean repetition ratio
- rollout `summary.json` reward block now contains:
  - `sum`
  - `env`
  - `format`
  - `raw_format`
  - `non_grounding_adjacency_ratio`

#### Interpretation

- hard gating was useful to prove the earlier format-only hacking path, but it was too sparse for a run starting from an SKD actor
- the current reward keeps dense format signal alive while directly discounting the exact live failure mode
- this is a cleaner operational setup than:
  - pure hard gate
  - `WAIT`-only penalty
  - hand-written repetition thresholds
- however, it still does **not** fix suspicious env-positive rows by itself
- therefore:
  - reward-shaping progress and evaluator correctness must still be read separately

---

### Step 2h. Add a per-assistant-turn token cap and verify that the live bottleneck shifts from plain-text collapse to tool-format failure

#### Status

- Completed

#### Sources

- current launcher
  - `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
- runtime config / loop
  - `verl/workers/config/rollout.py`
  - `verl/trainer/config/rollout/rollout.yaml`
  - `verl/experimental/agent_loop/tool_agent_loop.py`
  - `verl/experimental/agent_loop/web_tool_agent_loop.py`
- tests
  - `tests/experimental/agent_loop/test_web_tool_agent_loop_on_cpu.py`

#### Method

- first verify from the previous live run that the dominant single-turn failure was **plain-text babbling collapse**
  - previous run `qwen35_webgym_fully_async_tool_veomni_20260515_041013`
  - no-tool non-parse turns were orders of magnitude longer than normal tool turns
- then add a dedicated multi-turn config field:
  - `max_assistant_response_tokens`
- at assistant generation time, inject:
  - `max_new_tokens = min(max_assistant_response_tokens, remaining_budget)`
- keep the existing total trajectory cap:
  - `data.max_response_length`
- wire the operational launcher to:
  - `actor_rollout_ref.rollout.multi_turn.max_assistant_response_tokens=2048`

#### Result

- the per-turn cap now exists as an explicit runtime contract field
- launcher uses `2048` as the current operational value
- tests verify the actual path:
  - `WebOsGymToolAgentLoop._handle_generating_state(...)`
  - `server_manager.generate(..., sampling_params=...)`
  receives `max_new_tokens` clipped by the remaining trajectory budget
- latest live run no longer shows the earlier huge single-turn text explosions

#### Interpretation

- the important distinction is:
  - `data.max_response_length`
    - total trajectory response budget
  - `multi_turn.max_assistant_response_tokens`
    - one assistant generation budget
- after this patch, the visible failure mode changed:
  - before:
    - single-turn plain-text babbling collapse dominated
  - after:
    - tool-call formatting / schema mismatch dominates

#### Latest live diagnosis

- latest run:
  - `logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260515_112641`
- aggregate summary:
  - `4000` summaries
  - `reward_zero_ratio ≈ 97.18%`
  - `env_positive_ratio ≈ 2.83%`
  - `raw_format_positive_ratio ≈ 29.25%`
  - `parse_error_summary_ratio ≈ 83.65%`
  - `invalid_action_summary_ratio ≈ 54.38%`
- aggregate turn-level:
  - assistant turns: `78758`
  - `tool_call_ratio ≈ 24.51%`
  - `parse_error_turn_ratio ≈ 72.75%`
  - `invalid_action_turn_ratio ≈ 13.88%`
  - `valid_exec_turn_ratio ≈ 10.64%`

#### Representative examples

- semantic plan looks plausible, but the environment never sees it because the tool block is malformed:
  - `step_22/web_bmi_03.../trajectory.jsonl`
  - the model explicitly recognizes the earlier `x coordinate` issue and the need to click `BMI Calculator for Children and Teens`
  - then emits malformed JSON:
    - `{\"action_type\": \"MOVE_TO\", \"x\": 508, 777}`
- format is valid, but the trajectory degenerates into a non-grounding loop and still gets `0`:
  - `step_123/prozilla_filefind_01.../summary.json`
  - `attempted=15`
  - `valid=15`
  - `parse=0`
  - `invalid=0`
  - `raw_format=0.85`
  - `non_grounding_adjacency_ratio=1.0`
  - `format=0.0`
  - `reward_score=0.0`
- suspicious positive reward still exists and must not be mistaken for real recovery:
  - `step_13/warn_prozilla_explorer_10.../summary.json`
  - `reward_score=1.0`
  - `valid_tool_call_count=2`
  - `parse_error_count=15`
  - `termination_reason=model_done`
  - this remains an evaluator / env-signal correctness issue

---

## Log Analysis Guide

### 1. If the target is a checkpoint

- use **`current_param_version`** as the primary anchor
- for checkpoint `global_step_N`, search:
  - `current_param_version:N`
  - and the nearby checkpoint-save line in `output.log`

### 2. If the target is quantitative rollout quality

- use the root rollout-data batch dumps:
  - `logs/rollout_data/.../N.jsonl`
- these rows contain the fields that matter most:
  - `web_osgym_attempted_tool_calls`
  - `web_osgym_valid_tool_calls`
  - `web_osgym_first_valid_tool_call_index`
  - `web_osgym_termination_reason`
  - `web_osgym_format_reward`
  - `web_osgym_raw_format_reward`
  - `web_osgym_non_grounding_adjacency_ratio`

### 3. If the target is qualitative trajectory reading

- first choose representative rows from root `N.jsonl`
- then take their `request_id`
- then find the matching trace under:
  - `logs/rollout_data/.../step_*/*/trajectory.jsonl`

### 4. Never use `step_*` directory number as a step axis

- `step_252` in the trace directory is **not**
  - W&B step 252
  - `training/global_step 252`
  - checkpoint `global_step_252`

- it is only a per-trajectory storage bucket

### 5. For format-first analysis, never read `score/format` alone

- always pair it with:
  - `score/format_raw`
  - `score/non_grounding_adjacency_ratio`
  - attempted / valid
  - termination distribution
  - action-type distribution
  - `WAIT`-only share
  - `response_length/clip_ratio`
  - at least a few representative trajectories

- current meaning:
  - `score/format`
    - effective format after repetition decay
  - `score/format_raw`
    - raw format before repetition decay
  - `score/non_grounding_adjacency_ratio`
    - how much the action stream stayed inside `{WAIT, HOTKEY, TYPING}` across adjacent pairs

### 6. Current practical workflow

1. choose the target axis
   - checkpoint / `current_param_version`
   - or `training/global_step`
2. read the surrounding `output.log` lines
3. read the matching root `N.jsonl` window
4. pick representative `request_id`s
5. read the matching `trajectory.jsonl`
6. explicitly test for
   - `WAIT`-only / `FAIL`-only reward hacking
   - or more generally long `{WAIT, HOTKEY, TYPING}` persistence
   before claiming format improvement

### Step 2. Unconstrained decode path isolation — CLICK payload root cause 최종 확인

#### Status

- Completed

#### Sources

- val run (structured output OFF):
  - `logs/rollout_data/qwen35_27b_webgym_fully_async_tool_veomni_only_val_20260515_173102/step_1`
- SKD turn log:
  - `logs/async_skd_turns_webgym_20260515_165049.jsonl`
- val script diff:
  - `WebOSWorld/val_run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
    - committed: `enable_qwen3_coder_structured_output=True` 포함
    - 실험용 수정: 해당 플래그 제거 → unconstrained run

#### Method

- `enable_qwen3_coder_structured_output=True` 버전(기존 27B probe)과
  `enable_qwen3_coder_structured_output=False` 버전(이번 run)을 직접 비교
- CLICK action의 x field 형태 분류: scalar integer vs list
- SKD turn log에서 동일 패턴 확인
- 프롬프트/스키마 전달 문제인지 vs 모델 선호 문제인지 판단

#### Result

- val run (structured output OFF) CLICK 분포
  - total CLICK actions: `203`
  - x_as_list: `187 / 203` (92.1%)
  - x_as_scalar: `16 / 203` (7.9%)
  - dominant error: `"Input should be a valid integer [input_value=[x, y]]"` — Pydantic
  - fatal trajectory error: `"unhashable type: 'list'"` (list가 dict key로 사용됨)

- 비교: structured output ON (기존 probe)
  - CLICK scalar: `85 / 85` (100%)
  - CLICK list: `0 / 85` (0%)

- SKD turn log (20260515_165049) 확인
  - total turn events: `405`
  - `invalid_action = 192` (47%)
  - CLICK x_as_list (invalid_action 중): `187 / 192` (97%)
  - pattern: `{"action_type": "CLICK", "x": [35, 175]}` — x에 좌표 pair 통째로
  - 일부는 y도 list: `{"action_type": "CLICK", "x": [239, 397], "y": [397]}`

- 스키마/프롬프트 전달 여부
  - prompt_len student/teacher 동일 확인 (이전 trace)
  - bundled schema에 `"x must be single integer, not list"` 명시
  - 그럼에도 unconstrained에서 x top-1 = `' ['`, mass ≈ 0.9946
  - **프롬프트 전달 문제 아님 — 모델의 unconstrained 선호 문제**

#### Interpretation

- `enable_qwen3_coder_structured_output` 유무가 CLICK payload mode를 결정적으로 분리함
- constraint ON → scalar 100%, constraint OFF → list 92%
- 이는 모델이 스키마를 보고도 unconstrained 생성에서는 `[x, y]` pair를 강하게 선호함을 의미
- constrained decoding 방식의 문제:
  - student는 constraint된 scalar 토큰을 생성
  - teacher는 unconstrained verify (list 선호)
  - distillation 신호 불일치 → SKD learning signal 품질 저하
  - inference 시에도 constraint 의존성이 남아 brittle한 정책
- **결론: 스키마를 모델의 자연 분포에 맞추는 것이 근본 해결책**
  - `x: integer, y: integer` → `coordinate: [x, y]`
  - constrained decoding 제거 → student/teacher 분포 정렬

---

### Step 3. Schema alignment + tolerate-at-boundary normalizer — 전체 action type 분포 분석 및 방향 확정

#### Status

- In progress (schema/파서 구현 중, val 검증 미완)

#### Sources

- val run (unconstrained): `logs/rollout_data/qwen35_27b_webgym_fully_async_tool_veomni_only_val_20260515_173102/step_1`
- `verl/tools/web_osgym_tool.py` — `_normalize_actions()` 코드 리뷰
- `verl/experimental/agent_loop/web_osgym_protocol.py` — `WebOsGymAction` Pydantic 모델

#### Method

- unconstrained val run 전체 trajectory에서 각 action type별 coordinate 필드 형태 분류 (scalar vs list)
- `_normalize_actions()` 코드 전체 읽기 — 핸들러 존재 여부 확인
- 실행 흐름 추적: `execute()` → `_parse_actions()` → `_normalize_actions()` → 에러 포인트 특정

#### Result

**action type별 unconstrained 출력 분포 (val run 173102)**

| action type | scalar `x,y` | list `[x,y]` in x | 현재 상태 |
|------------|-------------|-------------------|----------|
| CLICK | 14% (38) | **85%** (223) | Pydantic ValidationError on list |
| DOUBLE_CLICK | 34% (17) | **66%** (33) | Pydantic ValidationError on list |
| MOVE_TO | 11% (1) | **88%** (8) | Pydantic ValidationError on list |
| RIGHT_CLICK | 62% (5) | 37% (3) | scalar도 `_normalize_actions` 핸들러 없어 ValueError |
| DRAG_TO | 0 | 0 | 핸들러 없어 ValueError (출력 시도 자체 없음) |
| MOUSE_DOWN | 0 | 0 | 핸들러 없어 ValueError |
| MOUSE_UP | 0 | 0 | 핸들러 없어 ValueError |
| SCROLL/TYPING/HOTKEY/WAIT/DONE/FAIL 등 | — | — | 정상 작동 |

**구현 누락 확인 (`_normalize_actions` 357–419줄)**
- `elif action_type not in {"WAIT","DONE","FAIL"}: raise ValueError("Unsupported action_type: ...")`
- RIGHT_CLICK, DRAG_TO, MOUSE_DOWN, MOUSE_UP 모두 이 경로로 떨어짐
- 에러는 `execute()` 624–631줄에서 catch되어 `"Invalid Web/OSGym action payload: Unsupported action_type: ..."` 로 반환
- HTTP 서버(`WebOsGymClient`)에는 한 번도 도달하지 않음

**에러 발생 레이어 정리**
- 리스트 coordinate (CLICK/DOUBLE_CLICK/MOVE_TO): Pydantic `ValidationError` (`x: int | None`, list 거부)
- scalar RIGHT_CLICK: Pydantic 통과 → `_normalize_actions` `ValueError`
- 두 경로 모두 `execute()` except 블록에서 동일하게 처리됨 → 구분 불가능

#### Interpretation

- **근본 문제는 두 계층 중첩**: (1) 좌표 필드 타입 불일치 (int vs list), (2) 핸들러 구현 누락
- 단순히 스키마만 `coordinate: [x, y]`로 바꾸면 Pydantic 계층은 해결되지만 핸들러 누락은 별도로 수정 필요
- **채택 방향: tolerate-at-boundary normalizer**
  - `_normalize_web_osgym_action_payload()` 에서 Pydantic 전에 세 가지 모델 출력 변형을 `coordinate: [x, y]`로 통일
  - `_normalize_actions()`에 RIGHT_CLICK, DRAG_TO, MOUSE_DOWN, MOUSE_UP 핸들러 추가
  - 서버 페이로드(`{x: int, y: int}`) 불변 — tool 레이어에서만 변환
- **반복 실험 계획 (schema → val → 분석 → 조정)**
  - Round 1 목표: parse_error_turn_ratio 50% 이하, invalid_action 중 coordinate-type 에러 소멸
  - Round 2: 남은 에러 패턴 재분류 (DRAG_TO 의미론, unknown action_type 문자열 등)
  - Round 3: parse_error_turn_ratio < 5% 수렴 확인 → RL 진입 판단
- **cursor 스케일링 주의**: `_require_coordinates()` 수정 시 cursor fallback은 이미 screen 좌표이므로 scale 불필요. 새 `coordinate` 필드를 읽을 때만 `_scale_xy_to_screen()` 호출. 기존 `if action.x is not None and action.y is not None` 조건을 `if action.coordinate is not None` 으로 교체해야 이중 스케일링 방지.

---

## Next Update Rule

- 이후 단계는 이 문서에 계속 추가한다
- 각 단계는 아래 형식을 유지한다
  - `Status`
  - `Sources`
  - `Method`
  - `Result`
  - `Interpretation`
