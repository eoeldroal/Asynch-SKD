# Web / OS Gym Integration Design

## 0. Purpose

이 문서는 `verl` agent-loop rollout에 WebGym / OSWorld 계열 stateful remote environment를 통합하는 설계를 정리한다.

현재 구현은 protocol-compatible real WebGym / OSWorld server surface를 직접 기준으로 삼는다. 과거 mock bring-up path는 초기 통합 단계의 자산이었고, 현재 canonical 설계의 일부는 아니다.

현재 canonical 상태는 다음과 같다.

- model-facing tool surface: bundled `computer(actions=[...])`
- common browser-control prompt: `WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt`
- teacher-only additive guidance: `WebOSWorld/webgym_rl/teacher_system_prompt_webgym_rl.txt`
- optional teacher-only structured few-shot transcript: `distillation.skd.teacher_fewshot_path`

## 1. Environment Model

환경은 stateless function이 아니라 trajectory-lifetime session을 가진 remote environment다.

Protocol operations:

- `start`: task와 session을 열고 initial observation을 반환한다.
- `action`: 같은 session에 action list를 적용하고 다음 observation을 반환한다.
- `reward`: 종료 후 scalar reward를 반환한다.

Session identity:

- `task_id`: dataset/server task identifier
- `session_id`: runtime/client-owned trajectory session identifier

`session_id`는 dataset에서 오지 않는다. agent loop가 trajectory 시작 시 생성하고, 같은 trajectory의 모든 request에서 재사용한다.

## 2. Design Goals

1. 기존 `ToolAgentLoop` / `SkdAgentLoop` 상태 기계를 유지한다.
2. environment lifecycle을 `PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`에 자연스럽게 대응시킨다.
3. screenshot image와 a11y/text observation을 모델 입력으로 제공한다.
4. teacher에게는 request-time에 teacher-only system guidance와 optional few-shot을 주입하고, observation은 student와 teacher가 동일하게 받는다.
5. `DONE` / `FAIL`과 system cutoff를 모두 reward 회수로 수렴시킨다.
6. async SKD scheduler/trainer가 Web/OSGym protocol을 직접 알 필요 없게 한다.

Current RL-specific extension:

- fully async WebGym RL can reuse the same rollout prompt-window metadata during actor update.
- post-rollout row building supports configurable supervision blocks via
  `actor_rollout_ref.rollout.multi_turn.web_osgym_window_supervision_block_size`.
- `1` means one emitted update row per assistant generation.
- values `>1` mean one emitted update row may contain multiple adjacent assistant generations, while earlier carried
  turns remain as context with `response_mask=0`.

## 3. State Mapping

바깥 상태 전이는 계속 `ToolAgentLoop`-compatible 하게 유지한다. 즉 top-level state machine은 여전히 `PENDING -> GENERATING -> PROCESSING_TOOLS -> TERMINATED`를 사용한다. 다만 canonical committed state에는 **완료된 assistant turn**과 **이미 commit된 observation bundle**만 들어가고, 진행 중인 assistant turn은 `GENERATING` 안의 turn-local state로 따로 든다.

### PENDING

`PENDING`은 environment `start` 단계다.

수행:

- runtime `session_id` 생성
- dataset/tool kwargs에서 `task_id` 확보
- server `start` request 전송
- initial screenshot/text 수신
- initial observation bundle 구성

이 단계는 단순 prompt 준비가 아니라 환경과 첫 동기화를 수행한다.

### GENERATING

모델이 다음 `computer` tool call을 생성한다.

이 상태는 단순히 "다음 토큰을 prompt 뒤에 바로 append하는 단계"가 아니다. `GENERATING`은 현재 assistant turn의 turn-local chunk buffer를 소유한다. SKD에서 teacher verification을 통과한 chunk라도 assistant turn이 아직 끝나지 않았다면 committed rollout에는 올리지 않고, 현재 turn-local buffer에만 유지한다.

모델은 Web/OSGym protocol JSON을 직접 생성하지 않는다. canonical model-facing schema는 bundled
`computer` tool이며, 하나의 tool call 안에 순서가 있는 Computer 13 action list를 담는다.

```json
{
  "actions": [
    {"action_type": "CLICK", "x": 100, "y": 200, "button": "left"}
  ]
}
```

좌표는 OSWorld-style `1000x1000` relative screen coordinate를 사용한다. `WebOsGymTool`은 이
model-facing 좌표를 현재 session screenshot 크기 기준 pixel 좌표로 후처리한 뒤 server protocol로
전송한다. 따라서 prompt/schema는 항상 `0..999` 좌표 계약을 말하고, server request만 실제 pixel
좌표를 본다.

request-time compact prompt view도 long-lived truth가 아니다. student / teacher request에 보내는 compact prompt ids는 항상 **committed state + 현재 pending turn state**를 합쳐서 만든 request-time view로 취급한다.

### PROCESSING_TOOLS

`WebOsGymTool`이 model-facing tool call을 server protocol request로 바꾼다.

수행:

- action list parse
- malformed payload safety handling
- server `action` request
- next observation 수신
- terminal action이면 reward 회수 준비

### TERMINATED

더 이상 action을 생성하지 않는다. 종료 시에는 reward를 회수해 `AgentLoopOutput.reward_score`로 넘긴다.

중요한 종료 규칙:

- `EOS`는 현재 pending assistant turn을 committed canonical state로 승격(promote)한다.
- 그 뒤에만 tool parsing을 수행해 `PROCESSING_TOOLS`로 갈지, 그대로 종료할지 결정한다.
- 반대로 budget exhaustion / empty chunk / max chunk / teacher context exhaustion 같은 forced cutoff는 pending assistant turn을 canonical prompt state로 commit하지 않는다.
- 이 경우 final output은 turn-local final text를 flush해서 사용자-visible response에는 포함하되, tool parsing 없이 종료한다.

참고: 이 규칙은 SKD / WebSKD의 assistant-turn boundary 의미다. 반면 explicit RL `web_tool_agent`
structured SGLang path는 grammar completion으로 tool-call wrapper를 끝까지 쓰게 하기 위해
`ignore_eos=True`를 사용한다. 즉 RL structured path와 SKD chunk path는 종료 semantics를 동일하게 두지 않는다.

추가로 SKD / WebSKD student generation은 `skip_tokenizer_init=True` token-in 경로를 쓰므로
`ignore_eos`를 명시적으로 건드리지 않는다. 대신 tokenizer의 `eos_token_id`를
`stop_token_ids`로 request에 넣어 EOS를 assistant-turn stop boundary로 강제한다.

## 4. Observation Policy

정상 visual observation:

- student와 teacher 모두 screenshot image만 본다. text/a11y는 포함하지 않는다.

Image-less failure observation:

- image가 없으면 실패 원인이 text에만 있을 수 있다.
- 이 경우 student와 teacher 모두 text만 본다.

이 예외는 action failure를 복구 가능한 환경 feedback으로 취급하기 위한 것이다.

즉 observation rule은 단순하다: image가 있으면 image만, image가 없으면 text만. student와 teacher는 동일한 observation을 받는다. teacher differentiation은 observation이 아니라 request-time에 `_build_teacher_messages()`가 주입하는 teacher-only system guidance와 optional few-shot으로만 이루어진다.

정리하면 observation의 정체성은 `image 유무`가 아니라 **commit된 environment feedback bundle**이다. image는 observation의 한 속성일 뿐이며, parser/training logic은 이를 step boundary의 유일한 source-of-truth로 삼으면 안 된다.

## 5. Text Observation Policy

Web/OSGym server는 `text` 필드로 observation text를 제공할 수 있다. WebSKD loop는 다음 규칙으로 text inclusion을 결정한다.

- image-bearing observation: text를 버린다. image만 observation으로 사용한다.
- image-less observation: text를 observation으로 사용한다. 이 경우 student와 teacher 모두 같은 text를 받는다.

이 규칙은 student와 teacher에 동일하게 적용된다. teacher-only text channel은 없다.

teacher differentiation은 observation이 아니라 request-time에 `_build_teacher_messages()`로 주입되는 다음 요소로만 이루어진다.

- teacher-only system guidance: `WebOSWorld/webgym_rl/teacher_system_prompt_webgym_rl.txt`
- optional teacher-only structured few-shot: `distillation.skd.teacher_fewshot_path`

`include_a11y` 플래그는 server가 a11y tree를 `text` 필드에 포함할지를 제어한다. 현재 기본값은 `False`이며, `custom.web_skd_include_a11y=True`로 켤 수 있다. a11y를 켜더라도 teacher-only channel이 되는 것이 아니라, 위의 image/text 선택 규칙에 따라 동일하게 처리된다.

## 6. Image Handling

Protocol wire format은 base64 PNG다.

```json
{
  "image": {
    "data": "...",
    "mimeType": "image/png"
  }
}
```

`web_osgym_protocol.py`는 이를 PIL image로 decode한다. 내부 multimodal data path는 기존 verl multimodal interface와 맞추기 위해 image object list를 사용한다.

다만 image metadata는 step마다 sparse/optional하다. text-only failure observation처럼 visual step이 아닌 경우에는 `image`/`images` placeholder를 만들지 않고 image data를 생략해야 한다.

현재 manager는 image를 다시 SGLang request payload에 실어 보낸다. 따라서 server에서 screenshot이 와도 SGLang student/teacher request에 image가 빠지는 문제는 이 계층에서 막아야 한다.

For backward compatibility, stored generation-window image metadata may appear as either `prompt_image_indices` or the
older `image_indices` key. The current RL row builder accepts both and normalizes them to integer indices before
slicing multimodal data.

## 7. Prompt State Policy

WebSKD는 **확정된 상태**와 **현재 진행 중인 assistant turn**을 분리한다.

현재 구현의 WebSKD는 local processor ids와 SGLang logical server ids를 분리하고, 동시에 **확정된 상태**와 **현재 진행 중인 assistant turn**을 분리한다.

- canonical committed state
  - `messages`
  - `image_data`
  - `prompt_ids`: 완료된 assistant turn과 commit된 observation bundle만 반영한 학생 쪽 expanded prompt ids
  - `teacher_prompt_ids`: 같은 경계의 teacher 쪽 expanded prompt ids
  - `response_mask`
  - teacher row lists
- current assistant turn state
  - 현재 turn에서 teacher verify를 통과한 token들
  - 그 token들에 대응하는 teacher rows
  - 직전에 생성한 raw chunk / verified chunk

`prompt_ids`와 `teacher_prompt_ids`는 **현재까지 확정된 prefix**만 담는다. 아직 `EOS`에 도달하지 않은 assistant chunk는 여기에 바로 섞지 않고, 별도의 현재 turn state에 둔다.

image가 포함되면 expanded prompt ids에는 image expansion이 반영될 수 있다. 따라서 image-bearing Qwen3 / Qwen3.5 request에서 expanded prompt ids를 raw-image SGLang boundary로 직접 보내면 안 된다.

현재 구현 원칙은 다음과 같다.

- canonical state는 expanded prompt ids를 유지한다.
- committed canonical state는 완료된 assistant turn과 committed observation bundle만 저장한다.
- SGLang에 보낼 compact request ids는 request 시점의 committed state에 현재 assistant turn state를 합쳐서 만든다.
- compact request ids와 teacher verify 길이 보정값은 long-lived truth가 아니라 request-time derived values다.
- active SKD chunk loop는 같은 현재 turn state를 이어 가지만, request prefix는 iteration마다 현재 committed state에서 다시 계산한다.
- teacher 전용 state가 없으면 student state로 대체하지 않는다. `teacher_prompt_ids` 또는 teacher request prefix가 없으면 바로 오류를 내고 중단한다. teacher messages는 request-time에 student messages로부터 파생되므로 별도 committed state로 저장하지 않는다.

### Teacher-only prompt composition

현재 SKD teacher prompt는 별도의 독립 prompt tree가 아니라 다음 합성 규칙을 따른다.

```text
teacher messages
  = student messages
  + teacher-only system guidance
  + optional teacher-only structured few-shot transcript
```

구성 요소:

- common runtime prompt
  - `WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt`
- teacher-only additive guidance
  - `WebOSWorld/webgym_rl/teacher_system_prompt_webgym_rl.txt`
- optional teacher-only structured few-shot
  - `distillation.skd.teacher_fewshot_path`

teacher few-shot은 role-separated multi-turn transcript로 로드되고, teacher message stream 앞쪽에만 prepend된다.
student message stream에는 들어가지 않는다. few-shot image도 teacher image stream에만 합쳐진다.

teacher-side extra prompt reserve는 현재 exact few-shot 길이 계산이 아니라 coarse reserve로 잡는다.
현재 distillation config가 사용하는 기본 reserve 상수는 `4096`이며, 이는 teacher system guidance가 켜진
경우의 보수적 여유값이지 exact few-shot length accounting은 아니다.

## 8. Tool Parse Recovery

현재 Web/OSGym loop는 recoverable tool-call parse failure를 즉시 종료로 처리하지 않는다.

동작:

1. parser가 malformed `<tool_call>` 또는 malformed bundled actions payload를 감지한다.
2. loop는 `Invalid tool call format: ...` 형태의 feedback observation을 구성한다.
3. 이 feedback을 다음 turn context에 넣고 generation을 다시 진행한다.

현재 retry budget은 사실상 매우 크게 열려 있다.

```text
max_tool_parse_error_retries = 9999
```

의도:

- malformed output 한두 번으로 trajectory가 쉽게 끊기지 않게 한다.
- 포맷 오류를 다음 turn에서 모델-visible feedback으로 돌려준다.

한계:

- parser가 recoverable error로 분류할 수 있는 경우에만 적용된다.
- `<tool_call>` wrapper 자체가 전혀 없는 출력 등 일부 malformed case는 일반적인 "tool call 없음" 경로로
  빠질 수 있다.

## 9. Assistant Turn Promotion and Atomic Observation Commit

chunk는 바로 assistant turn이 아니다. WebSKD는 다음 순서를 지킨다.

1. student가 chunk를 생성한다.
2. teacher가 그 chunk를 verify하고 필요하면 일부 token을 교체한다.
3. verify를 통과한 token만 현재 assistant turn state에 누적한다.
4. `EOS`가 나올 때만 현재 assistant turn 전체를 **완성된 assistant turn**으로 승격해서 canonical state에 반영한다.

즉 `prompt_ids`, `teacher_prompt_ids`, `response_mask`, teacher rows는 chunk마다 조금씩 늘어나는 것이 아니라, 정상 경로에서는 `EOS`에서 한 번에 확정된다.

forced cutoff는 예외다.

- `budget_exhausted`
- `max_chunks`
- `teacher_context_exhausted`
- `empty_chunk`

같은 종료는 unfinished assistant turn을 최종 output에는 드러낼 수 있지만, tool parsing이나 `PROCESSING_TOOLS`로 넘어가면 안 된다.

tool-result observation은 별도의 atomic commit unit이다. screenshot이 있든 없든, environment/tool 결과는 "다음 turn이 읽는 observation bundle 하나"로 commit되거나 아예 commit되지 않아야 한다.

Web observation은 다음 canonical state를 함께 바꾼다.

- image data
- messages
- prompt ids
- teacher prompt ids
- response mask
- dummy teacher rows

compact request view는 필요할 때 현재 canonical state에서 다시 계산한다.

- `server_prompt_ids`
- `teacher_server_prompt_ids`
- `teacher_sglang_prefix_surplus`

이 값들은 guard 검사나 request 직전 계산에는 쓸 수 있지만, canonical state로 오래 저장하지 않는다. observation bundle commit에서 원자적으로 맞춰야 하는 것은 compact request cache가 아니라 **expanded canonical state 자체**다. 일부만 commit되면 carryover, cutoff, teacher verification에서 상태가 깨진다.

중요한 점은 image-bearing boundary를 compact delta append로 넘기지 않는다는 것이다. 새 image가 붙으면 compact request view는 이전 값에 suffix를 더하지 않고, **현재 canonical state에서 다시 계산**한다.

Cutoff나 teacher context guard에 걸리면 observation bundle 전체를 commit하지 않는다.

## 10. Async SKD Scheduling Model

WebSKD는 fully async trainer가 아니라, rollout manager 내부에서만 bounded lookahead를 수행하는 async SKD scheduler 위에 올라간다.

핵심 용어:

- `current work`: 이번 trainer step에 반드시 포함되어야 하는 sample
- `lookahead`: idle worker slot에서 미래 sample을 미리 시작하는 speculative execution
- `promoted`: step barrier 전에 terminal completion까지 끝나 이번 step batch 뒤에 append될 수 있는 sample
- `carryover`: terminal까지는 안 끝났지만 exportable boundary에서 partial state로 저장되어 다음 step current work로 재진입하는 sample

중요한 점:

- token correctness와 teacher alignment는 여전히 `SkdAgentLoop` / `WebSkdAgentLoop`가 담당한다.
- async manager는 sample-level scheduling만 담당하며, SKD token semantics를 재정의하지 않는다.
- partial snapshot은 handler 내부 임의 시점이 아니라 **exportable boundary**에서만 허용된다.

## 11. Validation and Data Source Ownership

Validation은 train-time teacher-guided SKD 재현이 아니라 student policy 평가다.

따라서 validation에서는:

- teacher verification을 사용하지 않는다.
- training `AsyncSkdDataSource`를 manager에서 detach한다.
- future sample reservation, promoted ledger, carryover ledger를 변경하지 않는다.

즉 validation은 training scheduler state를 오염시키면 안 된다.

## 12. Metadata and Identity Boundary

Async SKD는 current, promoted, carryover output을 함께 concat/union한다. 이때 non-tensor metadata key set이 경로별로 다르면 `DataProto` 경계에서 실패한다.

특히 windowed training에서는 one trajectory가 multiple trainer rows로 확장되므로, 다음 join key는 hard input-owned identity로 유지되어야 한다.

- `uid`
- `index`
- `input_pos`

원칙:

- output metadata가 input-owned join key를 덮어쓰면 안 된다.
- fresh completion과 carryover completion은 같은 completed-batch envelope를 만들어야 한다.
- key set 정렬은 metadata boundary helper에서 처리하고, 개별 경로에서 ad hoc patch를 하지 않는다.

## 13. Prompt Views and Teacher Verification

Teacher verification의 target은 항상 **학생이 방금 생성한 chunk**다.

Teacher verification의 target은 항상 **학생이 방금 생성한 chunk**다.

이를 위해 teacher 쪽은 다음 두 종류의 값을 구분한다.

- canonical expanded teacher prefix
  - `teacher_prompt_ids`
- request-time compact teacher prefix
  - `teacher_server_prompt_ids`
- multimodal expansion gap
  - `teacher_sglang_prefix_surplus`

핵심 원칙은 다음과 같다.

- `teacher_prompt_ids`는 확정된 teacher prefix를 나타내는 canonical state다.
- `teacher_server_prompt_ids`와 `teacher_sglang_prefix_surplus`는 long-lived truth가 아니라, **verify request를 보내기 직전 계산하는 파생값**이다.
- verify request prefix는 **현재 teacher messages / image_data / 현재 assistant turn token buffer** 기준으로 다시 만든다.
- verify 중인 active chunk loop는 현재 turn state를 이어 가지만, request prefix의 기준은 항상 현재 committed state다.
- teacher verification에 필요한 값이 없거나 모순되면 조용히 0이나 student prefix로 복구하지 않고 즉시 실패한다.

즉 teacher verification은 stale compact snapshot을 신뢰해서 이어 붙이는 방식이 아니라, **현재 상태에서 request-time view를 다시 만들고 그 위에 현재 chunk를 올리는 방식**으로 본다.

## 14. Teacher Context Guard

Teacher는 teacher-only system guidance, optional few-shot, 그리고 image expansion 때문에 student보다 긴 context를 가질 수 있다.

guard 위치는 두 군데다.

1. SKD verification 직전
   - `teacher_server_prompt_ids + chunk` 길이 검사
   - verify 직전에 current teacher state 기준 compact snapshot 재계산
2. Web observation commit 직전
   - non-terminal observation을 넣은 뒤 최소 1 future verified token 공간이 남는지 검사

초과 시 sample은 `teacher_context_exhausted`로 정상 종료한다. terminal observation/reward path는 더 이상 future verification이 없으므로 같은 guard를 적용하지 않는다.

## 15. Windowed Training Contract

Completed WebSKD trajectory는 actor update 전에 current mini-step 중심의 bounded training rows로 확장될 수 있다.

원칙:

- loss target boundary는 항상 original contiguous `response_mask == 1` assistant run이다.
- mini-step reconstruction은 **student topology**를 기준으로 한다.
- image metadata는 sparse/optional step attribute다.
- text-only failure observation도 정상 observation step으로 남는다.
- trailing observation-only suffix는 learnable row로 승격하지 않는다.

즉 window parser는 image count로 step을 정의하지 않고, student trajectory topology를 따라 step을 자른다.

## 16. Action Semantics

Action schema는 Computer 13 계열 low-level action을 따른다. 현재 canonical tool surface는
bundled `computer(actions=[...])` schema다. 현재 WebGym Async SKD와 fully async RL launcher는 모두
이 bundled surface를 쓴다. 코드에 남아 있는 legacy action-named compatibility path도 의미론은 같은
Computer 13 action list여야 한다.

Supported action names:

```text
MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP, RIGHT_CLICK, DOUBLE_CLICK,
DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY,
WAIT, DONE, FAIL
```

`DONE`과 `FAIL`은 표면상 action이지만 loop 의미상 terminal request다. 단독 action으로 보내는 것이 원칙이다.

또한 `DONE` / `FAIL`은 **새로운 learnable mini-step observation을 하나 더 추가하는 신호가 아니다.** terminal action 이후 최종 reward는 별도 reward request로 회수하며, training-side mini-step reconstruction은 terminal action 자체를 마지막 assistant target으로 취급해야 한다.

Coordinate contract:

- model-facing `x` / `y`는 항상 `0..999` 정수다.
- `(0, 0)`은 screenshot의 top-left, `(999, 999)`는 bottom-right다.
- tool implementation은 현재 session의 `screen_width` / `screen_height`를 사용해 relative 좌표를
  pixel 좌표로 project한다.
- `CLICK` / `DOUBLE_CLICK` 등 cursor-relative fallback을 허용하는 action은 cursor state가 없으면
  fail-fast해야 한다. visual grounding을 위해 가능한 한 명시적 좌표를 선호한다.

### Action postprocess boundary

현재 action postprocess는 한 파일에만 있지 않고, 두 계층으로 나뉜다.

1. `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
   - model tool call들을 실행 가능한 `actions` payload로 묶는다.
   - named action tool surface를 bundled `computer(actions=[...])` shape로 변환한다.
   - tool의 `postprocess_tool_arguments()` hook를 호출한다.
2. `verl/tools/web_osgym_tool.py`
   - key alias / casing / HOTKEY combo-string split 같은 canonicalization을 수행한다.
   - single-key action의 combo-string reject, cursor fallback, coordinate projection, terminal-action validation을 수행한다.
   - 최종 server payload를 만든다.

즉 `web_osgym_loop_mixin.py`는 orchestration layer이고, `web_osgym_tool.py`는 canonicalization +
validation layer다.

현재 Linux-focused canonical keyboard policy:

- shortcut modifier는 `Control` 중심 canonical form을 쓴다.
- literal OS modifier는 `Meta` canonical form을 쓴다.
- `PRESS`, `KEY_DOWN`, `KEY_UP`는 single key only다.
- `HOTKEY`는 single key name list를 canonical form으로 쓴다.
- `HOTKEY(["ctrl+a"])` 같은 combo-string payload는 `verl`에서 `["Control", "A"]`로 분해/정규화한 뒤
  server에 전달한다.

이 경계는 나중에 benchmark harness를 만들 때도 중요하다. benchmark harness가 현재 training/runtime과 같은
행동 의미를 재현하려면, server 호출만 맞추는 것이 아니라 **동일한 action postprocess contract**를 재사용하거나
문서 그대로 복제해야 한다.

## 17. Failure Semantics

Transport-level failure와 action failure는 다르다.

- transport failure: server request 자체가 실패한 것
- action failure: 환경이 action을 처리했지만 실패 feedback을 반환한 것

Action failure는 policy가 다음 행동을 고치는 데 필요한 observation일 수 있다. image가 없으면 text feedback을 student와 teacher 모두에게 제공한다.

Malformed model action payload도 가능하면 action failure observation으로 바꿔 trajectory를 계속 진행한다.

따라서 mini-step reconstruction은 다음 두 경우를 모두 동일한 observation step으로 다룬다.

- visual observation: screenshot image가 붙은 step
- text-only failure observation: image는 없고 text feedback만 있는 step

## 18. Reward Semantics

Reward source of truth는 environment server다.

Loop는 종료 시점에 reward를 회수하고, trainer는 protocol을 직접 알 필요 없이 기존 batch path에서 reward를 받는다.

종료 이유:

- model `DONE`
- model `FAIL`
- response/model length cutoff
- max chunks
- teacher context exhausted
- 기타 system termination

종료 이유와 무관하게 가능한 경우 reward request는 한 번 수행한다.

Operational timeout contract:

- Web/OSGym client config can declare `minimum_safe_action_timeout`; the actual client `timeout` must not be smaller
  than that floor.
- On client-side `ReadTimeout` during `action`, verl does not retry the action request. The server may already have
  executed the action, so retrying could duplicate `CLICK`, `TYPING`, or `SCROLL`.
- Instead, the client sends a best-effort `reward` request using the copied session identifiers so the server can close
  the session if possible, then terminates the local trajectory.
- In fully async RL, that timeout failure is dropped before queue insertion so one timed-out session does not leak a
  partial GRPO group into trainer assembly.

## 19. Boundary Findings and Simplification Direction

최근 Qwen3 / Qwen3.5 + SGLang raw-image 경계를 직접 조사하면서 다음 사실이 확인되었다.

- `prompt_ids`는 local에서는 processor-derived expanded ids다.
- SGLang raw-image multimodal path는 `input_ids`를 그대로 신뢰하지 않고, image placeholder 구간을 다시 해석한다.
- 실제 local probe에서는:
  - `expanded prompt_ids + raw image_data`는 Qwen3-VL / Qwen3.5 계열에서 실패했다.
  - `compact placeholder ids + raw image_data`는 성공했고, SGLang이 최종 expanded ids를 local `prompt_ids`와 같게 복원했다.

따라서 현재까지의 결론은 다음과 같다.

- raw-image SGLang boundary에는 expanded ids를 직접 밀어 넣지 않는다.
- local canonical state는 expanded 상태로 유지할 수 있다.
- 대신 SGLang request 직전에 compact server-side prompt view를 만들고, request 후에는 이를 장기 runtime truth로 보존하지 않는 방향이 더 단순하다.

특히 image-bearing boundary에서는 이전 compact prefix에 delta를 누적하는 것보다, **현재 messages + image_data로 compact ids를 다시 만드는 쪽**이 더 안전하다.

이 문서 기준으로 정리하면:

- **current implementation**
  - WebSKD는 long-lived `server_prompt_ids` / `teacher_server_prompt_ids`를 유지한다.
- **current direction**
  - local canonical expanded state는 유지
  - compact server prompt view는 boundary-local derived state로 축소

주의:

- generic non-WebOS `single_turn_agent` / `tool_agent` 경로는 여전히 processor-derived `prompt_ids`를 직접 SGLang에 넘긴다.
- 따라서 generic 경로는 Qwen3 / Qwen3.5 raw-image boundary의 안전성에 대한 증거가 아니라, 아직 별도 검증이 필요한 경로로 봐야 한다.

## 20. Runtime Requirements and Retired Workarounds

현재 canonical runtime requirement는 다음과 같다.

- WebSKD student/teacher generation은 ids-only token streams를 source-of-truth로 사용한다.
- old native-text multimodal generate workaround는 retired되었다.
- actor-side Qwen3.5 multimodal stall은 parser/trainer logic이 아니라 runtime/library 축 문제였다.

현재 알려진 operational requirement:

- `torch 2.9.1` line을 유지하는 경우, B200 WebGym SKD actor path에서는 `nvidia-cudnn-cu12==9.15.1.9`가 known-good baseline이다.

즉 과거 RCA에 사용했던:

- native-text student request 우회
- end-to-end tracing probe
- exact forward snapshot/replay helper
- one-off SGLang replay benchmark bundle

은 현재 설계의 일부가 아니며, repo 기본 경로에서도 retire되었다.

## 21. Current Implementation Map

현재 구현의 source-of-truth는 다음 파일들이다.

Protocol / tool boundary:

- `verl/experimental/agent_loop/web_osgym_protocol.py`
  - `POST /` with `op=start|action|reward`
  - wire field `session_id`
  - base64 PNG image parsing
- `verl/tools/web_osgym_tool.py`
  - model-facing `computer` tool
  - `actions: [...]` parsing
  - protocol request construction
  - malformed action payload safety path
  - `DONE` / `FAIL` terminal action handling

Loop integration:

- `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
  - runtime-owned `web_osgym_session_id`
  - session restore for partial/carryover state
  - final reward fetch
- `verl/experimental/agent_loop/web_skd_agent_loop.py`
  - registered as `web_skd_agent`
  - `include_a11y` defaults to `False`; configurable via `custom.web_skd_include_a11y`
  - pending initial observation bundle
  - tool observation atomic commit
  - unified observation rule: image-bearing → image only; image-less failure → text only (same for student and teacher)
  - request-time teacher messages derived from student messages via `_build_teacher_messages()`
  - server prompt ids and teacher server prompt ids maintenance
  - teacher context guard before non-terminal observation commit
  - final environment reward propagation
- `verl/experimental/agent_loop/web_tool_agent_loop.py`
  - registered as `web_tool_agent`
  - inherits from `ToolAgentLoop`, not `SkdAgentLoop`
  - starts one Web/OSGym session before generation
  - reuses that same session for every `computer` tool call in a trajectory
  - fetches environment reward once at `DONE` / `FAIL` / budget / system-stop termination
  - propagates reward through `AgentLoopOutput.reward_score`, allowing fully async RL postprocess to create `rm_scores`

Async SKD scheduling / batch boundary:

- `verl/experimental/async_skd/manager.py`
  - current/lookahead/promoted/carryover scheduling
- `verl/experimental/async_skd/worker.py`
  - completed-window postprocess boundary
- `verl/experimental/async_skd/windowed_training.py`
  - completed trajectory -> bounded mini-step training rows
- `verl/experimental/teacher_loop/teacher_manager.py`
  - teacher replica routing and sticky carryover binding

Trainer entrypoint:

- `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`

Tool config:

- `WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml`
  - canonical bundled `computer(actions=[...])` schema used by both fully async RL and the current Async
    SKD WebGym launcher
- `WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml`
  - legacy compatibility schema for older or debugging paths
  - not the canonical current WebGym RL / Async SKD model-facing contract

Prompt files:

- `WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt`
  - common browser-control prompt used by the student runtime
- `WebOSWorld/webgym_rl/teacher_system_prompt_webgym_rl.txt`
  - teacher-only additive guidance
  - SKD teacher messages are built as `student messages + teacher-only system guidance`, not as an independent prompt tree
- `distillation.skd.teacher_fewshot_path`
  - optional teacher-only structured few-shot transcript path
  - loaded as role-separated messages plus teacher-only images
  - prepended only to the teacher message/image stream

Launcher prompt arguments:

- `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
  - positional arg 1: common system prompt txt path
  - positional arg 2: teacher-only system prompt txt path
- `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
  - positional arg 1: common system prompt txt path
  - positional arg 2: accepted for interface symmetry, but unused because fully async RL has no teacher path

## 22. One-line Summary

Web / OS Gym integration은 `tool_agent`/`skd_agent` 위에 stateful remote environment session을 얹고, screenshot/text observation과 final environment reward를 기존 rollout/training path로 전달하는 구조다. SKD 경로는 `web_skd_agent`, 순수 fully async RL 경로는 `web_tool_agent`를 사용하며, async SKD scheduler와 windowed training은 이 Web observation contract를 깨지 않는 범위에서만 sample scheduling과 actor context를 제한한다.
