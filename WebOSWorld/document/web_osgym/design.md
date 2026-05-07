# Web / OS Gym Integration Design

## 0. Purpose

이 문서는 `verl` agent-loop rollout에 WebGym / OSWorld 계열 stateful remote environment를 통합하는 설계를 정리한다.

현재 구현은 protocol-compatible real WebGym / OSWorld server surface를 직접 기준으로 삼는다. 과거 mock bring-up path는 초기 통합 단계의 자산이었고, 현재 canonical 설계의 일부는 아니다.

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
4. student와 teacher의 observation visibility를 분리한다.
5. `DONE` / `FAIL`과 system cutoff를 모두 reward 회수로 수렴시킨다.
6. async SKD scheduler/trainer가 Web/OSGym protocol을 직접 알 필요 없게 한다.

## 3. State Mapping

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

## 4. Observation Policy

정상 visual observation:

- student sees screenshot image
- teacher sees screenshot image + a11y/text

Image-less failure observation:

- image가 없으면 실패 원인이 text에만 있을 수 있다.
- 이 경우 student와 teacher 모두 text를 본다.

이 예외는 action failure를 복구 가능한 환경 feedback으로 취급하기 위한 것이다.

정리하면 observation의 정체성은 `image 유무`가 아니라 **commit된 environment feedback bundle**이다. image는 observation의 한 속성일 뿐이며, parser/training logic은 이를 step boundary의 유일한 source-of-truth로 삼으면 안 된다.

## 5. A11y/Text Policy

a11y tree는 server가 `text` 필드에 제공하는 observation text다. WebSKD loop는 정상 image-bearing observation에서 이 text를 teacher-only channel에 둔다.

중요한 점:

- server가 특정 a11y 포맷을 강제할 필요는 없다.
- loop는 image 유무를 기준으로 student text visibility를 결정한다.
- image가 있으면 student는 text/a11y를 보지 않는다.
- image가 없으면 failure feedback으로 보고 student도 text를 본다.

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

## 7. Prompt Stream Policy

현재 구현의 WebSKD는 local processor ids와 SGLang logical server ids를 분리한다.

- `prompt_ids`: student local ids
- `teacher_prompt_ids`: teacher local ids
- `server_prompt_ids`: student SGLang logical ids
- `teacher_server_prompt_ids`: teacher SGLang logical ids

image가 포함되면 local ids에는 image expansion이 반영될 수 있다. server prompt ids는 SGLang `/generate`와 prompt-logprob delta contract의 기준이다.

따라서 WebSKD에서 local ids를 server ids로 fallback하면 안 된다.

## 8. Atomic Observation Commit

Web observation은 다음 상태를 함께 바꾼다.

- image data
- messages
- prompt ids
- server prompt ids
- teacher prompt ids
- teacher server prompt ids
- response mask
- dummy teacher rows

이 상태는 모두 성공한 뒤 한 번에 commit되어야 한다. 일부만 commit되면 carryover, cutoff, teacher verification에서 상태가 깨진다.

Cutoff나 teacher context guard에 걸리면 observation bundle 전체를 commit하지 않는다.

## 9. Async SKD Scheduling Model

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

## 10. Validation and Data Source Ownership

Validation은 train-time teacher-guided SKD 재현이 아니라 student policy 평가다.

따라서 validation에서는:

- teacher verification을 사용하지 않는다.
- training `AsyncSkdDataSource`를 manager에서 detach한다.
- future sample reservation, promoted ledger, carryover ledger를 변경하지 않는다.

즉 validation은 training scheduler state를 오염시키면 안 된다.

## 11. Metadata and Identity Boundary

Async SKD는 current, promoted, carryover output을 함께 concat/union한다. 이때 non-tensor metadata key set이 경로별로 다르면 `DataProto` 경계에서 실패한다.

특히 windowed training에서는 one trajectory가 multiple trainer rows로 확장되므로, 다음 join key는 hard input-owned identity로 유지되어야 한다.

- `uid`
- `index`
- `input_pos`

원칙:

- output metadata가 input-owned join key를 덮어쓰면 안 된다.
- fresh completion과 carryover completion은 같은 completed-batch envelope를 만들어야 한다.
- key set 정렬은 metadata boundary helper에서 처리하고, 개별 경로에서 ad hoc patch를 하지 않는다.

## 12. Prompt Streams and Teacher Verification

현재 구현의 WebSKD는 local processor ids와 SGLang logical server ids를 분리한다.

- `prompt_ids`: student local ids
- `teacher_prompt_ids`: teacher local ids
- `server_prompt_ids`: student SGLang logical ids
- `teacher_server_prompt_ids`: teacher SGLang logical ids

image-bearing prompt에서는 local ids에 image expansion이 반영될 수 있으므로, local ids를 server ids로 fallback하면 안 된다.

pending initialization 이후 teacher verification의 runtime truth는 다음 세 값이다.

- `teacher_prompt_ids`
- `teacher_server_prompt_ids`
- `teacher_sglang_prefix_surplus`

따라서 verify 단계는 teacher ids를 messages에서 다시 rebuild하면 안 된다. 검증 request와 row trimming은 위 tracked runtime state만 소비해야 한다.

현재 student multimodal generation도 historical native-text workaround를 쓰지 않고, 다시 **ids-only request contract**로 돌아와 있다. image-bearing request는 `input_ids + image_data`로 보내고, teacher/student alignment truth는 계속 tracked token streams가 담당한다.

## 13. Teacher Context Guard

Teacher는 a11y/text와 image expansion 때문에 student보다 긴 context를 가진다.

guard 위치는 두 군데다.

1. SKD verification 직전
   - `teacher_server_prompt_ids + chunk` 길이 검사
   - messages 기반 teacher id rebuild 금지
2. Web observation commit 직전
   - non-terminal observation을 넣은 뒤 최소 1 future verified token 공간이 남는지 검사

초과 시 sample은 `teacher_context_exhausted`로 정상 종료한다. terminal observation/reward path는 더 이상 future verification이 없으므로 같은 guard를 적용하지 않는다.

## 14. Windowed Training Contract

Completed WebSKD trajectory는 actor update 전에 current mini-step 중심의 bounded training rows로 확장될 수 있다.

원칙:

- loss target boundary는 항상 original contiguous `response_mask == 1` assistant run이다.
- mini-step reconstruction은 **student topology**를 기준으로 한다.
- image metadata는 sparse/optional step attribute다.
- text-only failure observation도 정상 observation step으로 남는다.
- trailing observation-only suffix는 learnable row로 승격하지 않는다.

즉 window parser는 image count로 step을 정의하지 않고, student trajectory topology를 따라 step을 자른다.

## 15. Action Semantics

Action schema는 Computer 13 계열 low-level action을 따른다. 현재 canonical tool surface는
bundled `computer(actions=[...])` schema다. Async SKD가 chunked generation / teacher verification
제약 때문에 action-named Qwen3.5 tool surface를 쓰는 경우에도, 의미론은 같은 Computer 13 action
list여야 한다.

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

## 16. Failure Semantics

Transport-level failure와 action failure는 다르다.

- transport failure: server request 자체가 실패한 것
- action failure: 환경이 action을 처리했지만 실패 feedback을 반환한 것

Action failure는 policy가 다음 행동을 고치는 데 필요한 observation일 수 있다. image가 없으면 text feedback을 student와 teacher 모두에게 제공한다.

Malformed model action payload도 가능하면 action failure observation으로 바꿔 trajectory를 계속 진행한다.

따라서 mini-step reconstruction은 다음 두 경우를 모두 동일한 observation step으로 다룬다.

- visual observation: screenshot image가 붙은 step
- text-only failure observation: image는 없고 text feedback만 있는 step

## 17. Reward Semantics

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

## 18. Boundary Findings and Simplification Direction

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

## 19. Runtime Requirements and Retired Workarounds

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

## 20. Current Implementation Map

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
  - `include_a11y=True`
  - pending initial observation bundle
  - tool observation atomic commit
  - student/teacher observation split
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
  - canonical bundled `computer(actions=[...])` schema used by fully async RL
- `WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml`
  - compatibility action-named Computer 13 schema, primarily for Async SKD paths that cannot yet use
    generation-time constrained decoding cleanly

Prompt files:

- `WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt`
  - common browser-control prompt used by the student runtime
- `WebOSWorld/webgym_rl/teacher_system_prompt_webgym_rl.txt`
  - teacher-only additive guidance
  - SKD teacher messages are built as `student messages + teacher-only system guidance`, not as an independent prompt tree

Launcher prompt arguments:

- `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
  - positional arg 1: common system prompt txt path
  - positional arg 2: teacher-only system prompt txt path
- `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
  - positional arg 1: common system prompt txt path
  - positional arg 2: accepted for interface symmetry, but unused because fully async RL has no teacher path

## 21. One-line Summary

Web / OS Gym integration은 `tool_agent`/`skd_agent` 위에 stateful remote environment session을 얹고, screenshot/a11y observation과 final environment reward를 기존 rollout/training path로 전달하는 구조다. SKD 경로는 `web_skd_agent`, 순수 fully async RL 경로는 `web_tool_agent`를 사용하며, async SKD scheduler와 windowed training은 이 Web observation contract를 깨지 않는 범위에서만 sample scheduling과 actor context를 제한한다.
