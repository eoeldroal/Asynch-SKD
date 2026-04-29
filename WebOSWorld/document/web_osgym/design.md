# Web / OS Gym Integration Design

## 0. Purpose

이 문서는 `verl` agent-loop rollout에 WebGym / OSWorld 계열 stateful remote environment를 통합하는 설계를 정리한다.

현재 구현은 real WebGym stack에 바로 의존하지 않고, protocol-compatible mock Web/OSGym server로 먼저 검증한다. mock 검증 후 같은 protocol surface를 real server에 붙이는 것이 기본 순서다.

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

모델은 Web/OSGym protocol JSON을 직접 생성하지 않는다. 모델-facing schema는 다음 형태다.

```json
{
  "actions": [
    {"action_type": "CLICK", "x": 100, "y": 200, "button": "left"}
  ]
}
```

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

현재 manager는 image를 다시 SGLang request payload에 실어 보낸다. 따라서 server에서 screenshot이 와도 SGLang student/teacher request에 image가 빠지는 문제는 이 계층에서 막아야 한다.

## 7. Prompt Stream Policy

WebSKD는 local processor ids와 SGLang logical server ids를 분리한다.

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

## 9. Teacher Context

Teacher는 a11y/text와 image expansion 때문에 student보다 긴 context를 가진다.

따라서 WebSKD는 observation commit 전 prospective teacher context를 검사한다. non-terminal observation을 넣은 뒤 최소 1 token을 teacher가 verify할 수 없다면 해당 observation을 commit하지 않고 sample을 `teacher_context_exhausted`로 종료한다.

Terminal observation/reward path는 더 이상 future verification이 없으므로 같은 guard를 적용하지 않는다.

## 10. Action Semantics

Action schema는 Computer 13 계열 low-level action을 따른다.

Supported action names:

```text
MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP, RIGHT_CLICK, DOUBLE_CLICK,
DRAG_TO, SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY,
WAIT, DONE, FAIL
```

`DONE`과 `FAIL`은 표면상 action이지만 loop 의미상 terminal request다. 단독 action으로 보내는 것이 원칙이다.

## 11. Failure Semantics

Transport-level failure와 action failure는 다르다.

- transport failure: server request 자체가 실패한 것
- action failure: 환경이 action을 처리했지만 실패 feedback을 반환한 것

Action failure는 policy가 다음 행동을 고치는 데 필요한 observation일 수 있다. image가 없으면 text feedback을 student와 teacher 모두에게 제공한다.

Malformed model action payload도 가능하면 action failure observation으로 바꿔 trajectory를 계속 진행한다.

## 12. Reward Semantics

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

## 13. Mock Server Role

Mock server는 policy quality를 평가하는 서버가 아니다. protocol과 trainer integration을 확인하는 서버다.

검증 대상:

- `GET /health`
- `POST /` with `op=start|action|reward`
- same `session_id` across a trajectory
- stable `task_id`
- base64 PNG observation
- optional a11y text
- action list logging
- reward return

Mock server를 통과한 뒤 real WebGym / Omnibox readiness와 task-specific 동작을 본다.

## 14. Current Implementation Map

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

Mock server assets:

- `WebOSWorld/mock_server/web_osgym_mock_server.py`
- `WebOSWorld/mock_server/web_osgym_mock_client.py`
- `WebOSWorld/mock_server/create_mock_web_osgym_dataset.py`
- `WebOSWorld/mock_server/reward_fn_mock_web_osgym.py`

Trainer entrypoint:

- `WebOSWorld/run_qwen35_web_mock_async_skd_tool_fsdp.sh`
- `WebOSWorld/run_qwen35_web_mock_fully_async_rl_tool_fsdp.sh`

Tool config:

- `examples/sglang_multiturn/config/tool_config/web_osgym_tool_config_webgym_rl.yaml`

## 15. Current Verification Flow

### Protocol smoke

Start the mock server:

```bash
cd /home/sogang_nlpy/verl
conda activate skd

python WebOSWorld/mock_server/web_osgym_mock_server.py \
  --host 127.0.0.1 \
  --port 18000 \
  --log-path logs/mock_web_osgym_requests.jsonl
```

Run the mock client:

```bash
python WebOSWorld/mock_server/web_osgym_mock_client.py \
  --base-url http://127.0.0.1:18000 \
  --session-id 777 \
  --task-id 12345
```

Expected request sequence:

```text
start -> action(CLICK) -> action(DONE) -> reward
```

All events must keep the same `session_id` and `task_id`.

### Trainer mock run

Generate the mock dataset if needed:

```bash
python WebOSWorld/mock_server/create_mock_web_osgym_dataset.py \
  --local-save-dir /home/sogang_nlpy/verl/data/mock_web_osgym \
  --num-samples 256
```

For fully async RL, generate the same protocol dataset with `agent_name=web_tool_agent`:

```bash
python WebOSWorld/mock_server/create_mock_web_osgym_dataset.py \
  --local-save-dir /home/sogang_nlpy/verl/data/mock_web_osgym_fully_async_rl \
  --num-samples 256 \
  --agent-name web_tool_agent
```

Run the mock server detached:

```bash
nohup python WebOSWorld/mock_server/web_osgym_mock_server.py \
  --host 127.0.0.1 \
  --port 18000 \
  --log-path logs/mock_web_osgym_requests.jsonl \
  > logs/mock_web_osgym_server.log 2>&1 &
```

Run the trainer:

```bash
nohup bash WebOSWorld/run_qwen35_web_mock_async_skd_tool_fsdp.sh \
  > logs/web_mock_async_skd_train.log 2>&1 &

tail -f logs/web_mock_async_skd_train.log
```

For fully async RL:

```bash
nohup bash WebOSWorld/run_qwen35_web_mock_fully_async_rl_tool_fsdp.sh \
  > logs/web_mock_fully_async_rl_train.log 2>&1 &

tail -f logs/web_mock_fully_async_rl_train.log
```

Default script shape:

```text
TRAIN_BATCH_SIZE=16
DATA_MAX_RESPONSE_LENGTH=1024
STUDENT_MAX_MODEL_LEN=3073
TEACHER_MAX_MODEL_LEN=8073
TOTAL_TRAINING_STEPS=4
default_agent_loop=web_skd_agent
```

Fully async RL script shape:

```text
data.max_response_length=8192
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16
default_agent_loop=web_tool_agent
checkpoint_engine.update_weights_bucket_megabytes=4096
```

## 16. Current Milestone Interpretation

`7a404f2f` is the current WebSKD mock GPU milestone.

It confirms:

- actual trainer path, not isolated smoke-only generation
- mock server protocol integration
- image-bearing student generation
- teacher SGLang prompt-logprob verification
- multimodal prefix surplus trimming
- async lookahead, promotion, carryover
- non-tensor metadata normalization
- distillation loss and actor update

It does not claim:

- real WebGym / Omnibox readiness
- browser task quality
- long-horizon dataset coverage
- policy convergence

Operational note: older 64-row mock datasets can exhaust the source before the final step when prefetch is enabled. Use 256 rows or lower prefetch for longer runs.

## 17. One-line Summary

Web / OS Gym integration은 `tool_agent`/`skd_agent` 위에 stateful remote environment session을 얹고, screenshot/a11y observation과 final environment reward를 기존 rollout/training path로 전달하는 구조다. SKD 경로는 `web_skd_agent`, 순수 fully async RL 경로는 `web_tool_agent`를 사용한다.
