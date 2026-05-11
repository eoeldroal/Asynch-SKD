# Web / OS Gym Protocol

이 문서는 VERL 측 client와 RL environment server가 공유하는 HTTP protocol 계약을 정리한다. 구현 내부의 학습 로직이나 trainer 세부사항은 포함하지 않는다.

## Overview

- Request sender: VERL
- Response sender: RL Environment
- Endpoint: `POST /`
- Health endpoint: `GET /health` 권장
- Request body: JSON
- Response body: JSON

지원 operation:

- `start`: task/session 시작 및 최초 observation 반환
- `action`: action 수행 및 다음 observation 반환
- `reward`: 최종 reward 반환

## Identifiers

### `task_id`

서버에 등록된 task를 식별한다.

예:

- WebGym: 5~7자리 숫자 string
- OSWorld: UUID string

### `session_id`

하나의 환경 session을 식별한다.

- VERL/client 측이 생성한다.
- server는 이 값을 session key로 사용한다.
- 같은 trajectory의 `start -> action* -> reward` 요청은 같은 `session_id`를 사용한다.
- integer를 사용한다.

`task_id`와 `session_id`는 서로 다른 개념이다. 같은 `task_id`를 여러 session에서 동시에 실행할 수 있다.

## Start

### Request

```json
{
  "session_id": 123,
  "task_id": "12345",
  "op": "start",
  "include_a11y": true
}
```

### Response

```json
{
  "session_id": 123,
  "task_id": "12345",
  "status": "ok",
  "text": "A11Y_TREE:\n...",
  "image": {
    "data": "...",
    "mimeType": "image/png"
  }
}
```

`include_a11y=true`일 때 server는 가능한 경우 `text`에 a11y tree 또는 그에 준하는 textual observation을 포함한다.

## Action

### Request

```json
{
  "session_id": 123,
  "task_id": "12345",
  "op": "action",
  "include_a11y": true,
  "actions": [
    {
      "action_type": "CLICK",
      "button": "left",
      "x": 100,
      "y": 200,
      "num_clicks": 1
    }
  ]
}
```

`actions`는 한 개 이상의 low-level computer actions를 순서대로 담는다.

### Model-facing vs wire contract

현재 `verl` WebGym RL / Async SKD 경로의 model-facing contract와 server wire contract는 구분된다.

- model-facing contract
  - bundled `computer(actions=[...])`
  - `qwen3_coder` tool-call serialization
  - tool schema source:
    - `WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml`
- wire contract
  - 이 문서에 적힌 `op=start|action|reward`
  - `actions=[{action_type: ...}]` payload

즉 모델은 직접 wire-level HTTP JSON을 생성하지 않는다. 모델은 bundled `computer` tool 안에 Computer 13
action list를 내고, `WebOsGymTool`이 이를 server wire request로 바꾼다.

### Success Response

```json
{
  "session_id": 123,
  "task_id": "12345",
  "status": "ok",
  "text": "A11Y_TREE:\n...",
  "image": {
    "data": "...",
    "mimeType": "image/png"
  }
}
```

### Action Failure Response

Action 자체가 실패했지만 server가 정상적으로 실패 원인을 반환할 수 있는 경우, HTTP error 대신 observation response로 반환할 수 있다.

```json
{
  "session_id": 123,
  "task_id": "12345",
  "status": "ok",
  "text": "At failed_action_index 2, action failed. Reason: target field was not focused",
  "image": null
}
```

Screenshot을 반환할 수 없는 실패에서는 `image`를 `null` 또는 빈 object로 둘 수 있고, 구현에 따라 필드를 생략할 수도 있으며, 실패 원인은 `text`에 포함한다.

이 응답은 여전히 하나의 environment observation bundle이다. 따라서 client / loop는 이를 "image가 없는 오류"가 아니라, 다음 assistant turn이 복구 행동을 결정하는 데 사용할 수 있는 **text-only failure observation**으로 취급할 수 있다.

## Reward

### Request

```json
{
  "session_id": 123,
  "task_id": "12345",
  "op": "reward"
}
```

### Response

```json
{
  "session_id": 123,
  "task_id": "12345",
  "status": "ok",
  "reward": 1.0
}
```

`reward`는 float이다. 성공/실패 task에서는 보통 `1.0` 또는 `0.0`을 반환할 수 있다.

## Image Field

`image.data`는 base64 encoded PNG payload다.

```json
{
  "data": "...",
  "mimeType": "image/png"
}
```

현재 계약에서는 `mimeType="image/png"`를 기본으로 한다.

`image` metadata는 observation마다 sparse/optional하다. wire-level response에서는 `image: null` 또는 field omission이 가능하다. 다만 training-side reconstructed rows에서는 screenshot이 없는 step에 대해 `images=[]` 같은 fake visual placeholder를 materialize하지 말아야 한다.

## Termination

Action type에는 `DONE`과 `FAIL`이 있다.

- `DONE`: 모델이 task 완료를 선언한다.
- `FAIL`: 모델이 task 실패 또는 진행 불가를 선언한다.

Server는 `DONE` 또는 `FAIL`을 받으면 reward를 계산하고 browser/session resource를 정리할 수 있다. 이후 client는 `reward` operation으로 최종 reward를 요청한다.

중요한 점:

- `DONE` / `FAIL`의 학습 의미는 terminal action 자체다.
- 최종 reward는 별도 `reward` operation으로 회수한다.
- client는 `DONE` / `FAIL` 이후에 추가적인 learnable observation step이 새로 생긴다고 가정하면 안 된다.
- 구현이 `DONE` / `FAIL`에 대한 `action` response에서 text나 image를 반환하더라도, 그것은 protocol 관점에서 reward fetch를 대체하지 않으며 별도의 후속 학습 step을 암시하지 않는다.

시스템 종료 이유는 client 내부에서 관리할 수 있으며, protocol request에는 반드시 포함될 필요는 없다.

예:

- `model_done`
- `model_fail`
- `max_step`
- `max_length`

## Action Detail

각 action은 다음 기본 형태를 가진다.

```json
{
  "action_type": "<ACTION_TYPE>"
}
```

Action별 parameters:

```text
MOVE_TO:
  x, y

CLICK:
  button, x, y, num_clicks

MOUSE_DOWN:
  button

MOUSE_UP:
  button

RIGHT_CLICK:
  x, y

DOUBLE_CLICK:
  x, y

DRAG_TO:
  x, y

SCROLL:
  dx, dy

TYPING:
  text

PRESS:
  key

KEY_DOWN:
  key

KEY_UP:
  key

HOTKEY:
  keys

WAIT:
  no parameters

FAIL:
  no parameters

DONE:
  no parameters
```

### Model-facing keyboard contract

현재 `verl`의 model-facing keyboard contract는 wire-level server contract보다 더 좁다.

- `PRESS`, `KEY_DOWN`, `KEY_UP`
  - `key`는 **single key name**이어야 한다.
  - `ctrl+a` 같은 combo string은 허용하지 않는다.
- `HOTKEY`
  - `keys`는 **single key name들의 배열**이어야 한다.
  - `["Control", "A"]`, `["Control", "Shift", "T"]` 같은 형태를 canonical form으로 본다.
  - `ctrl+a` 같은 combo string은 model-facing 출력에서 나올 수 있지만, 현재 `verl` runtime은 이를
    `["Control", "A"]` 같은 배열로 분해/정규화한 뒤 server로 보낸다.

현재 Linux-focused canonical key normalization은 다음과 같다.

- modifier canonical form
  - shortcut intent: `ctrl`, `control`, `cmd`, `command`, `controlormeta` -> `Control`
  - literal OS modifier: `meta`, `super`, `win`, `windows` -> `Meta`
  - `option` -> `Alt`
- special key aliases
  - `return` -> `Enter`
  - `esc` -> `Escape`
  - `del`, `suppr` -> `Delete`
  - `bksp` -> `Backspace`
  - `pgup` -> `PageUp`
  - `pgdn` -> `PageDown`
- function keys
  - `f1` .. `f12` -> `F1` .. `F12`
- HOTKEY alphabetic keys
  - canonical form은 `A`, `C`, `V`, `T`처럼 대문자를 사용한다.

중요한 점:

- server wire contract는 여전히 concrete Playwright key names를 기대한다.
- `ControlOrMeta` 같은 cross-platform abstract key name은 현재 canonical wire contract가 아니다.

## Logging Recommendation

Server-side request logging is strongly recommended for debugging. Each log record should include at least:

- timestamp
- `op`
- `session_id`
- `task_id`
- action count and action payload for `action`
- reward for `reward`
- response status

This makes it possible to verify that all requests in one trajectory use the same `session_id` and expected `task_id`.
