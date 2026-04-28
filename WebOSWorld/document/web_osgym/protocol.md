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

Screenshot을 반환할 수 없는 실패에서는 `image`를 `null` 또는 빈 object로 둘 수 있으며, 실패 원인은 `text`에 포함한다.

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

## Termination

Action type에는 `DONE`과 `FAIL`이 있다.

- `DONE`: 모델이 task 완료를 선언한다.
- `FAIL`: 모델이 task 실패 또는 진행 불가를 선언한다.

Server는 `DONE` 또는 `FAIL`을 받으면 reward를 계산하고 browser/session resource를 정리할 수 있다. 이후 client는 `reward` operation으로 최종 reward를 요청한다.

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
