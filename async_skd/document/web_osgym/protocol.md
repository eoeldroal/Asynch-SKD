Overview

Protocol Overview

Request: Volcano Engine (VERL)

Response: RL Environment (Modified Web Gym / OSWorld)



Request Type

Start : Task 지정

Action : Action 수행

Reward : 결과 반환



Protocol Detail

Start

Request

VERL 측에서 어떤 task를 어떤 세션으로 관리할지 지정하여 요청

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "op" : "start"
  "include_a11y": boolean,
  // ... Ignored
}



Response

RL_ENV에서 지정된 task에 대한 환경 설정 수행 후 최초의 스크린샷(+ a11y tree)을 반환, request_id, task_id 는 유지 

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "status": "ok",
  "text": "A11Y_TREE:\n...",
  "image": {
    "data": "...",
    "mimeType": "image/png"
  }
}




Action

Request

VERL 측에서 어떤 task를 어떤 세션으로 관리할지 지정하여 요청

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "op" : "action"
  "include_a11y": boolean,
  "actions": [
    <action1>, <action2>, ...
  ]
}



Response

RL_ENV에서 지정된 action을 수행한 후 스크린샷(+ a11y tree)을 반환

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "status": "ok",
  "text": "A11Y_TREE:\n...",
  "image": {
    "data": "...",
    "mimeType": "image/png"
  }
}



action 실패 시 아래와 같이 반환

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "status": "ok",
  "text": "At failed_action_index 2, action Failed. Reason : target field was not focused",
  "image": {}
}



Reward

Request

VERL 측에서 리워드 요청

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "op" : "reward"
  // ... Ignored
}



Response

RL_ENV에서 결과 확인 후 반단

{
  "request_id": <req_id>,
  "task_id": <task_id>,
  "status": "ok",
  "reward": 1.0, 0.0
}



Implementation Detail

image.data는 base64로 encoding

Action Type에는 Done과 Fail 이 있으며, VERL은 해당 두 액션으로 종료 이후 reward를 요청한다. 종료 이유는 다음과 같다.

termination_reason:
  model_done - 모델이 스스로 
  model_fail
  max_step
  max_length

RL_ENV 는 리소스 관리를 위해 DONE 과 FAIL 을 받을 경우 해당 브라우저 세션에서 reward를 미리 계산하고 브라우저 세션을 종료한다.

<req_id> 는 integer

<task_id> 는 OSWorld 및 WebGym TaskID형식을 따른다.

WebGym : 5~7 자리 숫자 string

OSWorld : "bb5e4c0d-f964-439c-97b6-bdb9747de3f4"



Action Detail

<action> 필드는 다음과 같다

{
  "action_type" : <Action_Type>
  ... parameters
}


<Action_Type> 과 parameter 는 아래 확인

MOVE_TO:
  parameters: x, y
CLICK:
  parameters: button, x, y, num_clicks
MOUSE_DOWN:
  parameters: button
MOUSE_UP:
  parameters: button
RIGHT_CLICK:
  parameters: x, y
DOUBLE_CLICK:
  parameters: x, y
DRAG_TO:
  parameters: x, y
SCROLL:
  parameters: dx, dy
TYPING:
  parameters: text
PRESS:
  parameters: key
KEY_DOWN:
  parameters: key
KEY_UP:
  parameters: key
HOTKEY:
  parameters: keys
WAIT:
  parameters: none
FAIL:
  parameters: none
DONE:
  parameters: none