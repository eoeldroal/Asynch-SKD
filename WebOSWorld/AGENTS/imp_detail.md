# Async SKD Implementation Detail

이 문서는 현재 코드의 구현 구조를 설명한다. 논문 draft가 아니라 작업용 source-of-truth다.

## 1. 전체 구조

Async SKD는 trainer 전체를 새로 만든 구조가 아니다. 기존 PPO trainer 경계는 유지하고, rollout manager 레이어에서 sample-level scheduling을 바꾼다.

```text
RayPPOTrainer
  -> AsyncSkdAgentLoopManager
    -> AsyncSkdAgentLoopWorker
      -> skd_agent / web_skd_agent
        -> student SGLang generation
        -> teacher SGLang prompt-logprob verification
  -> DataProto
  -> actor update / distillation loss
```

순수 fully async RL 경로는 별도 trainer를 사용한다.

```text
FullyAsyncTaskRunner
  -> FullyAsyncTrainer
    -> actor update
    -> async checkpoint weight transfer
  -> FullyAsyncRollouter
    -> FullyAsyncAgentLoopManager
      -> web_tool_agent
        -> student SGLang generation
        -> persistent Web/OSGym session actions
        -> final environment reward
  -> message queue
```

핵심 분리:

- `skd_agent_loop.py`: SKD token semantics
- `manager.py`: async scheduling semantics
- `source.py`: training source / promoted / carryover ledger
- `metadata.py`: DataProto metadata normalization
- `teacher_manager.py`: teacher routing and SGLang request boundary
- `web_tool_agent_loop.py`: fully async RL Web/OSGym session semantics without SKD

## 2. SKD token semantics

`SkdAgentLoop`는 student가 chunk를 생성하고 teacher가 해당 chunk를 검증하는 구조다.

기본 흐름:

1. student가 `chunk_size`만큼 생성한다.
2. teacher는 `teacher_server_prompt_ids + chunk`에 대한 prompt logprobs를 계산한다.
3. student token이 teacher top-k 안에 있으면 accept한다.
4. 첫 reject가 나오면 teacher top-1로 교체하고 chunk suffix를 버린다.
5. 실제 commit된 token에 대해서만 teacher rows를 누적한다.

tool/user span은 KD 대상이 아니므로 dummy teacher row와 `response_mask=0`으로 정렬한다.

## 3. Teacher prompt stream

teacher에는 student와 다른 prompt가 들어갈 수 있다.

- math 경로: teacher-only planning system prompt
- WebSKD 경로: teacher-only a11y/text observation

따라서 student prompt ids와 teacher prompt ids는 별도 stream이다. 특히 WebSKD에서는 local ids와 server ids까지 분리된다.

```text
prompt_ids
teacher_prompt_ids
server_prompt_ids
teacher_server_prompt_ids
```

의미:

- `prompt_ids`: student local processor/chat-template ids
- `teacher_prompt_ids`: teacher local processor/chat-template ids
- `server_prompt_ids`: student SGLang logical ids
- `teacher_server_prompt_ids`: teacher SGLang logical ids

SGLang prompt-logprob delta는 server logical ids 기준으로 계산한다.

## 4. SGLang teacher delta extraction

teacher verification은 native SGLang `/generate` prompt-logprob output을 사용한다.

text-only 경로에서는 대체로 다음 길이가 맞는다.

```text
returned prompt_logprob rows == chunk length
```

multimodal 경로에서는 다르다. image placeholder가 SGLang 내부에서 여러 rows로 expansion될 수 있으므로 반환 rows가 다음처럼 보일 수 있다.

```text
multimodal prefix surplus rows + chunk suffix rows
```

현재 구현은 surplus를 동적으로 계산한다.

```text
expected_mm_prefix_surplus =
    len(teacher_prompt_ids) - len(teacher_server_prompt_ids)
```

그리고 반환 rows에서 앞쪽 surplus를 제거한 뒤 chunk suffix rows만 SKD 검증에 사용한다. 이 값은 이미지마다 고정되어 있다고 가정하지 않는다.

## 5. Teacher context budget

teacher는 student보다 긴 context를 가질 수 있다. 그래서 student response budget과 별개로 teacher context guard가 필요하다.

구현 위치:

- `skd_agent_loop.py`
  - teacher verification 직전 검사
- `web_skd_agent_loop.py`
  - Web observation commit 직전 검사
- `teacher_manager.py`
  - routing-key별 max model len 조회

검사 원칙:

```text
teacher_required_len <= teacher_max_model_len
```

검증 직전에는 `teacher_server_prompt_ids + chunk`를 본다. Web observation commit 직전에는 observation을 넣은 뒤 최소 1 future verified token 공간이 남는지 본다.

초과하면 SGLang call을 보내지 않고 `teacher_context_exhausted`로 sample을 정상 종료한다.

## 6. Async manager semantics

`AsyncSkdAgentLoopManager`는 step 내부에서 current work와 lookahead work를 함께 관리한다.

Current work:

- 이번 trainer step에 반드시 포함되어야 하는 sample
- terminal completion까지 진행한다.

Lookahead:

- current가 남아 있을 때 idle worker slot으로 미래 sample을 미리 시작한다.
- terminal이면 promoted로 현재 step 뒤에 붙을 수 있다.
- partial이면 exportable boundary에서 carryover로 다음 step에 넘긴다.

Drain:

- current work가 모두 끝나면 새 lookahead admission을 멈춘다.
- 이미 활성화된 lookahead만 terminal/promoted 또는 partial/carryover로 정리한다.

## 7. Exportable boundary

carryover partial은 아무 시점에서나 자르지 않는다.

현재 export 가능한 상태는 handler가 바깥으로 돌아온 뒤, 다음 generation을 시작할 수 있는 안정 상태다.

조건의 의미:

- tool-call 처리 중간이 아니다.
- teacher row alignment가 깨지지 않았다.
- prompt ids / response ids / masks / teacher rows가 같은 commit 경계에 있다.

이 원칙 때문에 Web observation도 atomic bundle로 commit한다.

## 8. Non-tensor metadata normalization

Async SKD는 여러 output 경로를 합친다.

- current fresh output
- promoted output
- carryover continuation output

각 경로의 `non_tensor_batch` key가 다르면 `DataProto.concat()`과 `DataProto.union()`이 실패한다. 현재는 `metadata.py`가 이 경계를 담당한다.

주요 helper:

- `missing_non_tensor_value()`
- `align_non_tensor_keys_for_concat()`
- `sync_output_non_tensor_with_input()`

원칙:

- output끼리 concat할 때는 key union을 만든다.
- trainer input과 generated output을 union할 때는 input metadata와 output metadata를 동기화한다.
- loop 내부에서 임의로 `agent_name` 같은 key를 땜질하지 않는다.

## 9. Training source

`AsyncSkdDataSource`는 training state다.

역할:

- current fresh sample 공급
- lookahead future sample reserve
- promoted input/output pair 보관
- carryover partial state 보관

validation 동안에는 이 source를 detach한다. validation이 이 source를 건드리면 training future state가 오염된다.

## 10. Teacher sticky carryover

carryover sample은 이전 step에서 이미 teacher replica를 사용했을 수 있다. 가능하면 같은 teacher replica로 다시 보내 KV locality를 활용할 수 있다.

제어 인자:

```text
actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover
```

현재 text/tool script와 Web mock script 모두 sticky carryover를 켠다.

관련 metric:

- `async_skd/teacher_pinned_carryover_count`
- `async_skd/teacher_fallback_carryover_count`

fallback은 pin이 불가능해 일반 routing으로 보낸 경우다.

## 11. WebSKD loop

`web_skd_agent`는 `skd_agent` 위에 Web/OSGym environment lifecycle을 얹은 loop다.

추가 책임:

- runtime-owned `web_osgym_session_id` 생성
- `start`로 initial observation 확보
- `computer` tool action을 environment `action`으로 전송
- terminal 시 `reward` 회수
- student/teacher observation split
- image data 유지
- server prompt ids 재계산

정상 visual observation:

- student: image
- teacher: image + a11y/text

image-less failure:

- student와 teacher 모두 text를 본다.

## 12. Web observation atomic commit

Pending start observation과 tool observation은 모두 bundle로 처리한다.

Bundle에 포함되는 상태:

- images
- student messages
- teacher messages
- local prompt ids
- server prompt ids
- teacher server prompt ids
- response ids/mask
- teacher dummy rows

모든 계산과 guard가 성공하면 commit한다. 실패하면 commit하지 않는다.

이 구조는 cutoff나 teacher context overflow에서 일부 상태만 남는 문제를 막는다.

## 13. WebOsGymTool

`WebOsGymTool`은 모델-facing schema와 server protocol을 분리한다.

모델은 다음 형태의 tool call을 낸다.

```json
{
  "actions": [
    {"action_type": "CLICK", "x": 100, "y": 200, "button": "left"}
  ]
}
```

서버 protocol은 별도다.

```json
{
  "session_id": 123,
  "task_id": "12345",
  "op": "action",
  "actions": [...]
}
```

모델이 protocol JSON을 직접 내는 것이 아니다.

Malformed model payload는 Python exception으로 trainer를 죽이지 않고 action failure observation으로 변환한다.

## 14. Mock Web/OSGym server

mock server는 real WebGym 전에 wire contract와 trainer integration을 검증하기 위한 수동 서버다.

제공 기능:

- `GET /health`
- `POST /` with `op=start|action|reward`
- client/runtime-owned integer `session_id`
- dataset-owned string `task_id`
- base64 PNG image
- optional a11y text
- JSONL request logging

검증 목표:

- 같은 trajectory에서 같은 `session_id`가 유지되는지
- `task_id`가 변하지 않는지
- action list가 protocol대로 들어오는지
- terminal 후 reward가 회수되는지

## 15. Completion milestone 해석

Web mock milestone에서 확인된 것은 모델이 좋은 웹 정책을 배웠다는 뜻이 아니다. 확인된 것은 다음 end-to-end plumbing이다.

- mock Web/OSGym server protocol
- WebSKD initial observation
- image-bearing student generation
- teacher prompt-logprob verification
- multimodal surplus trimming
- malformed action safety path
- async lookahead / carryover / promotion
- metadata-normalized batch assembly
- distillation loss 계산
- actor update

즉 현재 검증 기준은 policy quality가 아니라 시스템 계약이다.
