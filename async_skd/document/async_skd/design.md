# Async SKD Design

## 0. Purpose

이 문서는 현재 `verl` 코드베이스에 구현된 async SKD 구조를 정리한다. 초기 아이디어 문서가 아니라, 현재 코드와 실행 스크립트가 따르는 source-of-truth 설계 문서다.

핵심 질문은 네 가지다.

1. SKD token semantics는 어디서 유지되는가
2. async scheduling은 어디서 일어나는가
3. validation, training source, metadata boundary는 어떻게 분리되는가
4. WebSKD / multimodal 확장은 어떤 추가 불변식을 갖는가

## 1. Problem

Tool-aware SKD rollout은 sample 길이 편차가 크다. 짧은 sample은 빨리 끝나지만, 긴 reasoning, tool loop, rejection, 긴 response를 가진 sample은 step tail을 만든다.

동기식 rollout에서는 긴 sample 하나가 step 전체를 늦추고 worker/GPU slot이 idle 상태가 된다. async SKD는 이 tail 구간에서 sample-level lookahead를 사용해 미래 sample을 미리 처리한다.

중요한 점은 async SKD가 fully asynchronous trainer가 아니라는 것이다. PPO/distillation trainer boundary는 유지하고, rollout manager 내부에서만 bounded speculative scheduling을 수행한다.

## 2. Scope

포함:

- `AsyncSkdAgentLoopManager` 기반 sample-level scheduling
- current / lookahead / promoted / carryover 관리
- `skd_agent`와 `web_skd_agent`의 teacher-guided rollout
- teacher prompt-logprob verification
- validation student-only 분리
- training source detach/restore
- non-tensor metadata normalization
- multimodal prompt-logprob surplus handling

비포함:

- fully async trainer 재설계
- importance sampling correction
- live KV migration
- arbitrary mid-handler interruption
- `rollout.n > 1` async generalization

## 3. Top-level Architecture

```text
RayPPOTrainer
  -> AsyncSkdAgentLoopManager.generate_sequences(...)
    -> AsyncSkdAgentLoopWorker.generate_skd_until_boundary(...)
      -> SkdAgentLoop / WebSkdAgentLoop
        -> student SGLang generation
        -> teacher SGLang verification
  -> DataProto
  -> actor update / distillation loss
```

Async SKD는 다음 config로 manager를 교체해 들어간다.

```text
actor_rollout_ref.rollout.agent.agent_loop_manager_class
  = verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager
```

## 4. Component Responsibilities

### `skd_agent_loop.py`

SKD token semantics의 중심이다.

- student chunk generation
- teacher verification
- first-rejection correction
- teacher row accumulation
- response mask / teacher row alignment
- teacher context guard

Async manager는 이 loop를 sample 단위로 실행할 뿐, token correctness를 재정의하지 않는다.

### `web_skd_agent_loop.py`

Web/OSGym 환경 lifecycle을 SKD loop에 얹는다.

- environment `start/action/reward`
- runtime-owned `session_id`
- screenshot image handling
- teacher-only a11y/text channel
- Web observation atomic commit
- Web observation teacher context guard
- logical server prompt stream 유지

### `manager.py`

Async scheduler의 중심이다.

- current work launch
- lookahead admission
- completion event 처리
- promoted/carryover bookkeeping
- finalize output

### `source.py`

Training data source와 async ledger를 담당한다.

- fresh current sample 공급
- future lookahead sample reserve
- promoted input/output pair 보관
- carryover partial 보관

Validation은 이 source를 건드리면 안 된다.

### `metadata.py`

`DataProto.concat()` / `DataProto.union()` 경계에서 non-tensor metadata를 정규화한다.

- fresh output
- promoted output
- carryover continuation output

이 셋은 서로 다른 경로에서 만들어지므로 key set과 length를 경계에서 맞춰야 한다.

### `teacher_manager.py`

Teacher server routing boundary다.

- teacher replica routing
- sticky carryover binding
- routing-key별 max model len 조회
- SGLang request 인자 전달

## 5. Core Terms

### Current work

이번 trainer step에 반드시 포함되어야 하는 sample 집합이다. terminal completion까지 진행한다.

### Lookahead

current work가 아직 남아 있을 때 idle worker slot이 미래 sample을 미리 시작하는 speculative execution이다.

### Promoted

lookahead sample이 step barrier 전에 terminal completion에 도달해 이번 step batch 뒤에 append될 수 있는 상태다.

### Carryover

lookahead sample이 terminal까지 끝나지 않았지만 exportable boundary에서 partial state로 저장되어 다음 step current work 앞쪽으로 들어가는 상태다.

### Exportable boundary

partial snapshot을 만들 수 있는 안정 지점이다. handler 내부 임의 시점이 아니다.

조건:

- next generation을 시작할 수 있다.
- teacher row alignment가 맞다.
- tool processing 중간이 아니다.
- prompt ids, response ids, masks, teacher rows가 같은 commit boundary에 있다.

## 6. Scheduling Flow

한 training step은 다음 순서로 진행된다.

1. current batch를 구성한다.
   - carryover partials first
   - fresh current second
   - appendable promoted pairs if available
2. current samples를 launch한다.
3. current work가 남아 있는 동안 idle worker slot에 lookahead를 admit한다.
4. current sample completion을 sample 단위로 처리한다.
5. current work가 모두 끝나면 drain phase로 들어간다.
6. active lookahead를 정리한다.
   - terminal이면 promoted
   - exportable partial이면 carryover
   - otherwise drop
7. output을 `DataProto`로 finalize한다.

`async_skd_prefetch_limit`은 한 step에서 새로 reserve할 lookahead sample 수의 상한이다. `async_skd_prefetch_worker_target`은 worker별 active target이다.

## 7. SKD Correctness

### First rejection

첫 rejection이 나오면 reject 이전 accepted prefix만 유지하고 reject 위치를 teacher top-1 token으로 교체한다. 그 뒤 student suffix는 버린다.

### Teacher row alignment

assistant-generated token은 KD 대상이다.

```text
assistant token -> response_mask=1, real teacher row
tool/user token -> response_mask=0, dummy teacher row
```

불변식:

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

### Tool macro-step

assistant tool-call completion, tool parsing, tool execution, tool result append, dummy teacher rows append는 하나의 environment transition으로 취급한다.

tool result token을 teacher-verified assistant token처럼 다루면 alignment가 깨진다.

## 8. Validation Semantics

Validation은 train-time teacher guidance 재현이 아니라 student policy 평가다.

현재 구현:

- validation entrypoint에서 `skd_agent`를 `tool_agent`로 전환한다.
- teacher verification을 사용하지 않는다.
- validation 동안 training `AsyncSkdDataSource`를 manager에서 detach한다.
- validation 후 source를 복원한다.

따라서 validation은 training future sample reservation, promoted ledger, carryover ledger를 변경하지 않아야 한다.

## 9. Metadata Boundary

Async SKD는 current, promoted, carryover output을 함께 합친다. non-tensor metadata key가 경로별로 다르면 `DataProto`가 실패한다.

현재 helper:

```text
align_non_tensor_keys_for_concat()
sync_output_non_tensor_with_input()
```

이 helper가 담당하는 것:

- output concat 전 key set 정렬
- missing key에 sentinel object 채움
- trainer input과 generated output union 전 metadata 동기화

이 레이어 바깥에서 `agent_name` 같은 key를 개별적으로 땜질하지 않는다.

## 10. Teacher Routing

Teacher는 별도 server pool이다. Student rollout처럼 update 후 sleep/reload를 반복하지 않는다.

Carryover sample은 같은 teacher replica로 다시 보내 KV locality를 활용할 수 있다.

Config:

```text
actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover=True
```

Metric:

```text
async_skd/teacher_pinned_carryover_count
async_skd/teacher_fallback_carryover_count
```

## 11. Multimodal Extension

WebSKD는 image-bearing prompt를 사용한다. 이때 local processor ids와 SGLang logical server ids가 다를 수 있다.

Prompt streams:

```text
prompt_ids
teacher_prompt_ids
server_prompt_ids
teacher_server_prompt_ids
```

SGLang이 image expansion 때문에 prompt-logprob rows를 더 반환할 수 있으므로, teacher verification은 `expected_mm_prefix_surplus`를 전달한다.

```text
expected_mm_prefix_surplus =
  len(teacher_prompt_ids) - len(teacher_server_prompt_ids)
```

Extraction은 반환 rows 앞쪽 surplus를 제거하고 suffix chunk rows만 사용한다.

## 12. Teacher Context Guard

Teacher context는 student보다 길 수 있다. 특히 WebSKD에서는 a11y text와 image expansion이 누적된다.

Guard 위치:

1. SKD verification 직전
   - `teacher_server_prompt_ids + chunk` 검사

2. Web observation commit 직전
   - observation을 넣은 뒤 최소 1 future verified token 공간이 남는지 검사

초과 시 SGLang call을 보내지 않고 `teacher_context_exhausted`로 정상 종료한다.

## 13. Operational Defaults

Text/tool script:

```text
async_skd/run_qwen35_math_async_skd_tool_fsdp.sh
default_agent_loop=skd_agent
train_batch_size=64
max_response_length=8192
async_skd_prefetch_limit=80
async_skd_prefetch_worker_target=16
async_skd_max_promoted_per_step=48
```

Web mock script:

```text
async_skd/run_qwen35_web_mock_async_skd_tool_fsdp.sh
default_agent_loop=web_skd_agent
train_batch_size=16
max_response_length=1024
student_max_model_len=3073
teacher_max_model_len=8073
```

Web mock prefetch defaults are derived from batch size and worker count.

## 14. Current Milestone Interpretation

`7a404f2f` validates the WebSKD mock RL system path on real GPUs. It does not claim policy quality. It verifies that the current code can pass through:

- mock Web/OSGym protocol
- image observation
- WebSKD prompt stream split
- student SGLang generation
- teacher SGLang prompt-logprob verification
- multimodal surplus trimming
- async lookahead / carryover / promotion
- metadata-normalized batch assembly
- distillation loss and actor update

This is the current implementation baseline.
