# Async SKD Design

## 0. 문서 목적

이 문서는 현재 `verl` 코드베이스에 구현되어 있는 **async SKD**의 실제 구조를 정리한다.

초점은 초기 아이디어나 미구현 계획이 아니라, 지금 코드가 어떤 구조와 경계를 가지고 돌아가는지에 있다. 특히 아래 다섯 가지를 빠르게 파악하는 것을 목표로 한다.

1. async SKD가 풀려는 문제가 무엇인가
2. 어떤 컴포넌트가 어떤 책임을 가지는가
3. current / lookahead / promoted / carryover가 어떻게 동작하는가
4. validation, teacher, training source가 어디서 분리되는가
5. 운영 중 어떤 설정과 로그를 먼저 보면 되는가

이 문서는 구현 완료 후의 **source-of-truth design**이다.

## 1. 문제 정의

tool-aware SKD는 sample 길이 편차가 매우 크다. 어떤 sample은 짧게 끝나지만, 어떤 sample은 긴 reasoning, repeated rejection, tool loop, 장문 response 때문에 step tail을 길게 만든다.

동기식 구조에서는 긴 sample 하나가 step 전체를 늦추고, 먼저 끝난 worker/GPU는 idle에 가까운 상태가 된다. async SKD는 이 tail 구간에서 **sample-level 비동기 스케줄링**을 사용해 유휴 자원을 미래 sample 처리에 재투입한다.

핵심은 다음 두 가지를 동시에 만족하는 것이다.

- systems 측면: tail latency를 줄이고 GPU 유휴를 줄인다
- correctness 측면: teacher alignment, tool-aware masking, current-step batch 의미를 깨뜨리지 않는다

즉 async SKD는 "완전 비동기 trainer"가 아니라, 기존 PPO/distillation trainer 경계는 유지하면서 rollout 내부만 비동기화한 구조다.

## 2. 범위와 비범위

### 포함 범위

- `AsyncSkdAgentLoopManager` 기반 sample-level async rollout
- current work / lookahead / promoted / carryover 관리
- tool-aware `skd_agent` chunk generation + teacher verification
- teacher-only prompt stream
- validation을 student-only path로 분리
- validation과 training source state 격리

### 비범위

- fully async trainer 전체 재설계
- exact importance sampling correction
- live KV migration
- arbitrary mid-handler interrupt/resume
- `rollout.n > 1` 일반화

현재 구현은 **bounded async rollout scheduler**이지, open-ended asynchronous training system이 아니다.

## 3. 최상위 구조

전체 조립은 기존 trainer 경계를 유지한다.

```text
RayPPOTrainer
  -> rollout / validation batch 준비
  -> AsyncSkdAgentLoopManager.generate_sequences(...)
  -> DataProto 수집
  -> 기존 actor update / distillation loss path
```

entry point는 실행 스크립트에서 manager class를 바꾸는 방식이다.

```text
actor_rollout_ref.rollout.agent.agent_loop_manager_class
  = verl.experimental.async_skd.manager.AsyncSkdAgentLoopManager
```

즉 async SKD는 trainer 전체를 포크하지 않고, **agent-loop manager 레이어에서 비동기 스케줄링을 삽입**하는 구조다.

## 4. 주요 구성요소

### 4.1 `skd_agent_loop.py`

역할:

- student chunk 생성
- teacher verification
- first-rejection correction
- tool result span 처리
- teacher row alignment 유지

이 파일은 async 여부와 무관하게 **SKD 의미론의 중심**이다. 비동기 manager는 이 루프를 조각내서 호출할 뿐, assistant token / tool token / teacher row의 의미는 여기서 결정된다.

### 4.2 `AsyncSkdAgentLoopWorker`

역할:

- sample 단위 실행 primitive 제공
- fresh sample과 partial carryover sample을 각각 적절한 경계까지 실행

중요한 점은 worker가 기존 batch API를 대체하는 것이 아니라, async SKD 전용 primitive를 추가한 별도 worker라는 것이다.

### 4.3 `AsyncSkdAgentLoopManager`

역할:

- current work launch
- lookahead admission
- sample completion event 처리
- promoted / carryover bookkeeping
- validation과 training data source 경계 유지

이 manager가 async SKD의 핵심 scheduler다.

### 4.4 `AsyncSkdDataSource`

역할:

- training current batch 원본 source
- lookahead fresh sample reserve
- promoted pair ledger
- carryover partial ledger

주의할 점은 이 source가 **training state**라는 점이다. validation은 이 source를 건드리면 안 된다.

### 4.5 teacher stack

teacher는 학습 대상이 아니라 rollout 시점의 verification / top-k logprob provider다.

역할:

- teacher prompt stream 유지
- top-k teacher rows 반환
- carryover sample에 대해 필요하면 sticky teacher routing 유지

teacher는 trainer update에 맞춰 weight sync되지 않는다.

## 5. 핵심 용어

### 5.1 current work

현재 step에서 실제로 끝까지 완료해야 하는 샘플 집합이다. trainer는 이 집합이 이번 step 학습 입력이 된다고 가정한다.

### 5.2 lookahead

current work가 아직 남아 있는 동안 idle worker가 미래 sample을 미리 시작하는 speculative execution이다.

### 5.3 promoted

lookahead sample이 step barrier 전에 terminal completion에 도달한 경우다. 이 경우 sample은 현재 step 학습 입력 뒤에 append될 수 있다.

### 5.4 carryover

lookahead sample이 끝까지 완료되지는 못했지만 exportable boundary에서 멈춰 다음 step current work로 넘길 수 있는 partial state다.

중요:

- **carryover는 promoted가 아니다**
- carryover는 다음 step current work 앞쪽으로 들어간다
- carryover가 많다는 것은 미래 step current quota가 줄어든다는 뜻이다

### 5.5 exportable boundary

partial snapshot을 만들 수 있는 경계다. handler 내부 임의 시점이 아니라, 바깥 handler가 return한 뒤에만 판단한다.

현재 의미는 대략 다음과 같다.

- `next_state == GENERATING`
- teacher row alignment valid
- tool-call 경계가 exportable

즉 async pause/resume은 임의 interruption이 아니라 **commit 가능한 경계에서만 허용**된다.

## 6. Async control flow

### 6.1 training step

training step에서 manager는 다음 순서로 동작한다.

1. current batch를 만든다
   - carryover current
   - fresh current
   - promoted input/output pair
2. current work를 sample 단위로 launch한다
3. current work가 남아 있는 동안, idle worker slot이 생기면 lookahead를 admit한다
4. current sample이 끝나면 즉시 completion event를 처리한다
5. current work가 모두 끝나면 drain 모드로 들어간다
6. lookahead는
   - terminal이면 promoted
   - partial이면 carryover
   - 아니면 drop
7. 최종 DataProto를 trainer update path로 넘긴다

핵심은 manager가 batch 전체 완료를 기다리지 않고 **sample completion 단위**로 의사결정을 한다는 점이다.

### 6.2 current와 lookahead의 차이

둘의 가장 큰 차이는 "이번 step에서 반드시 끝내야 하느냐"다.

- current sample:
  - 이번 step 학습 입력으로 이미 약속된 sample
  - terminal까지 완료해야 함
- lookahead sample:
  - 아직 speculative sample
  - terminal이면 promoted
  - 아니면 carryover 또는 drop 가능

즉 pause가 허용되는 것은 lookahead 쪽이지, current 쪽이 아니다.

### 6.3 drain phase

current work가 모두 끝나면 더 이상 새로운 lookahead를 admit하지 않는다. 이후에는 이미 활성화된 lookahead task만 정리한다.

이 설계 때문에 특정 step에서 `started`가 `prefetch_limit`보다 1 작게 끝나는 경우가 나올 수 있다. 마지막 current completion 시점에는 drain gate가 먼저 걸리기 때문이다.

## 7. Tool-aware SKD correctness

async 구조는 scheduling만 바꾸고, SKD token semantics는 유지해야 한다.

### 7.1 first-rejection correction

student chunk에서 첫 rejection이 나오면:

- reject 이전 accepted prefix만 유지
- reject 위치는 teacher top-1로 교체
- 그 뒤 student suffix는 버린다

따라서 distillation target은 항상 **실제로 commit된 trajectory**에 대해서만 형성된다.

### 7.2 response mask와 teacher row alignment

assistant-generated token은 KD 대상이고, tool/user/interact span은 KD 대상이 아니다.

즉:

```text
assistant token: response_mask = 1, actual teacher row
tool/user token: response_mask = 0, dummy teacher row
```

절대 깨면 안 되는 불변식:

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

### 7.3 tool macro-step

tool-aware trajectory에서는 assistant tool-call 생성과 tool result append를 하나의 환경 전이로 본다.

즉 아래 전체를 atomic하게 취급한다.

```text
assistant tool-call completion
+ tool parsing
+ tool execution
+ tool result append
+ response_mask zero-span append
+ dummy teacher row append
```

tool result가 중간에 KD chunk처럼 취급되면 alignment가 무너진다.

## 8. Training source와 validation 분리

이 항목은 현재 구현에서 특히 중요하다.

validation은 student-only path로 돌아야 한다. 현재는 `skd_agent -> tool_agent`로 바꿔서 teacher-guided validation을 피한다.

하지만 그것만으로는 충분하지 않다. validation이 같은 async manager를 재사용하는 동안 training `AsyncSkdDataSource`가 붙어 있으면, validation 중에도 training future sample을 prefetch해서 promoted / carryover ledger를 오염시킬 수 있다.

현재 trainer는 validation 동안:

1. manager에서 training `AsyncSkdDataSource`를 잠깐 떼고
2. validation을 수행한 뒤
3. 끝나면 원래 source를 복원한다

이 설계 덕분에 validation 이후에는:

- validation current는 student-only로만 처리되고
- training carryover / promoted count가 validation 때문에 변하지 않아야 한다

과거 `carryover_count > base_batch_size` 류의 문제는 바로 이 분리가 없을 때 발생했다.

## 9. Teacher routing과 sticky carryover

teacher는 별도 pool이며, student처럼 update 후 sleep / weight reload를 반복하지 않는다. 이 때문에 carryover sample을 같은 teacher replica로 다시 보내 KV locality를 노릴 수 있다.

현재 구현은 step 간 teacher sticky carryover reuse를 지원하지만, 항상 강제되는 것은 아니다.

제어 인자:

```text
actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover
```

의미:

- `True`
  - carryover는 가능하면 같은 teacher replica로 hard pin
  - fresh sample은 그 pinned load를 반영한 뒤 rebalance
- `False`
  - carryover도 fresh처럼 다시 분배

현재 기본 실행 스크립트는 `False` 쪽을 사용한다. 따라서 최근 실험 로그를 볼 때는 teacher step-cross KV reuse를 기대하지 않는 쪽이 맞다.

## 10. 주요 설정값

async SKD에서 특히 의미가 큰 인자는 다음이다.

### 10.1 async scheduler

```text
actor_rollout_ref.rollout.agent.async_skd_mode=lookahead
actor_rollout_ref.rollout.agent.async_skd_prefetch_limit
actor_rollout_ref.rollout.agent.async_skd_prefetch_worker_target
actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover
```

해석:

- `prefetch_limit`: 한 step에서 새로 시작할 speculative lookahead sample 총수 상한
- `prefetch_worker_target`: worker당 active sample target
- `teacher_sticky_carryover`: teacher carryover pin reuse on/off

### 10.2 SKD 본체

```text
distillation.skd.chunk_size
distillation.skd.verify_top_k
distillation.skd.max_chunks_per_sample
```

해석:

- `chunk_size`: student generation chunk 크기
- `verify_top_k`: teacher verification 비교 폭
- `max_chunks_per_sample`: 개별 sample hard stop

### 10.3 distillation loss

현재 주요 supervised path는 다음 조합이다.

```text
distillation.distillation_loss.loss_mode=forward_kl_topk
distillation.distillation_loss.topk=32
+distillation.distillation_loss.forward_kl_topk_impl=logsumexp_gather
distillation.distillation_loss.use_policy_gradient=False
```

즉 현재 기준선은 **top-k teacher rows를 사용한 supervised forward KL**이다.

## 11. 운영 중 먼저 볼 로그

### 11.1 step summary

대표 로그:

```text
started
promoted
carryover_next
continued_partial
worker_active_max
```

의미:

- `started`: speculative lookahead 시작 수
- `promoted`: 현재 step으로 끌어온 완료 sample 수
- `carryover_next`: 다음 step current로 넘어가는 partial 수
- `continued_partial`: partial continuation 누적량
- `worker_active_max`: worker active pressure

### 11.2 validation 분리 확인

초기 validation 직후에는 다음이 기대된다.

```text
started=0
promoted=0
carryover_next=0
```

만약 validation에서 training source를 잘못 건드리면, 이 구간에서 lookahead 흔적이 생긴다.

### 11.3 distillation health

W&B에서 먼저 보는 값:

```text
actor/distillation/loss
actor/distillation/teacher_mass
actor/distillation/teacher_mass_min
```

해석:

- `teacher_mass` 평균이 1 근처면 top-k coverage가 대체로 충분
- `teacher_mass_min`만 가끔 낮은 것은 일부 토큰에서 teacher 분포가 넓다는 뜻일 수 있음
- 평균까지 같이 낮아지면 top-k coverage 부족이나 정렬 문제를 의심

## 12. 현재 설계의 한계

현재 async SKD는 실용적인 구조이지만, 다음 제약을 갖는다.

1. validation도 여전히 async manager를 탄다
   - correctness는 확보했지만 systems 측면에선 여전히 무겁다

2. lookahead는 teacher/backlog-aware admission control이 아니다
   - 현재는 sample count와 worker active target 중심이다

3. stale prefix를 별도 수학적 보정으로 correction하지 않는다
   - 현재는 bounded speculative execution으로만 제어한다

4. student는 step 간 KV reuse 대상이 아니다
   - update / offload / residency 전환이 있기 때문이다

## 13. 실무적으로 기억할 것

이 시스템을 짧게 요약하면 다음과 같다.

- SKD semantics의 중심은 `skd_agent_loop.py`
- async behavior의 중심은 `AsyncSkdAgentLoopManager`
- training state의 중심은 `AsyncSkdDataSource`
- validation은 student-only여야 하고 training source와도 격리되어야 한다
- carryover는 promoted가 아니다
- teacher sticky carryover reuse는 옵션이지 항상-on 전제가 아니다

즉 현재 async SKD를 읽을 때 가장 중요한 구분은 이것이다.

**teacher-guided token semantics**와 **async scheduling semantics**를 섞어 생각하지 말 것.

전자는 `skd_agent_loop.py`의 correctness 문제이고, 후자는 manager / source / validation isolation 문제다. 실제 운영에서 생기는 많은 버그는 둘 중 어디가 깨졌는지를 구분하는 것만으로도 절반은 정리된다.
