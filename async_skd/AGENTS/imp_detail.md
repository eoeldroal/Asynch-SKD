# On-Policy SKD 구현 상세

이 문서는 [`AGENTS/onboarding.md`](/home/sogang_nlpy/verl/async_skd/AGENTS/onboarding.md)를 읽고 들어온 독자를 위한 구현 상세 문서다.  
빠른 진입은 `onboarding.md`가 맡고, 본 문서는 논문용 implementation detail에 가까운 관점에서 설계 선택과 시스템 최적화를 정리한다.

## 개요

본 작업은 `verl`의 표준 PPO/agent-loop 경로 위에 다음 세 축을 결합한 것이다.

1. **Speculative Knowledge Distillation (SKD)**  
   학생이 chunk를 제안하고, 교사가 top-k 검증을 통해 first-rejection 교정을 수행한다.
2. **Tool-aware multi-turn rollout**  
   수학 문제를 직접 풀거나, 풀이를 검토하거나, 필요시 `code_interpreter`를 호출하는 mixed-task trajectory를 지원한다.
3. **Teacher-side asymmetric conditioning**  
   학생 rollout은 그대로 유지하되, 교사는 별도 prompt stream을 사용해 tool-grounded verification prior를 더 강하게 반영한다.

핵심 목표는 다음과 같다.

1. SKD의 speculative verification semantics를 정확히 보존할 것
2. shared-prefix cache reuse와 delta prompt-logprob 경로를 활용해 teacher 비용을 최대한 줄일 것
3. tool/user span이 섞인 multi-turn trajectory에서도 distillation target 정렬을 깨지 않게 할 것
4. validation은 student policy 자체를 평가하도록 train-time teacher guidance와 분리할 것

## SKD 루프의 기본 구조

SKD의 본체는 [`verl/experimental/agent_loop/skd_agent_loop.py`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/skd_agent_loop.py)에 있다.

각 sample에 대해 학생은 현재 committed prefix를 기준으로 길이 `chunk_size` 이하의 청크를 제안한다. 이후 다음 절차를 반복한다.

1. 학생이 현재 prefix 위에서 chunk를 생성한다.
2. 교사는 `prefix + chunk`에 대해 prompt-logprob을 계산한다.
3. 각 학생 토큰이 교사 top-`verify_top_k` 안에 있는지 검사한다.
4. 첫 rejection이 발생하면 해당 위치의 학생 토큰을 교사 top-1로 교체한다.
5. accepted prefix와 replacement token만 rollout state에 커밋한다.
6. 수정된 committed prefix를 기준으로 다음 chunk를 이어간다.

즉 본 구현은 **first-rejection 기반의 엄격한 speculative verification**을 따른다. rejection 이후의 학생 suffix는 유지하지 않는다.

## Backend-agnostic Teacher Interface

교사 모델은 [`AsyncTeacherLLMServerManager.compute_teacher_logprobs_single()`](/home/sogang_nlpy/verl/verl/experimental/teacher_loop/teacher_manager.py)에 의해 접근된다.

이 인터페이스는 backend별 차이를 내부에서 흡수한다.

- vLLM: native `prompt_logprobs`
- SGLang: input logprob 경로 + `top_logprobs_num`

SKD는 결국 다음만 소비한다.

- teacher top-k token ids
- teacher top-k logprobs

즉 SKD 알고리즘 자체는 backend-specific payload format에 직접 의존하지 않는다.

## SGLang 사용 방식: Frontend DSL이 아니라 Runtime 직접 호출

현재 구현은 논문에서 소개한 SGLang front-end language(`gen`, `select`, `fork`, `join`, `@function`)를 사용하지 않는다.  
대신 `verl`이 자체 agent loop와 prompt state를 직접 관리하고, SGLang은 **SRT request API를 직접 호출하는 backend runtime**으로 사용한다.

즉 현재 구조는 다음에 가깝다.

- `verl`: prompt/message state 관리, tool loop, SKD accept/reject, 병렬화(`asyncio`)
- SGLang: `GenerateReqInput` 기반 generation, input logprob, prefix cache, memory management

이 구분은 중요하다. 현재 구현이 직접 활용하는 SGLang의 핵심은 front-end language의 분기 primitive가 아니라, **fresh request들 사이의 shared-prefix reuse와 input-logprob interface**다.

## Sticky Routing과 Teacher Cache Locality

교사 검증 요청이 prefix cache를 실제로 활용하려면, 같은 trajectory의 연속 teacher request가 항상 같은 teacher instance로 가야 한다. 이를 위해 현재 구현은 동일 trajectory에 대해 **동일한 바깥쪽 `request_id`를 sticky routing key로 재사용**한다.

이 설계는 [`verl`의 sticky routing](/home/sogang_nlpy/verl/verl/experimental/agent_loop/agent_loop.py)과 결합된다.

- 같은 trajectory
- 같은 teacher replica
- 점진적으로 증가하는 verified prefix

다만 여기서 중요한 점이 하나 있다. 현재 구현은 **SGLang session/continuation API를 사용하지 않는다**.  
`request_id`는 load balancer가 같은 replica를 고르기 위한 키일 뿐이고, 실제 backend request의 `rid`는 매 호출마다 새로 생성된다.

즉 현재 teacher path는:

- 같은 replica로 라우팅되는 fresh request의 연속
- full `prefix + chunk`를 다시 보내는 content-based prefix reuse

구조다. 따라서 현재 구조가 직접 활용하는 것은 논문식 `fork` 기반의 rich tree sharing보다는, **persistent shared-prefix reuse**에 더 가깝다.
이 세 조건이 함께 보장되면, SGLang의 RadixAttention/prefix cache가 실제 효과를 낼 수 있다.

## SGLang Delta Prompt-Logprob 최적화

teacher verification의 핵심 최적화는 SGLang의 `prompt_logprobs_start_len` 경로를 사용해 **현재 suffix에 필요한 row만 반환받는 것**이다.

순진한 구현에서는 매 step마다 `prefix + chunk` 전체에 대한 input logprob row를 materialize한다. 그러나 SKD가 실제로 소비하는 row는 현재 speculative suffix뿐이다.

현재 구현은 다음 두 조건을 동시에 만족한다.

1. 교사에게는 여전히 전체 `prefix + chunk`를 전달한다.  
   즉 prefix-cache match는 유지된다.
2. 반환 payload는 현재 suffix row로 제한한다.  
   즉 teacher serialization / Python postprocess / tensor materialization 비용은 `O(chunk)`에 가까워진다.

구체적으로 `logprob_start_len = len(teacher_prompt_ids) - 1`로 설정되며, 이 값은 청크가 커밋될 때마다 앞으로 전진한다. 덕분에 건너뛰는 대상이 두 종류로 나뉜다.

- **원본 프롬프트 위치 (0 ~ P-2)**: `response_mask=0`이라 supervision 대상이 아니다. 첫 청크부터 영구히 건너뛴다.
- **이전 청크에서 이미 누적된 행 (P-1 ~ current_start-2)**: `teacher_ids_list`에 이미 저장된 행을 중복 요청하지 않는다. 청크가 쌓일수록 이 구간이 늘어난다.

또한 `-1` 오프셋 덕분에 반환된 local index 0번 행이 청크의 첫 토큰을 예측하는 분포가 되어, accept/reject 검증과 teacher target 누적에 곧바로 사용할 수 있다.

이는 teacher-side systems overhead를 줄이면서도 verification semantics를 바꾸지 않는다.  
다만 이 최적화의 주된 효과는 **host-side serialization/materialization 비용 절감**이며, teacher compute 측 핵심 전제는 여전히 shared-prefix cache reuse다.

## Tool-aware SKD

본 구현은 SKD를 single-turn reasoning에만 제한하지 않고, `ToolAgentLoop` 위에 올려 **tool-aware multi-turn SKD**로 확장했다.

현재 구조에서 trajectory는 다음 span이 섞일 수 있다.

- assistant-generated token
- tool response token
- interaction / user response token

이때 `response_mask`는 다음 의미를 갖는다.

- `1`: 학생이 실제로 생성한 assistant token
- `0`: tool/user/intermediate observation token

즉 SKD가 supervise해야 하는 위치와, merely appended observation span이 같은 response 안에 공존한다.

## Distillation Target의 Online 정렬

tool-aware SKD에서 가장 중요한 구현 포인트는 **teacher target을 response-token aligned하게 유지하는 것**이다.

학생이 생성한 assistant token에 대해서만 teacher row를 누적하면, tool response가 중간에 삽입될 때 뒤쪽 assistant span이 teacher row 기준으로 밀린다. 이 문제를 피하기 위해 현재 구현은 online invariant를 사용한다.

- assistant span이 append될 때: 실제 teacher row를 append
- tool/user span이 append될 때: 같은 길이의 dummy teacher row를 append

즉 항상 다음을 유지한다.

- `len(response_mask) == len(teacher_ids_list) == len(teacher_logprobs_list)`

이 설계 덕분에:

- `mask == 1` 위치는 항상 실제 teacher supervision
- `mask == 0` 위치는 placeholder

가 되어, downstream loss가 `response_mask`만 보고도 안전하게 동작한다.

## Teacher Tensor 재조립과 Left-shift Contract

SKD는 rollout 도중 response-aligned teacher row를 온라인으로 누적한다. 반면 distillation loss 경로는 `verl`의 표준 next-token left-shift contract를 기대한다. 따라서 [`AgentLoopWorker._compute_teacher_logprobs()`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/agent_loop.py) 에서는 SKD teacher row를 full-sequence tensor로 재구성한다.

형식은 다음과 같다.

- 앞: `prompt dummy x (prompt_len - 1)`
- 중간: response-token aligned teacher rows
- 끝: final dummy 1개

이 구조를 통해 기존 padding / slicing / actor loss 경로를 그대로 유지하면서, teacher target 공급만 online 형태로 바꿀 수 있었다.

## Teacher-only Asymmetric Prompting

tool-aware setting에서는 교사가 학생과 동일한 prompt만 볼 필요가 없다. 현재 구현은 teacher-only system prompt를 지원하며, 이를 위해 [`SkdAgentLoop`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/skd_agent_loop.py) 안에서 **teacher 전용 prompt stream**을 따로 유지한다.

- `agent_data.prompt_ids`: 학생 rollout용 prefix
- `agent_data.extra_fields["teacher_prompt_ids"]`: teacher verification용 prefix

초기 `PENDING` 단계에서는:

- student messages로 student prompt ids 생성
- teacher-only system guidance를 merge한 messages로 teacher prompt ids 생성

이후 assistant span은 양쪽 stream에 동일하게 append하고, tool/user span도 student delta를 기준으로 teacher stream에 동기화한다.  
teacher verification 시에는 항상 `teacher_prompt_ids + chunk`를 사용하고, `logprob_start_len`도 teacher prefix 기준으로 계산한다.

이 설계의 의미는 분명하다.

- 학생 rollout policy는 유지
- 교사는 tool-grounded verification prior를 더 강하게 반영
- train-time teacher guidance와 student evaluation policy를 분리

## Teacher-only Prompt를 위한 Teacher Budget Reserve

teacher-only system prompt를 추가하면 teacher prefix 길이가 student prefix보다 길어진다. 이를 반영하지 않으면 teacher verify 단계에서 context budget 초과가 발생할 수 있다.

현재 구현은 [`DistillationConfig.__post_init__()`](/home/sogang_nlpy/verl/verl/workers/config/distillation.py)에서 teacher-only prompt가 활성화된 경우 **고정 512-token reserve**를 teacher inference budget에 자동으로 더한다.

적용 대상:

- `teacher_model.inference.max_model_len`
- `teacher_model.inference.max_num_batched_tokens`
- `teacher_model.inference.prompt_length`

이 정책은 run script에 별도 산술을 넣지 않고도 teacher-only asymmetric prompting을 안전하게 사용할 수 있게 해 준다.

## Tool Runtime의 시스템 안정화

tool-aware SKD를 실제로 대규모 batch에서 돌리려면 rollout 알고리즘만 맞다고 충분하지 않다. 본 작업에서는 tool runtime 경로도 함께 정리했다.

### 1. Dataset 단계의 schema-only tool loading

[`RLHFDataset`](/home/sogang_nlpy/verl/verl/utils/dataset/rl_dataset.py)는 prompt-length filtering을 위해 tool schema가 필요하다. 현재 구현은 dataset 단계에서 실제 tool backend를 초기화하지 않고, YAML에서 **static `tool_schema`만 읽는다**.

이 설계는 다음을 방지한다.

- dataset init 시 불필요한 tool actor 생성
- startup 단계의 sandbox backend side effect
- prompt filtering과 runtime execution의 책임 혼합

### 2. Worker-level agent loop reuse

tool runtime은 sample마다 새로 초기화하면 actor churn이 급격히 커진다. 현재 [`AgentLoopWorker`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/agent_loop.py)는 agent loop를 **worker scope에서 재사용**한다.

즉:

- 동일 worker 안에서는 `skd_agent` / `tool_agent`를 한 번만 instantiate
- tool backend도 worker-level lifetime을 갖는다

이 설계는 96+/128 batch 규모에서 tool actor churn을 크게 줄이는 데 중요하다.

### 3. Sandbox execution path의 rate-limit-free support

tool backend는 global rate limiter가 없는 설정도 정상 처리하도록 수정되었다. 이를 통해 local sandbox execution 환경에서 불필요한 singleton rate-limiter failure chain을 피할 수 있다.

## Validation Protocol의 분리

validation은 train-time teacher guidance를 그대로 재현하는 단계가 아니라, **student policy 자체를 평가하는 단계**로 정의한다. 현재 구현은 validation에서도 agent-loop 경로를 타지만, `validate=True`일 때 distillation 경로가 비활성화되므로 train-time teacher-guided rollout과 동일하지 않다.

이 구조의 의미는 다음과 같다.

- train: teacher-guided SKD rollout
- validation: student-only evaluation path

또한 validation sampling은 multi-sample 평가를 위해 `val_kwargs.n > 1`을 사용할 수 있고, trainer는 이를 기반으로 `mean@N`, `best@N` 계열 metric을 자동 집계한다.

## 현재 VeOmni 런에서 실제로 보는 메모리 최적화 축

코드베이스에는 여러 메모리 최적화 옵션이 있지만, 현재 VeOmni async SKD 런에서 진짜로 중요하게 작동하는 축과 “이름은 있으나 현재 비주역인 축”을 분리해서 봐야 한다.

### 현재 활성 축

1. **gradient checkpointing**  
   현재 student model은 gradient checkpointing을 켠다. actor backward peak를 줄이는 가장 기본적인 축이다.

2. **dynamic batch sizing**  
   실제 actor peak는 `ppo_mini_batch_size`보다 `use_dynamic_bsz=True`와 `ppo_max_token_len_per_gpu`가 더 직접적으로 결정한다. 즉 현재는 “몇 row냐”보다 “한 GPU가 한 번에 몇 token을 먹느냐”가 더 중요하다.

3. **parameter / optimizer offload**  
   이 둘은 단순한 메모리 절약 옵션이 아니라, actor residency를 어느 phase에 둘 것인지 결정한다. offload를 켜면 actor util을 잃고, 끄면 rollout resume과 충돌할 수 있다.

4. **rollout SGLang `gpu_memory_utilization`**  
   이 값은 추상적인 tuning knob가 아니라 runtime `mem_fraction_static`으로 직접 들어간다. student rollout weight-resume 안정성에 직접 듣는다.

5. **distillation `forward_kl_topk_impl`**  
   현재 FSDP distillation forward는 `logsumexp_gather` 경로를 선택할 수 있다. 이 경로는 exact forward를 유지하면서 full `log_softmax` materialization을 피하는 방향의 더 가벼운 구현이다. 다만 full-vocab reduction 자체를 없애 주는 마법은 아니다.

### 현재 비주역 / 비활성 축

1. **TiledMLP**  
   코드베이스와 config 옵션은 존재하지만, 현재 VeOmni async SKD 경로의 핵심 활성 최적화로 보면 안 된다.

2. **entropy chunking / entropy checkpointing**  
   옵션은 있으나 현재 pure distillation 위주의 actor update에서는 주역이 아니다.

3. **activation offload**  
   관련 context 경로는 존재하지만, 현재 기본은 `enable_activation_offload=False`다. 즉 현재 메인 해법으로 읽으면 안 된다.

이 구분을 하지 않으면, 실제 런의 memory bug를 TiledMLP나 entropy 쪽에서 찾는 식의 오판이 생긴다.

## 최근 시스템 통합 메모

최근 안정화 과정에서 알고리즘 자체보다 시스템 통합 이슈가 반복적으로 드러났다. 현재 구현을 읽을 때는 아래 네 가지를 함께 염두에 둔다.

1. student rollout과 teacher inference의 SGLang backend는 현재 `triton` / `triton_attn`으로 명시한다. Blackwell 환경에서 `fa3`가 잡히면 startup 단계에서 깨질 수 있다.
2. 포트 충돌은 한 레이어가 아니다. Ray worker group master port와 SGLang 내부 `nccl_port`는 서로 다른 경로라서, 하나를 고쳐도 다른 하나가 남을 수 있다.
3. async SKD는 `skd_agent` registration에 의존한다. 이 loop는 decorator 등록형이므로, `skd_agent_loop.py`가 startup import chain에 포함되지 않으면 registry에서 빠진다.
4. 현재 student 학습 엔진은 RL 경로와 최대한 맞추기 위해 VeOmni 쪽 설정을 사용하는 방향으로 정리 중이다. rollout 계층과 학습 엔진 계층을 같은 문제로 섞지 않는다.

## 최근 메모리 장애를 읽는 방법

최근 런에서 메모리 장애는 세 종류가 따로 나타났다.

1. **actor backward OOM**
   - 위치: `update_actor -> loss.backward()`
   - 해석: actor training peak 문제
   - 주 레버: `ppo_max_token_len_per_gpu`, train batch, offload

2. **student rollout weight-resume OOM**
   - 위치: `update_weights -> rollout.resume(tags=["weights"])`
   - 해석: actor residency와 student rollout SGLang weight resume의 충돌
   - 주 레버: rollout `gpu_memory_utilization`, offload

3. **unsupported offload 조합 assert**
   - 위치: `BaseEngine.to()`
   - 해석: `param_offload=False`, `optimizer_offload=True`는 현재 계약상 허용되지 않음

즉 최근 메모리 디버깅에서 가장 중요한 습관은 “전부 OOM”으로 묶지 않고, **어느 phase에서 죽는지 먼저 자르는 것**이다.

## Dataset Standardization: Nemotron-Cascade-RL-Math

현재 실험에서는 [`Nemotron-Cascade-RL-Math`](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-RL-Math) train split을 tool-aware parquet로 전처리한다.

전처리의 핵심 정책은 다음과 같다.

- `problem` 전체를 user prompt로 사용
- `answer`는 가공 없이 reward ground truth로 사용
- `source`, `task_type`, 원문 `problem`, `answer`는 `extra_info`에 보존
- mixed-task dataset 내부의 final-answer carrier instruction은 제거하고, 전역 system prompt의 `\boxed{}` 규칙으로 통일
- 단, `decimal`, `base 10`, `without units` 같은 semantic formatting constraints는 유지

즉 dataset-level task 의미는 유지하되, answer-carrier convention만 system prompt 수준에서 통일한다.

## 현재 구현이 의미하는 바

정리하면, 현재 구현의 핵심 기여는 다음 다섯 가지로 요약할 수 있다.

1. SKD를 agent-loop 수준에서 구현하여 backend 교체 가능성을 유지
2. teacher sticky routing + SGLang delta prompt-logprob 경로를 결합해 teacher 검증 비용을 줄임
3. tool-aware multi-turn trajectory에서도 distillation target을 online으로 정확히 정렬
4. teacher-only prompt stream을 분리해 asymmetric teacher conditioning을 지원
5. bounded async SKD lookahead로 current work 중 빈 worker slot을 다음 step sample 처리에 사용
6. dataset/tool runtime/validation protocol을 함께 정리해 실제 대규모 on-policy 실험이 가능한 시스템으로 만들었음

즉 본 구현은 단순히 speculative decoding 아이디어를 붙인 수준이 아니라, **tool-aware on-policy distillation을 backend 효율, target 정렬, validation semantics까지 포함해 end-to-end로 구성한 시스템 구현**으로 볼 수 있다.

## Bounded Async SKD Lookahead

현재 async 구현은 별도 fully-async trainer가 아니다. `AsyncSkdAgentLoopManager`가 rollout step 내부에서 current work와 lookahead work를 함께 관리한다.

현재 mode는 다음 값으로 결정된다.

```text
actor_rollout_ref.rollout.agent.async_skd_mode
```

지원 값은 `sync`, `disabled`, `none`, `sample_async`, `lookahead`다. `sample_async`와 `lookahead`는 `rollout.n == 1`만 지원한다.

`lookahead`에서 current work는 다음 순서로 구성된다.

```text
carryover partials first
fresh samples second
```

lookahead admission은 다음 두 값으로 제한된다.

```text
actor_rollout_ref.rollout.agent.async_skd_prefetch_limit
actor_rollout_ref.rollout.agent.async_skd_prefetch_worker_target
```

`async_skd_prefetch_limit`은 step 전체에서 새로 reserve할 lookahead sample 수의 상한이다. `async_skd_prefetch_worker_target`은 worker별 active request target이다. 값이 `0` 이하이면 worker capacity를 target으로 사용한다.

completed lookahead sample은 promoted sample이다. source ledger가 promoted input/output pair를 보관하고, trainer는 current batch 뒤에 붙인다. 단, final train batch가 DP size로 나누어지도록 append 가능한 수만 붙인다. 남은 promoted pair는 pending으로 남는다.

unfinished lookahead sample은 partial carryover다. drain 시점에 exportable boundary까지 진행한 뒤 carryover로 기록된다. 현재 구현은 별도 `async_skd_max_old_gen_chunks` hard cap을 사용하지 않는다. sample-level generation의 종료 cap은 SKD 본체의 `distillation.skd.max_chunks_per_sample`이다.

`PREFETCH_LIMIT=0`은 lookahead prefetch만 비활성화한다. 이 경우에도 current batch는 `AsyncSkdAgentLoopManager`의 sample-level scheduling을 탄다. 동기 SKD baseline과 동일하다고 주장하려면 동기 SKD 스크립트를 사용한다.

## Async SKD Observability

훈련 프로세스는 `VERL_ASYNC_SKD_EVENT_LOG`가 설정되어 있으면 JSONL event log를 남긴다. 주요 event는 다음이다.

```text
sample_launch
sample_finish
lookahead_admit
drain_start
carryover_record
rollout_summary
chunk_commit
replica_request_start
replica_request_finish
```

대시보드는 이 JSONL을 tail해서 scheduler worker slot, student generation replica, teacher verification replica, LT candidate, anomaly를 보여준다.

W&B에는 step-level summary만 남긴다. worker별 상세 분포와 replica별 상태는 event log/dashboard에서 본다.
