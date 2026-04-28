# Async SKD Details

이 문서는 최근 구현과 실행에서 확인된 세부 운영 지식을 정리한다. 오래된 단발성 장애 메모는 제거하고, 지금도 재사용할 수 있는 판단 기준만 남긴다.

## 1. Validation은 training source와 분리한다

validation은 teacher-guided rollout이 아니라 student policy 평가다.

현재 trainer는 validation 시:

1. `skd_agent`를 `tool_agent`로 바꾼다.
2. `AsyncSkdAgentLoopManager`에서 training `AsyncSkdDataSource`를 잠깐 detach한다.
3. validation 이후 source를 복원한다.

이 격리가 없으면 validation 중에도 future training samples가 reserve되어 promoted/carryover ledger가 오염된다.

정상 validation 직후에는 async scheduling metric에서 lookahead 흔적이 없어야 한다.

```text
lookahead_started_count=0
lookahead_promoted_count=0
lookahead_carryover_count=0
```

## 2. Async lookahead는 dataset row를 실제로 소비한다

lookahead는 가짜 예약이 아니라 `AsyncSkdDataSource`에서 future row를 reserve한다. 따라서 작은 dataset에서는 prefetch가 source를 빠르게 소모한다.

Web mock milestone에서 `Training Progress 75% (3/4)`까지 간 뒤 종료된 것은 rollout crash가 아니라 64-row dataset과 prefetch 설정의 조합으로 source가 고갈된 결과다.

실험을 길게 보려면:

- dataset row 수를 늘린다.
- `ASYNC_SKD_PREFETCH_LIMIT`을 낮춘다.
- `TOTAL_TRAINING_STEPS`를 dataset size에 맞춘다.

## 3. DataProto non-tensor metadata는 boundary에서 정규화한다

Async SKD에서는 fresh output, promoted output, carryover continuation output이 서로 다른 경로에서 나온다. 이때 `agent_name`, `tools_kwargs`, `extra_info` 같은 non-tensor keys가 조금만 어긋나도 `DataProto.concat()` 또는 `DataProto.union()`에서 터진다.

현재 정리된 원칙:

- worker나 loop 내부에서 ad hoc하게 key를 맞추지 않는다.
- `verl/experimental/async_skd/metadata.py` helper를 source of truth로 둔다.
- concat 전에는 `align_non_tensor_keys_for_concat()`.
- trainer union 전에는 `sync_output_non_tensor_with_input()`.

이 경계 정규화가 없으면 다음 류의 에러가 난다.

```text
AssertionError: Key 'agent_name' is not present
AssertionError: `agent_name` ... are not the same object
AssertionError: key agent_name length ... is not equal to batch size
```

## 4. Teacher row alignment 불변식

SKD target은 assistant-generated token에 대해서만 실제 teacher row를 가진다. tool/user/interact span은 KD 대상이 아니므로 dummy row를 넣고 `response_mask=0`이어야 한다.

깨면 안 되는 불변식:

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

tool result를 중간에 append하면서 dummy row를 빼먹으면 distillation loss가 잘못된 token에 붙는다.

## 5. First-rejection correction

teacher verification에서 첫 rejection이 나오면:

1. reject 이전 accepted prefix만 유지한다.
2. reject 위치는 teacher top-1 token으로 교체한다.
3. 그 뒤 student suffix는 버린다.

따라서 실제 commit된 trajectory와 teacher target이 항상 같은 sequence 위에 있어야 한다. async carryover도 이 commit 경계 밖에서만 export된다.

## 6. WebSKD observation은 atomic bundle로 commit한다

Web tool observation은 여러 상태를 동시에 바꾼다.

- image data
- student messages
- teacher messages
- local prompt ids
- server prompt ids
- teacher server prompt ids
- response mask
- dummy teacher rows

이 중 일부만 들어가면 다음 carryover나 teacher verification에서 상태가 어긋난다. 현재 WebSKD는 pending observation과 tool observation 모두 tokenization/recompute/guard가 성공한 뒤 한 번에 commit한다.

response length cutoff나 teacher context guard에 걸리면 bundle 전체가 들어가지 않는다.

## 7. WebSKD prompt stream은 두 층이다

WebSKD에서는 local prompt ids와 server prompt ids가 다르다.

- local prompt ids: processor-expanded ids. image가 들어가면 길이가 커질 수 있다.
- server prompt ids: SGLang logical prompt ids. `/generate` 요청과 prompt-logprob delta 계산 기준이다.

따라서 WebSKD에서 `server_prompt_ids` / `teacher_server_prompt_ids`를 local ids로 fallback하는 것은 위험하다. 현재 WebSKD는 prompt stream이 없으면 fail-fast하거나 messages에서 재계산한다.

## 8. Multimodal prefix surplus는 동적으로 처리한다

이미지는 SGLang 내부에서 여러 token row로 확장될 수 있다. 이 expansion 크기는 이미지 수, 모델 processor, backend 구현에 따라 달라질 수 있으므로 고정값으로 보면 안 된다.

현재 teacher logprob extraction은 다음 구조를 따른다.

```text
teacher local prefix length
- teacher server logical prefix length
= expected multimodal prefix surplus
```

SGLang이 반환한 prompt-logprob rows에서 surplus에 해당하는 앞부분은 SKD 검증 대상이 아니므로 버리고, 뒤쪽 suffix chunk rows만 사용한다.

여러 턴에서 이미지가 누적되어도 prefix length 차이로 매번 계산하므로 고정 `219` 같은 값을 가정하지 않는다.

## 9. Teacher context overflow는 graceful termination으로 처리한다

교사는 student보다 긴 context를 볼 수 있다.

- teacher-only system prompt
- a11y tree
- image expansion
- tool observation text

이 때문에 student는 아직 괜찮아도 teacher call이 max model len을 넘을 수 있다. 이 경우 SGLang server-side ValueError로 터뜨리지 않고 loop가 정상 종료해야 한다.

현재 guard:

- SKD chunk verify 직전: `teacher_server_prompt_ids + chunk` 길이를 검사한다.
- Web observation commit 직전: observation을 넣은 뒤 최소 1 verified token 공간이 남는지 검사한다.

초과 시 `teacher_context_exhausted`로 종료한다.

## 10. Malformed computer tool payload

모델이 `actions`를 schema대로 내지 않을 수 있다. 예를 들어 list 안에 dict가 아니라 string이 들어올 수 있다.

이 경우를 Python `TypeError`로 trainer까지 터뜨리면 안 된다. 현재 `WebOsGymTool`은 malformed action을 action failure observation으로 바꿔 loop에 돌려준다. image가 없기 때문에 이 text는 student와 teacher 모두에게 제공된다.

이 처리는 모델 형식 오류를 환경 feedback처럼 처리하기 위한 것이며, transport failure와 다르다.

## 11. Session ID와 Task ID

`task_id`는 dataset row에서 온다. 서버에 어떤 task를 열지 지정한다.

`session_id`는 agent loop/runtime이 만든다. 같은 trajectory 안의 `start`, `action`, `reward` 요청은 모두 같은 `session_id`를 사용한다.

이 둘을 섞으면 병렬 Web/OSGym training에서 서로 다른 trajectory가 같은 서버 세션을 공유하거나, 같은 trajectory가 중간에 세션을 잃는 문제가 생긴다.

## 12. SGLang backend와 model len

현재 실행 스크립트는 SGLang backend를 명시한다.

```text
attention_backend=triton
mm_attention_backend=triton_attn
```

Blackwell 계열 환경에서 자동 선택된 backend가 맞지 않으면 student/teacher server startup 단계에서 깨질 수 있다.

Web mock script는 teacher max model len을 student보다 크게 둔다.

```text
STUDENT_MAX_MODEL_LEN=3073
TEACHER_MAX_MODEL_LEN=8073
```

teacher-only 정보가 늘어나면 teacher max model len과 max num batched tokens를 같이 조정한다.

## 13. Actor memory 문제는 phase별로 나눈다

모든 OOM을 같은 문제로 보면 안 된다.

1. actor backward OOM
   - 위치: `update_actor -> loss.backward()`
   - 주요 레버: `ppo_max_token_len_per_gpu`, batch size, offload

2. student rollout weight-resume OOM
   - 위치: `update_weights -> rollout.resume(tags=["weights"])`
   - 주요 레버: rollout `gpu_memory_utilization`, actor residency, offload

3. unsupported offload 조합
   - `param_offload=False`, `optimizer_offload=True`는 현재 invariant와 맞지 않는다.

## 14. W&B teardown noise

학습 종료 시점에 다음과 같은 로그가 나올 수 있다.

```text
Exception ignored in atexit callback
RuntimeError: unable to perform operation on <UnixTransport closed=True ...>
```

이 로그만으로 SKD 실패라고 판단하지 않는다. 직전의 trainer step metric, Ray traceback, 실제 exception 위치를 먼저 본다. Web mock milestone에서는 이 로그가 나왔지만 step metric은 정상 기록되었다.

## 15. Debug log 레벨

```text
VERL_SKD_DEBUG=0  # quiet
VERL_SKD_DEBUG=1  # chunk summary
VERL_SKD_DEBUG=2  # token/alignment detail
```

Web mock script는 기본 `VERL_SKD_DEBUG=2`와 `VERL_ASYNC_SKD_TRACE=2`를 사용한다. 장시간 run에서는 로그량과 성능 영향을 고려해 낮출 수 있다.

## 16. 현재 가장 먼저 볼 metric

Async:

- `async_skd/lookahead_started_count`
- `async_skd/lookahead_promoted_count`
- `async_skd/lookahead_carryover_count`
- `async_skd/teacher_pinned_carryover_count`
- `async_skd/teacher_fallback_carryover_count`

Distillation:

- `actor/distillation/loss`
- `actor/distillation/teacher_mass`
- `actor/distillation/teacher_mass_min`
- `actor/distillation/loss_max`

Rollout:

- `response_length/max`
- `response_length/clip_ratio`
- `response/aborted_ratio`
- `[SKD] ... done=... rate=...`

Web server:

- `logs/mock_web_osgym_requests.jsonl`
- same `session_id` across `start -> action* -> reward`
- expected `task_id`
