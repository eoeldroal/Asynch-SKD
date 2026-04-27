# On-Policy SKD 빠른 온보딩

## 이 문서의 역할

이 문서는 이 작업 영역의 **entry point** 다.  
목표는 처음 들어온 사람이 짧은 시간 안에 다음 네 가지를 바로 파악하게 하는 것이다.

1. 현재 실험이 무엇을 하려는지
2. 어디부터 코드를 읽어야 하는지
3. 어떤 로그를 먼저 봐야 하는지
4. 무엇이 correctness 이슈이고, 무엇이 systems/perf 이슈인지

더 자세한 설명은 [`AGENTS/imp_detail.md`](/home/sogang_nlpy/verl/async_skd/AGENTS/imp_detail.md) 로 내려간다.

## 현재 작업의 한 줄 요약

현재 작업은 **tool-aware speculative knowledge distillation (SKD)** 를 `verl`의 agent-loop 위에 구현하고,  
학생 `Qwen3.5-9B`를 교사 `Qwen3.5-27B`로부터 on-policy distillation 하는 것이다.

현재 실험의 주요 특징은 다음과 같다.

- `skd_agent` 기반 chunked generation + teacher verification
- tool-aware multi-turn rollout (`code_interpreter`)
- teacher-only system prompt 지원
- bounded async SKD lookahead (`AsyncSkdAgentLoopManager`)
- distillation `forward_kl_topk` with `topk=32`
- FSDP distillation forward는 현재 `logsumexp_gather` 구현 선택 가능하며, 현 스크립트는 그 경로를 사용한다
- event-log 기반 dashboard 관측
- train: `Nemotron-Cascade-RL-Math`
- validation: `AIME-2024` + `MATH500`
- validation sampling: `n=4` (`mean@4`, `best@4`), `skd_agent -> tool_agent` student-only path

## 가장 먼저 볼 파일

다음 순서로 보면 가장 빠르다.

1. 실행 스크립트
   [`run_qwen35_math_async_skd_tool_fsdp.sh`](/home/sogang_nlpy/verl/async_skd/run_qwen35_math_async_skd_tool_fsdp.sh)
   현재 bounded async SKD 실험의 기본 entry point다. `AsyncSkdAgentLoopManager`, lookahead prefetch, teacher distillation 설정, 학생 학습 엔진, student rollout / teacher inference SGLang backend가 여기서 결정된다. 현재 기본값은 `MAX_PROMPT_LENGTH=1024`, `MAX_RESPONSE_LENGTH=8192`, `MODEL_ENGINE=veomni`다.

2. SKD 루프 본체
   [`skd_agent_loop.py`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/skd_agent_loop.py)
   chunk generate, teacher verify, tool-aware teacher alignment, teacher-only prompt stream, `skd_agent` registration이 여기에 있다.

3. async SKD manager / source
   [`manager.py`](/home/sogang_nlpy/verl/verl/experimental/async_skd/manager.py), [`data_source.py`](/home/sogang_nlpy/verl/verl/experimental/async_skd/data_source.py)
   current work, lookahead, promoted, carryover, per-worker admission target, source ledger가 여기에 있다.

4. teacher interface
   [`teacher_manager.py`](/home/sogang_nlpy/verl/verl/experimental/teacher_loop/teacher_manager.py)
   teacher prompt-logprob contract와 backend별 차이를 여기서 흡수한다.

5. SGLang wrapper
   [`async_sglang_server.py`](/home/sogang_nlpy/verl/verl/workers/rollout/sglang_rollout/async_sglang_server.py)
   `prompt_logprobs_start_len`, teacher payload 반환 규약, backend 선택(`triton` / `triton_attn`), single-node `nccl_port` 계산이 여기서 나온다.

6. teacher tensor 재구성 / backend validation
   [`agent_loop.py`](/home/sogang_nlpy/verl/verl/experimental/agent_loop/agent_loop.py)
   SKD가 online으로 모은 teacher rows를 downstream distillation 경로가 이해하는 형태로 다시 맞추고, async SKD backend contract validation을 담당한다.

7. trainer 조립
   [`ray_trainer.py`](/home/sogang_nlpy/verl/verl/trainer/ppo/ray_trainer.py)
   actor/rollout/teacher resource pool 조립, validation, async rollout manager 생성, Ray worker group wiring이 여기 있다.

## 현재 실험에서 중요한 입력/리소스

### 모델
- student: [`models/Qwen3.5-9B`](/home/sogang_nlpy/verl/models/Qwen3.5-9B)
- teacher: [`models/Qwen3.5-27B`](/home/sogang_nlpy/verl/models/Qwen3.5-27B)

현재 student `Qwen3.5-9B`는 이름상 9B지만, 실제 checkpoint total size 기준 bf16 parameter count는 약 `9.653B`다.  
즉 actor memory를 볼 때는 막연히 “9B라 여유 있을 것”으로 보면 안 된다.

### 데이터
- train: [`data/nemotron_cascade_rl_math_multiturn_w_tool/train.parquet`](/home/sogang_nlpy/verl/data/nemotron_cascade_rl_math_multiturn_w_tool/train.parquet)
- val:
  - [`data/aime-2024.parquet`](/home/sogang_nlpy/verl/data/aime-2024.parquet)
  - [`data/math500/test.parquet`](/home/sogang_nlpy/verl/data/math500/test.parquet)

### teacher prompt
- current: [`teacher_system_prompt_math_planning.txt`](/home/sogang_nlpy/verl/async_skd/test_mixedgen/teacher_system_prompt_math_planning.txt)

### async SKD 기본값
- `TRAIN_BATCH_SIZE=64`
- `PREFETCH_LIMIT=64`
- `PREFETCH_WORKER_TARGET=16`
- `SKD_CHUNK_SIZE=128`
- `SKD_VERIFY_TOP_K=5`
- `SKD_MAX_CHUNKS=512`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=8192`
- `max_model_len=9217` (= MAX_RESPONSE + MAX_PROMPT + 1, student rollout·teacher 공통)
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288`
- `MODEL_ENGINE=veomni`
- `DISTILLATION_TOPK=32`
- `FORWARD_KL_TOPK_IMPL=logsumexp_gather`
- `distillation.distillation_loss.loss_max_clamp=10.0`
- `distillation.distillation_loss.log_prob_min_clamp=-10.0`
- `distillation.distillation_loss.use_task_rewards=False`
- `distillation.distillation_loss.use_policy_gradient=False`
- student rollout backend: `sglang` with `attention_backend=triton`, `mm_attention_backend=triton_attn`
- teacher inference backend: `sglang` with `attention_backend=triton`, `mm_attention_backend=triton_attn`
- current script offload:
  - `param_offload=True`
  - `optimizer_offload=True`
- current script rollout memory fraction:
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.9`
- tool config path: `examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml`
- multi_turn format: `qwen3_coder`

환경 변수 (스크립트 상단에서 설정):
- `VERL_SKD_DEBUG=1`: per-chunk diagnostics 활성화 (`=2`이면 batch당 첫 3 sample의 token-level alignment 검증 로그 추가 출력)
- `SGLANG_NUMA_BIND_V2=0`: NUMA binding 비활성화 (multi-GPU 환경에서 SGLang process 고정 방지)
- `SGLANG_ENABLE_TORCH_INFERENCE_MODE=1`: SGLang 내부 `torch.inference_mode()` 활성화

기타 주요 config:
- `data.truncation=error`: 오버롱 prompt는 silently truncate하지 않고 에러로 처리
- `data.shuffle=False`: curriculum / 순서 보존을 위해 shuffle 끔
- `data.return_raw_chat=True`: raw message list를 dataset에서 반환
- `trainer.use_legacy_worker_impl=disable`: legacy worker 구현 경로 비활성화 (신규 VeOmni engine 경로 사용)
- `trainer.resume_mode=disable`: 체크포인트 resume 비활성화 (fresh start)

`PREFETCH_LIMIT=0`은 lookahead prefetch만 끈다. 이 경우에도 `AsyncSkdAgentLoopManager`의 sample-level current scheduling은 남으므로 기존 동기 SKD와 완전히 같다고 보면 안 된다. 동기 baseline은 동기 SKD 스크립트를 따로 사용한다.

## 구현에서 기억할 핵심 계약

### 1. SKD는 first-rejection 방식이다

학생 chunk 안에서 첫 rejection이 나오면:
- reject 이전 accepted prefix만 유지
- reject 위치는 teacher top-1로 교체
- 그 뒤 학생 suffix는 버린다

즉 distillation target도 항상 **실제로 커밋된 경로**에만 맞아야 한다.

### 2. tool/user span은 `response_mask=0`이다

tool-aware trajectory에서는 response 안에:
- assistant token
- tool response token
- interaction/user token

이 섞인다.  
현재 구현은 tool/user span에도 dummy teacher row를 같이 넣어서,

- `len(response_mask) == len(teacher_ids_list)`

를 유지한다.

### 3. teacher는 별도 prompt stream을 쓴다

teacher-only system prompt가 들어가면 student prefix와 teacher prefix가 달라진다.  
그래서 현재는:

- student: `agent_data.prompt_ids`
- teacher: `agent_data.extra_fields["teacher_prompt_ids"]`

를 따로 유지한다.

teacher verification은 항상 teacher prompt stream 기준으로 이뤄진다.

### 4. teacher-only prompt를 쓰면 teacher budget reserve가 자동으로 붙는다

teacher prompt가 student prompt보다 길어지므로,
config 레벨에서 teacher inference budget에 고정 `512` 토큰 reserve를 자동으로 더한다.

즉 run script의 raw `max_model_len` 값만 보면 teacher 실제 예산을 과소평가할 수 있다.

### 5. bounded async SKD는 promoted와 carryover를 분리한다

`lookahead` sample이 step barrier 전에 `TERMINATED`가 되면 promoted sample이다. trainer는 promoted input/output pair를 현재 train batch 뒤에 붙인다. 단, final train batch가 DP size로 나누어지도록 append 가능한 수만 붙이고, 남은 promoted pair는 source ledger에 pending으로 남긴다.

`lookahead` sample이 완료되지 못하면 partial carryover다. carryover는 다음 step의 current work 앞쪽에 들어가고, 그만큼 fresh quota가 줄어든다. promoted row는 next-step fresh quota에서 차감하지 않는다.

즉 **carryover는 promoted가 아니다.**  
carryover가 많다는 로그는 “이번 step이 길고 무거웠다”는 간접 신호일 수는 있지만, 그 자체가 곧 현재 actor update batch에 append된 promoted sample이라는 뜻은 아니다.

### 5-1. teacher sticky pin은 carryover에 대해 step 간 유지될 수 있다

현재 구현은 teacher 쪽에 한해 carryover sample의 sticky routing을 step 간으로 연장하는 기능을 지원한다.

- carryover partial은 `extra_fields["teacher_replica_id"]`를 함께 들고 간다
- manager는 `sample_id -> real teacher server_id`와 `sample_id -> teacher_routing_key`를 함께 유지한다
- resumed carryover는 가능한 한 같은 teacher pool 안의 같은 real teacher server로 **hard pin** 된다
- 새 base sample은 carryover pinned load를 먼저 반영한 뒤, 그 위에서 rebalance된다

기능이 켜져 있을 때 현재 정책은:

- carryover: reuse 우선
- base: carryover를 고려한 재분배

중요한 점은, 여기서 말하는 pin 대상이 `teacher-replica-0` 같은 manager 내부 가상 이름이 아니라는 것이다. 실제 source of truth는 teacher load balancer가 아는 **real server_id**다. planner가 실제 teacher server ID를 모르면 bind하지 않고 fallback acquire 경로를 타는 쪽이 맞다.

다만 student는 update phase에서 sleep / weight residency 전환을 거치므로, teacher처럼 step 간 KV reuse의 직접 대상이라고 보면 안 된다.

이 기능은 `actor_rollout_ref.rollout.agent.async_skd_teacher_sticky_carryover`로 켜고 끌 수 있다. 현재 실행 스크립트는 이 값을 `False`로 두고 있으므로, 최근 런을 읽을 때는 carryover가 fresh처럼 다시 배치된다고 보는 편이 맞다.

### 5-2. validation은 student-only일 뿐 아니라 training source와도 격리된다

validation batch는 `tool_agent`로 돌기 때문에 teacher-guided SKD rollout이 아니다. 하지만 그것만으로는 충분하지 않다. validation이 같은 `AsyncSkdAgentLoopManager`를 재사용하는 동안 training `AsyncSkdDataSource`가 붙어 있으면, validation 중에도 training future sample을 prefetch해서 promoted/carryover state를 오염시킬 수 있다.

현재 trainer는 `_validate()` 동안 manager에서 training source를 잠깐 떼고 끝나면 다시 붙인다. 따라서 validation 직후에는:

- validation carryover가 training source에 남지 않고
- 다음 training step의 fresh quota / carryover count가 validation 때문에 변하지 않아야 한다

### 6. 현재 VeOmni memory 레버는 active/inactive를 구분해서 본다

현재 런에서 실제로 메모리/속도에 크게 듣는 축은 다음이다.

- `enable_gradient_checkpointing=True`
- `use_dynamic_bsz=True`
- `ppo_max_token_len_per_gpu`
- `param_offload`, `optimizer_offload`
- rollout SGLang `gpu_memory_utilization`

반대로 코드베이스에 있어도 현재 VeOmni async SKD 런에서 핵심 축으로 보면 안 되는 것들이 있다.

- `TiledMLP`: 코드베이스에는 있으나 현재 VeOmni 경로의 활성 최적화로 보면 안 된다
- entropy chunking / entropy checkpointing: 현재 pure distillation 런에서는 주역이 아니다
- activation offload context: 코드 경로는 있으나 현재 기본은 `enable_activation_offload=False`다

## 로그에서 먼저 볼 것

### correctness
- `teacher_mass_max`
- `distillation/loss_max`
- `AssertionError`
- `Prompt length ... exceeds ...`
- `distillation/loss_max` ≫ 10이면 teacher row 정렬 문제 가능성이 높다. `VERL_SKD_DEBUG=2`로 재실행하면 batch당 첫 3 sample의 token-level alignment 로그를 확인할 수 있다.

### rollout dynamics
- `[SKD] ... done=eos / max_chunks / budget_exhausted`
- `avg_tok/chunk`
- `accept`, `reject`, `rate`
- `student=...ms`, `teacher=...ms`
- `[ASYNC_SKD] step_input`
- `[ASYNC_SKD] rollout`
- `[ASYNC_SKD] train`

### async SKD dashboard

dashboard는 event log를 tail해서 scheduler worker, student replica, teacher replica, LT candidate, carryover/promoted 상태를 보여준다.

현재 teacher carryover pin 관련해서는 다음 두 metric만 보면 된다.

- `async_skd/teacher_pinned_carryover_count`
- `async_skd/teacher_fallback_carryover_count`

콘솔은 일반 carryover/rollout 상태만 요약하고, sticky 관측은 위 두 metric을 source of truth로 본다.

```bash
cd /home/sogang_nlpy/verl

nohup /home/sogang_nlpy/miniconda3/envs/skd/bin/python -m verl.experimental.async_skd.dashboard \
  --event-log /home/sogang_nlpy/verl/logs/async_skd_events_live.jsonl \
  --host 0.0.0.0 \
  --port 10001 \
  > /home/sogang_nlpy/verl/logs/async_skd_dashboard_live.log 2>&1 &
```

훈련은 `VERL_ASYNC_SKD_EVENT_LOG`가 가리키는 JSONL에 이벤트를 쓴다. 기록 보존이 필요하면 run별 event log를 만들고 `logs/async_skd_events_live.jsonl`을 symlink로 연결한다.

### training / systems
- `step:`
- `time/step`
- `throughput`
- `torch.OutOfMemoryError`
- `ActorDiedError`

## 현재까지 자주 나온 문제 유형

1. **teacher row alignment 문제**  
   tool response가 중간에 들어가는데 teacher row를 assistant token만 기준으로 쌓으면 `1280` pathology가 재발한다.

2. **teacher prompt로 인한 context budget 초과**  
   teacher-only prompt를 넣으면 teacher prefix가 길어지므로 별도 reserve가 필요하다.

3. **tool runtime startup side effect**  
   dataset 단계에서 tool backend를 실제로 띄우면 startup이 불안정해진다.

4. **SGLang backend mismatch**  
   Blackwell 환경에서 `fa3` backend가 잡히면 teacher/student SGLang server가 startup 단계에서 실패한다. 현재 기준으로는 `attention_backend=triton`, `mm_attention_backend=triton_attn`를 명시해야 한다.

5. **포트 충돌이 레이어별로 따로 난다**  
   Ray worker group master port 충돌과 SGLang 내부 `nccl_port` 충돌은 다른 문제다. 하나를 고쳤다고 다른 하나가 자동으로 해결되지는 않는다.

6. **agent registry import 누락**  
   `skd_agent`처럼 decorator 등록형 loop는 startup import가 빠지면 registry에서 사라진다. `Agent loop skd_agent not registered`가 뜨면 registration decorator 자체보다 package import chain을 먼저 본다.

7. **actor-side OOM**  
   긴 response + 큰 batch + update_actor backward에서 자주 난다. 이 경우 teacher가 아니라 actor mini-batch budget 문제로 보는 게 맞다.

8. **unsupported offload 조합**  
   현재 코드 계약상 `param_offload=False`, `optimizer_offload=True`는 지원되지 않는다. train/eval context 진입 시 `engine.to(model=False, optimizer=True, grad=False)`가 만들어지고, base invariant assert에 걸린다.

9. **student rollout resume OOM**  
   actor update는 통과했는데 그 다음 `update_weights -> rollout.resume(tags=["weights"])`에서 `resume_memory_occupation` timeout과 `SGLangHttpServer ... out of memory`가 나면, 이건 actor backward가 아니라 student rollout weight resume 실패다. 특히 `param_offload=False`일 때 actor residency가 GPU에 남아 있으면 이 문제가 잘 난다.

## 무엇을 먼저 의심할 것인가

문제가 생기면 아래 순서로 본다.

1. config/context 문제인가  
   - `teacher_system_prompt_path`
   - teacher budget
   - backend override (`triton` / `triton_attn`)
   - script가 실제로 `model_engine=veomni`를 타는지
   - validation `n`

2. teacher alignment 문제인가  
   - `teacher_mass_max`
   - `loss_max`

3. rollout 자체가 너무 긴가  
   - `done=budget_exhausted`
   - `avg_tok/chunk`
   - `resp_len`

4. systems startup 문제인가  
   - `EADDRINUSE`
   - `DistNetworkError`
   - `Agent loop skd_agent not registered`
   - `fa3` / `Blackwell`

5. actor update memory 문제인가  
   - traceback이 `update_actor` / `loss.backward()`인지 확인

6. student rollout resume 문제인가  
   - traceback이 `update_weights -> rollout.resume(tags=["weights"])`인지 확인
   - `resume_memory_occupation`
   - `SGLangHttpServer ... out of memory`
   - HTTP timeout 3회

## 현재 문서 이후 읽기

구현 의도와 시스템 최적화를 더 자세히 보려면 바로 다음 문서로 간다.

- [`AGENTS/imp_detail.md`](/home/sogang_nlpy/verl/async_skd/AGENTS/imp_detail.md)
