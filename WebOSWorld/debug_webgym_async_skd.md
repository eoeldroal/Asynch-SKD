# Async SKD Debug Notes

이 문서는 `WebOSWorld` Async SKD 디버깅 중 확인된 사실을 누적 기록하는 작업 메모다.

원칙:
- 확인된 사실과 추측을 분리한다.
- 로그/코드로 재확인한 내용만 `Confirmed`에 적는다.
- 설계 가설이나 다음 조사 포인트는 `Open Questions`에 적는다.

## Current Setup

- launcher: `WebOSWorld/run_qwen35_webgym_async_skd_tool_veomni.sh`
- distillation:
  - `distillation.enabled=True`
  - `distillation.distillation_loss.loss_mode=forward_kl_topk`
  - `distillation.distillation_loss.topk=32`
  - `distillation.distillation_loss.use_task_rewards=False`
  - `distillation.distillation_loss.use_policy_gradient=False`
- generation-time verification:
  - `distillation.skd.verify_top_k=10`
- actor:
  - `actor_rollout_ref.actor.calculate_entropy=False`
- rollout sampling:
  - launcher override는 `temperature=0.6`, `top_p=0.95`, `top_k=20`, `repetition_penalty=1.0`
  - 이 값은 Qwen3.5 precise coding recommendation과 일치한다.
- student stop behavior:
  - `verl/experimental/agent_loop/skd_agent_loop.py`에서 `stop_token_ids=[tokenizer.eos_token_id]`를 student sampling params에 넣는다.

### Important runtime note about sampling params

- 현재 Async SKD student generation 경로는 launcher의 `repetition_penalty` 값을 그대로 쓰지 않는다.
- `verl/experimental/agent_loop/agent_loop.py`에서 sampling params를 만들 때 `repetition_penalty=1.0`을 하드코딩한다.
- `presence_penalty`는 현재 sampling params에 아예 들어가지 않는다.
- `verl/experimental/agent_loop/skd_agent_loop.py`의 student chunk generation은 이 sampling params를 복사한 뒤 `max_tokens`와 `stop_token_ids`만 추가해 request를 보낸다.
- 따라서 다음 실험에서 penalty를 바꾸려면, launcher override만이 아니라 sampling param assembly path도 같이 수정해야 한다.

## Confirmed Fixes

### 1. Teacher exact-position contract

- WebSKD teacher verify 경로에서 request-time teacher prefix full rebuild를 복구한 뒤,
  이전의 `SGLang SKD delta prompt_logprobs exact-position contract is inconsistent` fatal error는 재발하지 않았다.

### 2. Student EOS stop

- student request에 `stop_token_ids`를 명시적으로 넣은 뒤,
  이전처럼 `</tool_call><|im_end|><|im_start|>user...`가 같은 chunk에 남는 현상은 최신 run에서 사라졌다.
- 이 수정은 post-hoc truncation이 아니라 student generation boundary 자체를 바꾼 표적 수정이었다.

## Confirmed Runtime Failure

### Checkpoint save failure

- 최신 종료 원인은 학습 로직/teacher verify가 아니라 checkpoint write failure였다.
- stack:
  - `ray_trainer.py::_save_checkpoint()`
  - `engine.save_checkpoint()`
  - `fsdp_checkpoint_manager.py`
  - `torch.save(optimizer_state_dict, ...)`
- error:
  - `PytorchStreamWriter failed writing file data/...: file write failed`
  - `unexpected pos ... vs ...`
- 당시 root filesystem은 사실상 full 상태였다.

## Confirmed Training Objective

- 현재 Async SKD 설정은 reward-driven PPO가 아니라 pure supervised distillation이다.
- 이유:
  - `use_task_rewards=False`
  - `use_policy_gradient=False`
- `verl/trainer/distillation/losses.py`에서 이 조합은 `is_supervised_distillation_only(...)`로 판정되고,
  최종 loss는 PPO loss와 결합되지 않고 `distill_loss`만 반환된다.
- 따라서 `critic/score/mean` 같은 reward 계열 수치는 관측 지표일 뿐, actor update를 직접 밀지 않는다.

## Confirmed Training-Dynamics Pattern

- 수치 폭주형 failure 증거는 현재 약하다.
  - `nan`, `inf`, overflow 계열은 로그에서 보지 못했다.
  - `actor/distillation/loss`는 초반 `28+`에서 후반 `~1` 수준으로 전반적으로 감소했다.
  - `actor/grad_norm`도 초반 `1000+`에서 후반 `20~60`대로 전반적으로 내려왔다.
- 반면 출력 품질은 악화되었다.
- 즉 현재 관측은 optimizer explosion보다는 objective mismatch 쪽에 더 가깝다.

정리:
- 모델은 현재 objective를 못 배우는 것이 아니라,
  malformed student trajectory 위에서도 distillation loss를 낮추는 방향으로는 잘 최적화되는 것으로 보인다.

## Confirmed Distillation-vs-Verification Split

- `distillation_loss.topk=32`와 `verify_top_k=10`은 역할이 다르다.
  - `topk=32`: loss 계산용 teacher top-k
  - `verify_top_k=10`: generation-time accept/reject 기준
- 현재 evidence로는 `topk=32` loss 근사 자체가 주된 문제로 보이지 않는다.

### Why top-32 does not currently look like the main problem

- 로그의 `actor/distillation/teacher_mass`는 거의 모든 step에서 `0.99+`였다.
- 전체 step 기준:
  - `teacher_mass` 평균: 약 `0.994`
  - `teacher_mass > 0.99` step: `43 / 44`
  - `teacher_mass_min < 0.4`인 step: `1 / 44`
- 해석:
  - teacher top-32가 teacher 분포 질량의 대부분을 이미 잡고 있다.
  - 따라서 `forward_kl_topk(topk=32)`의 수치 근사 자체는 평균적으로 꽤 정확한 편으로 보인다.
- 반면 `student_mass_min`은 매우 작은 row가 반복적으로 존재했다.
  - 이는 top-32 근사 자체가 틀렸다기보다,
    student가 teacher support를 잘 못 따라가는 row가 있다는 뜻으로 읽는 편이 맞다.

### More likely issue

- 현재 더 의심되는 것은 `verify_top_k=10` 기반 generation-time 제약과 local replacement 방식이다.
- teacher verification은:
  - student token이 teacher top-10 안에 있으면 통과시키고
  - 첫 mismatch에서 teacher top-1 한 토큰만 replacement한다.
- 즉 full teacher forcing이나 teacher rollout imitation이 아니라,
  student rollout 위에 local teacher admissibility check를 얹는 구조다.
- multi-turn tool-use failure는 sequence-level 구조 붕괴인데,
  현재 verification과 loss는 둘 다 token-local signal에 더 가깝다.

## Confirmed Output-Quality Timeline

### Step-level qualitative summary

- `step 10`: partially broken
  - task intent는 남아 있지만 `tool_call` markup bleed, malformed coordinate/json, restart가 빠르게 나타난다.
- `step 15`: partially broken, but better than step 10 and step 20
  - on-task sample이 아직 꽤 남아 있다.
  - failure signature는 `tool_call` serialization drift, repeated `<parameter=...>` tags, truncated payloads.
- `step 20`: clearly collapsed
  - 반복 parameter block, `MOUSE_DOWN`/`CLICK` loop, parser-busting tool-call corruption이 지배적이다.

### Step choice that looked best qualitatively

- `10 vs 15`: `step 15`가 더 나았다.
- `15 vs 20`: `step 15`가 더 나았다.
- `15 vs 20 vs 25`: `step 15`가 가장 나았고, `step 20`이 가장 나빴다.
- `10 vs 15 vs 20`: 질적 기준 선택은 `step 15`

## Confirmed Repetition-Collapse Pattern

### Chunk-level repetition is already strong by step 15

- 누적 `tail`이 아니라 `async_skd_events_webgym_20260512_030331.jsonl`의 `chunk_commit.verified_chunk`를 직접 디코드해서 반복 정도를 봤다.
- 반복 강도는 간단한 token repetition ratio로 측정했다.
  - `rep >= 0.35`: 의미 있는 반복
  - `rep >= 0.55`: 사실상 severe repetition loop

### Step 10

- `chunk_commit`: `126`
- `rep >= 0.55`: `0`
- `accepted_mean`: `16.17`
- `rejected=0`: `36`
- `rejected>0`: `90`

해석:
- step 10은 아직 repetition collapse가 본격화되기 전으로 보인다.

### Step 15

- `chunk_commit`: `1153`
- `rep >= 0.55`: `796`
- `accepted_mean`: `188.1`
- `rejected=0`: `886`
- `rejected>0`: `267`

더 중요한 분해:
- severe repetitive chunk `796`개 중
  - `rejected=0`: `795`
  - `rejected>0`: `1`

대표 signature:
- repeated `<parameter=button>`
- repeated coordinate field payload
- repeated `MOVE_TO`-like action payload block

해석:
- repetition loop는 step 15에서 이미 강하다.
- 그리고 그 loop 대부분은 teacher verification에 거의 걸리지 않고 그대로 통과한다.

### Step 20

- `chunk_commit`: `1582`
- `rep >= 0.55`: `1210`
- `accepted_mean`: `200.5`
- `rejected=0`: `1240`
- `rejected>0`: `343`

더 중요한 분해:
- severe repetitive chunk `1210`개 중
  - `rejected=0`: `1208`
  - `rejected>0`: `2`

대표 signature:
- repeated `<parameter=action_type>`
- repeated `MOUSE_DOWN`
- malformed function / parameter block repetition

해석:
- step 20 collapse는 teacher replacement artifact라기보다,
  student가 반복 chunk를 그대로 생성하고 그것이 대체로 accepted되는 구조에 더 가깝다.

### Step 46

- `chunk_commit`: `611`
- `rep >= 0.55`: `210`
- `accepted_mean`: `108.54`

더 중요한 분해:
- severe repetitive chunk `210`개 전부 `rejected=0`

대표 signature:
- repeated `<parameter=y>`
- repeated `RIGHT_CLICK` family

### Main implication

- 현재 repetition collapse는 "teacher가 자꾸 바꿔서 이상해진다"보다,
  student decoding 자체가 반복 loop로 들어가고 그것이 verification에 많이 통과하는 쪽으로 보는 게 더 맞다.
- 이 관찰은 `verify_top_k=10`의 local admissibility check가 sequence-level repetition loop를 충분히 자르지 못한다는 해석과 맞는다.

## Confirmed Penalty Tuning Direction

### Presence penalty vs repetition penalty

- 현재 failure mode는 generic natural-language repetition보다는
  repeated structure token / action token reuse에 가깝다.
- 대표 예:
  - `<parameter=...>`
  - `</function>`
  - `MOUSE_DOWN`
  - `RIGHT_CLICK`
- 이런 패턴에는 `presence_penalty`보다 `repetition_penalty`가 더 직접적이다.

현재 판단:
- 첫 실험은 `presence_penalty`보다 `repetition_penalty`가 우선이다.
- `presence_penalty`는 tool-call schema에서도 정상적으로 재등장해야 하는 token까지 거칠게 누를 위험이 더 크다.

### Why `1.05` looked too weak

- severe repetition chunk가 이미 매우 많고,
  그 chunk 대부분이 `rejected=0`으로 통과한다.
- 따라서 `1.05`는 no-op는 아니더라도,
  지금 보이는 수준의 repetition loop를 깨기엔 약할 가능성이 높다.

### Current best estimate for the next try

- 현재 로그 기준 다음 실험 시작값은 `repetition_penalty=1.10`이 더 타당해 보인다.
- 이유:
  - `1.05`보다 확실히 stronger하다.
  - 하지만 `1.15+`처럼 정상 tool serialization까지 크게 해칠 정도로 과격하지는 않다.
- 더 보수적으로 가려면 `1.08`도 후보지만,
  현재 evidence만 보면 `1.10` 쪽이 더 근거가 있다.

## Confirmed Async SKD Scheduler Behavior

- prefetch:
  - `prefetch_limit=16`, `prefetch_worker_target=4`는 실제 runtime에서 채워졌다.
- carryover:
  - partial lookahead가 다음 step current work로 resumed_current로 재투입되는 것을 확인했다.
- sticky teacher carryover:
  - resumed_current로 넘어간 sample에서 `teacher_replica_id` 유지가 확인됐다.
- promoted:
  - step-level promoted count는 정상적으로 증가한다.
  - 다만 per-sample promoted trace는 현재 로그에서 직접 보이지 않아 observability는 약하다.

## Confirmed RL Rollout Old-LogProb Semantics

### Scope

- 이 항목은 `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh` 기준 fully async RL rollout path를 대상으로 한다.
- 현재 RL rollout은:
  - `grammar_backend=xgrammar`
  - `enable_qwen3_coder_structured_output=True`
  - `top_p=0.95`
  - `top_k=20`
  - `calculate_log_probs=True`
  설정을 사용한다.

### What is currently stored as `old_log_prob`

- rollout sample의 `old_log_probs`는 agent loop postprocess에서 `output.response_logprobs`를 그대로 batch에 넣는 경로를 탄다.
- 따라서 RL에서 trainer가 쓰는 `old_log_probs`는 rollout server가 반환한 token log-prob 값이다.

### What changes the stored value

- SGLang rollout path에서 grammar mask / repetition penalty / presence penalty / 기타 logit bias는 sampling 직전에 logits에 직접 적용된다.
- 따라서 현재 stored `old_log_prob`는:
  - raw actor log-prob는 아니고
  - 최소한 constraint-decoding / penalty / temperature 영향을 받은 값이다.

### What does NOT directly change the stored value

- 현재 SGLang sampler 구현 기준으로 `top_k/top_p`는 sampled token을 고르는 단계에 쓰인다.
- 반면 `return_logprob`로 저장하는 sampled-token log-prob는 그보다 앞선 분포에서 읽는다.
- 즉 현재 stored `old_log_prob`는:
  - `top_k/top_p` 이후의 renormalized behavior policy log-prob가 아니라
  - `top_k/top_p` 이전 분포의 log-prob다.

정리:
- 현재 stored `old_log_prob`는
  - **pre-top_k/top_p**
  - **post-grammar/post-penalty/post-temperature**
  분포에 더 가깝다.

### Consequence

- 따라서 현재 rollout `old_log_prob`는 두 가지 중 어느 쪽도 아니다.
  - exact raw actor log-prob: 아님
  - exact rollout behavior log-prob: 아님
- exact rollout behavior policy correction을 하려면, sampled token 기준으로 `top_k/top_p` 이후 재정규화까지 반영한 log-prob가 필요하다.
- 반대로 raw actor old/new policy space를 맞추고 싶다면, `top_k/top_p`는 rescoring의 직접 원인이 아니고 grammar/penalty 쪽이 직접 원인이다.

### Important clarification about sampled tokens

- sampled token에 대해서 denominator support가 0이 되는 문제는 실질적으로 없다.
- 실제로 반환된 token은 rollout behavior가 양의 확률을 줬기 때문에 sampled-token ratio 자체는 정의된다.
- 현재 논점의 핵심은 support failure가 아니라:
  - stored `old_log_prob`가 exact behavior policy냐
  - 아니면 pre-top_k/top_p proxy냐
  이다.

### Current best interpretation

- 현재 RL rollout `old_log_prob`는 “behavior-aware hybrid value”로 해석하는 편이 가장 정확하다.
- 즉:
  - grammar / penalty / temperature는 반영
  - `top_k/top_p` 이후 exact behavior correction은 미반영
  상태다.

### Implication for the next RL discussion

- exact behavior correction을 더 중시하면:
  - `top_k/top_p`를 끄거나
  - sampled token에 대해 post-top_k/top_p renormalized log-prob를 저장하는 쪽이 더 정합적이다.
- raw old/new policy space consistency를 더 중시하면:
  - old log-prob를 raw actor로 rescoring하는 쪽이 더 맞지만,
  - 그 경우 actual rollout behavior correction은 일부 포기하는 셈이다.

## Working Hypothesis

- 현재 붕괴는 reward hacking보다는,
  `pure forward-KL distillation on student-generated malformed trajectories`
  쪽 설계 한계일 가능성이 더 높다.
- 특히 multi-turn tool-use 형식 안정성은 현재 objective의 직접 타깃이 아니다.
- `topk=32` 근사 오차보다는,
  `verify_top_k=10`의 느슨한 generation-time 제약과
  malformed student prefix 위에서도 계속 distillation-only 학습을 하는 구조가 더 핵심 원인일 가능성이 높다.
- repetition 관점에서는,
  malformed prefix가 살아남은 뒤 local distillation만으로도 severe token loop가 안정화될 수 있는 구조로 보인다.
- 현재 sampling recipe가 Qwen3.5 recommended precise-coding defaults와 일치한다는 사실은,
  문제의 1차 원인이 단순 sampling misconfiguration이 아닐 가능성을 높인다.
- 다만 현재 penalty path가 사실상 `repetition_penalty=1.0`, `presence_penalty` 미사용 상태인 것은,
  once-bad-prefix regime에서 반복을 적극적으로 끊어 주지 못하는 증폭 요인일 수 있다.

## Open Questions

1. malformed student rollout을 어떤 지점에서 batch에 포함시키지 말아야 하는가?
2. `tool_call` 형식 붕괴를 막는 신호를 distillation-only regime에 넣을 수 있는가?
3. reward를 학습에 넣지 않더라도, invalid action / malformed tool-call trajectory를 더 일찍 차단할 수 있는가?
4. current logging에 `promoted_record`나 `promoted_merge` 같은 per-sample trace를 추가할 필요가 있는가?
5. 다음 try에서 `verify_top_k=10 -> 5`로 줄였을 때 student drift가 의미 있게 줄어드는가?
6. 다음 try에서 `repetition_penalty=1.10`이 severe repetition chunk 비율을 실제로 낮추는가?
7. `presence_penalty`를 전혀 쓰지 않는 것이 맞는가, 아니면 `repetition_penalty`만으로 부족할 때 `0.05~0.1`의 작은 값을 추가해야 하는가?
