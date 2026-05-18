# KILL IT 2

## 목적

이 문서는 현재 WebGym fully async RL 실험을 **왜 다시 base Qwen3.5-9B에서 시작했는지**,
그리고 **무엇을 검증하려는 실험인지**를 정리한다.

이번 실험의 핵심 질문은 하나다.

- **기존 Async SKD actor를 RL 시작점으로 쓴 판단 자체가 잘못되었는가?**

즉 지금 보고 있는 문제를

1. `SKD actor initialization` 문제로 볼지,
2. 아니면 `RL reward / rollout / termination / throughput` 문제로 볼지

를 분리해 보려는 실험이다.

---

## 지금까지 확인된 기존 문제

### 1. Async SKD actor는 내부 품질이 완전히 깨진 것은 아니었다

기존 `KILL_IT.md`에서 이미 정리한 요지는 이렇다.

- early / mid phase에서 semantic intent 자체는 종종 plausible했다.
- 하지만 그 intent가 **tool formatting / schema mismatch** 때문에 환경에 제대로 반영되지 못하는 경우가 많았다.
- 이후 local fused-gating patch 이후에는 초기에 지배적이던 `no-tool <|im_end|>`류 내부 cutoff 문제는 많이 줄었다.
- 그 뒤 남은 병목은 점점 **tool-call payload quality**, **invalid action**, **parse error** 쪽으로 이동했다.

즉 기존 SKD actor는:

- 완전히 쓸 수 없는 actor라기보다,
- **WebGym tool contract compliance가 불완전한 actor**

로 보는 편이 더 정확했다.

### 2. fully async RL로 넘어온 뒤의 주요 병목은 SKD와 별개였다

기존 RL probe와 최근 live run에서 확인된 핵심 병목은 다음이었다.

- `tool_response_budget_exhausted` 비율이 매우 높았다.
- `system_stop`도 적지 않았다.
- zero-group 비율이 높아서 LLM judge 의존도가 커졌다.
- trainer-side reward stage에서 judge latency가 매우 컸다.
- `format_reward_alpha=0.0` 상태라 format-related shaping은 final reward에 실질적으로 들어가지 않았다.

즉 RL 단계에서 보인 성능 정체는 곧바로

- "SKD actor가 나빠서 그렇다"

로 결론 내릴 수 있는 상태가 아니었다.

### 3. format reward는 그대로 켜면 해킹 위험이 크다

현재 format reward는 본질적으로:

- valid tool call 비율
- 첫 valid tool call의 빠르기
- budget exhausted penalty

를 본다.

이 신호는 정답성 자체가 아니라 **형식적 도구 사용 품질**을 보기 때문에,
양의 shaping으로 켜면

- 실제 task progress는 없고
- 형식적으로 valid한 tool call만 빠르게 내는 방향

으로 해킹되기 쉽다.

그래서 현재 launcher는:

- `format_reward_alpha=0.0`

를 유지하고 있다.

### 4. `system_stop`은 지금 너무 거친 termination bucket이다

현재 `system_stop`은 단순히

- `max_assistant_response_tokens`에 걸렸다는 뜻이 아니다.

실제 코드상 `system_stop`은:

- 전체 response budget 도달
- assistant turn cap
- user turn cap
- tool call 없이 generation 종료
- 그 외 종료

를 상당 부분 한 bucket으로 묶는다.

즉 현재 `system_stop`만 보고는

- reasoning이 중간에 잘렸는지,
- tool-call 없이 멈췄는지,
- turn cap에 걸렸는지

를 정확히 분해하기 어렵다.

---

## 왜 base Qwen3.5-9B로 다시 돌리나

이번 실험의 목적은 **정책 시작점을 바꿔서 원인 축을 분리**하는 것이다.

기존에는:

- actor initialization: `Async SKD actor checkpoint`
- judge: `gpt-5.4`
- reward path: zero-group trainer-side judge

였다.

이 상태에서 성능이 안 오르면,

- SKD actor의 tool contract 품질이 충분히 안 좋았던 것인지,
- 아니면 RL 자체의 reward / throughput / termination 문제가 더 큰 것인지

를 분리하기가 어렵다.

그래서 이번에는 의도적으로:

- **actor를 base Qwen3.5-9B로 되돌리고**
- **judge는 `gpt-5.4-mini`로 낮춰 reward latency를 줄이고**
- **resume을 끄고 완전히 새 run으로 시작**했다.

이 실험은 다음 질문을 겨냥한다.

- base 9B로 시작하면 오히려 더 빨리 무너지는가?
- 아니면 기존 SKD actor가 RL에 불리한 bias를 들고 있었는가?
- reward / termination 병목을 base actor에서도 그대로 보는가?

---

## 현재 실험 계약

현재 live run에서 실제로 확인된 계약은 다음과 같다.

- supervisor: 실행 중
- launcher: `WebOSWorld/run_qwen35_webgym_fully_async_rl_tool_veomni.sh`
- actor model path: `/home/sogang_nlpy/model/Qwen3.5-9B`
- `rollout.n=6`
- `val_kwargs.n=6`
- `llm_judge_model=gpt-5.4-mini`
- `llm_judge_max_concurrency=6`
- `trainer.resume_mode=disable`
- experiment name: `qwen35_9b_base_fully_async_webgym_tool`
- rollout dir:
  - `/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_20260517_145957`

즉 이번 run은:

- **SKD checkpoint를 resume하지 않고**
- **base 9B에서**
- **judge 비용을 조금 낮춘 상태로**

새로 시작한 비교 실험이다.

---

## 이 실험으로 검증할 수 있는 것

### 검증 가능

1. base 9B가 current WebGym tool contract에서
   - parse error
   - invalid action
   - no-progress zero-group
   를 얼마나 많이 내는지

2. SKD actor를 제거했을 때도
   - budget exhaustion
   - system_stop
   - zero-group ratio
   - judge latency pressure
   가 그대로 큰지

3. 기존 RL 정체가
   - actor initialization 문제인지
   - RL runtime / reward 문제인지
   의 상대적 비중

### 이 실험만으로는 아직 검증 못 하는 것

1. format reward를 positive shaping으로 켰을 때의 장기 안정성
2. `system_stop` 세부 원인 분해
3. base 9B가 장기 RL에서 실제로 더 낫다는 결론

즉 이번 run은 **원인 분리용 probe**이지,
곧바로 새로운 최종 학습 recipe를 확정하는 실험은 아니다.

---

## 현재까지의 초기 관찰

현재 rollout dir의 summary 기준 초반 snapshot:

- summary count: `576`
- termination:
  - `model_done: 185`
  - `tool_response_budget_exhausted: 243`
  - `system_stop: 132`
  - `model_fail: 16`
- `env_mean: 0.2135`
- `sum_mean: 0.4191`

이 수치는 아직 초반 스냅샷이라 결론으로 쓰면 안 된다.

다만 이미 이것만으로도 볼 수 있는 점은 있다.

- base 9B로 바꿔도 `budget_exhausted`와 `system_stop` 축이 즉시 사라진 것은 아니다.
- 즉 기존 정체의 전부를 SKD initialization 탓으로 돌리기는 어렵다.

### `system_stop`에 대한 추가 실측

현재 run의 `system_stop` sample 중, trajectory가 실제로 남아 있는 sample의 **마지막 assistant generation**을 다시 세어 보면
다음 패턴이 강하게 보인다.

- analyzed `system_stop` samples: `147`
- last assistant generation token length:
  - `p50: 4095`
  - `p75: 4096`
  - `p90: 4096`
  - `p95: 4096`
  - `max: 4096`
  - `mean: 2736.3`
- `>= 3800 tokens`: `89`
- `< 1000 tokens`: `47`
- last assistant turn:
  - `no <tool_call>`: `147 / 147`
  - `tool_call_count = 0`: `147 / 147`

이건 다음 해석을 지지한다.

- 현재 `system_stop`의 큰 덩어리는
  - **마지막 assistant generation이 `max_assistant_response_tokens=4096`에 걸려**
  - **tool call 없이 끝나는 패턴**
  로 보인다.

하지만 동시에:

- `system_stop` 전부가 그 원인 하나는 아니다.
- 짧은 no-tool stop도 적지 않다.

즉 현재는:

- "`system_stop`은 대부분 per-generation cap과 관련이 있다"
  는 해석은 가능하지만,
- "`system_stop`은 전부 4096 cap 때문이다"
  라고 단정하면 안 된다.

---

## 이 실험에서 반드시 같이 봐야 하는 지표

### 1. actor-side contract quality

- parse error rate
- invalid action count
- valid tool call rate
- zero-group share

### 2. termination mix

- `model_done`
- `tool_response_budget_exhausted`
- `system_stop`
- `model_fail`

특히 `system_stop`은 이후 세분화가 필요하다.

### 3. reward path pressure

- `timing_s/reward`
- `timing_s/step`
- judge start/done elapsed time
- trainer idle ratio

### 4. 실제 성능

- `score/env`
- `score/sum`
- success rate by task family

---

## 현재 읽는 방법

이번 실험은 이렇게 읽어야 한다.

- **만약 base 9B가 tool boundary에서 바로 더 심하게 깨지면**
  - 기존 SKD actor initialization은 최소한 완전히 잘못된 선택은 아니었다.

- **만약 base 9B와 SKD actor가 비슷하게 budget exhaustion / system_stop / zero-group 병목을 보이면**
  - 문제의 중심은 initialization보다 RL runtime / reward / termination 설계 쪽이다.

- **만약 base 9B가 오히려 더 안정적으로 env reward를 올리면**
  - 기존 SKD actor가 downstream RL에 불리한 bias를 남겼을 가능성을 더 진지하게 봐야 한다.

---

## 현재 작업 원칙

이 문서 기준 현재 원칙은 다음이다.

1. 지금 run은 **원인 분리 실험**으로 읽는다.
2. 아직 format reward positive shaping은 켜지 않는다.
3. `system_stop`은 원인 bucket을 더 잘게 나누기 전까지 단정적으로 해석하지 않는다.
4. `gpt-5.4-mini` judge는 품질 최적화가 아니라 **reward latency 완화용 선택**으로 본다.
5. base 9B 결과를 보고도 병목이 같으면, 다음 타깃은 actor initialization이 아니라 RL runtime 구조다.

---

## 현재 task difficulty 버킷

이 버킷은 **dataset row가 아니라 `task_id` 기준**이다.

- current train set:
  - rows: `41`
  - unique `task_id`: `38`
- duplicate row가 있는 task:
  - `prozilla_file_delete_02`
  - `prozilla_file_edit_02`
  - `prozilla_file_delete_folder_01`

즉 아래 분류는 **unique task_id 기준 provisional classification**이다.

### Easy

현재 policy가 이미 자주 성공시키는 task들이다.

- `prozilla_file_find_01`
- `prozilla_file_find_02`
- `prozilla_app_search_01`
- `prozilla_file_delete_01`
- `prozilla_setting_desktop_01`
- `web_task_apple_04`
- `web_task_apple_05`

공통점:

- single-app 또는 short navigation
- low branching
- target state가 명확
- tool sequence가 짧다

### Medium

일부는 풀지만 안정적으로 높다고 보기엔 아직 이른 task들이다.

- `prozilla_setting_taskbar_01`
- `prozilla_app_filter_01`
- `prozilla_file_delete&file_properties_01`
- `web_task_apple_03`
- `web_sheet_bmi_reference_child_01`

공통점:

- single-goal보다는 길지만, 완전히 hopeless한 수준은 아님
- 일부는 꾸준히 성공하지만 termination variance가 아직 큼
- `web_sheet_bmi_reference_child_01`은 `BMI/Sheet` family 중 유일하게 상대적으로 살아 있는 편

### Hard

현재 base 9B run과 previous SKD-init run 모두에서

- env success가 매우 낮고
- budget exhaustion 또는 system stop 압력이 높고
- turns / attempted tool calls가 길다

는 이유로 hard로 본다.

- `prozilla_setting_appearance_01`
- `prozilla_setting_appearance_02`
- `prozilla_setting_taskbar_02`
- `prozilla_file_edit_01`
- `prozilla_setting_taskbar&setting_desktop_01`
- `prozilla_setting_appearance&window_minimize_01`
- `prozilla_setting_taskbar&setting_autolaunch_01`
- `prozilla_app_filter&setting_autolaunch_01`
- `prozilla_setting_autolaunch_01`
- `prozilla_file_delete_02`
- `prozilla_file_edit_02`
- `prozilla_file_delete_folder_01`
- `web_sheet_bmi_record_01`
- `web_sheet_bmi_record_us_01`
- `web_sheet_bmi_filter_obesity_01`
- `web_sheet_bmi_reference_adult_01`
- `web_sheet_bmi_child_record_01`
- `web_sheet_bmi_child_filter_healthy_01`
- `web_sheet_bmi_child_filter_healthy_us_01`
- `web_task_apple_02`
- `prozilla_setting_taskbar&app_01`
- `prozilla_appfilter&autolaunch01`
- `prozilla_app&auto_launch_01`

핵심 hard family는 두 축이다.

1. `BMI + Sheet`
- cross-site state carry
- repeated calculator entry
- structured sheet write-back
- filtering / reference extraction

2. long horizon configuration
- `file_edit`
- `taskbar / autolaunch / combined settings`
- heavy Apple configurator

### Audit

난이도라기보다 **task/evaluator mismatch 후보**다.

- `prozilla_file_properties_01`
- `prozilla_setting_appearance_03`
- `web_task_apple_01`

이 셋은 current base 9B run에서:

- `model_done`가 높게 나오는데
- `env=0`이 유지된다

즉 pure difficulty보다 먼저:

- final-state criterion
- evaluator strictness
- task instruction ambiguity

를 다시 봐야 한다.

### 해석

현재 task set은 단순히 “전반적으로 어렵다”가 아니다.

- `easy`는 이미 policy가 다룰 수 있다
- `hard`는 현재 policy에게 과하게 어렵다
- `audit`은 난이도보다 정의/evaluator 문제 가능성이 있다

따라서 이후 train sampling은

- flat mixture

가 아니라

- `easy`
- `easy + medium`
- `hard` late introduction
- `audit` holdout

식으로 가는 것이 더 자연스럽다.

---

## 다음 해석 단계

다음으로 필요한 건 세 가지다.

1. base 9B run에서
   - parse/invalid-action
   - zero-group
   - termination mix
   를 SKD-initialized RL과 직접 비교

2. `system_stop`을 세분화
   - no tool call
   - response budget reached
   - assistant turn cap
   - user turn cap

3. task sampling을 `easy / medium / hard / audit` 기준으로 다시 볼지 결정

4. judge latency와 reward-stage throughput이
   - `gpt-5.4-mini`
   - `n=6`
   에서 어느 정도 완화되는지 확인

이 네 가지를 보고 나서야,

- base actor 유지
- SKD actor 복귀
- curriculum sampling 적용
- reward 설계 변경

중 어디로 갈지 결정할 수 있다.
