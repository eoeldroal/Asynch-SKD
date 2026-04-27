# On-Policy SKD 주의 사항

이 문서는 이 작업 영역에서 에이전트가 반드시 지켜야 하는 운영 규칙을 정리한다.  
구현 상세는 [`AGENTS/imp_detail.md`](/home/sogang_nlpy/verl/async_skd/AGENTS/imp_detail.md), 빠른 진입은 [`AGENTS/onboarding.md`](/home/sogang_nlpy/verl/async_skd/AGENTS/onboarding.md)를 본다. 본 문서는 환경, 실행, 수정 범위, 로그 해석에서의 주의점만 다룬다.

## 서론

이 영역의 실수는 대부분 모델 구조보다 운영 방식에서 발생했다.  
특히 다음 네 가지를 지키지 않으면 같은 오류가 반복되기 쉽다.

1. 올바른 conda 환경을 쓰지 않는 경우
2. 읽기 전용 환경을 수정하려는 경우
3. sandbox/tool readiness 확인 없이 RL을 먼저 태우는 경우
4. teacher 문제, student 문제, actor memory 문제를 섞어 해석하는 경우

## 본론

### 1. 기본 작업 환경은 `skd`다

코드 확인, 테스트, 전처리 점검, 로그 분석은 기본적으로 `skd` 환경에서 수행한다.

```bash
source /home/sogang_nlpy/miniconda3/etc/profile.d/conda.sh
conda activate skd
```

별도 지시가 없으면:

- Python 확인
- `pytest`
- `py_compile`
- 데이터셋 점검
- 실행 스크립트 sanity check

는 모두 `skd`에서 수행한다.

### 2. `skd` 환경은 읽기 전용으로 취급한다

`skd`는 현재 실험 재현용 기준 환경이다. 에이전트는 이 환경을 수정하지 않는다.

금지 사항:

- `pip install`
- `uv pip install`
- `conda install`
- `poetry install`
- 패키지 업그레이드/다운그레이드
- 환경 변수 파일, site-packages, interpreter 경로 수정

즉 `skd`는 **사용만 하고 변경하지 않는 환경**으로 취급한다.  
환경 문제가 생기면 코드나 실행 설정을 먼저 본다. `skd` 자체를 손보는 쪽으로 가지 않는다.

### 3. SandboxFusion 서버는 별도 환경에서 다룬다

`code_interpreter` 경로를 쓰는 실험은 RL보다 먼저 SandboxFusion readiness를 확인해야 한다.

서버 구동은 보통 `sandbox` 같은 별도 환경에서 처리한다.

```bash
source /home/sogang_nlpy/miniconda3/etc/profile.d/conda.sh
conda activate sandbox
```

최소 확인 순서:

1. 서버가 떠 있는지 확인
2. `curl` 또는 단일 `code_interpreter` smoke test 수행
3. 실제 출력이 반환되는지 확인
4. 그 다음에만 RL 실행

tool call 로그가 보인다고 해서 sandbox execution이 실제로 성공했다고 가정하면 안 된다.

### 4. RL 실행 전에 기존 프로세스를 먼저 정리한다

새 실험을 시작하기 전에는 기존 rollout / trainer / sandbox 관련 프로세스가 남아 있지 않은지 먼저 확인한다.  
이 영역은 장시간 실행 프로세스가 많기 때문에, 이전 런이 남아 있으면 로그 해석과 자원 상태를 쉽게 오염시킨다.

특히 다음 계열은 중복 실행을 피한다.

- `main_ppo`
- `TaskRunner`
- `AgentLoopWorker`
- `SGLangHttpServer`
- `ExecutionWorker`
- `wandb`
- `gpu_stats`

### 5. 포트 충돌은 레이어를 구분해서 본다

최근 실험에서 포트 충돌은 한 번이 아니라 **서로 다른 레이어**에서 반복됐다.  
이 둘을 같은 문제로 취급하면 원인 분석이 계속 어긋난다.

- `RayWorkerGroup` master port 충돌
- `SGLangHttpServer` 내부 `nccl_port` 충돌

즉 `EADDRINUSE`가 떴다고 해서 항상 같은 수정으로 해결된다고 보면 안 된다.  
어느 traceback에서 났는지, 어떤 프로세스/actor가 죽었는지를 먼저 구분한다.

### 6. 수정은 코드와 설정에 한정하고, 환경 자체는 건드리지 않는다

문제가 발생했을 때 우선순위는 다음과 같다.

1. 실행 스크립트 설정 확인
2. config/dataclass 계약 확인
3. rollout / training 코드 확인
4. tool readiness 확인

환경 수정은 마지막 수단이 아니라, 이 작업 영역에서는 원칙적으로 피하는 선택이다.

### 7. Validation은 train-time teacher guidance와 분리해서 본다

이 축에서 validation은 student policy 자체를 평가하는 단계다.  
따라서 validation 동작을 해석할 때는 teacher-guided train rollout과 구분해야 한다.  
현재 구현은 validation에서도 agent-loop 경로를 타지만, trainer validation entrypoint에서 `skd_agent`를 `tool_agent`로 바꿔 student-only validation으로 처리하므로 teacher-guided training rollout과 동일하다고 보면 안 된다.

또 하나 더 중요하다. validation은 training `AsyncSkdDataSource`를 건드리면 안 된다. 현재 trainer는 `_validate()` 동안 manager에서 training source를 잠깐 분리하므로, validation이 training future sample을 prefetch하거나 promoted/carryover를 training source에 기록해서는 안 된다. 이 보호가 깨지면 validation이 끝난 뒤 다음 training step에서 carryover quota가 비정상적으로 커지는 leakage가 발생한다.

추가로 `best@4`, `mean@4`를 보려면:

- `val_kwargs.n=4`
- `do_sample=True`

가 함께 필요하다.  
`n=4`만 주고 sampling이 꺼져 있으면 사실상 greedy 복제에 가깝다.

### 8. SGLang backend 문제와 actor 문제를 섞지 않는다

최근 런에서는 Blackwell 환경에서 SGLang vision backend가 `fa3`로 잡혀 startup이 깨지는 문제가 있었다.  
현재 기준으로는 student rollout과 teacher inference 모두:

- `attention_backend=triton`
- `mm_attention_backend=triton_attn`

를 명시해야 한다.

`fa3`, `Blackwell`, `Scheduler hit an exception`이 보이면 actor 학습 문제보다 rollout backend 설정을 먼저 본다.

### 9. Teacher 문제와 actor OOM을 같은 문제로 보지 않는다

이 축에서 자주 생기는 오판은 다음과 같다.

- teacher가 느리다
- 그래서 teacher가 메모리 크래시의 원인일 것이다

실제로는 다를 수 있다.  
최근 런에서는 teacher 병목과 별개로, 최종 크래시는 `update_actor -> loss.backward()`에서의 actor-side OOM으로 나타났다.

따라서 로그를 볼 때는 다음을 분리한다.

- teacher latency 문제
- student rollout 누적 시간 문제
- actor backward memory 문제

특히 `torch.OutOfMemoryError`가 `update_actor` 아래에서 나면, teacher가 아니라 actor training memory로 본다.

### 10. `ppo_max_token_len_per_gpu`는 sample truncation이 아니다

이 값은 “sample을 여기까지만 학습한다”가 아니라, GPU 한 장이 한 번에 처리하는 총 token budget 상한에 가깝다.  
즉 값을 줄이면 sample을 버리는 것이 아니라, 더 작은 micro-batch로 나눠 처리하게 된다.

이 점을 모르면:

- token budget 조정
- batch size 조정

의 의미를 혼동하기 쉽다.

### 10-1. 여러 micro-batch로 나눠도 update 단위 residency 문제는 남는다

현재 actor 학습은 dynamic micro-batch 분할을 쓰지만, `train_mini_batch()` 바깥에서 `train_mode()`를 한 번 열고 그 안에서 여러 micro-batch를 순차 처리한다.  
즉 micro-batch를 여러 개로 나눈다고 해서 update 경계의 residency 문제가 자동으로 사라지지는 않는다.

특히 최근 실제 장애는:

- actor backward OOM
- 그 다음 단계의 `update_weights -> rollout.resume(tags=["weights"])` OOM

으로 나뉘어 나타났다.  
따라서 token budget 조정은 actor backward peak를 낮추는 데는 직접적이지만, rollout resume weight memory 충돌까지 자동으로 해결해 주지는 않는다.

### 10-2. `param_offload=False`, `optimizer_offload=True` 조합은 현재 코드에서 지원되지 않는다

현재 `BaseEngine.to()` 계약은 model 없이 optimizer/grad만 따로 옮기는 호출을 금지한다.  
따라서 다음 조합은 아이디어상 타협안처럼 보여도, 코드 수정 없이 그대로는 쓸 수 없다.

- `param_offload=False`
- `optimizer_offload=True`

이 조합을 넣으면 train/eval context 진입 시 assert로 바로 죽는다.  
현재 코드 수정 없이 가능한 안정적인 조합은 사실상 다음 둘이다.

- 둘 다 끄기
- 둘 다 켜기

즉 optimizer만 따로 offload하는 해법은, **현 코드베이스에서는 제안 가능하지만 실행 가능한 설정은 아니다.**

### 11. Teacher-only prompt를 쓰면 teacher budget reserve를 염두에 둔다

teacher-only system prompt를 넣으면 teacher prefix는 student prefix보다 길어진다.  
따라서 teacher context budget은 student 기준 값과 같다고 가정하면 안 된다.

현재 구현은 config 레이어에서 reserve를 자동으로 더하지만, 로그 해석이나 실행 설정을 볼 때는 이 사실을 알고 있어야 한다.

### 12. `skd_agent` registration은 import chain에 의존한다

`skd_agent`는 decorator 등록형 agent loop다.  
따라서 `skd_agent_loop.py`가 startup 시 import되지 않으면 registry에서 빠진다.

다음 오류가 보이면:

- `Agent loop skd_agent not registered`

registration decorator 자체보다 `verl.experimental.agent_loop.__init__`의 import chain을 먼저 본다.

### 13. AGENTS 문서를 수정할 때는 편의보다 규칙을 따른다

이 디렉터리의 문서는 에이전트가 반복적으로 읽는 문서다.  
따라서 수정할 때는:

- 구현 상세는 `imp_detail.md`
- 빠른 진입은 `onboarding.md`
- 운영 규칙은 `warning.md`

처럼 역할을 분리한다.  
문서 길이와 중복도 함께 관리한다.

### 14. `gpu_memory_utilization`은 추상적인 감이 아니라 SGLang static memory fraction이다

student rollout SGLang 쪽 `gpu_memory_utilization`은 runtime에서 그대로 `mem_fraction_static`으로 들어간다.  
즉 숫자 `0.90 -> 0.75 -> 0.60`은 “조금 보수적으로 바꿨다” 수준이 아니라, 실제 reserved/static budget을 크게 바꾸는 값이다.

예를 들어 총 `178.35 GiB` GPU 기준 단순 환산은 대략 다음과 같다.

- `0.90` -> `160.52 GiB`
- `0.75` -> `133.76 GiB`
- `0.60` -> `107.01 GiB`

즉 이 값은 rollout throughput과 resume 안정성에 직접 영향을 주는 레버다.

### 15. `data.truncation=error`는 silently 넘어가지 않는다

현재 스크립트는 `data.truncation=error`를 사용한다. 오버롱 prompt가 들어오면 truncate 하지 않고 에러로 올린다. `filter_overlong_prompts=True`와 함께 쓰이기 때문에 정상 흐름에서는 에러가 나지 않아야 한다. 만약 `Prompt length ... exceeds ...` 에러가 뜨면 filter 단계가 제대로 작동하지 않은 것이므로, data pipeline을 먼저 본다.

### 16. `VERL_SKD_DEBUG` 레벨은 목적에 맞게 선택한다

- `VERL_SKD_DEBUG=0` (기본): 디버그 로그 없음
- `VERL_SKD_DEBUG=1` (현재 스크립트 기본): chunk 단위 diagnostics 출력 (`accept/reject rate`, `avg_tok/chunk`, 종료 이유 등)
- `VERL_SKD_DEBUG=2`: 위에 더해 batch당 첫 3 sample의 token-level alignment 검증 로그 출력. teacher row 정렬 문제가 의심될 때만 사용한다. 로그가 매우 많아지고 성능에 영향을 줄 수 있다.

## 결론

이 작업 영역에서 가장 중요한 준수 사항은 다음 네 가지다.

1. 기본 작업은 `conda activate skd`에서 수행한다.
2. `skd` 환경은 읽기 전용으로 취급하고 수정하지 않는다.
3. tool 실험은 SandboxFusion readiness를 먼저 확인한 뒤 실행한다.
4. Ray master port, SGLang `nccl_port`, backend mismatch, actor memory를 서로 다른 문제로 분리해서 해석한다.

위 네 가지를 지키면, 같은 유형의 시행착오를 상당 부분 줄일 수 있다.
