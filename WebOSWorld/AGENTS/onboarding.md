# Async SKD Onboarding

이 문서는 `verl/async_skd` 작업 영역을 처음 다시 잡을 때 보는 빠른 지도다. 자세한 구현 설명은 `imp_detail.md`, 운영상 주의점은 `warning.md`, 장애/런 메모는 `details.md`를 본다.

## 현재 기준 상태

현재 브랜치의 핵심 마일스톤은 두 개다.

1. `e104f23a Milestone: finalize async SKD implementation runbook`
   - text/tool 기반 async SKD 실행 경로를 정리한 기준점이다.
   - validation은 student-only로 분리되고, training `AsyncSkdDataSource`와 격리된다.
   - lookahead, promoted, carryover, teacher sticky carryover가 trainer 경로에서 동작한다.

2. `7a404f2f Milestone: validate WebSKD mock RL on real GPUs`
   - mock Web/OSGym 서버와 실제 GPU 기반 `web_skd_agent` trainer 경로를 연결해 검증한 기준점이다.
   - image observation, student SGLang generation, teacher SGLang prompt-logprob verification, multimodal prefix surplus trimming, lookahead prefetch, carryover, promotion, batch assembly, actor update까지 실제 경로를 통과했다.
   - `Training Progress 75% (3/4)`에서 보인 종료는 SKD crash가 아니라 64-row mock dataset을 작은 step 수 + prefetch로 소모한 결과다.

## 주요 실행 경로

### Text/tool async SKD

```bash
cd /home/sogang_nlpy/verl
conda activate skd

bash async_skd/run_qwen35_math_async_skd_tool_fsdp.sh
```

특징:

- `default_agent_loop=skd_agent`
- SandboxFusion tool config 사용
- teacher-only math planning prompt 사용
- 기본 train batch 64, response length 8192
- validation은 `tool_agent`로 전환되어 teacher guidance 없이 student policy를 평가한다.

### WebSKD mock RL

먼저 mock Web/OSGym 서버를 띄운다.

```bash
cd /home/sogang_nlpy/verl
conda activate skd

nohup python async_skd/mock_server/web_osgym_mock_server.py \
  --host 127.0.0.1 \
  --port 18000 \
  --log-path logs/mock_web_osgym_requests.jsonl \
  > logs/mock_web_osgym_server.log 2>&1 &
```

데이터셋이 없으면 생성한다.

```bash
python async_skd/mock_server/create_mock_web_osgym_dataset.py \
  --local-save-dir /home/sogang_nlpy/verl/data/mock_web_osgym \
  --num-samples 64
```

훈련 경로는 다음 스크립트를 사용한다.

```bash
bash async_skd/run_qwen35_web_mock_async_skd_tool_fsdp.sh
```

특징:

- `default_agent_loop=web_skd_agent`
- tool config: `examples/sglang_multiturn/config/tool_config/web_osgym_tool_config_webgym_rl.yaml`
- mock server 기본 endpoint: `http://127.0.0.1:18000`
- 기본 train batch 16, response length 1024
- teacher max model len은 기본 8073으로 student보다 여유를 둔다.
- prefetch 기본값은 batch/worker 비율에서 계산된다.

## 핵심 코드 위치

### SKD 의미론

- `verl/experimental/agent_loop/skd_agent_loop.py`
  - student chunk generation
  - teacher verification
  - first-rejection correction
  - teacher row / response mask alignment
  - teacher context overflow guard

### Async scheduling

- `verl/experimental/async_skd/manager.py`
  - current / lookahead / promoted / carryover scheduling
  - output finalize
- `verl/experimental/async_skd/worker.py`
  - fresh sample과 partial carryover sample 실행 primitive
- `verl/experimental/async_skd/source.py`
  - future sample reservation, promoted ledger, carryover ledger
- `verl/experimental/async_skd/metadata.py`
  - `DataProto.concat()` / `DataProto.union()` 경계에서 non-tensor metadata key 정규화

### Teacher / SGLang

- `verl/experimental/teacher_loop/teacher_manager.py`
  - teacher routing, sticky binding, routing-key별 max model len 조회
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
  - SGLang prompt-logprob delta extraction
  - multimodal prefix surplus trimming

### Web / OSGym

- `verl/experimental/agent_loop/web_skd_agent_loop.py`
  - `web_skd_agent`
  - initial observation commit
  - tool observation commit
  - student/teacher observation split
  - image observation과 logical server prompt stream 유지
- `verl/experimental/agent_loop/web_osgym_protocol.py`
  - `POST /` protocol client
- `verl/experimental/agent_loop/web_osgym_loop_mixin.py`
  - runtime-owned `web_osgym_session_id` allocation and restore
- `verl/tools/web_osgym_tool.py`
  - model-facing `computer` tool
  - `actions: [...]` parsing
  - `DONE` / `FAIL` terminal handling
- `async_skd/mock_server`
  - session-aware mock Web/OSGym server, client, dataset generator, reward function

## WebSKD의 현재 입력 계약

Dataset row는 `task_id`만 제공한다. `session_id`는 dataset에서 오지 않는다.

- `task_id`: 서버에 등록된 환경 task 식별자
- `session_id`: agent loop/runtime이 trajectory 시작 시 생성하는 환경 세션 식별자

같은 trajectory에서는 `start -> action* -> reward`가 모두 같은 `session_id`를 사용해야 한다. 이 계약은 이후 GRPO 계열 강화학습에서 여러 세션을 병렬로 굴릴 때 중요하다.

## Student / Teacher observation split

정상 visual observation:

- student: screenshot image를 본다.
- teacher: screenshot image와 a11y/text를 본다.

Image-less failure observation:

- image가 없으면 action 실패 원인이 `text`에만 들어있을 수 있다.
- 이 경우 student와 teacher 모두에게 text를 제공한다.

이 예외는 복구 가능한 환경 feedback을 student에게 숨기지 않기 위한 것이다.

## Prompt stream 구분

WebSKD에서는 두 종류의 prompt id가 의도적으로 다르다.

- local prompt ids
  - processor/chat-template 적용 후의 로컬 ids
  - image expansion이 반영될 수 있다.
- server prompt ids
  - SGLang `/generate`로 보내는 logical ids
  - teacher prompt-logprob delta의 기준이다.

따라서 `server_prompt_ids`와 `teacher_server_prompt_ids`가 없으면 WebSKD는 추측하지 않고 fail-fast하거나 재계산해야 한다. local ids를 server ids로 대체하면 multimodal logprob alignment가 조용히 깨진다.

## Multimodal teacher delta

이미지가 포함된 prompt에서는 SGLang이 내부적으로 image token을 확장한다. 그 결과 teacher가 반환하는 prompt-logprob rows에는 logical suffix chunk 외에 multimodal prefix surplus가 섞일 수 있다.

현재 구현은 surplus를 고정값으로 가정하지 않는다.

- 현재 teacher local prefix length
- 현재 teacher server logical prefix length
- SGLang 반환 rows
- 요청한 chunk length

이 정보를 바탕으로 매 검증 시점마다 앞쪽 surplus rows를 버리고, 실제 SKD 검증 대상인 suffix chunk rows만 사용한다.

## Teacher context guard

학생 쪽은 response cutoff에 가까운 하드 budget으로 종료된다. 교사 쪽은 teacher-only prompt, a11y text, multimodal expansion 때문에 student보다 길어질 수 있으므로 별도 guard가 필요하다.

현재 guard는 두 군데 있다.

1. teacher verification 직전
   - `teacher_server_prompt_ids + chunk`가 teacher max model len을 넘으면 teacher call 전에 정상 종료한다.

2. Web tool observation commit 직전
   - non-terminal observation을 teacher context에 넣은 뒤 최소 1 token verify 공간이 남지 않으면 observation을 commit하지 않고 정상 종료한다.

종료 reason은 `teacher_context_exhausted`로 기록된다.

## Async source와 dataset size

lookahead prefetch는 실제 dataset row를 미리 reserve한다. 작은 dataset으로 여러 step을 돌릴 때는 current batch 외에 prefetch가 row를 더 소비한다.

예를 들어 64-row mock dataset, train batch 16, prefetch가 켜진 4-step run에서는 마지막 step 전에 source가 고갈될 수 있다. 이것은 rollout crash가 아니라 source exhaustion이다. 여러 step을 안정적으로 보려면 dataset row 수를 batch + prefetch 총량보다 여유 있게 잡는다.

## 로그에서 먼저 볼 것

SKD health:

- `[SKD_DBG]`
- `[SKD] ... done=... accept=... reject=... rate=...`
- `teacher_prefix_len`
- `teacher_server_prefix_len`
- `teacher_mm_prefix_surplus`
- `teacher_logprobs_accumulated`

Async scheduling:

- `[ASYNC_SKD] rollout`
- `lookahead_started_count`
- `lookahead_promoted_count`
- `lookahead_carryover_count`
- `teacher_pinned_carryover_count`
- `teacher_fallback_carryover_count`

Trainer:

- `actor/distillation/loss`
- `actor/distillation/teacher_mass`
- `response/aborted_ratio`
- `critic/score/mean`
- `Training Progress`

W&B 종료 시점의 `Exception ignored in atexit callback` / closed `UnixTransport` 로그는 Ray/W&B teardown noise일 수 있다. 직전 step metric과 traceback의 실제 예외 위치를 먼저 본다.

## 문서 읽는 순서

1. 이 문서
2. `AGENTS/imp_detail.md`
3. `AGENTS/details.md`
4. `AGENTS/warning.md`
5. `document/async_skd/design.md`
6. `document/web_osgym/design.md`
7. `document/web_osgym/protocol.md`
