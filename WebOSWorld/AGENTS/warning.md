# Async SKD Warnings

이 문서는 반복해서 터졌거나 앞으로도 쉽게 재발할 수 있는 주의사항만 모은다.

## 1. 기본 환경

작업과 확인은 기본적으로 `skd` conda 환경에서 한다.

```bash
cd /home/sogang_nlpy/verl
conda activate skd
```

패키지 설치나 환경 변경은 명시적으로 필요할 때만 한다. 실행 코드, hydra config, import 가능 여부를 볼 때는 항상 이 환경 기준으로 판단한다.

## 2. 건드리지 말 파일

다음 문서는 논문 draft 성격이므로 일반 문서 정리에서 수정하지 않는다.

```text
WebOSWorld/document/paper_draft/Implementation Detail.md
```

## 3. Real WebGym보다 mock server를 먼저 본다

WebSKD 문제를 디버깅할 때 real WebGym / Omnibox stack을 바로 붙이면 원인이 섞인다.

먼저 mock server에서 확인한다.

- protocol shape
- `session_id` 유지
- action list shape
- image/a11y response
- reward 회수
- trainer integration

mock에서 깨지면 WebGym 문제가 아니라 우리 loop/protocol 문제다.

## 4. 테스트 코드는 서버 런처가 아니다

mock client는 이미 떠 있는 서버에 protocol request를 보내는 용도다. 서버 lifecycle을 복잡하게 숨기는 테스트 런처를 만들면 readiness와 conda 환경 문제가 섞인다.

서버는 명시적으로 띄우고, client나 trainer는 그 endpoint를 바라보게 한다.

## 5. `task_id`와 `session_id`를 섞지 않는다

- `task_id`: dataset에서 온다. 어떤 환경 task인지 나타낸다.
- `session_id`: runtime이 만든다. 한 trajectory의 서버 세션이다.

dataset에 `session_id`를 넣지 않는다. 같은 task를 여러 trajectory에서 동시에 풀 수 있어야 한다.

## 6. WebSKD에서 local ids를 server ids로 대체하지 않는다

multimodal prompt에서는 processor-expanded local ids와 SGLang logical server ids가 다를 수 있다.

위험한 fallback:

```python
server_prompt_ids = list(agent_data.prompt_ids)
```

WebSKD에서는 이 fallback이 조용히 teacher delta alignment를 깨뜨릴 수 있다. 필요한 경우 messages에서 server prompt ids를 재계산하거나 fail-fast한다.

## 7. Image expansion 크기를 고정값으로 보지 않는다

특정 run에서 surplus가 `219`처럼 보였어도 고정 계약이 아니다.

이미지 수, processor, backend, 모델 구현에 따라 달라질 수 있다. 현재 구현처럼 매 verification 시점에 prefix length 차이로 계산해야 한다.

## 8. Tool observation은 부분 commit하지 않는다

image만 들어가고 prompt ids가 실패하거나, teacher messages만 들어가고 dummy rows가 빠지는 식의 상태는 carryover에서 반드시 문제를 만든다.

Web observation은 bundle 전체가 들어가거나 전체가 들어가지 않아야 한다.

## 9. Teacher context overflow는 server error로 보내지 않는다

teacher-only a11y/text와 image expansion 때문에 teacher context가 먼저 찰 수 있다. 이 상태에서 SGLang에 요청을 보내면 server-side ValueError로 run이 터질 수 있다.

반드시 teacher call 전, Web observation commit 전 guard를 거쳐 `teacher_context_exhausted`로 정상 종료해야 한다.

## 10. 작은 dataset + prefetch는 source exhaustion을 만든다

기존 64-row dataset으로 4-step run을 돌릴 때, current batch뿐 아니라 lookahead prefetch가 row를 추가로 소비한다.

`Training Progress 75% (3/4)` 근처에서 종료되었다면 먼저 source exhaustion을 의심한다. rollout crash로 단정하지 않는다. 기본 smoke dataset은 256 rows로 생성한다.

## 11. Non-tensor metadata key를 임의로 땜질하지 않는다

`agent_name` 하나가 빠져도 `DataProto.concat()`이 실패한다. 반대로 길이가 batch size와 맞지 않아도 실패한다.

정리 위치는 `verl/experimental/async_skd/metadata.py`다. 새 경계를 추가한다면 이 helper를 통한다.

## 12. Validation에서 training source를 건드리지 않는다

validation 중에 lookahead reservation이 생기면 training ledger가 오염된다. validation은 student-only 평가이며, training source detach/restore가 유지되어야 한다.

## 13. W&B teardown noise를 실제 SKD 실패와 구분한다

종료 시 다음 로그는 W&B/Ray 종료 순서 문제일 수 있다.

```text
Exception ignored in atexit callback
RuntimeError: UnixTransport closed
```

이 로그만 보고 실패라고 판단하지 않는다. 실제 fatal traceback이 trainer / RayTaskError / SGLang request 어디에서 났는지 본다.

## 14. SGLang backend를 명시한다

현재 검증된 실행 경로는 다음 backend override를 사용한다.

```text
attention_backend=triton
mm_attention_backend=triton_attn
```

자동 backend 선택이 바뀌면 startup 실패가 날 수 있다.

## 15. Ray port와 SGLang/NCCL port는 다른 문제다

포트 충돌이 나면 어떤 레이어의 포트인지 먼저 나눈다.

- Ray worker / dashboard / head port
- SGLang server port
- NCCL/distributed init port
- mock Web/OSGym server port
- real WebGym / Omnibox port

한쪽 포트를 바꿨다고 다른 충돌이 해결되지는 않는다.

## 16. Process cleanup은 범위를 좁힌다

강화학습 프로세스를 내릴 때 mock server, code interpreter, 외부 WebGym stack까지 무차별 종료하지 않는다. 사용 중인 서버가 무엇인지 먼저 확인한다.

확인 후보:

```bash
ps -ef | rg 'verl.trainer.main_ppo|ray|sglang|uvicorn'
```

## 17. Debug level은 비용이 있다

`VERL_SKD_DEBUG=2`는 alignment를 보기에 좋지만 로그가 많고 느려질 수 있다. milestone run이나 alignment 문제 재현에는 유용하지만, 장시간 throughput run에서는 낮추는 것을 고려한다.

## 18. Real WebGym integration은 mock 통과 후 본다

real WebGym에서 timeout이나 browser readiness 문제가 나면 먼저 다음을 분리한다.

1. mock server + trainer가 되는가
2. protocol client가 real server health/action/reward를 받을 수 있는가
3. Omnibox/browser session이 뜨는가
4. 실제 task가 action을 처리하는가

이 순서를 건너뛰면 server readiness 문제와 SKD alignment 문제를 구분하기 어렵다.
