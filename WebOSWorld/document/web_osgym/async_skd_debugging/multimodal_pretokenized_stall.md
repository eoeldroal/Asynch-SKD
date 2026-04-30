# Async SKD Multimodal Pretokenized Stall

## 0. Status

이 문서는 WebGym 기반 Async SKD 실행에서 관측된 긴 stall의 원인 조사와 해결 방향을 정리한다.

현재 결론:

- 병목은 WebGym action server, screenshot base64/PIL decode, tokenizer, 일반 multimodal preprocess가 아니다.
- 병목은 SGLang에 `input_ids + image_data`로 multimodal generate를 요청하는 pretokenized student generation path다.
- 같은 이미지와 같은 WebGym loop라도 `text + image_data` native path는 정상 속도로 동작한다.
- 따라서 해결 방향은 WebSKD image-bearing student generation의 SGLang request만 native text multimodal path로 보내고, 학습용 token stream과 teacher verification 좌표계는 현재 불변식을 유지하는 것이다.
- 학습 중에는 느린 `input_ids + image_data` fallback을 사용하지 않는다. 대신 tokenization drift를 충분히 로깅하고, returned suffix token ids만 canonical stream에 append한다.

## 1. Problem

WebGym Async SKD에서 다음 증상이 반복되었다.

- SKD trace는 정상적으로 올라오다가 generation 구간에서 긴 정지가 발생한다.
- GPU util은 어느 정도 보이지만 power draw는 B200 기준 매우 낮게 보이는 경우가 많다.
- tool call 이후 WebGym action request가 찍힌 뒤 다음 generation까지 긴 간격이 생긴다.
- text-only math/tool SKD에서는 같은 길이 설정에서도 이런 stall이 재현되지 않는다.

초기 의심 후보는 다음과 같았다.

1. WebGym server action latency
2. screenshot base64 -> PIL -> base64 변환 비용
3. Qwen-VL image preprocessing 비용
4. SGLang scheduler queue/batch selection 문제
5. SGLang multimodal prompt-logprob/top-k materialization 문제
6. SGLang multimodal pretokenized input path 문제

## 2. Relevant Code Path

현재 WebSKD student chunk generation은 다음 경로를 탄다.

```text
WebSkdAgentLoop / SkdAgentLoop
  -> self.server_manager.generate(...)
    -> AsyncLLMServerManager.generate(...)
      -> SGLangHttpServer.generate(...)
        -> GenerateReqInput(input_ids=prompt_ids, image_data=image_data)
```

핵심 코드 경계:

- `verl/experimental/agent_loop/skd_agent_loop.py`
  - `server_prompt_ids`를 student SGLang prompt로 사용한다.
  - `image_data`를 함께 넘긴다.
- `verl/experimental/agent_loop/web_skd_agent_loop.py`
  - `prompt_ids`: local processor-expanded training ids
  - `server_prompt_ids`: SGLang logical ids
  - `teacher_prompt_ids`: teacher local ids
  - `teacher_server_prompt_ids`: teacher SGLang logical ids
- `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
  - 현재 `GenerateReqInput`에 항상 `input_ids=prompt_ids`를 넣는다.
  - image가 있으면 `image_data=image_data`도 넣는다.

즉 현재 student generate는 image가 있는 경우에도 `text + image_data`가 아니라 `input_ids + image_data` request다.

## 3. Coordinate Background

WebSKD는 local ids와 SGLang server ids를 분리해야 한다.

```text
local processor ids:
  Qwen-VL processor가 image placeholder를 expanded image token ids로 펼친 ids

SGLang logical server ids:
  tokenizer chat template 기준 ids
  실제 image bytes/base64는 image_data로 따로 전달
```

이 분리는 teacher prompt-logprob delta correctness 때문에 필요하다.

중요한 점:

- local ids를 server ids로 대체하면 teacher/student alignment가 깨진다.
- 반대로 SGLang generate request까지 반드시 `input_ids`로만 보내야 한다는 뜻은 아니다.
- SGLang이 native `text + image_data` path를 잘 처리한다면, generate request는 text path로 보내되 training/alignment stream은 기존 ids를 유지할 수 있다.

## 4. Experiments

### 4.1 Fixed SGLang Server Pool

SGLang server를 고정으로 4개 띄우고 같은 서버에 반복 요청을 보냈다.

```text
ports: 37000, 37001, 37002, 37003
model: Qwen3.5-9B
tp: 1 per server
backend: triton / triton_attn
stream: true
concurrency: 16
```

이 실험의 목적은 model loading, server startup, first server readiness를 제거하고 request shape만 비교하는 것이다.

### 4.2 Synthetic Verl-path Replay

먼저 WebGym server 없이, verl prompt 구성과 유사한 synthetic image replay를 사용했다.

핵심 결과:

```text
pretokenized_image_student, 3 images:
  total avg ~= 52.3s
  TTFT avg ~= 51.8s

native_image_student, 3 images:
  total avg ~= 1.24s
  TTFT avg ~= 0.66s

pretokenized_text_student, same text length:
  total avg ~= 0.57s
  TTFT avg ~= 0.16s

native_text_student, same text length:
  total avg ~= 0.55s
  TTFT avg ~= 0.15s
```

결론:

- text-only pretokenized path는 빠르다.
- image native path도 빠르다.
- 느린 것은 `input_ids + image_data` 조합이다.

### 4.3 Real WebGym Loop Replay

다음으로 실제 WebGym server `127.0.0.1:18001`에 연결했다.

재현한 loop:

```text
WebGym start
  -> SGLang stream generate
  -> parse/fallback tool call
  -> real WebOsGymTool.execute
  -> WebGym action
  -> screenshot 누적
  -> next generate
```

실험 조건:

```text
sessions: 16
turns: 3
task_id: counter
include_a11y: true
student text visibility: image가 있으면 text/a11y hidden
fallback action: CLICK(x=696, y=475)
```

fallback action을 강제한 이유는 model tool-call 품질을 평가하려는 실험이 아니라, 실제 WebGym screenshot 누적과 SGLang request latency를 분리해서 보기 위함이다.

결과:

```text
pretokenized_image, real WebGym images

turn 1:
  TTFT avg  ~= 17.0s
  total avg ~= 17.4s
  tool avg  ~= 0.85s

turn 2:
  TTFT avg  ~= 34.3s
  total avg ~= 34.8s
  tool avg  ~= 0.83s

turn 3:
  TTFT avg  ~= 51.4s
  total avg ~= 51.9s
  tool avg  ~= 0.77s
```

같은 조건에서 native path:

```text
native_image, real WebGym images

turn 1:
  TTFT avg  ~= 0.13s
  total avg ~= 0.62s
  tool avg  ~= 0.90s

turn 2:
  TTFT avg  ~= 0.34s
  total avg ~= 0.88s
  tool avg  ~= 0.96s

turn 3:
  TTFT avg  ~= 0.48s
  total avg ~= 1.09s
  tool avg  ~= 0.75s
```

결론:

- WebGym action latency는 0.7-1.2s 수준이다.
- stall은 tool call/result 사이 HTTP action이 아니라 SGLang generate prefill 구간이다.
- image 수가 누적될수록 `pretokenized_image` TTFT가 약 17s 단위로 증가한다.
- native image path는 같은 WebGym image 누적에서도 정상이다.

## 5. Server Trace Evidence

대표 pretokenized WebGym request:

```text
request_id='webgymloop-pretokenized_image-s0000-t1-a24682a5'
image_count=1
base_input_ids_len=1789
final_input_ids_len=2748

mm_process_done:          ~18.6ms
scheduler_dispatch_done:  ~8.1ms
first_response_ready:     ~17471.2ms
prefill_launch_latency:   ~17.399s
```

대표 3-image request:

```text
request_id='webgymloop-pretokenized_image-s0007-t3-7dcb7244'
image_count=3
base_input_ids_len=1897
final_input_ids_len=4774

mm_process_done:          ~56.2ms
scheduler_dispatch_done:  ~28.8ms
first_response_ready:     ~49374.2ms
```

해석:

- image load / preprocess / rope metadata 생성은 수십 ms 수준이다.
- tokenizer manager에서 scheduler로 보내는 데도 수십 ms 수준이다.
- detokenizer/IPC 반환도 첫 token 이후에는 수 ms 수준이다.
- 긴 구간은 model runner prefill launch / first token 이전 구간이다.

## 6. What This Is Not

### 모델 GPU 로딩 문제가 아니다

SGLang servers는 이미 떠 있었고 GPU memory에 model weights가 올라간 상태였다.

또한 같은 고정 서버에서 `native_image`는 빠르게 응답했다.

따라서 이것은 model loading warmup이 아니다.

### 일반적인 첫 요청 warmup만도 아니다

일부 cold request-shape warmup은 있을 수 있다.

하지만 real WebGym loop에서는:

```text
turn 1: ~17s
turn 2: ~34s
turn 3: ~51s
```

처럼 image 누적에 따라 반복적으로 커진다.

따라서 단순히 "첫 요청만 느리다"가 아니다.

### WebGym action server가 주범이 아니다

실제 WebGym action round trip은 대부분 0.7-1.2s다.

`pretokenized_image`의 17/34/51s와 scale이 다르다.

### Base64/PIL 변환이 주범이 아니다

SGLang trace에서 `mm_process_done`이 수십 ms 수준이다.

실제 screenshot transport/decode 비용은 존재하지만, 현재 stall의 order-of-magnitude와 맞지 않는다.

### Teacher prompt-logprob/top-k만의 문제로 보기 어렵다

이번에 가장 강하게 재현된 stall은 student generation path에서 발생했다.

Teacher prompt-logprob path는 별도 주의가 필요하지만, 현재 17/34/51s stall의 직접 원인은 student `input_ids + image_data` generate path다.

## 7. Root Cause Hypothesis

현재 가장 강한 가설:

```text
SGLang의 multimodal pretokenized generate path
  = GenerateReqInput(input_ids=logical_prompt_ids, image_data=images)

가 Qwen-VL image-bearing prompt에서 prefill/model-runner path를 비정상적으로 느리게 만든다.
```

반면:

```text
GenerateReqInput(text=rendered_chat_prompt, image_data=images)
```

는 같은 image count와 같은 WebGym loop에서 정상 속도로 처리된다.

따라서 문제는 "multimodal 자체"가 아니라 "multimodal + pretokenized input_ids generate path"다.

## 8. Fix Direction

### Principle

WebSKD의 correctness stream과 SGLang serving request stream을 분리한다.

유지해야 하는 것:

- `prompt_ids`: local processor-expanded training ids
- `server_prompt_ids`: SGLang logical ids for alignment / fallback
- `teacher_prompt_ids`: teacher local ids
- `teacher_server_prompt_ids`: teacher logical ids
- teacher prompt-logprob delta contract
- response mask / teacher row alignment
- atomic Web observation commit

바꿀 수 있는 것:

- student generation을 위한 SGLang request representation

### Proposed Student Generate Path

image가 없는 경우:

```text
기존 input_ids path 유지 가능
```

image가 있는 경우:

```text
canonical state:
  A = server_prompt_ids

request view:
  T = tokenizer.decode(A, skip_special_tokens=False, clean_up_tokenization_spaces=False)

SGLang request:
  text = T
  image_data = accumulated images

SGLang internal prompt:
  B = SGLang tokenizer(text=T)

SGLang output:
  S = generated suffix token ids

Training/alignment state:
  server_prompt_ids <- A + accepted_or_corrected_suffix(S)
  prompt_ids / response_mask / teacher rows는 기존 방식 유지
```

즉, SGLang generate만 native image path로 보낸다. `B`는 SGLang 내부의 transient request view일 뿐이며, WebSKD의 canonical state로 승격하지 않는다.

### No Training-time Fallback

학습 중에는 drift가 관측되어도 기존 `input_ids + image_data` path로 fallback하지 않는다.

이유:

- fallback path가 현재 stall의 직접 원인이다.
- drift가 있다고 해서 즉시 crash가 나는 것은 아니며, 위험도는 drift 종류에 따라 다르다.
- 먼저 실제 학습 경로에서 drift 규모와 유형을 계측해야 한다.

따라서 정책은 다음과 같다.

```text
image-bearing WebSKD student generation:
  always use text + image_data request view
  never fallback to input_ids + image_data during training
  log round-trip drift and request-shape metrics

teacher verification:
  keep existing teacher_server_prompt_ids + suffix prompt-logprob path
```

허용하지 않는 것은 fallback이지, 장애 은폐가 아니다. SGLang request 자체가 실패하거나 output contract가 깨지면 기존처럼 error를 올린다.

### Required Implementation Boundary

필요한 변화는 대략 다음 경계다.

1. `WebSkdAgentLoop`가 student messages를 계속 보존한다.
   - 이미 `agent_data.messages`가 student-visible messages다.
   - tool observation commit 때 messages도 atomic하게 갱신된다.
2. student generate 호출 시 image가 있으면 `server_prompt_ids`를 request text로 detokenize할 수 있어야 한다.
   - `tokenizer.decode(server_prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)`
   - 같은 tokenizer로 round-trip encode를 수행해 drift를 기록한다.
3. `AsyncLLMServerManager.generate()` / `SGLangHttpServer.generate()`가 optional `prompt_text`를 받을 수 있어야 한다.
4. `SGLangHttpServer.generate()`는 다음 중 하나를 선택한다.
   - `prompt_text is not None`: `GenerateReqInput(text=prompt_text, image_data=image_data)`
   - otherwise: `GenerateReqInput(input_ids=prompt_ids, image_data=image_data)`
5. output token ids는 기존처럼 response tokens로 받아 teacher verification을 거친 뒤 `server_prompt_ids += new_tokens`에 append한다.

### Why This Is Safe in Principle

SGLang native text path가 tokenizer를 다시 거치므로, prompt tokenization이 server-side에서 수행된다. 따라서 엄밀한 생성 조건은 다음과 같다.

```text
S ~ p(. | B)
B = SGLangTokenizer(T)
T = LocalTokenizer.decode(A)
A = server_prompt_ids
```

반면 학습과 teacher alignment의 canonical stream은 계속 `A`다.

```text
A <- A + accepted_or_corrected_suffix(S)
```

이것은 완전한 수학적 항등성을 주장하는 것이 아니다. 대신 다음 절충이다.

- 기존 canonical ids, masks, teacher rows를 중간에 갈아끼우지 않는다.
- 느린 pretokenized multimodal path를 피한다.
- `A`와 `decode(A)` round-trip drift를 로깅해서 실제 위험도를 관측한다.

이 방식에서 남는 위험은 behavior-policy prefix drift다. 이는 fallback stall보다 작은 위험으로 보고, 실제 drift 통계를 근거로 후속 개선을 결정한다.

## 9. Risks

### Risk 1. Text path tokenization drift

local `server_prompt_ids`와 SGLang native text path 내부 tokenization이 다르면, student는 `B`를 보고 생성했는데 학습 state는 `A + S`로 기록된다.

대응:

- `A -> decode -> encode` round-trip 결과를 매 student chunk마다 로깅한다.
- exact match 여부, 길이 차이, first mismatch 위치, mismatch 주변 token ids, prompt text 길이를 기록한다.
- 학습 중 fallback은 하지 않는다.
- structural drift 여부는 로그 분석 단계에서 판단한다.

### Risk 2. Teacher verification path와 섞으면 안 된다

teacher prompt-logprob delta는 여전히 `teacher_server_prompt_ids + chunk`에 대한 suffix logprob를 정확히 받아야 한다.

현재 확인된 stall은 student generate path가 중심이다.

따라서 첫 수정은 student generate native path에 한정한다. teacher logprob path는 별도 실험 없이 같이 바꾸지 않는다.

### Risk 3. Prefix cache behavior changes

native text path는 SGLang 내부 prefix cache key가 input_ids path와 다를 수 있다.

하지만 현재 input_ids image path가 17/34/51s stall을 만든다. prefix cache 이득보다 path 병목 비용이 압도적으로 크다.

### Risk 4. General agent loop 영향

이 변경은 WebSKD image-bearing student generation에만 적용해야 한다.

text-only SKD, math SKD, non-Web tool SKD는 기존 path를 유지한다.

### Risk 5. Returned suffix의 의미

SGLang이 반환하는 `output_ids`는 generated suffix다. 이 suffix만 canonical `server_prompt_ids`에 append해야 한다.

금지:

```text
server_prompt_ids <- B + S
```

허용:

```text
server_prompt_ids <- A + accepted_or_corrected_suffix(S)
```

`B`로 canonical state를 교체하면 이전 chunk의 teacher rows, response mask, accepted/rejected 기록이 밀릴 수 있다.

## 10. Validation Plan

### Unit / CPU-level

1. WebSKD가 image-bearing prompt에서 `server_prompt_ids`를 `prompt_text` request view로 detokenize하는지 확인한다.
2. drift가 있어도 fallback하지 않고 drift metadata만 기록하는지 확인한다.
3. image가 없는 경우 기존 `input_ids` path를 유지하는지 확인한다.
4. SGLang request builder가 `prompt_text`가 있으면 `text`, 없으면 `input_ids`를 쓰는지 확인한다.
5. Web observation commit 후 messages, server ids, local ids가 함께 갱신되는지 확인한다.
6. teacher rows / response mask alignment invariant가 유지되는지 확인한다.

### Fixed SGLang replay

다음 두 경로를 재실행한다.

```text
pretokenized_image_student
native_image_student
```

기대:

- WebSKD patched student path가 native_image 수준으로 내려와야 한다.
- 3 image / concurrency 16에서 TTFT가 수십 초가 아니라 1초 안팎이어야 한다.

### Real WebGym loop replay

실제 WebGym fixed server loop를 다시 실행한다.

기대:

```text
turn 1/2/3 모두 native_image와 같은 order
tool latency는 기존처럼 0.7-1.2s 수준
generate TTFT는 17/34/51s 패턴이 사라짐
```

### Full Async SKD smoke

작은 설정으로 full trainer를 돌린다.

확인:

- `[SKD_DBG] student=...ms`가 수십 초로 튀지 않는지
- teacher verify time이 정상인지
- teacher row alignment test가 깨지지 않는지
- actor update까지 도달하는지

## 11. Artifacts

실험 스크립트:

```text
WebOSWorld/sglang_replay_bench/verl_path_replay_bench.py
WebOSWorld/sglang_replay_bench/webgym_fixed_server_loop_bench.py
```

주요 결과 로그:

```text
logs/sglang_replay_bench/verl_path_stream_student_t3.jsonl
logs/sglang_replay_bench/webgym_fixed_server_loop_t3_forced.jsonl
logs/sglang_replay_bench/verl_path_fixed_gpu*_port3700*.log
```

대표 요약:

```text
pretokenized_image + real WebGym:
  turn 1 TTFT ~= 17s
  turn 2 TTFT ~= 34s
  turn 3 TTFT ~= 51s

native_image + real WebGym:
  turn 1 TTFT ~= 0.13s
  turn 2 TTFT ~= 0.34s
  turn 3 TTFT ~= 0.48s
```

## 12. Decision

다음 구현은 WebSKD image-bearing student generation에서 SGLang native multimodal path를 사용하도록 바꾸는 것을 1순위로 한다.

구체적으로는 다음 결정이다.

- `server_prompt_ids`는 계속 canonical state다.
- image-bearing student generation에서는 `server_prompt_ids`를 detokenize한 `prompt_text`를 SGLang에 보낸다.
- SGLang output suffix는 teacher verification 이후 canonical streams에 append한다.
- 학습 중에는 느린 pretokenized multimodal fallback을 쓰지 않는다.
- drift는 충분히 로깅하고, 후속 분석에서 위험도를 판단한다.

단, 이 변경은 serving request representation만 바꾸는 것이다. SKD correctness를 담당하는 token streams, teacher streams, response masks, teacher rows, atomic commit contract는 유지한다.
