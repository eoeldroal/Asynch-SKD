# Async SKD Multimodal Pretokenized Stall

## 0. Status

이 문서는 WebGym 기반 Async SKD 실행에서 관측된 긴 stall의 원인 조사와 해결 방향을 정리한다.

주의: student native-text 우회는 historical RCA다. 현재 디자인의 source of truth는 ids-only runtime state와 teacher verify contract이며, 이 문서는 stall 해석의 기록으로 읽어야 한다.

현재 결론:

- 병목은 WebGym action server, screenshot base64/PIL decode, tokenizer, 일반 multimodal preprocess가 아니다.
- 초기 강한 병목 중 하나는 SGLang에 `input_ids + image_data`로 multimodal generate를 요청하는 pretokenized student generation path였다.
- 같은 이미지와 같은 WebGym loop라도 `text + image_data` native path는 정상 속도로 동작했고, 이 근거로 WebSKD student image generation은 native text path로 우회했다.
- 이 native-text 우회는 당시 stall을 줄이기 위한 historical workaround였고, 현재는 SGLang upgrade 이후 generation stall의 직접 대응책으로 유지되는 설계가 아니다.
- native text 우회 이후에도 stall이 완전히 사라진 것은 아니었다. 이전 로그 기준으로는 다음 두 종류의 SGLang-local stall이 관측되었다.
  1. tokenizer-manager가 request dispatch를 끝낸 뒤 scheduler process가 실제로 request를 받기까지 수십 초가 걸리는 internal handoff stall
  2. teacher `input_ids + image_data + prompt_logprobs` prefill에서 실제 `forward_batch_generation(...)`가 48-49초까지 걸리는 true model-forward stall
- 하지만 더 낮은 레벨까지 계측한 최신 실행에서는, 긴 stall의 대부분이 더 이상 handoff/postprocess 쪽이 아니라 **`TpModelWorker -> model_runner.forward(...)` 내부**에서 소비되는 것으로 좁혀졌다.
- 즉, 현재 이해는 다음과 같다.
  - historical logs: handoff stall + true forward stall이 모두 있었다
  - latest low-level run: 재현된 큰 stall은 거의 전부 true forward stall이었다
- 현재 관점에서 이 우회는 "stall을 막기 위해 계속 필요한 활성 workaround"라기보다, upgrade 이전 증상을 설명하는 historical branch다. generation stall 자체는 SGLang upgrade 이후 현재 blocker로 남아 있지 않다.

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

## 6. Addendum: Actor-Side Multimodal Forward Stall

The pretokenized student-generate stall above was not the last major blocker.
After the ids-only runtime cleanup and the windowed-training parser fixes, a
second and much lower-level stall remained in actor updates.

### Observed Symptom

In the WebGym Async SKD actor path, `self.module()` inside the VeOmni training
worker became the dominant wall time by a huge margin.

Representative logs before the fix:

- local `input_ids_shape` commonly in the `14k-19k` range
- local `pixel_values_shape` commonly in the `23k-34k` range
- `veomni.forward_step_module_forward_done` often at `90s-170s`

This persisted even after:

- sparse visual observation parsing was fixed
- fresh/carryover window identity was fixed
- Ulysses sequence parallelism was enabled

Enabling `ulysses_parallel_size=2` alone did not solve the stall. It reduced
local sequence length, but dynamic packing immediately used the extra budget to
pack heavier multimodal microbatches, and actor forward remained extremely
slow.

### Root-Cause Hypothesis

The decisive clue was that the Qwen vision path enters through a `Conv3d`
patch-embedding layer under bf16/autocast, while the environment was pinned to:

- `torch==2.9.1+cu128`
- `nvidia-cudnn-cu12==9.10.2.21`

This matched public PyTorch regression reports around `Conv3d + bf16/AMP` on
the 2.9 line closely enough to justify a single-variable environment test.

### Final Fix

Keep the rest of the shared stack on the torch 2.9.1 line, but raise cuDNN in
the cloned RL environment:

```text
nvidia-cudnn-cu12: 9.10.2.21 -> 9.15.1.9
```

This was validated in a cloned env (`skd-cudnn`) rather than by mutating the
original shared env first.

### Outcome

After the cuDNN upgrade, the same actor path changed from:

- `module_forward_ms ~= 90s-170s`

to:

- `module_forward_ms ~= 0.4s-4.8s`

in live RL runs, with the run continuing past multiple logical steps and actor
updates.

### Practical Conclusion

For this WebGym Async SKD Qwen3.5 setup on B200:

1. parser/alignment bugs and actor-kernel stalls were separate problems
2. fixing the parser was necessary, but not sufficient
3. the decisive actor-stall fix was the cuDNN move to `9.15.1.9`
4. Ulysses and actor packing settings still matter, but they were not the
   final root cause of the extreme `self.module()` stall

## 7. What This Is Not

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

## 8. Root Cause Hypothesis

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

## 9. Fix Direction

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

## 10. Risks

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

## 11. Validation Plan

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

## 12. Artifacts

당시 사용한 실험 스크립트:

```text
WebOSWorld/sglang_replay_bench/verl_path_replay_bench.py
WebOSWorld/sglang_replay_bench/webgym_fixed_server_loop_bench.py
```

주의:

- 위 벤치 스크립트들은 RCA 당시의 일회성 재현 자산이었다.
- stall의 최종 원인이 runtime/cuDNN 쪽으로 정리된 뒤에는 기본 코드 경로를
  더럽히지 않기 위해 repo에서 제거했다.
- 아래 로그 경로는 historical result reference로만 남긴다.

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

## 13. Decision

다음 구현은 WebSKD image-bearing student generation에서 SGLang native multimodal path를 사용하도록 바꾸는 것을 1순위로 한다.

구체적으로는 다음 결정이다.

- `server_prompt_ids`는 계속 canonical state다.
- image-bearing student generation에서는 `server_prompt_ids`를 detokenize한 `prompt_text`를 SGLang에 보낸다.
- SGLang output suffix는 teacher verification 이후 canonical streams에 append한다.
- 학습 중에는 느린 pretokenized multimodal fallback을 쓰지 않는다.
- drift는 충분히 로깅하고, 후속 분석에서 위험도를 판단한다.

단, 이 변경은 serving request representation만 바꾸는 것이다. SKD correctness를 담당하는 token streams, teacher streams, response masks, teacher rows, atomic commit contract는 유지한다.

## 14. Updated Stall Anatomy After Native Text Reroute

이 문서의 앞부분은 "왜 student `input_ids + image_data` path를 native `text + image_data`로 우회해야 하는가"를 설명한다. 그 결론은 여전히 맞다.

하지만 이후 실제 Async SKD 학습 로그를 더 깊게 추적해 보니, native text 우회 이후에도 stall은 남아 있었고, 그 내부 구조는 하나가 아니라 둘이었다.

### 13.1 Two Stall Classes

현재 관측되는 긴 stall은 다음 두 클래스로 나뉜다.

1. **Internal handoff stall**
   - tokenizer / multimodal preprocess는 이미 끝났다.
   - tokenizer-manager도 request dispatch를 끝냈다.
   - 그런데 scheduler process가 그 request를 실제로 받기까지 수십 초가 걸린다.
   - 이 경우 긴 시간은 model forward 이전에 소비된다.

2. **True forward stall**
   - scheduler가 request를 받아 batch를 구성한다.
   - `forward_mode='1'` prefill batch가 실제 model worker로 들어간다.
   - `forward_batch_generation(...)` 자체가 48-49초까지 걸린다.
   - 이 경우 긴 시간은 실제 model forward 내부에서 소비된다.

둘은 증상이 비슷하게 "응답이 안 오는 것"처럼 보이지만, 로그 경계가 완전히 다르다.

### 13.2 Why `dispatch -> recv` Is a Separate Stall

SGLang trace에서 다음 두 로그는 서로 다른 프로세스 경계를 나타낸다.

```text
sglang.scheduler_dispatch_done
  = tokenizer-manager가 _send_one_request(...)를 끝낸 시점

sglang.scheduler_loop_recv_done
  = scheduler process가 recv_requests()로 실제 request를 받은 시점
```

즉,

```text
dispatch_done -> recv_done
```

구간이 길다면, 그 시간은 다음 중 어느 것도 아니다.

- WebGym tool HTTP
- base64 / PIL image decode
- tokenizer
- Qwen-VL multimodal preprocess
- model forward

이 시간은 **SGLang 내부에서 tokenizer-manager -> scheduler로 request가 넘어가는 handoff 경계**에서 사라진다.

### 13.3 Decisive Student Example: `3d0fae...`

대표 학생 request:

```text
request_id='3d0fae45fe214145a92b9bf6cd8f7a99'
request_input_kind='text'
image_count=2
prompt_len=1969
prompt_text_len=7886
```

핵심 timeline:

```text
tokenize_done:                ~35.9ms
scheduler_dispatch_done:      ~19.6ms
tokenizer_generate_waiting:    5.0s
tokenizer_generate_waiting:   10.0s
scheduler_loop_recv_done:     44.4s
scheduler_loop_get_batch_done:44.7s
scheduler_model_forward_begin:44.7s
first_response_ready:         45.2s
generate_request_done:        45.2s
```

해석:

- tokenizer와 multimodal preprocess는 정상이다.
- request dispatch도 정상이다.
- model forward는 마지막 수백 ms만 차지한다.
- 전체 45초 중 대부분은 scheduler가 request를 실제로 받기 전 단계에서 사라진다.

즉 이 request는 **student native text path에서도 internal handoff stall이 실제로 발생함**을 보여준다.

### 13.4 Decisive Teacher Example: `16e625...`

대표 teacher request:

```text
request_id='16e625d64c0a4f7187cf7c48271929c3'
request_input_kind='input_ids'
image_count=3
prompt_logprobs=32
logprob_start_len=4609
```

핵심 timeline:

```text
tokenizer_generate_waiting:    5.0s
tokenizer_generate_waiting:   10.0s
scheduler_model_forward_done: 48.6s
first_response_ready:         48.8s
generate_request_done:        48.8s
```

그리고 같은 request의 scheduler-side trace:

```text
forward_mode='1'
return_logprob=True
seq_lens_sum=tensor(4674)
prefill_launch_latency=48.650s
```

이 경우는 internal handoff가 아니라, **teacher multimodal prompt-logprob prefill 자체가 48-49초짜리 true model-forward stall**이라는 뜻이다.

### 13.5 Mixed Case: `52b278...`

`52b278aa...` 같은 request는 두 종류의 stall이 섞일 수 있음을 보여준다.

이 request는:

```text
request_input_kind='input_ids'
image_count=3
generate_request_done: ~43.0s
```

그런데 잡힌 scheduler-side forward trace는:

```text
scheduler_model_forward_done: ~0.77s
batch_size=3
seq_lens_sum=tensor(14354)
```

즉, 전체 요청은 43초가 걸렸는데, 실제 관측된 forward는 0.77초다.

이런 패턴은:

- forward collapse만으로 설명되지 않고
- request lifecycle의 다른 구간, 특히 internal handoff / scheduling admission 쪽에 큰 stall이 있었음을 시사한다.

### 13.6 Replica-local Pathology

긴 요청은 모든 replica에 균등하게 나타나지 않았다.

최근 로그 집계 기준:

```text
long text requests (>30s):
  pid 843881: 8건
  pid 843878: 2건
  pid 843882: 1건
  pid 843877: 1건

long input_ids requests (>30s):
  pid 842996: 5건
  pid 842911: 1건
  pid 842907: 1건
  pid 842829: 1건
```

즉, 이것은 단순한 "모든 서버가 똑같이 느리다" 문제가 아니라, **특정 SGLang replica가 pathological state에 더 자주 빠지는 경향**도 함께 보인다.

### 13.7 What This Means

현재 WebGym Async SKD stall을 한 문장으로 요약하면 다음과 같다.

```text
student pretokenized multimodal path는 분명히 잘못된 경로였고 native text path 우회는 필요했다.

하지만 우회 이후에도 SGLang 내부에는
  (1) tokenizer-manager -> scheduler handoff stall
  (2) teacher multimodal prompt-logprob prefill forward collapse
라는 두 개의 서로 다른 stall class가 남아 있다.
```

따라서 현재 병목 분석은 더 이상 "student pretokenized path 하나의 문제"로 끝나지 않는다.

남은 후속 실험은 다음을 분리해서 봐야 한다.

1. **student native text request**
   - dispatch -> recv gap
   - recv -> first token gap

2. **teacher prompt-logprob request**
   - scheduler admitted 이후 actual forward time
   - replica별 반복성

이 분리가 되어야, 이후 최적화가

- request ingress / IPC / internal queue 문제인지
- teacher multimodal prompt-logprob serving 문제인지

정확히 갈라진다.

## 15. Lower-level Instrumentation Update

이후 SGLang manager 레벨보다 한 단계 더 낮은 위치에 계측을 추가했다.

추가한 주요 경계는 다음과 같다.

- `recv_requests()` 내부 세부 시간
  - `sglang.scheduler_recv_phase_done`
- overlap scheduler loop 전체 iteration 분해
  - `sglang.scheduler_overlap_iteration_done`
- `TpModelWorker.forward_batch_generation(...)` 내부 분해
  - `sglang.tp_worker_forward_batch_init_done`
  - `sglang.tp_worker_model_runner_forward_done`
  - `sglang.tp_worker_sample_done`
  - `sglang.tp_worker_compute_logprobs_only_done`
- prefill 후처리 분해
  - `sglang.prefill_logprob_return_values_done`
  - `sglang.prefill_cache_update_done`
  - `sglang.scheduler_stream_collect_done`
  - `sglang.scheduler_send_to_detokenizer_done`

이 계측의 목적은 다음 세 가설을 분리하는 것이었다.

1. scheduler ingress / recv stall
2. true prefill forward stall
3. forward 이후 CPU postprocess / detokenizer send stall

### 14.1 What the New Traces Ruled Out

최신 실행 로그에서는 다음 패턴이 보이지 않았다.

```text
sglang.scheduler_recv_phase_done total_ms >= 1s
sglang.scheduler_stream_collect_done elapsed_ms >= 1s
sglang.scheduler_send_to_detokenizer_done elapsed_ms >= 1s
sglang.prefill_logprob_return_values_done elapsed_ms >= 1s
sglang.prefill_cache_update_done elapsed_ms >= 1s
sglang.scheduler_overlap_iteration_done
sglang.scheduler_delayed_sample_done
sglang.scheduler_recv_skipped
```

즉 최신 실행 기준으로는:

- scheduler ingress / ZMQ receive / broadcast가 수십 초를 먹지 않았다
- detokenizer send나 CPU-side result packing이 병목이 아니었다
- overlap scheduling이나 delayed sample path가 stall의 원인이 아니었다

### 14.2 Decisive Student Example: `934145...`

대표 student native-text request:

```text
request_id='9341453d2af44d2abf530c960a534a74'
request_input_kind='text'
image_count=3
forward_mode='1'
seq_lens_sum=tensor(4935)
```

핵심 timeline:

```text
tp_worker_model_runner_forward_done:   49758.7ms
scheduler_model_forward_done:          49759.5ms
scheduler_run_batch_done:              49822.2ms
prefill_req_loop_done:                     1.0ms
prefill_cache_update_done:                0.9ms
scheduler_stream_collect_done:            0.0ms
scheduler_send_to_detokenizer_done:       0.1ms
prefill_postprocess_done:                 1.4ms
scheduler_loop_iteration_done:
  recv_ms=21.5
  process_ms=73.2
  schedule_ms=6.5
  run_ms=49822.4
  post_ms=1.5
```

이 request는 최신 계측 기준으로 매우 결정적이다.

- request ingress는 정상이다.
- prefill 후처리도 정상이다.
- 긴 시간의 거의 전부가 `run_ms`이고,
- 그 `run_ms`는 다시 `tp_worker_model_runner_forward_done`와 거의 동일하다.

즉 이 케이스의 stall은 **scheduler 바깥/주변이 아니라 `model_runner.forward(...)` 내부**다.

### 14.3 Decisive Teacher Example: `b7338f...`

대표 teacher prompt-logprob request:

```text
request_id='b7338fcd73154cdc8c98b3154b26e497'
request_input_kind='input_ids'
image_count=3
return_logprob=True
forward_mode='1'
seq_lens_sum=tensor(4815)
```

핵심 timeline:

```text
tp_worker_model_runner_forward_done:   51994.2ms
scheduler_model_forward_done:          51995.3ms
scheduler_run_batch_done:              51996.2ms
prefill_logprob_tolist_done:               0.0ms
prefill_logprob_return_values_done:        0.0ms
prefill_cache_update_done:                 0.3ms
scheduler_stream_collect_done:             0.1ms
prefill_postprocess_done:                  0.8ms
scheduler_loop_iteration_done:
  recv_ms=23.0
  process_ms=72.3
  schedule_ms=6.7
  run_ms=51996.4
  post_ms=0.9
```

teacher 경로도 해석은 같다.

- `prompt_logprobs=32` 후처리 자체는 거의 공짜다
- 결과 packing도 거의 공짜다
- 실제 52초는 `model_runner.forward(...)` 안에서 소모된다

즉 최신 실행 기준으로는, teacher stall도 더 이상
"prompt logprob row trimming / return-value packing"
같은 Python-side postprocess로 설명되지 않는다.

### 14.4 What This Changes

이전에는 "handoff stall"과 "true forward stall"을 동등한 두 축으로 취급했다.

그 해석은 **historical logs를 설명하는 데에는 여전히 유효**하다. 실제로 이전에는

- `dispatch_done -> recv_done`가 수십 초였던 case
- 전체 요청은 43초인데 scheduler-side forward trace는 0.77초였던 mixed case

가 존재했다.

하지만 최신 저수준 계측 실행에서는 큰 stall이 재현될 때마다 경계가 다음처럼 정렬되었다.

```text
tp_worker_model_runner_forward_done
  ~= scheduler_model_forward_done
  ~= scheduler_run_batch_done
  >> prefill postprocess
  >> scheduler send / detokenizer send
```

따라서 **현재 재현되는 dominant pathology는 `model_runner.forward(...)` 내부의 multimodal EXTEND/prefill collapse**로 보는 것이 가장 정확하다.

### 14.4.1 Sticky Carryover Clarification

여기서 한 가지 표현을 더 정확히 해야 한다.

이 문서 앞선 논의에서 teacher verification을 "긴 multimodal prefix에 대해 매 chunk마다 fresh prefill scoring을 다시 한다"고 요약한 적이 있다. 이 표현은 **방향은 맞지만 너무 거칠다.**

실제 구현은 다음 두 사실을 동시에 가진다.

1. **Sticky carryover는 실제로 존재한다.**
   - design 문서도 carryover sample을 같은 teacher replica로 다시 보내 **KV locality**를 활용한다고 명시한다.
   - `AsyncLLMServerManager` docstring도 multi-turn request를 같은 server로 보내 **automatic prefix caching**을 노린다고 적고 있다.
   - 실제 teacher path는 `bind_sticky_request(...)`를 통해 request id를 특정 teacher replica에 묶는다.

2. **하지만 현재 teacher verify path는 true session continuation은 아니다.**
   - teacher verify는 매 chunk마다 `verify_sequence = teacher_server_prompt_ids + chunk`를 다시 만들고,
   - `compute_teacher_logprobs_single(...)`에서 그 전체 `sequence_ids`를 다시 `generate(...)`로 넘긴다.
   - 이 request는 일반 `GenerateReqInput` 경로를 사용하며, SGLang이 별도로 제공하는 `session_params` / `open_session` / `continue_generation` 기반의 "old kv cache를 이어 쓰는" contract는 사용하지 않는다.

즉 현재 teacher path를 가장 정확히 설명하면 다음과 같다.

```text
same-replica repeated full-sequence scoring request
+ prefix-cache / KV locality opportunity
- true live-session continuation
- completely cold random-replica prefill
```

이 구분은 중요하다.

- **"매번 완전히 cold prefill"**이라고 말하면 sticky 구현의 효과를 과소평가하게 된다.
- 반대로 **"이미 KV를 다 재사용하니 prefix cost는 거의 없다"**고 말하면 현재 request contract를 과대평가하게 된다.

현재 stall 해석은 이 중간점 위에 서야 한다.
Sticky carryover는 분명 teacher 부담을 줄이는 장치다. 다만 지금의 verify 호출은 여전히 **multimodal EXTEND path를 반복해서 밟는 request shape**이므로, prefix locality가 있더라도 pathological `model_runner.forward(...)`를 완전히 피하지는 못한다.

### 14.5 Remaining Open Question

이 문서 수준에서 더 이상 못 자르는 마지막 질문은 이것이다.

```text
model_runner.forward(...) 내부에서
정말 dense GPU prefill compute가 오래 도는 것인지,
아니면 attention backend / CUDA stream / kernel launch / sync 계열의
internal stall인지
```

새 로그는 다음을 이미 보여준다.

- manager-level ingress는 병목이 아니다
- prefill req-loop나 detokenizer send도 병목이 아니다
- 시간은 `model_runner.forward(...)` 호출 경계 안에 있다

하지만 이 문서 수준의 계측만으로는, 그 내부가

- 실제 matmul-heavy prefill compute
- multimodal backend-specific pathological wait
- CUDA graph 비적용 경로에서의 synchronization

중 정확히 무엇인지는 아직 확정하지 못했다.

즉 다음 조사 목표는 자연스럽게 **SGLang `model_runner` / attention backend / model forward implementation 내부 계측**이다.
