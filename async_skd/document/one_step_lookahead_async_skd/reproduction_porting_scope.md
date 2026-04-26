# Async SKD Reproduction Porting Scope

## 0. Purpose

이 문서는 현재 APSKD 저장소에서 구현된 **bounded one-step lookahead async SKD**를 새 verl 코드베이스로 이식할 때 가져가야 하는 변경 범위를 정리한다.

목표는 두 가지다.

1. Async SKD 구현이 완성된 시점까지의 변경만 안정적으로 재현한다.
2. 이후에 시작된 qwen3.5, FP8, SGLang 0.5.10 호환성, local code interpreter 실험 변경은 섞지 않는다.

따라서 source of truth는 `HEAD`가 아니라 아래 git 범위다.

```text
base commit: 293cdb72^
end commit:  9030c3c1
range:       293cdb72^..9030c3c1
```

`9030c3c1`은 다음 커밋이다.

```text
9030c3c1 Merge pull request #5 from eoeldroal/async-skd-boundary-worker
```

이 문서에서 말하는 "Async SKD 재현 패키지"는 `293cdb72^..9030c3c1`에 들어간 변경을 기준으로 한다.

## 1. Cutoff Rule

### 1.1 Include

다음 범위는 포함한다.

```bash
git diff 293cdb72^ 9030c3c1
```

이 범위에는 다음 작업이 들어 있다.

- on-policy distillation 기반 SKD agent loop
- async SKD manager, worker, state, data source
- lookahead scheduling
- carry-over partial state
- promoted sample assembly
- dynamic promoted batch handling
- worker slot refill
- export predicate 기반 partial snapshot
- async SKD event log, dashboard, runtime diagnostics
- qwen3 math long-context async SKD launcher
- tool-aware math reproduction support
- 관련 테스트

### 1.2 Exclude

다음 커밋과 현재 working tree 변경은 제외한다.

```text
79b00dc1 script: add updated sglang uv installer
52c5c431 Use matching FlashAttention wheel in installer
e48b0c58 chore: update sglang runtime compatibility
548f7a74 tools: add local code interpreter endpoint
73e37cbb examples: add qwen3.5 skd tool run
7e45bc82 config: raise tool response length limit
```

특히 다음 변경은 Async SKD 재현 범위가 아니다.

- `examples/on_policy_distillation_trainer/run_qwen35_math_skd_fp8.sh`
- qwen3.5 / qwen3.6 FP8 실험 스크립트
- local code interpreter endpoint
- `launch_local_code_interpreter.py`
- SGLang 0.5.10 호환성 패치
- FlashAttention wheel installer 보정
- 최근 tool response length limit 변경
- 현재 dirty working tree에 있는 후속 실험/호환성 변경

## 2. Prerequisite

새 verl 코드베이스가 이미 on-policy distillation과 teacher colocate 기반을 갖고 있다면 `293cdb72^..9030c3c1`만 이식하면 된다.

만약 target verl에 on-policy distillation 자체가 없다면 아래 선행 커밋 계열도 먼저 필요하다.

```text
455e44c6 [fsdp,megatron,vllm,trainer,algo] feat: On-Policy Distillation
51da8fea refactor: Teacher colocate mode
2e8ab5ac feat: Teacher colocate mode
```

이 선행 범위는 Async SKD 자체는 아니지만, `skd_agent_loop.py`, teacher verification server, distillation loss/config 경로가 의존하는 바닥이다.

## 3. Commit Timeline

이식 시에는 가능하면 아래 순서를 유지한다. 중간 커밋을 squash해도 되지만, 충돌을 줄이려면 논리 순서는 유지하는 편이 안전하다.

```text
293cdb72 feat: add APSKD async distillation work
cd384f41 feat: add skd boundary run api
c190bbe2 test: cover async skd boundary worker
195ed97a feat: add async skd boundary worker primitive
7b13959b docs: clarify async skd boundary worker contract
6e309371 docs: fix lookahead admission and quota accounting
b09407f4 feat: add async skd lookahead manager scheduling
0e8a9e98 feat: add async skd data source
ea321d25 feat: add async skd carryover assembly
7a0a7c5a chore: preserve manual skd script permissions
272492b0 feat: add async skd input batch assembly
2bde7cd0 feat: add async skd trainer batch helpers
a35f41e1 docs: clarify async skd reuse plan
dbd8c2c6 feat: assemble async skd promoted inputs
3cacf3ec docs: replace skd safe point labels with export predicates
1c244f72 docs: plan async skd export predicate refactor
79b5cadb refactor: remove async skd safe point state labels
95674e99 refactor: use export predicates for skd partial snapshots
96c259a2 test: update async skd tests for export predicates
6d1bca8c chore: remove residual skd safe point references
7a972fdf docs: add worker slot refill skd scheduling plan
4b7d3c6b docs: add async skd worker slot prefetch plan
b2289a13 merge: include async skd export predicate refactor
7887466d test: specify async skd worker slot refill
c49d0ca8 feat: add async skd worker slot refill
9dcce283 feat: record rollout server id in agent loop outputs
58f4a097 feat: add async skd worker slot metrics
eed3c192 docs: describe async skd worker slot refill implementation
0b5db3db test: specify carryover lookahead scheduling
3f73e3c1 feat: reuse lookahead scheduler for carryover work
213a3651 feat: prepare async skd current input rows
e70026f6 feat: cap lookahead continuation by skd chunks
06c4d9a8 script: add 8gpu 4x4 async skd longctx launcher
155e22b7 fix: preserve async skd source state across epochs
6b835b29 fix: select async skd manager in longctx launcher
8100b724 fix: allow async skd agent config overrides
8ae12288 feat: add async skd runtime diagnostics
6b6f084f chore: log async skd carryover resume chunks
e8253167 fix: keep async skd input snapshot internal
d1c36652 chore: log async skd lookahead drain events
c40915cf fix: make async skd promoted batches dynamic
b39c0441 feat: add async skd observability tooling
9030c3c1 Merge pull request #5 from eoeldroal/async-skd-boundary-worker
```

## 4. Core Async SKD Files

이 절은 Async SKD 구현 자체다. 새 verl이 기존 APSKD와 유사한 구조라면 반드시 가져가야 한다.

### 4.1 Agent Loop

```text
verl/experimental/agent_loop/skd_agent_loop.py
```

역할:

- student generation chunk 생성
- teacher top-k verification
- first-rejection replacement commit
- assistant token에 `response_mask=1` 부여
- tool/user/interact span에 dummy teacher rows와 `response_mask=0` 부여
- partial export boundary 제공
- SKD alignment assert 수행

주의:

- 최근 분석한 빈 출력/짧은 출력 문제도 이 파일의 first-rejection replacement 및 EOS 처리와 직접 연결된다.
- 재현 이식에서는 우선 `9030c3c1` 상태를 그대로 가져가고, 별도 버그픽스는 후속 패치로 분리하는 편이 안전하다.

### 4.2 Async SKD Package

```text
verl/experimental/async_skd/__init__.py
verl/experimental/async_skd/state.py
verl/experimental/async_skd/worker.py
verl/experimental/async_skd/manager.py
verl/experimental/async_skd/data_source.py
verl/experimental/async_skd/events.py
verl/experimental/async_skd/dashboard.py
```

역할:

- `AsyncSkdAgentLoopWorker`: single-sample execution primitive와 boundary execution primitive 제공
- `AsyncSkdAgentLoopManager`: base batch, lookahead, carry-over, promoted sample scheduling
- `AsyncSkdDataSource`: promoted input/output pair accounting
- `state.py`: partial sample, source state, promoted/carry-over/drop 상태 표현
- `events.py`: JSONL event logging
- `dashboard.py`: async SKD event dashboard

핵심 semantics:

- MVP는 `rollout.n == 1`을 전제로 한다.
- lookahead는 current step의 idle slot에서 `current_step + 1` sample만 가져온다.
- terminal lookahead는 promoted sample이 된다.
- non-terminal lookahead는 handler-return export boundary에서 carry-over가 된다.
- current batch로 편입된 carry-over는 이번 step에서 terminal까지 완료해야 한다.

## 5. Integration Files

### 5.1 Agent Loop Manager Integration

```text
verl/experimental/agent_loop/agent_loop.py
verl/experimental/agent_loop/tool_agent_loop.py
```

역할:

- agent loop output schema 확장
- rollout server id / observability metadata 전달
- async SKD worker가 기존 agent loop worker 경계와 호환되도록 지원

### 5.2 Teacher Loop

```text
verl/experimental/teacher_loop/teacher_manager.py
verl/experimental/teacher_loop/teacher_model.py
```

역할:

- teacher verification request/response 경로
- SKD teacher top-k row 생성
- teacher colocate / server routing 경계 지원

### 5.3 PPO Trainer

```text
verl/trainer/ppo/ray_trainer.py
```

역할:

- async SKD manager 선택
- promoted input/output batch assembly
- dynamic promoted batch 처리
- async SKD meta info 및 metrics 수집
- validation 경로와 distillation rollout 경로의 충돌 방지

### 5.4 Config

```text
verl/trainer/config/distillation/distillation.yaml
verl/workers/config/distillation.py
verl/workers/config/rollout.py
```

역할:

- SKD 및 async SKD 설정 필드 추가
- launcher에서 `actor_rollout_ref.rollout.agent.agent_loop_manager_class`로 manager override 가능
- `async_skd_prefetch_limit`, `async_skd_prefetch_worker_target` 등 scheduling 관련 값 전달
- `distillation.skd.max_chunks_per_sample` 등 SKD cap 전달

이식 시 주의:

- 최신 verl에 이미 `teacher_models` / `teacher_key` 기반 multi-teacher OPD 구조가 있으면 이를 유지한다.
- APSKD의 오래된 single-teacher `DistillationConfig.teacher_model` 구조로 덮어쓰지 않는다.
- `distillation.skd.*`는 기존 OPD config 아래에 작은 SKD 전용 블록으로 추가한다.
- `teacher_system_prompt_path`가 설정되면 teacher-only prompt prefix를 감안해 teacher inference budget에 512-token margin을 더한다.
- `rollout.n == 1` 제한은 config dataclass에서 전역으로 막지 않고 `AsyncSkdAgentLoopManager`의 runtime guard에 맡긴다.
- `distillation_loss.memory_efficient`는 Async SKD core가 아니다. target verl의 기존 full-vocab `log_softmax` loss 경로를 그대로 쓰고, launcher에서도 해당 override를 제거한다. 실제 OOM이 확인되면 별도 FSDP distillation loss memory optimization 패치로 분리한다.

### 5.5 SGLang Rollout Server

```text
verl/workers/rollout/sglang_rollout/async_sglang_server.py
```

역할:

- rollout server id 관측
- async SKD event/metric에서 어느 server가 처리했는지 추적

주의:

- `9030c3c1` 이후의 SGLang 0.5.10 호환성 변경은 제외한다.
- 새 verl에서 이미 SGLang server 구조가 달라졌다면 이 파일은 patch를 그대로 적용하기보다 관측 필드와 server id 전달만 이식한다.

## 6. Reproduction Support Files

사용자가 요청한 "실험 재현" 관점에서는 아래 파일도 같이 가져가는 편이 좋다. 엄밀히 말하면 모두 Async SKD core는 아니지만, qwen3 math tool-aware SKD 실험을 같은 조건으로 재현하려면 필요하다.

### 6.1 On-Policy Distillation Trainer Examples

```text
examples/on_policy_distillation_trainer/AGENTS/details.md
examples/on_policy_distillation_trainer/AGENTS/imp_detail.md
examples/on_policy_distillation_trainer/AGENTS/onboarding.md
examples/on_policy_distillation_trainer/AGENTS/warning.md
examples/on_policy_distillation_trainer/document/draft/Implementation Detail.md
examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/design.md
examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd/implementation_plan.md
```

역할:

- 실험 운영상의 경고와 온보딩 문서
- one-step lookahead async SKD 설계 문서
- implementation invariant 정리

### 6.2 Prompts And Tool Config

```text
examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v1.txt
examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v2.txt
examples/on_policy_distillation_trainer/config/tool_config/computer_13_tool_config.yaml
examples/on_policy_distillation_trainer/config/tool_config/sandbox_fusion_tool_config.yaml
```

역할:

- teacher prompt variants
- tool-aware math rollout 설정
- sandbox/tool execution 설정

### 6.3 Launchers To Include

```text
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k_base.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_gspo_tool_longctx16k.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_gspo_tool_longctx16k_base.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_6x2_skd_tool.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt.sh
examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt_v2.sh
```

역할:

- qwen3 math SKD / async SKD / GSPO baseline 재현
- long context 16k async SKD 설정 재현

제외:

```text
examples/on_policy_distillation_trainer/run_qwen35_math_skd_fp8.sh
examples/on_policy_distillation_trainer/run_qwen3_5_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh
```

이 둘은 qwen3.5/FP8 후속 실험 계열이므로 이번 이식 범위에서 제외한다.

### 6.4 Reward, Tools, Dataset Support

```text
examples/on_policy_distillation_trainer/reward_fn_math_verify.py
examples/data_preprocess/deepscaler_preview.py
examples/data_preprocess/math_multiturn_w_tool.py
examples/data_preprocess/nemotron_cascade_rl_math_multiturn_w_tool.py
verl/tools/computer_13_tool.py
verl/tools/sandbox_fusion_tools.py
verl/utils/dataset/rl_dataset.py
verl/utils/reward_score/math_verify.py
verl/experimental/reward_loop/reward_loop.py
```

역할:

- math verification reward
- tool-aware math data preprocessing
- computer_13 / sandbox fusion tool support
- dataset field handling
- reward loop 호환성

주의:

- 이 절은 core Async SKD만 이식할 때는 제외 가능하다.
- 하지만 동일 실험 재현을 목표로 하면 가져가는 편이 안전하다.
- 후속 local code interpreter endpoint는 제외한다.

## 7. Tests To Port

이식 시 테스트는 함께 가져간다. 실행은 target 환경에서 dependency를 맞춘 뒤 단계적으로 수행한다.

### 7.1 Async SKD Unit Tests

```text
tests/skd/__init__.py
tests/skd/test_skd_logic.py
tests/skd/test_async_skd_worker_boundary.py
tests/skd/test_async_skd_worker.py
tests/skd/test_async_skd_state.py
tests/skd/test_async_skd_replica_events.py
tests/skd/test_async_skd_manager_lookahead.py
tests/skd/test_async_skd_manager.py
tests/skd/test_async_skd_events.py
tests/skd/test_async_skd_data_source.py
tests/skd/test_async_skd_dashboard.py
tests/skd/manual/launch_test_servers.sh
tests/skd/manual/skd_integration_manual.py
tests/skd/manual/sweep_chunk_size.sh
```

### 7.2 Trainer And Config Tests

```text
tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
tests/trainer/ppo/test_ray_trainer_validation_on_cpu.py
tests/workers/config/test_distillation_config_on_cpu.py
tests/workers/config/test_rollout_config_on_cpu.py
```

### 7.3 Tool, Dataset, Reward Support Tests

```text
tests/tools/test_computer_13_tool.py
tests/utils/test_sandbox_fusion_tools_on_cpu.py
tests/utils/dataset/test_nemotron_cascade_rl_math_preprocess_on_cpu.py
```

### 7.4 Agent Loop Compatibility Tests

```text
tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py
tests/experimental/agent_loop/test_agent_loop_server_observability.py
tests/experimental/agent_loop/test_basic_agent_loop.py
tests/experimental/agent_loop/test_multi_modal.py
```

## 8. Patch Extraction Commands

### 8.1 Full Reproduction Patch

아래 명령은 Async SKD core와 실험 재현 지원 파일을 함께 추출한다.

```bash
git diff 293cdb72^ 9030c3c1 -- \
  verl/experimental/agent_loop/skd_agent_loop.py \
  verl/experimental/async_skd \
  verl/experimental/agent_loop/agent_loop.py \
  verl/experimental/agent_loop/tool_agent_loop.py \
  verl/experimental/teacher_loop/teacher_manager.py \
  verl/experimental/teacher_loop/teacher_model.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/trainer/config/distillation/distillation.yaml \
  verl/workers/config/distillation.py \
  verl/workers/config/rollout.py \
  verl/workers/rollout/sglang_rollout/async_sglang_server.py \
  examples/on_policy_distillation_trainer/AGENTS \
  examples/on_policy_distillation_trainer/document/draft \
  examples/on_policy_distillation_trainer/document/one_step_lookahead_async_skd \
  examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v1.txt \
  examples/on_policy_distillation_trainer/config/prompts/teacher_system_prompt_v2.txt \
  examples/on_policy_distillation_trainer/config/tool_config/computer_13_tool_config.yaml \
  examples/on_policy_distillation_trainer/config/tool_config/sandbox_fusion_tool_config.yaml \
  examples/on_policy_distillation_trainer/reward_fn_math_verify.py \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k_base.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_gspo_tool_longctx16k.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_gspo_tool_longctx16k_base.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_6x2_skd_tool.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt.sh \
  examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_4x4_skd_tool_teacher_prompt_v2.sh \
  examples/data_preprocess/deepscaler_preview.py \
  examples/data_preprocess/math_multiturn_w_tool.py \
  examples/data_preprocess/nemotron_cascade_rl_math_multiturn_w_tool.py \
  verl/tools/computer_13_tool.py \
  verl/tools/sandbox_fusion_tools.py \
  verl/utils/dataset/rl_dataset.py \
  verl/utils/reward_score/math_verify.py \
  verl/experimental/reward_loop/reward_loop.py \
  tests/skd \
  tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py \
  tests/trainer/ppo/test_ray_trainer_validation_on_cpu.py \
  tests/tools/test_computer_13_tool.py \
  tests/utils/test_sandbox_fusion_tools_on_cpu.py \
  tests/utils/dataset/test_nemotron_cascade_rl_math_preprocess_on_cpu.py \
  tests/workers/config/test_distillation_config_on_cpu.py \
  tests/workers/config/test_rollout_config_on_cpu.py \
  tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py \
  tests/experimental/agent_loop/test_agent_loop_server_observability.py \
  tests/experimental/agent_loop/test_basic_agent_loop.py \
  tests/experimental/agent_loop/test_multi_modal.py
```

### 8.2 Core-Only Patch

target verl에 tool/data/reward reproduction support가 이미 있거나, 우선 Async SKD scheduler만 이식하려면 아래처럼 줄인다.

```bash
git diff 293cdb72^ 9030c3c1 -- \
  verl/experimental/agent_loop/skd_agent_loop.py \
  verl/experimental/async_skd \
  verl/experimental/agent_loop/agent_loop.py \
  verl/experimental/agent_loop/tool_agent_loop.py \
  verl/experimental/teacher_loop/teacher_manager.py \
  verl/experimental/teacher_loop/teacher_model.py \
  verl/trainer/ppo/ray_trainer.py \
  verl/trainer/config/distillation/distillation.yaml \
  verl/workers/config/distillation.py \
  verl/workers/config/rollout.py \
  verl/workers/rollout/sglang_rollout/async_sglang_server.py \
  tests/skd \
  tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py \
  tests/workers/config/test_distillation_config_on_cpu.py \
  tests/workers/config/test_rollout_config_on_cpu.py
```

## 9. Suggested Porting Procedure

### 9.1 Prepare Target

1. Confirm whether target verl already has on-policy distillation and teacher colocate support.
2. Create an isolated branch or worktree in the target repository.
3. Apply prerequisite distillation commits first if needed.
4. Apply the Async SKD patch using `git apply --3way`.

### 9.2 Resolve Conflicts In This Order

1. Config schema:
   - `verl/workers/config/distillation.py`
   - `verl/workers/config/rollout.py`
   - `verl/trainer/config/distillation/distillation.yaml`
2. Agent loop boundaries:
   - `verl/experimental/agent_loop/agent_loop.py`
   - `verl/experimental/agent_loop/tool_agent_loop.py`
   - `verl/experimental/agent_loop/skd_agent_loop.py`
3. Teacher loop:
   - `verl/experimental/teacher_loop/teacher_manager.py`
   - `verl/experimental/teacher_loop/teacher_model.py`
4. Trainer assembly:
   - `verl/trainer/ppo/ray_trainer.py`
5. Rollout server observability:
   - `verl/workers/rollout/sglang_rollout/async_sglang_server.py`
6. Tool/data/reward reproduction support:
   - `verl/tools/*`
   - `verl/utils/dataset/rl_dataset.py`
   - `verl/utils/reward_score/math_verify.py`
   - `verl/experimental/reward_loop/reward_loop.py`
7. Tests and launchers.

### 9.3 Verify Semantics Before Running Full Experiment

Check these invariants in code review before launching a long run.

```text
len(response_mask) == len(teacher_ids_list)
len(response_mask) == len(teacher_logprobs_list)
```

Assistant-generated spans:

```text
response_mask = 1
teacher rows = real teacher top-k rows
```

Tool/user/interact spans:

```text
response_mask = 0
teacher rows = dummy rows
```

Lookahead:

```text
lookahead_step <= current_step + 1
current batch carry-over must complete to terminal
non-terminal lookahead may pause only at exportable handler-return boundary
```

Batch assembly:

```text
base rows + promoted rows are both represented in output
promoted input/output pairs are consumed exactly once
trainer dynamic batch sizing accounts for promoted rows
```

## 10. Verification Plan

이 문서는 테스트를 실행하지 않고 이식 범위를 정리한다. target 코드베이스에 적용한 뒤에는 아래 순서로 검증한다.

### 10.1 Fast CPU Tests

```bash
pytest -q tests/skd/test_async_skd_state.py
pytest -q tests/skd/test_async_skd_events.py
pytest -q tests/skd/test_async_skd_data_source.py
pytest -q tests/skd/test_async_skd_manager.py
pytest -q tests/skd/test_async_skd_manager_lookahead.py
pytest -q tests/trainer/ppo/test_ray_trainer_async_skd_helpers_on_cpu.py
pytest -q tests/workers/config/test_distillation_config_on_cpu.py
pytest -q tests/workers/config/test_rollout_config_on_cpu.py
```

### 10.2 Tool And Dataset Tests

```bash
pytest -q tests/tools/test_computer_13_tool.py
pytest -q tests/utils/test_sandbox_fusion_tools_on_cpu.py
pytest -q tests/utils/dataset/test_nemotron_cascade_rl_math_preprocess_on_cpu.py
```

### 10.3 Agent Loop Compatibility Tests

```bash
pytest -q tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py
pytest -q tests/experimental/agent_loop/test_agent_loop_server_observability.py
pytest -q tests/experimental/agent_loop/test_basic_agent_loop.py
pytest -q tests/experimental/agent_loop/test_multi_modal.py
```

### 10.4 Manual Integration Smoke

```bash
bash tests/skd/manual/launch_test_servers.sh
python tests/skd/manual/skd_integration_manual.py
```

### 10.5 Experiment Smoke

Use the qwen3 async SKD launcher, not the qwen3.5/FP8 launcher.

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_math_fsdp_8gpu_4x4_async_skd_tool_longctx16k.sh
```

For a smoke run, reduce batch size, rollout length, and total training steps in the target environment rather than changing the ported implementation.

## 11. Known Risk From Rollout Dump Analysis

최근 rollout dump와 event log 분석에서 가장 큰 품질 문제는 다음이었다.

- 빈 출력
- 너무 짧은 출력
- 중간에 끊긴 출력

의심 경로:

```text
skd_agent_loop.py
  rejected token at position 0
  -> commit accepted_tokens + [teacher_replacement]
  -> new_tokens length == 1
  -> teacher_replacement is EOS/special
  -> done=eos
  -> batch_decode(..., skip_special_tokens=True)
  -> visible response == ""
```

이 현상은 Async SKD 이식 범위와는 별도 버그 분석 항목이다. 새 verl로 재현 패키지를 옮길 때는 우선 `9030c3c1` 의미를 보존하고, 이후 다음 계측을 추가하는 후속 패치로 분리하는 것을 권장한다.

추가 계측 후보:

- first rejected position
- teacher replacement token id
- teacher replacement token text
- teacher top-k ids/text at rejected position
- tokenizer eos/pad/special token ids
- prompt tail
- whether replacement id is in EOS/special/control set
- visible decoded response length with and without `skip_special_tokens`

## 12. Final Boundary

이식 기준은 다음 한 줄로 고정한다.

```text
Port Async SKD reproduction from APSKD range 293cdb72^..9030c3c1, excluding all qwen3.5/FP8, local code interpreter, and post-9030c3c1 runtime compatibility changes.
```
