# Async SKD Live Mixed-Gen Probe

This directory contains a manual runtime probe for Async SKD mixed generation
against real SGLang student and teacher servers. It is not a CI test and it is
not a full trainer run. The goal is to exercise the mixed-generation path with
small, inspectable inputs while keeping all diagnostic logging in test-only
wrappers and offline analysis.

The intended first live target is:

- Student: Qwen3.5-9B
- Teacher: Qwen3.5-27B BF16

## What This Probe Verifies

The live probe is meant to answer narrow runtime questions:

- SGLang delta rows: teacher logprob rows line up with generated suffix tokens.
- Direct SKD: student chunks are checked against teacher top-k rows and replaced
  when the student token is outside the accepted set.
- Boundary partial/resume: partial mixed-generation state can stop at a boundary
  and resume later.
- Test-only mixed current: one carryover partial and fresh prompts can be run in
  the same probe invocation, without importing the full trainer/Ray scheduler.
- Short or empty output diagnosis: the offline analyzer reconstructs whether a
  first-token teacher replacement was a special token or EOS.

## What It Does Not Verify Yet

This probe does not verify full trainer integration. In particular, it does not
prove promoted-row append behavior, checkpoint save/load behavior, or trainer
data-source integration across long jobs. Those need a separate trainer-level
test once the runtime behavior is understood. The `mixed` mode is intentionally
sequential: it exports one carryover partial, resumes it, and then runs fresh
prompts in the same probe process. It does not prove trainer/Ray queue
interleaving.

## Required Environment

Set these paths before launching servers or running the probe:

```bash
export STUDENT_MODEL_PATH=/path/to/qwen3.5-9b
export TEACHER_MODEL_PATH=/path/to/qwen-27b
```

Set these URLs when the probe should attach to existing SGLang servers:

```bash
export STUDENT_URL=http://127.0.0.1:31000
export TEACHER_URL=http://127.0.0.1:31001
```

GPU placement is controlled by the server launcher:

```bash
export STUDENT_CUDA_VISIBLE_DEVICES=0
export TEACHER_CUDA_VISIBLE_DEVICES=1
```

The launcher defaults to one GPU for the student and one GPU for the teacher.
Override `TEACHER_TP_SIZE` if the teacher tensor parallel size differs from the
number of visible teacher GPUs.

For non-default teacher loading experiments, pass SGLang flags through
`TEACHER_SGLANG_ARGS`. Example names vary by SGLang version, so check the local
`python -m sglang.launch_server --help` output before a large run:

```bash
export TEACHER_SGLANG_ARGS="--quantization fp8"
```

## Recommended First Run

Use a small run before increasing response length or prompt count:

```bash
export PROBE_NUM_DIRECT=2
export PROBE_MAX_RESPONSE=96
export PROBE_CHUNK_SIZE=32
export PROBE_MAX_CHUNKS=2
export PROBE_MODE=mixed
export VERIFY_TOP_K=5
```

The prompt file has eight deterministic math prompts. Start with the defaults,
confirm that rows and decoded text look sane, then increase prompt count,
response length, or chunk count.

## One-Command Dataset Probe

For the default Qwen3.5-9B student plus Qwen3.5-27B BF16 teacher setup, run:

```bash
cd /home/sogang_nlpy/verl
async_skd/test_mixedgen/run_qwen35_27b_bf16_dataset_probe.sh
```

This wrapper sets the usual defaults, converts a few prompts from
`/home/sogang_nlpy/APSKD/data/aime-2024.parquet` into probe JSONL, launches the
student and teacher SGLang servers, waits for their ports, runs the live probe
with terminal streaming enabled, and stops the servers on exit.

The default model paths are:

- Student: `/home/sogang_nlpy/model/Qwen3.5-9B`
- Teacher: `/home/sogang_nlpy/APSKD/checkpoints/Qwen3.5-27B`
- Teacher-only prompt: `/home/sogang_nlpy/verl/async_skd/test_mixedgen/teacher_system_prompt_math_planning.txt`

The teacher path is the BF16 checkpoint, so the wrapper leaves
`TEACHER_SGLANG_ARGS` empty by default. The probe also enables a teacher-only
planning prompt by default so teacher verification follows a more explicit
stepwise math style. To check the resolved configuration and generated prompts
without launching servers:

```bash
PROBE_DRY_RUN=1 async_skd/test_mixedgen/run_qwen35_27b_bf16_dataset_probe.sh
```

Useful optional overrides:

```bash
PROBE_DATASET_PATH=/home/sogang_nlpy/APSKD/data/math500/test.parquet \
PROBE_NUM_DIRECT=4 \
TEACHER_CUDA_VISIBLE_DEVICES=1 \
PROBE_TEACHER_SYSTEM_PROMPT_PATH=/path/to/custom_teacher_prompt.txt \
async_skd/test_mixedgen/run_qwen35_27b_bf16_dataset_probe.sh
```

## Launch SGLang Servers

From the repo root:

```bash
cd /home/sogang_nlpy/verl
async_skd/test_mixedgen/launch_two_sglang_servers.sh
```

The launcher writes:

- `logs/live_mixedgen_student_sglang.log`
- `logs/live_mixedgen_teacher_sglang.log`
- `logs/live_mixedgen_student.pid`
- `logs/live_mixedgen_teacher.pid`

It prints the student and teacher URLs. Export those URLs if you changed ports:

```bash
export STUDENT_URL=http://127.0.0.1:${STUDENT_PORT:-31000}
export TEACHER_URL=http://127.0.0.1:${TEACHER_PORT:-31001}
```

Stop the servers with the recorded pids when finished:

```bash
kill "$(cat logs/live_mixedgen_student.pid)" "$(cat logs/live_mixedgen_teacher.pid)"
```

## Run Probe and Analyzer

After both SGLang servers are healthy:

```bash
cd /home/sogang_nlpy/verl
async_skd/test_mixedgen/run_live_mixedgen_probe.sh
```

The wrapper creates timestamped files:

- `logs/live_mixedgen_probe_<timestamp>.jsonl`
- `logs/live_mixedgen_probe_<timestamp>_summary.json`

`run_live_mixedgen_probe.sh` requires `STUDENT_MODEL_PATH`.
`TEACHER_MODEL_PATH` is optional in the wrapper because some probe versions use
it only for metadata when attaching to an already-running teacher server.

Useful knobs:

```bash
export STUDENT_URL=http://127.0.0.1:31000
export TEACHER_URL=http://127.0.0.1:31001
export PROBE_PROMPTS=async_skd/test_mixedgen/prompts.jsonl
export PROBE_MAX_PROMPT=1024
export PROBE_MAX_RESPONSE=96
export PROBE_CHUNK_SIZE=32
export PROBE_MAX_CHUNKS=2
export PROBE_NUM_DIRECT=2
export PROBE_MODE=mixed
export PROBE_PREFETCH_LIMIT=2
export PROBE_PREFETCH_WORKER_TARGET=1
export PROBE_TEMPERATURE=0.6
export PROBE_TOP_P=0.95
export PROBE_TOP_K=20
export VERIFY_TOP_K=5
export LOSS_TOP_K=32
export SHORT_CHARS=16
```

## Live Streaming

Enable the optional terminal streamer when you want to watch the JSONL probe log
as the run progresses:

```bash
export PROBE_STREAM=1
async_skd/test_mixedgen/run_live_mixedgen_probe.sh
```

The wrapper starts `stream_probe.py` before `live_mixedgen_probe.py`, follows the
same `PROBE_LOG`, and stops the streamer when the wrapper exits. Streaming is
chunk-level, not token-by-token: each display update corresponds to a logged
probe chunk or final result. The wrapper uses `conda run --no-capture-output`
for the streamer so lines appear while the probe is still running.

Color meanings:

- Cyan: student text
- Magenta: teacher text
- Green: accepted or final text
- Yellow: rejected or short text
- Red: EOS, special token, error, empty output, or mismatch
- Dim: metadata such as request id, chunk index, and top-k rows

Example shape, with ASCII markers shown here instead of relying on color:

```text
PROBE_START mode=mixed prompts=8 student=/path/to/qwen3.5-9b teacher=/path/to/qwen-27b
STUDENT req=8f1a0c2d44 chunk_len=32 stop_reason=None text='We need solve this step by step...'
TEACHER req=8f1a0c2d44 rows=32 width=32
ACCEPT_ALL req=8f1a0c2d44 accept_count=32 first_reject_pos=None
REJECT req=21c4d77b91 accept_count=3 first_reject_pos=3 student=1000:' wrong' teacher_top1=1001:' right'
REPLACE_EOS req=994feab010 accept_count=0 first_reject_pos=0 student=1000:' wrong' teacher_top1=248046:'<|im_end|>'
EMPTY sample_id=math_0001 mode=mixed_carryover response_len=1 text=''
```

Streamer knobs:

```bash
export PROBE_STREAM_POLL_INTERVAL=0.2
export PROBE_STREAM_MAX_TOKEN_WIDTH=24
export PROBE_STREAM_SHOW_TOP_K=3
export PROBE_STREAM_NO_COLOR=1
```

Replay an existing log without following for new writes:

```bash
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python \
  async_skd/test_mixedgen/stream_probe.py \
  --probe-log logs/live_mixedgen_probe_<timestamp>.jsonl \
  --tokenizer "${STUDENT_MODEL_PATH}" \
  --verify-top-k "${VERIFY_TOP_K:-5}"
```

## Interpreting First-Token Special or EOS Replacements

The analyzer reconstructs each request from the JSONL boundary logs. For every
student chunk, it checks whether the student token appears in the teacher
top-k row. The first position outside teacher top-k is the rejection position,
and the teacher row's first token is the replacement.

When `final_result` records are present, the top-level `empty_outputs` and
`short_outputs` counts refer to final sample outputs. Request-level fragment
counts are still reported separately as `request_empty_outputs` and
`request_short_outputs`, which is useful for boundary/resume runs where the
first partial fragment is short by construction.

If a sample is empty or very short and the summary reports:

- `first_token_reject: true`
- `replacement_is_special: true`
- `replacement_is_eos: true` or `replacement_decoded_with_special` matching the
  tokenizer EOS token

then the short output was likely caused by the teacher replacing the first
student token with a special/EOS token. That points to prompt formatting,
tokenizer mismatch, chat-template mismatch, or teacher logprob row alignment
before it points to trainer checkpoint logic.

If the replacement is not special, inspect the decoded replacement text and the
teacher row width. A normal text replacement followed by truncation usually
means the probe stopped at a configured boundary such as `PROBE_MAX_CHUNKS` or
`PROBE_MAX_RESPONSE`, not an EOS diagnosis.
