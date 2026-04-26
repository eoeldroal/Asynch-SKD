# Live Mixed-Gen Probe Plan

Goal: add a test-only probe under `async_skd/test_mixedgen` that can attach to
real SGLang student and teacher servers, run small Async SKD trajectories, and
diagnose empty or very short outputs without adding noisy logging to production
SKD code.

## Scope

- Add only test/runbook files under `async_skd/test_mixedgen`.
- Do not modify production trainer, SKD loop, teacher loop, or SGLang runtime
  files for logging.
- Use external observer wrappers around the student and teacher manager
  boundary.
- Keep the first live target simple: Qwen3.5-9B student tokenizer/model path,
  Qwen 27B teacher server, and user-supplied SGLang quantization flags such as
  8bit or FP8.
- Exercise direct SKD, boundary partial/resume, and a lightweight
  carryover-plus-fresh mixed shape.

## Implemented Files

- `async_skd/test_mixedgen/observer.py`
  - Thread-safe JSONL writer.
  - Student manager wrapper logs prompt length/tail, sampling params, generated
    token IDs, stop reason, and extra fields.
  - Teacher manager wrapper logs sequence length/tail, `logprob_start_len`,
    expected suffix length, returned teacher IDs/logprobs shape, and errors.

- `async_skd/test_mixedgen/analyze_probe.py`
  - Parses probe JSONL.
  - Reconstructs student chunk versus teacher top-k verification.
  - Detects first-token rejection, teacher top-1 replacement, EOS/special
    replacement, row-length mismatch, unpaired chunks, and teacher errors.
  - Separates request-fragment counts from final sample output counts so
    boundary/resume probes do not look short only because the exported partial
    is short.

- `async_skd/test_mixedgen/live_mixedgen_probe.py`
  - Attaches to existing SGLang HTTP `/generate` endpoints for student and
    teacher.
  - Builds a small text-only `SkdAgentLoop` directly.
  - Supports `--mode direct`, `--mode boundary`, and `--mode mixed`.
  - `mixed` mode is intentionally sequential: export one carryover partial,
    resume it, then run fresh prompts in the same probe process. It does not
    prove trainer/Ray queue interleaving.

- `async_skd/test_mixedgen/run_live_mixedgen_probe.sh`
  - Runs the live probe and analyzer.
  - Defaults to `PROBE_MODE=mixed`.

- `async_skd/test_mixedgen/launch_two_sglang_servers.sh`
  - Convenience launcher for separate student and teacher SGLang servers.
  - Passes teacher quantization/custom flags through `TEACHER_SGLANG_ARGS`.

- `async_skd/test_mixedgen/prompts.jsonl`
  - Small deterministic math prompt set.

- `async_skd/test_mixedgen/README.md`
  - Runbook, environment variables, and interpretation notes.

## Deliberate Limitations

- This is not a full trainer integration test.
- It does not verify promoted-row append behavior, checkpoint save/load, long
  training data-source plumbing, or real Ray scheduler queue interleaving.
- It does not start SGLang inside Python. Servers are either launched by the
  helper shell script or attached through `STUDENT_URL` and `TEACHER_URL`.
- It keeps the known SKD behavior where teacher top-1 EOS replacement can end
  the trajectory; the probe diagnoses that behavior rather than changing it.

## Verification Commands

Run from `/home/sogang_nlpy/verl`:

```bash
PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python -m py_compile \
  async_skd/test_mixedgen/observer.py \
  async_skd/test_mixedgen/analyze_probe.py \
  async_skd/test_mixedgen/live_mixedgen_probe.py

bash -n async_skd/test_mixedgen/run_live_mixedgen_probe.sh
bash -n async_skd/test_mixedgen/launch_two_sglang_servers.sh

PYTHONPATH=/home/sogang_nlpy/verl conda run -n kd python \
  async_skd/test_mixedgen/live_mixedgen_probe.py --help

git diff --check
```

