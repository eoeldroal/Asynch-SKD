#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

: "${STUDENT_MODEL_PATH:?set STUDENT_MODEL_PATH}"

STUDENT_URL="${STUDENT_URL:-http://127.0.0.1:31000}"
TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:31001}"
PROMPTS="${PROBE_PROMPTS:-async_skd/test_mixedgen/prompts.jsonl}"
TS="$(date +%Y%m%d_%H%M%S)"
PROBE_LOG="logs/live_mixedgen_probe_${TS}.jsonl"
SUMMARY="logs/live_mixedgen_probe_${TS}_summary.json"
RESPONSE_TOKENS_WAS_EXPLICIT=0
if [[ -n "${PROBE_MAX_RESPONSE+x}" || -n "${MAX_RESPONSE_TOKENS+x}" || -n "${MAX_TOKENS+x}" ]]; then
  RESPONSE_TOKENS_WAS_EXPLICIT=1
fi
PROBE_MAX_PROMPT_VALUE="${PROBE_MAX_PROMPT:-${MAX_PROMPT_TOKENS:-1024}}"
PROBE_MAX_RESPONSE_VALUE="${PROBE_MAX_RESPONSE:-${MAX_RESPONSE_TOKENS:-${MAX_TOKENS:-96}}}"
PROBE_CHUNK_SIZE_VALUE="${PROBE_CHUNK_SIZE:-${CHUNK_TOKENS:-32}}"
if [[ -n "${PROBE_MAX_CHUNKS+x}" ]]; then
  PROBE_MAX_CHUNKS_VALUE="${PROBE_MAX_CHUNKS}"
elif [[ -n "${MAX_CHUNKS+x}" ]]; then
  PROBE_MAX_CHUNKS_VALUE="${MAX_CHUNKS}"
elif [[ "${RESPONSE_TOKENS_WAS_EXPLICIT}" == "1" ]]; then
  PROBE_MAX_CHUNKS_VALUE="$(( (PROBE_MAX_RESPONSE_VALUE + PROBE_CHUNK_SIZE_VALUE - 1) / PROBE_CHUNK_SIZE_VALUE ))"
else
  PROBE_MAX_CHUNKS_VALUE=2
fi

mkdir -p logs

STREAM_PID=""
cleanup_probe_stream() {
  if [[ -n "${STREAM_PID}" ]] && kill -0 "${STREAM_PID}" 2>/dev/null; then
    kill "${STREAM_PID}" 2>/dev/null || true
    wait "${STREAM_PID}" 2>/dev/null || true
  fi
  pkill -f "stream_probe.py --probe-log ${PROBE_LOG}" 2>/dev/null || true
}
trap cleanup_probe_stream EXIT

PROBE_ARGS=(
  --student-model "${STUDENT_MODEL_PATH}"
  --prompts "${PROMPTS}"
  --probe-log "${PROBE_LOG}"
  --student-url "${STUDENT_URL}"
  --teacher-url "${TEACHER_URL}"
  --max-prompt "${PROBE_MAX_PROMPT_VALUE}"
  --max-response "${PROBE_MAX_RESPONSE_VALUE}"
  --chunk-size "${PROBE_CHUNK_SIZE_VALUE}"
  --verify-top-k "${VERIFY_TOP_K:-5}"
  --loss-top-k "${LOSS_TOP_K:-32}"
  --max-chunks "${PROBE_MAX_CHUNKS_VALUE}"
  --prefetch-limit "${PROBE_PREFETCH_LIMIT:-2}"
  --prefetch-worker-target "${PROBE_PREFETCH_WORKER_TARGET:-1}"
  --temperature "${PROBE_TEMPERATURE:-0.6}"
  --top-p "${PROBE_TOP_P:-0.95}"
  --top-k "${PROBE_TOP_K:-20}"
  --min-p "${PROBE_MIN_P:-0.0}"
  --presence-penalty "${PROBE_PRESENCE_PENALTY:-0.0}"
  --repetition-penalty "${PROBE_REPETITION_PENALTY:-1.0}"
  --num-direct "${PROBE_NUM_DIRECT:-2}"
  --mode "${PROBE_MODE:-mixed}"
)

if [[ "${PROBE_STUDENT_STREAM:-${PROBE_STREAM:-0}}" == "1" ]]; then
  PROBE_ARGS+=(--student-stream)
fi

if [[ -n "${TEACHER_MODEL_PATH:-}" ]]; then
  PROBE_ARGS+=(--teacher-model "${TEACHER_MODEL_PATH}")
fi

if [[ -n "${PROBE_TEACHER_SYSTEM_PROMPT_PATH:-}" ]]; then
  PROBE_ARGS+=(--teacher-system-prompt-path "${PROBE_TEACHER_SYSTEM_PROMPT_PATH}")
fi

if [[ "${PROBE_STREAM:-0}" == "1" ]]; then
  : > "${PROBE_LOG}"
  STREAM_ARGS=(
    --probe-log "${PROBE_LOG}"
    --tokenizer "${STUDENT_MODEL_PATH}"
    --verify-top-k "${VERIFY_TOP_K:-5}"
    --follow
    --poll-interval "${PROBE_STREAM_POLL_INTERVAL:-0.2}"
    --max-token-width "${PROBE_STREAM_MAX_TOKEN_WIDTH:-24}"
    --show-top-k "${PROBE_STREAM_SHOW_TOP_K:-3}"
    --view "${PROBE_STREAM_VIEW:-sequence}"
    --token-delay "${PROBE_STREAM_TOKEN_DELAY:-0.01}"
    --max-final-chars "${PROBE_STREAM_MAX_FINAL_CHARS:-420}"
  )

  if [[ "${PROBE_STREAM_NO_COLOR:-0}" == "1" ]]; then
    STREAM_ARGS+=(--no-color)
  elif [[ "${PROBE_STREAM_FORCE_COLOR:-1}" == "1" ]]; then
    STREAM_ARGS+=(--force-color)
  fi

  PYTHONPATH="${ROOT}" conda run --no-capture-output -n kd python async_skd/test_mixedgen/stream_probe.py \
    "${STREAM_ARGS[@]}" &
  STREAM_PID=$!
fi

PYTHONPATH="${ROOT}" conda run -n kd python async_skd/test_mixedgen/live_mixedgen_probe.py \
  "${PROBE_ARGS[@]}"

PYTHONPATH="${ROOT}" conda run -n kd python async_skd/test_mixedgen/analyze_probe.py \
  --probe-log "${PROBE_LOG}" \
  --tokenizer "${STUDENT_MODEL_PATH}" \
  --verify-top-k "${VERIFY_TOP_K:-5}" \
  --short-chars "${SHORT_CHARS:-16}" \
  --out "${SUMMARY}"

echo "probe_log=${PROBE_LOG}"
echo "summary=${SUMMARY}"
