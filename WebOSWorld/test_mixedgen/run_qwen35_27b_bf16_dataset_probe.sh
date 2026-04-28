#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"

export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/home/sogang_nlpy/model/Qwen3.5-9B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/home/sogang_nlpy/APSKD/checkpoints/Qwen3.5-27B}"
export PROBE_DATASET_PATH="${PROBE_DATASET_PATH:-/home/sogang_nlpy/APSKD/data/aime-2024.parquet}"
export PROBE_DATASET_PROMPT_KEY="${PROBE_DATASET_PROMPT_KEY:-prompt}"
export PROBE_DATASET_OFFSET="${PROBE_DATASET_OFFSET:-0}"
export PROBE_NUM_DIRECT="${PROBE_NUM_DIRECT:-3}"
export PROBE_DATASET_LIMIT="${PROBE_DATASET_LIMIT:-${PROBE_NUM_DIRECT}}"

export STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-0}"
export TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-1}"
export TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
export STUDENT_DTYPE="${STUDENT_DTYPE:-bfloat16}"
export TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
export STUDENT_PORT="${STUDENT_PORT:-31000}"
export TEACHER_PORT="${TEACHER_PORT:-31001}"
export STUDENT_NCCL_PORT="${STUDENT_NCCL_PORT:-32000}"
export TEACHER_NCCL_PORT="${TEACHER_NCCL_PORT:-32001}"
export STUDENT_URL="${STUDENT_URL:-http://127.0.0.1:${STUDENT_PORT}}"
export TEACHER_URL="${TEACHER_URL:-http://127.0.0.1:${TEACHER_PORT}}"
export REUSE_SGLANG_SERVERS="${REUSE_SGLANG_SERVERS:-1}"
export STOP_SERVERS_ON_EXIT="${STOP_SERVERS_ON_EXIT:-0}"

export PROBE_MODE="${PROBE_MODE:-mixed}"
export PROBE_STREAM="${PROBE_STREAM:-1}"
RESPONSE_TOKENS_WAS_EXPLICIT=0
if [[ -n "${PROBE_MAX_RESPONSE+x}" || -n "${MAX_RESPONSE_TOKENS+x}" || -n "${MAX_TOKENS+x}" ]]; then
  RESPONSE_TOKENS_WAS_EXPLICIT=1
fi
export PROBE_MAX_PROMPT="${PROBE_MAX_PROMPT:-${MAX_PROMPT_TOKENS:-1024}}"
export PROBE_MAX_RESPONSE="${PROBE_MAX_RESPONSE:-${MAX_RESPONSE_TOKENS:-${MAX_TOKENS:-96}}}"
export PROBE_CHUNK_SIZE="${PROBE_CHUNK_SIZE:-${CHUNK_TOKENS:-32}}"
if [[ -n "${PROBE_MAX_CHUNKS+x}" ]]; then
  export PROBE_MAX_CHUNKS
elif [[ -n "${MAX_CHUNKS+x}" ]]; then
  export PROBE_MAX_CHUNKS="${MAX_CHUNKS}"
elif [[ "${RESPONSE_TOKENS_WAS_EXPLICIT}" == "1" ]]; then
  export PROBE_MAX_CHUNKS="$(( (PROBE_MAX_RESPONSE + PROBE_CHUNK_SIZE - 1) / PROBE_CHUNK_SIZE ))"
else
  export PROBE_MAX_CHUNKS=2
fi
export VERIFY_TOP_K="${VERIFY_TOP_K:-5}"
export LOSS_TOP_K="${LOSS_TOP_K:-32}"
export SHORT_CHARS="${SHORT_CHARS:-16}"
export PROBE_TEMPERATURE="${PROBE_TEMPERATURE:-0.6}"
export PROBE_TOP_P="${PROBE_TOP_P:-0.95}"
export PROBE_TOP_K="${PROBE_TOP_K:-20}"
export PROBE_MIN_P="${PROBE_MIN_P:-0.0}"
export PROBE_PRESENCE_PENALTY="${PROBE_PRESENCE_PENALTY:-0.0}"
export PROBE_REPETITION_PENALTY="${PROBE_REPETITION_PENALTY:-1.0}"

export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
export SGLANG_ENABLE_TORCH_INFERENCE_MODE="${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-1}"
export PROBE_TEACHER_SYSTEM_PROMPT_PATH="${PROBE_TEACHER_SYSTEM_PROMPT_PATH:-${ROOT}/WebOSWorld/test_mixedgen/teacher_system_prompt_math_planning.txt}"

# Default to the BF16 teacher checkpoint. Keep extra SGLang args empty unless
# the caller explicitly wants to test a different serving mode.
export TEACHER_SGLANG_ARGS="${TEACHER_SGLANG_ARGS:-}"

PROMPT_DIR="${PROBE_PROMPT_DIR:-logs/live_mixedgen_prompts}"
mkdir -p "${PROMPT_DIR}" logs
export PROBE_PROMPTS="${PROBE_PROMPTS:-${PROMPT_DIR}/dataset_prompts_${RUN_TS}.jsonl}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "${label} does not exist: ${path}" >&2
    exit 1
  fi
}

wait_for_port() {
  local url="$1"
  local label="$2"
  local pid_file="${3:-}"
  local timeout="${SERVER_START_TIMEOUT:-900}"
  local deadline=$((SECONDS + timeout))
  echo "waiting for ${label} at ${url} (timeout=${timeout}s)"
  while (( SECONDS < deadline )); do
    if URL="${url}" conda run -n kd python -c "import os, socket, urllib.parse; u=urllib.parse.urlparse(os.environ['URL']); s=socket.create_connection((u.hostname, u.port or 80), timeout=2); s.close()" >/dev/null 2>&1; then
      echo "${label} port is open: ${url}"
      local launch_status=""
      local status_file="${pid_file%.pid}.status"
      if [[ -n "${pid_file}" && -f "${status_file}" ]]; then
        launch_status="$(cat "${status_file}")"
      fi
      if [[ "${launch_status}" != "reused" ]]; then
        sleep "${SERVER_READY_EXTRA_SLEEP:-5}"
      fi
      return 0
    fi
    if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "${label} server process exited before opening ${url}" >&2
        tail -120 "logs/live_mixedgen_${label}_sglang.log" >&2 || true
        return 1
      fi
    fi
    sleep 5
  done
  echo "timed out waiting for ${label}: ${url}" >&2
  return 1
}

cleanup_servers() {
  if [[ "${STOP_SERVERS_ON_EXIT:-1}" != "1" ]]; then
    return
  fi
  for pid_file in logs/live_mixedgen_student.pid logs/live_mixedgen_teacher.pid; do
    local status_file="${pid_file%.pid}.status"
    if [[ -f "${status_file}" ]] && [[ "$(cat "${status_file}")" == "reused" ]]; then
      continue
    fi
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    fi
  done
}
trap cleanup_servers EXIT

require_path "${STUDENT_MODEL_PATH}" "student model"
require_path "${TEACHER_MODEL_PATH}" "teacher model"
require_path "${PROBE_DATASET_PATH}" "probe dataset"
require_path "${PROBE_TEACHER_SYSTEM_PROMPT_PATH}" "teacher system prompt"

PYTHONPATH="${ROOT}" conda run -n kd python WebOSWorld/test_mixedgen/dataset_to_probe_prompts.py \
  --dataset "${PROBE_DATASET_PATH}" \
  --out "${PROBE_PROMPTS}" \
  --prompt-key "${PROBE_DATASET_PROMPT_KEY}" \
  --limit "${PROBE_DATASET_LIMIT}" \
  --offset "${PROBE_DATASET_OFFSET}"

cat <<EOF
resolved live mixed-gen probe config:
  student_model=${STUDENT_MODEL_PATH}
  teacher_model=${TEACHER_MODEL_PATH}
  dataset=${PROBE_DATASET_PATH}
  prompts=${PROBE_PROMPTS}
  student_gpu=${STUDENT_CUDA_VISIBLE_DEVICES}
  teacher_gpus=${TEACHER_CUDA_VISIBLE_DEVICES}
  teacher_tp=${TEACHER_TP_SIZE}
  student_nccl_port=${STUDENT_NCCL_PORT}
  teacher_nccl_port=${TEACHER_NCCL_PORT}
  student_url=${STUDENT_URL}
  teacher_url=${TEACHER_URL}
  teacher_system_prompt=${PROBE_TEACHER_SYSTEM_PROMPT_PATH}
  max_prompt=${PROBE_MAX_PROMPT}
  max_response=${PROBE_MAX_RESPONSE}
  chunk_size=${PROBE_CHUNK_SIZE}
  max_chunks=${PROBE_MAX_CHUNKS}
  temperature=${PROBE_TEMPERATURE}
  top_p=${PROBE_TOP_P}
  top_k=${PROBE_TOP_K}
  min_p=${PROBE_MIN_P}
  presence_penalty=${PROBE_PRESENCE_PENALTY}
  repetition_penalty=${PROBE_REPETITION_PENALTY}
  reuse_sglang_servers=${REUSE_SGLANG_SERVERS}
  stop_servers_on_exit=${STOP_SERVERS_ON_EXIT}
  stream=${PROBE_STREAM}
  mode=${PROBE_MODE}
EOF

if [[ "${PROBE_DRY_RUN:-0}" == "1" ]]; then
  echo "PROBE_DRY_RUN=1, stopping after prompt generation."
  exit 0
fi

conda run --no-capture-output -n kd bash WebOSWorld/test_mixedgen/launch_two_sglang_servers.sh

if ! wait_for_port "${STUDENT_URL}" "student" "logs/live_mixedgen_student.pid"; then
  tail -80 logs/live_mixedgen_student_sglang.log >&2 || true
  exit 1
fi
if ! wait_for_port "${TEACHER_URL}" "teacher" "logs/live_mixedgen_teacher.pid"; then
  tail -80 logs/live_mixedgen_teacher_sglang.log >&2 || true
  exit 1
fi

WebOSWorld/test_mixedgen/run_live_mixedgen_probe.sh
