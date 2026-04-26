#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
export SGLANG_ENABLE_TORCH_INFERENCE_MODE="${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-1}"

: "${STUDENT_MODEL_PATH:?set STUDENT_MODEL_PATH}"
: "${TEACHER_MODEL_PATH:?set TEACHER_MODEL_PATH}"

STUDENT_PORT="${STUDENT_PORT:-31000}"
TEACHER_PORT="${TEACHER_PORT:-31001}"
STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-0}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-1}"
STUDENT_DTYPE="${STUDENT_DTYPE:-bfloat16}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
STUDENT_NCCL_PORT="${STUDENT_NCCL_PORT:-32000}"
TEACHER_NCCL_PORT="${TEACHER_NCCL_PORT:-32001}"
REUSE_SGLANG_SERVERS="${REUSE_SGLANG_SERVERS:-1}"

mkdir -p logs

port_is_open() {
  local port="$1"
  python - "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=2):
    pass
PY
}

STUDENT_PID="reused"
if [[ "${REUSE_SGLANG_SERVERS}" == "1" ]] && port_is_open "${STUDENT_PORT}"; then
  echo "reusing existing student server on http://127.0.0.1:${STUDENT_PORT}"
  echo "reused" > logs/live_mixedgen_student.status
else
  CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
  python -m sglang.launch_server \
    --model-path "${STUDENT_MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${STUDENT_PORT}" \
    --nccl-port "${STUDENT_NCCL_PORT}" \
    --tp-size 1 \
    --trust-remote-code \
    --dtype "${STUDENT_DTYPE}" \
    > logs/live_mixedgen_student_sglang.log 2>&1 &
  STUDENT_PID="$!"
  echo "${STUDENT_PID}" > logs/live_mixedgen_student.pid
  echo "launched" > logs/live_mixedgen_student.status
fi

TEACHER_EXTRA_ARGS=()
if [[ -n "${TEACHER_SGLANG_ARGS:-}" ]]; then
  read -r -a TEACHER_EXTRA_ARGS <<< "${TEACHER_SGLANG_ARGS}"
fi

TEACHER_PID="reused"
if [[ "${REUSE_SGLANG_SERVERS}" == "1" ]] && port_is_open "${TEACHER_PORT}"; then
  echo "reusing existing teacher server on http://127.0.0.1:${TEACHER_PORT}"
  echo "reused" > logs/live_mixedgen_teacher.status
else
  CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
  python -m sglang.launch_server \
    --model-path "${TEACHER_MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${TEACHER_PORT}" \
    --nccl-port "${TEACHER_NCCL_PORT}" \
    --tp-size "${TEACHER_TP_SIZE}" \
    --trust-remote-code \
    --dtype "${TEACHER_DTYPE}" \
    "${TEACHER_EXTRA_ARGS[@]}" \
    > logs/live_mixedgen_teacher_sglang.log 2>&1 &
  TEACHER_PID="$!"
  echo "${TEACHER_PID}" > logs/live_mixedgen_teacher.pid
  echo "launched" > logs/live_mixedgen_teacher.status
fi

echo "student_url=http://127.0.0.1:${STUDENT_PORT}"
echo "teacher_url=http://127.0.0.1:${TEACHER_PORT}"
echo "student_pid=${STUDENT_PID}"
echo "teacher_pid=${TEACHER_PID}"
echo "student_nccl_port=${STUDENT_NCCL_PORT}"
echo "teacher_nccl_port=${TEACHER_NCCL_PORT}"
echo "student_log=logs/live_mixedgen_student_sglang.log"
echo "teacher_log=logs/live_mixedgen_teacher_sglang.log"
