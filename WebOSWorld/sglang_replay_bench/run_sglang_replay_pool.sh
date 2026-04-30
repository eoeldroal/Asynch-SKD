#!/usr/bin/env bash
set -euo pipefail

cd /home/sogang_nlpy/verl

export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
export SGLANG_ENABLE_TORCH_INFERENCE_MODE="${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-1}"
export VERL_ASYNC_SKD_TRACE="${VERL_ASYNC_SKD_TRACE:-2}"

MODEL_PATH="${MODEL_PATH:-/home/sogang_nlpy/verl/models/Qwen3.5-9B}"
PYTHON_BIN="${PYTHON_BIN:-/home/sogang_nlpy/miniconda3/envs/skd/bin/python}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-33000}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-34000}"
LOG_DIR="${LOG_DIR:-logs/sglang_replay_bench}"
REUSE_SGLANG_SERVERS="${REUSE_SGLANG_SERVERS:-1}"

mkdir -p "${LOG_DIR}"

port_is_open() {
  local port="$1"
    "${PYTHON_BIN}" - "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
with socket.create_connection(("127.0.0.1", port), timeout=2):
    pass
PY
}

idx=0
urls=()
for gpu in ${GPU_IDS}; do
  port=$((BASE_PORT + idx))
  nccl_port=$((NCCL_PORT_BASE + idx))
  log_file="${LOG_DIR}/sglang_replay_gpu${gpu}_port${port}.log"
  pid_file="${LOG_DIR}/sglang_replay_gpu${gpu}_port${port}.pid"

  if [[ "${REUSE_SGLANG_SERVERS}" == "1" ]] && port_is_open "${port}"; then
    echo "reusing http://127.0.0.1:${port} on gpu=${gpu}"
  else
    echo "launching http://127.0.0.1:${port} on gpu=${gpu}; log=${log_file}"
    nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --nccl-port "${nccl_port}" \
      --tp-size 1 \
      --trust-remote-code \
      --dtype bfloat16 \
      --load-format auto \
      --mem-fraction-static 0.80 \
      --max-running-requests 512 \
      --log-level error \
      --attention-backend triton \
      --mm-attention-backend triton_attn \
      --disable-cuda-graph \
      --enable-memory-saver \
      --skip-server-warmup \
      > "${log_file}" 2>&1 &
    echo "$!" > "${pid_file}"
  fi

  urls+=("http://127.0.0.1:${port}")
  idx=$((idx + 1))
done

printf 'urls='
(IFS=,; echo "${urls[*]}")
echo "logs=${LOG_DIR}"
