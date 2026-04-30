#!/usr/bin/env bash
set -euo pipefail

cd /home/sogang_nlpy/verl

export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"
export SGLANG_ENABLE_TORCH_INFERENCE_MODE="${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-1}"
export VERL_ASYNC_SKD_TRACE="${VERL_ASYNC_SKD_TRACE:-2}"

PYTHON_BIN="${PYTHON_BIN:-/home/sogang_nlpy/miniconda3/envs/skd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/sogang_nlpy/verl/models/Qwen3.5-9B}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-37000}"
NCCL_PORT_BASE="${NCCL_PORT_BASE:-38000}"
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

wait_for_http() {
  local port="$1"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if "${PYTHON_BIN}" - "${port}" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

port = int(sys.argv[1])
payload = {
    "text": "ping",
    "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    if resp.status != 200:
        raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 5
  done
  echo "server on port ${port} did not become ready" >&2
  return 1
}

idx=0
urls=()
for gpu in ${GPU_IDS}; do
  port=$((BASE_PORT + idx))
  nccl_port=$((NCCL_PORT_BASE + idx))
  log_file="${LOG_DIR}/verl_path_fixed_gpu${gpu}_port${port}.log"
  pid_file="${LOG_DIR}/verl_path_fixed_gpu${gpu}_port${port}.pid"

  if [[ "${REUSE_SGLANG_SERVERS}" == "1" ]] && port_is_open "${port}"; then
    echo "[verl-path-pool] reusing http://127.0.0.1:${port} gpu=${gpu}"
  else
    echo "[verl-path-pool] launching http://127.0.0.1:${port} gpu=${gpu} log=${log_file}"
    setsid bash -c '
      exec env CUDA_VISIBLE_DEVICES="$1" "$2" -m sglang.launch_server \
        --model-path "$3" \
        --host 127.0.0.1 \
        --port "$4" \
        --nccl-port "$5" \
        --tp-size 1 \
        --trust-remote-code \
        --dtype bfloat16 \
        --load-format auto \
        --mem-fraction-static 0.80 \
        --max-running-requests 512 \
        --log-level error \
        --attention-backend triton \
        --mm-attention-backend triton_attn \
        --enable-memory-saver \
        --skip-server-warmup
    ' _ "${gpu}" "${PYTHON_BIN}" "${MODEL_PATH}" "${port}" "${nccl_port}" > "${log_file}" 2>&1 &
    echo "$!" > "${pid_file}"
    wait_for_http "${port}"
    echo "[verl-path-pool] ready http://127.0.0.1:${port} gpu=${gpu}"
  fi

  urls+=("http://127.0.0.1:${port}")
  idx=$((idx + 1))
done

printf '[verl-path-pool] urls='
(IFS=,; echo "${urls[*]}")
echo "[verl-path-pool] logs=${LOG_DIR}"
