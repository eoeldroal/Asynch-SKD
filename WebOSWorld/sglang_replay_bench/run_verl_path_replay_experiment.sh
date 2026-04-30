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
OUTPUT_JSONL="${OUTPUT_JSONL:-${LOG_DIR}/verl_path_replay_results.jsonl}"

mkdir -p "${LOG_DIR}"

pids=()
urls=()

cleanup() {
  set +e
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

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
for gpu in ${GPU_IDS}; do
  port=$((BASE_PORT + idx))
  nccl_port=$((NCCL_PORT_BASE + idx))
  log_file="${LOG_DIR}/verl_path_gpu${gpu}_port${port}.log"
  echo "[verl-path-experiment] launch gpu=${gpu} port=${port} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
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
    --enable-memory-saver \
    --skip-server-warmup \
    > "${log_file}" 2>&1 &
  pids+=("$!")
  urls+=("http://127.0.0.1:${port}")
  wait_for_http "${port}"
  echo "[verl-path-experiment] ready gpu=${gpu} port=${port}"
  idx=$((idx + 1))
done

URLS="$(IFS=,; echo "${urls[*]}")"
echo "[verl-path-experiment] urls=${URLS}"

"${PYTHON_BIN}" WebOSWorld/sglang_replay_bench/verl_path_replay_bench.py \
  --urls "${URLS}" \
  --model-path "${MODEL_PATH}" \
  --modes "${MODES:-native_image_student,pretokenized_image_student}" \
  --concurrency "${CONCURRENCY:-16}" \
  --requests-per-mode "${REQUESTS_PER_MODE:-16}" \
  --turns "${TURNS:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --teacher-suffix-tokens "${TEACHER_SUFFIX_TOKENS:-64}" \
  --top-logprobs-num "${TOP_LOGPROBS_NUM:-32}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  "$@"

echo "[verl-path-experiment] output=${OUTPUT_JSONL}"
