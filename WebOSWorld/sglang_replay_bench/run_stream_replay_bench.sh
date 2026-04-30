#!/usr/bin/env bash
set -euo pipefail

cd /home/sogang_nlpy/verl

mkdir -p logs/sglang_replay_bench
PYTHON_BIN="${PYTHON_BIN:-/home/sogang_nlpy/miniconda3/envs/skd/bin/python}"

"${PYTHON_BIN}" WebOSWorld/sglang_replay_bench/stream_replay_bench.py \
  --urls "${URLS:-http://127.0.0.1:33000,http://127.0.0.1:33001,http://127.0.0.1:33002,http://127.0.0.1:33003}" \
  --modes text_multiturn,image_multiturn \
  --concurrency 16 \
  --requests-per-mode "${REQUESTS_PER_MODE:-16}" \
  --turns 3 \
  --stream \
  --max-new-tokens 64 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --repetition-penalty 1.0 \
  --output-jsonl "${OUTPUT_JSONL:-logs/sglang_replay_bench/stream_replay_results.jsonl}" \
  "$@"
