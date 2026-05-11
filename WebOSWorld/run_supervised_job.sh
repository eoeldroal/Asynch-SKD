#!/usr/bin/env bash
set -euo pipefail

VERL_ROOT=/home/sogang_nlpy/verl
SURFGYM_ROOT=/home/sogang_nlpy/goonco/surfgym
RL_ENV=skd-cudnn
TOOL_ENV=surfgym
MAX_RETRIES="${MAX_RETRIES:-3}"
RESET_SLEEP_SECONDS="${RESET_SLEEP_SECONDS:-30}"
CONDA_SH=/home/sogang_nlpy/miniconda3/etc/profile.d/conda.sh

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash WebOSWorld/run_supervised_job.sh <target_script> [target_args...]

Notes:
  - Launch this supervisor itself with nohup if you want it to survive session close.
  - The target script is always executed in:
      env: skd-cudnn
      cwd: /home/sogang_nlpy/verl
  - Recovery always runs in this order after a failed attempt:
      1) ray stop --force
      2) surfgym stop_all.bash
      3) sleep 30s
      4) surfgym launch_all.bash
      5) retry target script
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TARGET_SCRIPT="$1"
shift

if [[ "$TARGET_SCRIPT" != /* ]]; then
  TARGET_SCRIPT="${VERL_ROOT}/${TARGET_SCRIPT}"
fi

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  log "Target script not found: ${TARGET_SCRIPT}"
  exit 2
fi

run_in_conda() {
  local env_name="$1"
  local cwd="$2"
  shift 2
  bash -c 'source "$1" && conda activate "$2" && cd "$3" && shift 3 && exec "$@"' _ "$CONDA_SH" "$env_name" "$cwd" "$@"
}

run_target() {
  run_in_conda "$RL_ENV" "$VERL_ROOT" bash "$TARGET_SCRIPT" "$@"
}

reset_ray() {
  log "Stopping Ray"
  run_in_conda "$RL_ENV" "$VERL_ROOT" ray stop --force || true
}

stop_tools() {
  log "Stopping surfgym stack"
  run_in_conda "$TOOL_ENV" "$SURFGYM_ROOT" bash scripts/stop_all.bash || true
}

launch_tools() {
  log "Launching surfgym stack"
  run_in_conda "$TOOL_ENV" "$SURFGYM_ROOT" bash scripts/launch_all.bash || true
}

attempt=1
max_attempts=$((MAX_RETRIES + 1))

while (( attempt <= max_attempts )); do
  log "Starting attempt ${attempt}/${max_attempts}: ${TARGET_SCRIPT}"
  if run_target "$@"; then
    log "Target finished successfully"
    exit 0
  fi

  status=$?
  log "Target failed on attempt ${attempt}/${max_attempts} with exit code ${status}"

  if (( attempt == max_attempts )); then
    log "Retry limit reached"
    exit "$status"
  fi

  reset_ray
  stop_tools
  log "Sleeping ${RESET_SLEEP_SECONDS}s before relaunch"
  sleep "$RESET_SLEEP_SECONDS"
  launch_tools

  attempt=$((attempt + 1))
done

exit 1
