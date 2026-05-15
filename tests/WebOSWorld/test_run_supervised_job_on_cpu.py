from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path


SUPERVISOR = Path("/home/sogang_nlpy/verl/WebOSWorld/run_supervised_job.sh")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _setup_fake_runtime(tmp_path: Path) -> dict[str, str]:
    verl_root = tmp_path / "verl"
    surfgym_root = tmp_path / "surfgym"
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "state"
    trace_file = tmp_path / "trace.log"
    conda_sh = tmp_path / "conda.sh"

    verl_root.mkdir()
    (surfgym_root / "scripts").mkdir(parents=True)
    bin_dir.mkdir()
    state_dir.mkdir()

    _write_executable(
        conda_sh,
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            conda() {{
              if [[ "$1" == "activate" ]]; then
                export ACTIVE_ENV="$2"
                export PATH="{bin_dir}:$PATH"
                return 0
              fi
              echo "unsupported conda command: $*" >&2
              return 1
            }}
            """
        ),
    )

    _write_executable(
        bin_dir / "ray",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf 'ray:%s:%s:%s\n' "${ACTIVE_ENV:-missing}" "$PWD" "$*" >> "$TRACE_FILE"
            exit "${RAY_EXIT_CODE:-0}"
            """
        ),
    )

    _write_executable(
        surfgym_root / "scripts" / "stop_all.bash",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf 'stop_tools:%s:%s\n' "${ACTIVE_ENV:-missing}" "$PWD" >> "$TRACE_FILE"
            exit "${STOP_TOOLS_EXIT_CODE:-0}"
            """
        ),
    )

    _write_executable(
        surfgym_root / "scripts" / "launch_all.bash",
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            printf 'launch_tools:%s:%s\n' "${ACTIVE_ENV:-missing}" "$PWD" >> "$TRACE_FILE"
            exit "${LAUNCH_TOOLS_EXIT_CODE:-0}"
            """
        ),
    )

    target_script = verl_root / "target.sh"
    _write_executable(
        target_script,
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            attempt_file="{state_dir / 'target_attempts'}"
            attempt=0
            if [[ -f "$attempt_file" ]]; then
              attempt="$(cat "$attempt_file")"
            fi
            attempt=$((attempt + 1))
            printf '%s' "$attempt" > "$attempt_file"
            printf 'target:%s:%s:%s\\n' "${{ACTIVE_ENV:-missing}}" "$PWD" "$attempt" >> "$TRACE_FILE"

            mode="${{TARGET_MODE:-fail_once}}"
            if [[ "$mode" == "fail_once" && "$attempt" -eq 1 ]]; then
              exit 17
            fi
            if [[ "$mode" == "always_fail" ]]; then
              exit "${{TARGET_EXIT_CODE:-23}}"
            fi
            exit 0
            """
        ),
    )

    return {
        "VERL_ROOT": str(verl_root),
        "SURFGYM_ROOT": str(surfgym_root),
        "CONDA_SH": str(conda_sh),
        "TRACE_FILE": str(trace_file),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RL_ENV": "skd-cudnn",
        "TOOL_ENV": "surfgym",
        "TARGET_SCRIPT": str(target_script),
    }


def test_run_supervised_job_retries_in_requested_order_with_requested_envs(tmp_path: Path):
    runtime = _setup_fake_runtime(tmp_path)
    env = {
        **os.environ,
        **runtime,
        "MAX_RETRIES": "1",
        "RESET_SLEEP_SECONDS": "0",
        "TARGET_MODE": "fail_once",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), runtime["TARGET_SCRIPT"]],
        cwd="/home/sogang_nlpy/verl",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    trace = Path(runtime["TRACE_FILE"]).read_text(encoding="utf-8").strip().splitlines()
    assert trace == [
        f"target:skd-cudnn:{runtime['VERL_ROOT']}:1",
        f"ray:skd-cudnn:{runtime['VERL_ROOT']}:stop --force",
        f"stop_tools:surfgym:{runtime['SURFGYM_ROOT']}",
        f"launch_tools:surfgym:{runtime['SURFGYM_ROOT']}",
        f"target:skd-cudnn:{runtime['VERL_ROOT']}:2",
    ]
    assert "Re-activating skd-cudnn before retry" in result.stdout


def test_run_supervised_job_preserves_target_exit_code_when_retries_exhausted(tmp_path: Path):
    runtime = _setup_fake_runtime(tmp_path)
    env = {
        **os.environ,
        **runtime,
        "MAX_RETRIES": "0",
        "RESET_SLEEP_SECONDS": "0",
        "TARGET_MODE": "always_fail",
        "TARGET_EXIT_CODE": "23",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), runtime["TARGET_SCRIPT"]],
        cwd="/home/sogang_nlpy/verl",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 23
    assert "exit code 23" in result.stdout


def test_run_supervised_job_fails_closed_when_surfgym_launch_fails(tmp_path: Path):
    runtime = _setup_fake_runtime(tmp_path)
    env = {
        **os.environ,
        **runtime,
        "MAX_RETRIES": "1",
        "RESET_SLEEP_SECONDS": "0",
        "TARGET_MODE": "fail_once",
        "LAUNCH_TOOLS_EXIT_CODE": "19",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), runtime["TARGET_SCRIPT"]],
        cwd="/home/sogang_nlpy/verl",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 19
    trace = Path(runtime["TRACE_FILE"]).read_text(encoding="utf-8").strip().splitlines()
    assert trace == [
        f"target:skd-cudnn:{runtime['VERL_ROOT']}:1",
        f"ray:skd-cudnn:{runtime['VERL_ROOT']}:stop --force",
        f"stop_tools:surfgym:{runtime['SURFGYM_ROOT']}",
        f"launch_tools:surfgym:{runtime['SURFGYM_ROOT']}",
    ]
