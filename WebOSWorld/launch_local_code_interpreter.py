#!/usr/bin/env python3
"""Local /run_code server compatible with verl SandboxFusionTool.

This is a lightweight smoke-test server for trusted local experiments. It is
not a hardened sandbox. For untrusted code, use a real SandboxFusion deployment
or another container/Kata based isolation boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import resource
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class RunCodeRequest(BaseModel):
    code: str
    stdin: str | None = None
    compile_timeout: int = Field(default=30, ge=1)
    run_timeout: int = Field(default=30, ge=1)
    memory_limit_MB: int = Field(default=1024, ge=128)
    language: str = "python"
    files: dict[str, str] = Field(default_factory=dict)
    fetch_files: list[str] = Field(default_factory=list)


def _limit_child(memory_limit_mb: int) -> None:
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def _run_result(status: str, stdout: str, stderr: str, return_code: int | None, execution_time: float | None) -> dict:
    return {
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "execution_time": execution_time,
    }


async def _execute_python(req: RunCodeRequest, work_dir: Path) -> dict[str, Any]:
    fd, path = tempfile.mkstemp(suffix=".py", prefix="apSkd_code_", dir=work_dir, text=True)
    os.close(fd)
    code_path = Path(path)
    code_path.write_text(req.code, encoding="utf-8")

    started = asyncio.get_running_loop().time()
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(code_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            preexec_fn=lambda: _limit_child(req.memory_limit_MB),
        )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate((req.stdin or "").encode("utf-8")),
                timeout=req.run_timeout,
            )
            elapsed = asyncio.get_running_loop().time() - started
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            top_status = "Success" if proc.returncode == 0 else "Failed"
            return {
                "status": top_status,
                "compile_result": None,
                "run_result": _run_result("Finished", stdout, stderr, proc.returncode, elapsed),
            }
        except asyncio.TimeoutError:
            proc.kill()
            stdout_b, stderr_b = await proc.communicate()
            elapsed = asyncio.get_running_loop().time() - started
            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            return {
                "status": "Failed",
                "compile_result": None,
                "run_result": _run_result("TimeLimitExceeded", stdout, stderr, None, elapsed),
            }
    finally:
        try:
            code_path.unlink()
        except FileNotFoundError:
            pass


def build_app(max_concurrency: int, work_dir: Path) -> FastAPI:
    app = FastAPI(title="APSKD Local Code Interpreter")
    semaphore = asyncio.Semaphore(max_concurrency)
    work_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run_code")
    async def run_code(req: RunCodeRequest) -> dict[str, Any]:
        if req.language != "python":
            return {
                "status": "Failed",
                "compile_result": {
                    "status": "Error",
                    "stderr": f"Unsupported language: {req.language}",
                    "return_code": 1,
                },
                "run_result": None,
            }
        if req.files or req.fetch_files:
            raise HTTPException(status_code=400, detail="files/fetch_files are not supported by this local server")

        async with semaphore:
            return await _execute_python(req, work_dir)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a local /run_code endpoint for APSKD tool smoke tests.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--work-dir", default="/tmp/apskd_code_interpreter")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(max_concurrency=args.max_concurrency, work_dir=Path(args.work_dir))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
