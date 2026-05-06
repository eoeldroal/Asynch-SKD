#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    from PIL import Image
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.multimodal_processor import (
        get_mm_processor,
        import_processors,
    )
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils.hf_transformers_utils import (
        get_processor,
        get_tokenizer_from_processor,
    )
except ImportError as exc:  # pragma: no cover - runtime guard for direct script use
    raise SystemExit(
        "Missing runtime dependency for this probe. "
        "Run it in a Python environment that has `sglang`, `transformers`, and `Pillow`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT_RELATIVE_PATH = Path("models")
IMAGE_DIR_RELATIVE_PATH = Path(
    "logs/rollout_data/qwen35_webgym_fully_async_tool_veomni/webgym_tool_trace/images"
)
DEFAULT_MODEL_NAMES = (
    "Qwen3-VL-2B-Instruct",
    "Qwen3.5-9B",
    "Qwen3.5-27B",
)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _iter_git_worktree_roots() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "worktree", "list", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return [REPO_ROOT]

    if result.returncode != 0:
        return [REPO_ROOT]

    roots: list[Path] = []
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.removeprefix("worktree ").strip()))
    return roots or [REPO_ROOT]


def _resolve_existing_worktree_path(relative_path: Path) -> Path:
    primary_candidate = REPO_ROOT / relative_path
    if primary_candidate.exists():
        return primary_candidate

    for worktree_root in _iter_git_worktree_roots():
        candidate = worktree_root / relative_path
        if candidate.exists():
            return candidate

    return primary_candidate


def _default_models_root() -> Path:
    return _resolve_existing_worktree_path(MODELS_ROOT_RELATIVE_PATH)


def _default_image_dir() -> Path:
    return _resolve_existing_worktree_path(IMAGE_DIR_RELATIVE_PATH)


@dataclass
class ProbeRow:
    model: str
    status: str
    compact_len: int | None = None
    expanded_len: int | None = None
    expanded_errors: bool | None = None
    compact_matches_local_expanded: bool | None = None
    detail: str = "-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe SGLang's multimodal boundary for local Qwen models by comparing "
            "processor-expanded prompt ids against compact tokenizer-only prompt ids."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_default_models_root(),
        help=f"Model directory root. Default: {_default_models_root()}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_default_image_dir(),
        help=f"WebOSGym screenshot directory. Default: {_default_image_dir()}",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Explicit screenshot path. If omitted, the first image under --image-dir is used.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Model directory name under --models-root. Repeat to override the default set.",
    )
    return parser.parse_args()


def _pick_image_path(image_dir: Path, image_path: Path | None) -> Path:
    if image_path is not None:
        resolved = image_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Probe image not found: {resolved}")
        return resolved

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    candidates = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(f"No probe images found under: {image_dir}")
    return candidates[0].resolve()


def _build_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise UI observer."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the current WebOSGym screenshot in one sentence.",
                },
                {"type": "image"},
            ],
        },
    ]


def _make_server_args(model_path: Path) -> ServerArgs:
    return ServerArgs(
        model_path=str(model_path),
        tokenizer_path=str(model_path),
        trust_remote_code=True,
        disable_fast_image_processor=True,
        device="cpu",
    )


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def _exception_summary(exc: Exception, *, max_len: int = 80) -> str:
    first_line = str(exc).splitlines()[0].strip() if str(exc).strip() else type(exc).__name__
    summary = f"{type(exc).__name__}:{first_line}"
    return summary if len(summary) <= max_len else summary[: max_len - 3] + "..."


def _format_row(row: ProbeRow) -> str:
    return (
        f"{row.model:<22} "
        f"{row.status:<13} "
        f"{str(row.compact_len or '-'):>11} "
        f"{str(row.expanded_len or '-'):>12} "
        f"{_bool_text(row.expanded_errors):>15} "
        f"{_bool_text(row.compact_matches_local_expanded):>12} "
        f"{row.detail}"
    )


def _row_satisfies_boundary_invariants(row: ProbeRow) -> bool:
    return (
        row.status == "ok"
        and row.compact_matches_local_expanded is True
        and row.expanded_errors is True
    )


async def _probe_model(model_path: Path, image_path: Path) -> ProbeRow:
    server_args = _make_server_args(model_path)
    model_config = ModelConfig.from_server_args(server_args)
    processor = get_processor(str(model_path), trust_remote_code=True)
    tokenizer = get_tokenizer_from_processor(processor)
    mm_processor = get_mm_processor(
        model_config.hf_config,
        server_args,
        processor,
        "default",
        model_config=model_config,
    )

    prompt_text = processor.apply_chat_template(
        _build_messages(),
        tokenize=False,
        add_generation_prompt=True,
    )

    with Image.open(image_path) as image_file:
        probe_image = image_file.convert("RGB")
        local_expanded_prompt_ids = (
            processor(
                text=[prompt_text],
                images=[probe_image],
                return_tensors="pt",
            )["input_ids"][0]
            .tolist()
        )

    local_compact_prompt_ids = tokenizer(text=[prompt_text], return_tensors="pt")["input_ids"][0].tolist()
    request_obj = SimpleNamespace(
        rid=f"boundary-probe:{model_path.name}",
        video_data=None,
        audio_data=None,
    )

    row = ProbeRow(
        model=model_path.name,
        status="ok",
        compact_len=len(local_compact_prompt_ids),
        expanded_len=len(local_expanded_prompt_ids),
    )

    try:
        compact_result = await mm_processor.process_mm_data_async(
            image_data=[str(image_path)],
            input_text=local_compact_prompt_ids,
            request_obj=request_obj,
            max_req_input_len=model_config.context_len,
        )
        row.compact_matches_local_expanded = compact_result.input_ids == local_expanded_prompt_ids
    except Exception as exc:  # noqa: BLE001
        row.status = "compact_error"
        row.compact_matches_local_expanded = False
        row.detail = _exception_summary(exc)

    try:
        await mm_processor.process_mm_data_async(
            image_data=[str(image_path)],
            input_text=local_expanded_prompt_ids,
            request_obj=request_obj,
            max_req_input_len=model_config.context_len,
        )
        row.expanded_errors = False
    except Exception as exc:  # noqa: BLE001
        row.expanded_errors = True
        if row.detail == "-":
            row.detail = _exception_summary(exc)

    return row


async def _run() -> int:
    args = _parse_args()
    import_processors("sglang.srt.multimodal.processors")

    models_root = args.models_root.expanduser().resolve()
    image_path = _pick_image_path(args.image_dir.expanduser().resolve(), args.image_path)
    model_names = tuple(args.models) if args.models else DEFAULT_MODEL_NAMES

    print(f"image_path={image_path}")
    print(
        f"{'model':<22} {'status':<13} {'compact_len':>11} {'expanded_len':>12} "
        f"{'expanded_errors':>15} {'compact_eq':>12} detail"
    )

    saw_available_model = False
    saw_failure = False

    for model_name in model_names:
        model_path = models_root / model_name
        if not model_path.exists():
            print(_format_row(ProbeRow(model=model_name, status="missing")))
            saw_failure = True
            continue

        saw_available_model = True
        row = await _probe_model(model_path, image_path)
        print(_format_row(row))
        if not _row_satisfies_boundary_invariants(row):
            saw_failure = True

    if not saw_available_model:
        saw_failure = True

    return 1 if saw_failure else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
