"""Create a small mock Web/OSGym RLHF dataset for SKD smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LOCAL_SAVE_DIR = "/home/sogang_nlpy/verl/data/mock_web_osgym"
DEFAULT_NUM_SAMPLES = 64
DEFAULT_TASK_ID_START = 12345


def _task_id(task_id_start: int, index: int) -> str:
    return f"{task_id_start + index:05d}"


def _prompt(task_id: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are controlling a browser environment. Use the computer tool to interact with the page. "
                "When the task is complete, call computer with DONE. If it cannot be completed, call computer with FAIL. "
                "Do not write Web/OSGym protocol JSON directly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Complete mock Web/OSGym task {task_id}. Inspect the page and finish when the task is complete."
            ),
        }
    ]


def _row(*, split: str, index: int, task_id_start: int) -> dict[str, Any]:
    task_id = _task_id(task_id_start, index)
    tools_kwargs = {"computer": {"create_kwargs": {"task_id": task_id}}}
    return {
        "data_source": "mock_web_osgym",
        "prompt": _prompt(task_id),
        "ability": "web_osgym",
        "reward_model": {"style": "mock", "ground_truth": "done"},
        "agent_name": "web_skd_agent",
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": task_id,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def build_rows(*, split: str, num_samples: int, task_id_start: int) -> list[dict[str, Any]]:
    return [_row(split=split, index=index, task_id_start=task_id_start) for index in range(num_samples)]


def write_split(*, local_save_dir: Path, split: str, num_samples: int, task_id_start: int) -> Path:
    output_path = local_save_dir / f"{split}.parquet"
    rows = build_rows(split=split, num_samples=num_samples, task_id_start=task_id_start)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock Web/OSGym train.parquet and val.parquet files for SKD trainer smoke runs."
    )
    parser.add_argument(
        "--local-save-dir",
        default=DEFAULT_LOCAL_SAVE_DIR,
        help=f"Directory where train.parquet and val.parquet are written. Default: {DEFAULT_LOCAL_SAVE_DIR}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of rows to generate in each split. Default: {DEFAULT_NUM_SAMPLES}",
    )
    parser.add_argument(
        "--task-id-start",
        type=int,
        default=DEFAULT_TASK_ID_START,
        help=(
            "First numeric task ID. IDs are emitted as zero-padded 5-digit strings "
            f"and increment by row index. Default: {DEFAULT_TASK_ID_START}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.task_id_start < 0:
        raise ValueError("--task-id-start must be non-negative")

    local_save_dir = Path(args.local_save_dir).expanduser()
    local_save_dir.mkdir(parents=True, exist_ok=True)

    train_path = write_split(
        local_save_dir=local_save_dir,
        split="train",
        num_samples=args.num_samples,
        task_id_start=args.task_id_start,
    )
    val_path = write_split(
        local_save_dir=local_save_dir,
        split="val",
        num_samples=args.num_samples,
        task_id_start=args.task_id_start,
    )

    print(f"Wrote {args.num_samples} train rows to {train_path}")
    print(f"Wrote {args.num_samples} val rows to {val_path}")


if __name__ == "__main__":
    main()
