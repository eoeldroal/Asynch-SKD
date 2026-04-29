"""Create a small mock Web/OSGym RLHF dataset for smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LOCAL_SAVE_DIR = "/home/sogang_nlpy/verl/data/mock_web_osgym"
DEFAULT_NUM_SAMPLES = 256
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


DEFAULT_AGENT_NAME = "web_skd_agent"


def _row(*, split: str, index: int, task_id_start: int, agent_name: str) -> dict[str, Any]:
    task_id = _task_id(task_id_start, index)
    tools_kwargs = {"computer": {"create_kwargs": {"task_id": task_id}}}
    return {
        "data_source": "mock_web_osgym",
        "prompt": _prompt(task_id),
        "ability": "web_osgym",
        "reward_model": {"style": "mock", "ground_truth": "done"},
        "agent_name": agent_name,
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": task_id,
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def build_rows(*, split: str, num_samples: int, task_id_start: int, agent_name: str = DEFAULT_AGENT_NAME) -> list[dict[str, Any]]:
    return [
        _row(split=split, index=index, task_id_start=task_id_start, agent_name=agent_name)
        for index in range(num_samples)
    ]


def write_split(
    *,
    local_save_dir: Path,
    split: str,
    num_samples: int,
    task_id_start: int,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> Path:
    output_path = local_save_dir / f"{split}.parquet"
    rows = build_rows(split=split, num_samples=num_samples, task_id_start=task_id_start, agent_name=agent_name)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mock Web/OSGym train.parquet and val.parquet files for trainer smoke runs."
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
    parser.add_argument(
        "--agent-name",
        default=DEFAULT_AGENT_NAME,
        help=(
            "Agent loop name written to the dataset's agent_name column. "
            f"Default: {DEFAULT_AGENT_NAME}. Use web_tool_agent for fully async RL smoke runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.task_id_start < 0:
        raise ValueError("--task-id-start must be non-negative")
    if not args.agent_name:
        raise ValueError("--agent-name must be non-empty")

    local_save_dir = Path(args.local_save_dir).expanduser()
    local_save_dir.mkdir(parents=True, exist_ok=True)

    train_path = write_split(
        local_save_dir=local_save_dir,
        split="train",
        num_samples=args.num_samples,
        task_id_start=args.task_id_start,
        agent_name=args.agent_name,
    )
    val_path = write_split(
        local_save_dir=local_save_dir,
        split="val",
        num_samples=args.num_samples,
        task_id_start=args.task_id_start,
        agent_name=args.agent_name,
    )

    print(f"Wrote {args.num_samples} train rows to {train_path} with agent_name={args.agent_name}")
    print(f"Wrote {args.num_samples} val rows to {val_path} with agent_name={args.agent_name}")


if __name__ == "__main__":
    main()
