"""Create veRL parquet datasets backed by webgym-rl task definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_TASK_FILE = "/home/sogang_nlpy/goonco/surfgym/tasks/tasks_kr_sites.json"
DEFAULT_LOCAL_SAVE_DIR = "/home/sogang_nlpy/verl/data/webgym_rl_counter"
DEFAULT_ASYNC_RL_SAVE_DIR = "/home/sogang_nlpy/verl/data/webgym_rl_counter_fully_async_rl"
DEFAULT_NUM_TRAIN_SAMPLES = 120
DEFAULT_NUM_VAL_SAMPLES = 15
DEFAULT_AGENT_NAME = "web_skd_agent"
DEFAULT_ASYNC_RL_AGENT_NAME = "web_tool_agent"


def load_tasks(task_file: str | Path) -> list[dict[str, Any]]:
    path = Path(task_file).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        raise ValueError(f"Unsupported task payload in {path}")
    return [dict(row) for row in data]


def select_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    task_ids: list[str] | None = None,
    include_localhost: bool = False,
) -> list[dict[str, Any]]:
    selected = []
    wanted = set(task_ids or [])
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        website = str(task.get("website", ""))
        if wanted and task_id not in wanted:
            continue
        if not include_localhost and ("127.0.0.1" in website or "localhost" in website):
            continue
        selected.append(task)
    if wanted:
        found = {str(task.get("task_id", "")) for task in selected}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"Unknown or filtered task_id(s): {', '.join(missing)}")
    if not selected:
        raise ValueError("No WebGym tasks selected")
    return selected


def _prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    instruction = str(task["task_name"])
    return [{"role": "user", "content": instruction}]


def _row(*, split: str, index: int, task: dict[str, Any], agent_name: str) -> dict[str, Any]:
    task_id = str(task["task_id"])
    tools_kwargs = {"web_osgym": {"create_kwargs": {"task_id": task_id}}}
    return {
        "data_source": "webgym_rl",
        "prompt": _prompt(task),
        "ability": "web_osgym",
        "reward_model": {"style": "webgym_rl", "ground_truth": "env_reward"},
        "agent_name": agent_name,
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": task_id,
            "task_name": str(task["task_name"]),
            "website": str(task["website"]),
            "need_tools_kwargs": True,
            "tools_kwargs": tools_kwargs,
        },
    }


def build_rows(
    *,
    split: str,
    tasks: list[dict[str, Any]],
    num_samples: int,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> list[dict[str, Any]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    return [
        _row(split=split, index=index, task=tasks[index % len(tasks)], agent_name=agent_name)
        for index in range(num_samples)
    ]


def write_split(
    *,
    local_save_dir: Path,
    split: str,
    tasks: list[dict[str, Any]],
    num_samples: int,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> Path:
    output_path = local_save_dir / f"{split}.parquet"
    rows = build_rows(split=split, tasks=tasks, num_samples=num_samples, agent_name=agent_name)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def write_standard_webgym_datasets(
    *,
    task_file: str | Path,
    skd_save_dir: str | Path,
    async_rl_save_dir: str | Path,
    num_train_samples: int,
    num_val_samples: int,
    include_localhost: bool = False,
) -> tuple[Path, Path]:
    tasks = select_tasks(load_tasks(task_file), include_localhost=include_localhost)

    skd_dir = Path(skd_save_dir).expanduser()
    skd_dir.mkdir(parents=True, exist_ok=True)
    write_split(
        local_save_dir=skd_dir,
        split="train",
        tasks=tasks,
        num_samples=num_train_samples,
        agent_name=DEFAULT_AGENT_NAME,
    )
    write_split(
        local_save_dir=skd_dir,
        split="val",
        tasks=tasks,
        num_samples=num_val_samples,
        agent_name=DEFAULT_AGENT_NAME,
    )

    async_dir = Path(async_rl_save_dir).expanduser()
    async_dir.mkdir(parents=True, exist_ok=True)
    write_split(
        local_save_dir=async_dir,
        split="train",
        tasks=tasks,
        num_samples=num_train_samples,
        agent_name=DEFAULT_ASYNC_RL_AGENT_NAME,
    )
    write_split(
        local_save_dir=async_dir,
        split="val",
        tasks=tasks,
        num_samples=num_val_samples,
        agent_name=DEFAULT_ASYNC_RL_AGENT_NAME,
    )

    return skd_dir, async_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train.parquet and val.parquet files for webgym-rl."
    )
    parser.add_argument("--task-file", default=DEFAULT_TASK_FILE)
    parser.add_argument("--local-save-dir", default=DEFAULT_LOCAL_SAVE_DIR)
    parser.add_argument("--async-rl-save-dir", default=DEFAULT_ASYNC_RL_SAVE_DIR)
    parser.add_argument("--num-train-samples", type=int, default=DEFAULT_NUM_TRAIN_SAMPLES)
    parser.add_argument("--num-val-samples", type=int, default=DEFAULT_NUM_VAL_SAMPLES)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument(
        "--skip-async-rl-copy",
        action="store_true",
        help="Only write the primary dataset instead of also emitting the fully async RL copy.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional task_id allowlist. Defaults to all non-localhost tasks.",
    )
    parser.add_argument(
        "--include-localhost",
        action="store_true",
        help="Include tasks whose website points at localhost/127.0.0.1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_async_rl_copy:
        local_save_dir = Path(args.local_save_dir).expanduser()
        local_save_dir.mkdir(parents=True, exist_ok=True)

        tasks = select_tasks(
            load_tasks(args.task_file),
            task_ids=args.task_ids,
            include_localhost=args.include_localhost,
        )
        train_path = write_split(
            local_save_dir=local_save_dir,
            split="train",
            tasks=tasks,
            num_samples=args.num_train_samples,
            agent_name=args.agent_name,
        )
        val_path = write_split(
            local_save_dir=local_save_dir,
            split="val",
            tasks=tasks,
            num_samples=args.num_val_samples,
            agent_name=args.agent_name,
        )

        task_ids = ", ".join(str(task["task_id"]) for task in tasks)
        print(f"Wrote {args.num_train_samples} train rows to {train_path}")
        print(f"Wrote {args.num_val_samples} val rows to {val_path}")
        print(f"Task ids: {task_ids}")
        return

    skd_dir, async_dir = write_standard_webgym_datasets(
        task_file=args.task_file,
        skd_save_dir=args.local_save_dir,
        async_rl_save_dir=args.async_rl_save_dir,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        include_localhost=args.include_localhost,
    )
    print(f"Wrote standard WebGym datasets to {skd_dir} and {async_dir}")


if __name__ == "__main__":
    main()
