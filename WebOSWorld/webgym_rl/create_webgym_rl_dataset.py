"""Create SKD and fully async RL parquet datasets backed by WebGym task definitions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_TASK_FILE = "/home/sogang_nlpy/goonco/surfgym/tasks/tasks_subset.json"
DEFAULT_SKD_SAVE_DIR = "/home/sogang_nlpy/verl/data/webgym_skd"
DEFAULT_RL_SAVE_DIR = "/home/sogang_nlpy/verl/data/webgym_rl"
DEFAULT_TRAIN_REPEATS_PER_TASK = 16
DEFAULT_VAL_REPEATS_PER_TASK = 1
DEFAULT_INCLUDE_LOCALHOST = True

SKD_TARGET = "skd"
RL_TARGET = "rl"
BOTH_TARGET = "both"
DATASET_TARGETS = (SKD_TARGET, RL_TARGET, BOTH_TARGET)

SKD_AGENT_NAME = "web_skd_agent"
RL_AGENT_NAME = "web_tool_agent"


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
    include_localhost: bool = DEFAULT_INCLUDE_LOCALHOST,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
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
    instruction = str(task.get("instruction") or task.get("task_name") or "")
    if not instruction:
        raise ValueError(f"Task {task.get('task_id', '<unknown>')} is missing both instruction and task_name")
    return [{"role": "user", "content": instruction}]


def _normalized_website(website: Any) -> list[dict[str, Any]]:
    if isinstance(website, str):
        return [{"id": "default", "url": website}]
    if isinstance(website, list):
        return deepcopy(website)
    raise ValueError(f"Unsupported website payload: {website!r}")


def _row(*, split: str, index: int, task: dict[str, Any], agent_name: str) -> dict[str, Any]:
    task_id = str(task["task_id"])
    instruction = str(task.get("instruction") or task.get("task_name") or "")
    if not instruction:
        raise ValueError(f"Task {task_id} is missing both instruction and task_name")
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
            "task_name": instruction,
            "website": _normalized_website(task["website"]),
            "need_tools_kwargs": True,
            "tools_kwargs": {"web_osgym": {"create_kwargs": {"task_id": task_id}}},
        },
    }


def build_rows(
    *,
    split: str,
    tasks: list[dict[str, Any]],
    num_samples: int,
    agent_name: str,
) -> list[dict[str, Any]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    return [
        _row(split=split, index=index, task=tasks[index % len(tasks)], agent_name=agent_name)
        for index in range(num_samples)
    ]


def _resolve_num_samples(
    explicit_num_samples: int | None,
    *,
    num_tasks: int,
    repeats_per_task: int,
) -> int:
    if explicit_num_samples is not None:
        return explicit_num_samples
    return num_tasks * repeats_per_task


def _write_split(
    *,
    output_dir: Path,
    split: str,
    tasks: list[dict[str, Any]],
    num_samples: int,
    agent_name: str,
) -> Path:
    output_path = output_dir / f"{split}.parquet"
    rows = build_rows(split=split, tasks=tasks, num_samples=num_samples, agent_name=agent_name)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def _write_dataset_variant(
    *,
    output_dir: str | Path,
    tasks: list[dict[str, Any]],
    num_train_samples: int,
    num_val_samples: int,
    agent_name: str,
) -> Path:
    resolved_dir = Path(output_dir).expanduser()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    _write_split(
        output_dir=resolved_dir,
        split="train",
        tasks=tasks,
        num_samples=num_train_samples,
        agent_name=agent_name,
    )
    _write_split(
        output_dir=resolved_dir,
        split="val",
        tasks=tasks,
        num_samples=num_val_samples,
        agent_name=agent_name,
    )
    return resolved_dir


def write_standard_webgym_datasets(
    *,
    task_file: str | Path,
    skd_save_dir: str | Path,
    async_rl_save_dir: str | Path,
    num_train_samples: int | None,
    num_val_samples: int | None,
    include_localhost: bool = DEFAULT_INCLUDE_LOCALHOST,
    target: str = BOTH_TARGET,
    task_ids: list[str] | None = None,
) -> dict[str, Path]:
    if target not in DATASET_TARGETS:
        raise ValueError(f"Unsupported target: {target}")

    tasks = select_tasks(
        load_tasks(task_file),
        task_ids=task_ids,
        include_localhost=include_localhost,
    )
    resolved_num_train_samples = _resolve_num_samples(
        num_train_samples,
        num_tasks=len(tasks),
        repeats_per_task=DEFAULT_TRAIN_REPEATS_PER_TASK,
    )
    resolved_num_val_samples = _resolve_num_samples(
        num_val_samples,
        num_tasks=len(tasks),
        repeats_per_task=DEFAULT_VAL_REPEATS_PER_TASK,
    )

    outputs: dict[str, Path] = {}
    if target in (SKD_TARGET, BOTH_TARGET):
        outputs[SKD_TARGET] = _write_dataset_variant(
            output_dir=skd_save_dir,
            tasks=tasks,
            num_train_samples=resolved_num_train_samples,
            num_val_samples=resolved_num_val_samples,
            agent_name=SKD_AGENT_NAME,
        )
    if target in (RL_TARGET, BOTH_TARGET):
        outputs[RL_TARGET] = _write_dataset_variant(
            output_dir=async_rl_save_dir,
            tasks=tasks,
            num_train_samples=resolved_num_train_samples,
            num_val_samples=resolved_num_val_samples,
            agent_name=RL_AGENT_NAME,
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WebGym parquet datasets for SKD, RL, or both.")
    parser.add_argument("--task-file", default=DEFAULT_TASK_FILE)
    parser.add_argument("--skd-save-dir", default=DEFAULT_SKD_SAVE_DIR)
    parser.add_argument("--rl-save-dir", default=DEFAULT_RL_SAVE_DIR)
    parser.add_argument("--target", choices=DATASET_TARGETS, default=BOTH_TARGET)
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help=f"Defaults to selected_task_count * {DEFAULT_TRAIN_REPEATS_PER_TASK}.",
    )
    parser.add_argument(
        "--num-val-samples",
        type=int,
        default=None,
        help=f"Defaults to selected_task_count * {DEFAULT_VAL_REPEATS_PER_TASK}.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional task_id allowlist.",
    )
    parser.add_argument(
        "--include-localhost",
        dest="include_localhost",
        action="store_true",
        default=DEFAULT_INCLUDE_LOCALHOST,
        help="Include tasks whose website points at localhost/127.0.0.1.",
    )
    parser.add_argument(
        "--exclude-localhost",
        dest="include_localhost",
        action="store_false",
        help="Exclude tasks whose website points at localhost/127.0.0.1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_standard_webgym_datasets(
        task_file=args.task_file,
        skd_save_dir=args.skd_save_dir,
        async_rl_save_dir=args.rl_save_dir,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        include_localhost=args.include_localhost,
        target=args.target,
        task_ids=args.task_ids,
    )
    for target, output_dir in outputs.items():
        print(f"Wrote {target} dataset to {output_dir}")


if __name__ == "__main__":
    main()
