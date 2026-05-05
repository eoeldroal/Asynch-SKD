from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("/home/sogang_nlpy/verl")
sys.path.insert(0, str(ROOT))

from WebOSWorld.webgym_rl.create_webgym_rl_dataset import write_standard_webgym_datasets


def _make_task(index: int) -> dict:
    return {
        "task_id": f"task_{index}",
        "task_name": f"Complete Korean site task {index}",
        "website": f"https://example{index}.kr/",
        "evaluation": {
            "mode": "all",
            "rules": [{"text_regex": f"task {index}", "match": "regex"}],
        },
    }


def test_write_standard_webgym_datasets_creates_skd_and_async_rl_variants(tmp_path: Path):
    task_file = tmp_path / "tasks_kr_sites.json"
    tasks = [_make_task(i) for i in range(15)]
    task_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    skd_dir, async_rl_dir = write_standard_webgym_datasets(
        task_file=task_file,
        skd_save_dir=tmp_path / "webgym_rl_counter",
        async_rl_save_dir=tmp_path / "webgym_rl_counter_fully_async_rl",
        num_train_samples=120,
        num_val_samples=15,
    )

    assert skd_dir.name == "webgym_rl_counter"
    assert async_rl_dir.name == "webgym_rl_counter_fully_async_rl"

    skd_train = pd.read_parquet(skd_dir / "train.parquet")
    skd_val = pd.read_parquet(skd_dir / "val.parquet")
    async_train = pd.read_parquet(async_rl_dir / "train.parquet")
    async_val = pd.read_parquet(async_rl_dir / "val.parquet")

    assert len(skd_train) == 120
    assert len(skd_val) == 15
    assert len(async_train) == 120
    assert len(async_val) == 15

    assert set(skd_train["agent_name"]) == {"web_skd_agent"}
    assert set(skd_val["agent_name"]) == {"web_skd_agent"}
    assert set(async_train["agent_name"]) == {"web_tool_agent"}
    assert set(async_val["agent_name"]) == {"web_tool_agent"}

    assert skd_train.iloc[0]["prompt"][0]["role"] == "user"
    assert skd_train.iloc[0]["prompt"][0]["content"] == "Complete Korean site task 0"

    train_task_ids = [row["task_id"] for row in skd_train["extra_info"]]
    assert train_task_ids[:15] == [f"task_{i}" for i in range(15)]
    assert train_task_ids[15:30] == [f"task_{i}" for i in range(15)]

    val_task_ids = [row["task_id"] for row in skd_val["extra_info"]]
    assert val_task_ids == [f"task_{i}" for i in range(15)]
