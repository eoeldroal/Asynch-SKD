from __future__ import annotations

import pandas as pd
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


def test_rl_dataset_injects_runtime_system_prompt_and_replaces_dataset_system(tmp_path):
    dataset_path = tmp_path / "train.parquet"
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("Injected runtime system prompt.", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "prompt": [
                    {"role": "system", "content": "Old dataset system prompt."},
                    {"role": "user", "content": "Make the counter value 5."},
                ],
                "data_source": "webgym_rl",
                "extra_info": {"index": 0},
            }
        ]
    ).to_parquet(dataset_path, index=False)

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "filter_overlong_prompts": False,
            "system_prompt_path": str(prompt_path),
        }
    )

    dataset = RLHFDataset(data_files=str(dataset_path), tokenizer=object(), config=config)

    raw_prompt = dataset[0]["raw_prompt"]
    assert raw_prompt == [
        {"role": "system", "content": "Injected runtime system prompt."},
        {"role": "user", "content": "Make the counter value 5."},
    ]


def test_rl_dataset_keeps_dataset_messages_when_runtime_system_prompt_is_unset(tmp_path):
    dataset_path = tmp_path / "train.parquet"

    pd.DataFrame(
        [
            {
                "prompt": [
                    {"role": "system", "content": "Original dataset system prompt."},
                    {"role": "user", "content": "Make the counter value 5."},
                ],
                "data_source": "webgym_rl",
                "extra_info": {"index": 0},
            }
        ]
    ).to_parquet(dataset_path, index=False)

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "filter_overlong_prompts": False,
            "system_prompt_path": None,
        }
    )

    dataset = RLHFDataset(data_files=str(dataset_path), tokenizer=object(), config=config)

    raw_prompt = dataset[0]["raw_prompt"]
    assert raw_prompt == [
        {"role": "system", "content": "Original dataset system prompt."},
        {"role": "user", "content": "Make the counter value 5."},
    ]
