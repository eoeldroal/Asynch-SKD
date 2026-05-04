from __future__ import annotations

import asyncio
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.web_osgym_rl_prompt_window import build_web_osgym_prompt_window
from verl.utils.dataset.rl_dataset import RLHFDataset


class _FakeServerManager:
    async def generate(self, *args, **kwargs):
        raise AssertionError("generate should not be called in apply_chat_template tests")

    async def generate_for_partial(self, *args, **kwargs):
        raise AssertionError("generate_for_partial should not be called in apply_chat_template tests")


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        return [101, 102]


class _StrictProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = False,
        **kwargs,
    ):
        del messages, tools, add_generation_prompt, return_dict, kwargs
        if tokenize:
            return [201, 202]
        return "<prompt>"

    def __call__(
        self,
        *,
        text: list[str],
        images: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
        video_metadata: Optional[list[Any]] = None,
        return_tensors: str,
        do_sample_frames: bool,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, do_sample_frames
        self.calls.append(
            {
                "text": text,
                "images": images,
                "videos": videos,
                "video_metadata": video_metadata,
            }
        )
        if images == []:
            raise AssertionError("empty image lists must be normalized to None before processor calls")
        if videos == []:
            raise AssertionError("empty video lists must be normalized to None before processor calls")
        return {"input_ids": torch.tensor([[301, 302, 303]], dtype=torch.long)}


def test_apply_chat_template_normalizes_text_only_web_osgym_window_images_to_none():
    async def _case():
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                    "model": {},
                },
                "data": {
                    "tool_config_path": None,
                    "apply_chat_template_kwargs": {},
                },
            }
        )

        prompt_window = build_web_osgym_prompt_window(
            base_messages=[{"role": "user", "content": "Recover after invalid action"}],
            images=[],
            steps=[
                {
                    "step_idx": 1,
                    "phase": "tool_observation",
                    "image_start": 0,
                    "image_end": 0,
                    "text": "DONE/FAIL must be sent as a standalone action list",
                }
            ],
            assistant_turns=[],
            history_n=5,
            max_images_per_sample=6,
        )
        assert prompt_window.images == []

        loop = SingleTurnAgentLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=_FakeServerManager(),
            tokenizer=_FakeTokenizer(),
            processor=_StrictProcessor(),
            dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(config.data),
        )

        prompt_ids = await loop.apply_chat_template(
            prompt_window.messages,
            images=prompt_window.images,
            videos=[],
        )

        assert prompt_ids == [301, 302, 303]
        assert loop.processor.calls == [
            {
                "text": ["<prompt>"],
                "images": None,
                "videos": None,
                "video_metadata": None,
            }
        ]

    asyncio.run(_case())
