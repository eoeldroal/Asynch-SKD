from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors import qwen_vl as qwen_vl_module
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


def _build_test_processor() -> QwenVLImageProcessor:
    processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    processor.hf_config = SimpleNamespace(
        model_type="qwen3_5",
        vision_config=SimpleNamespace(spatial_merge_size=1, tokens_per_second=None),
    )
    processor.model_type = "qwen3_5"
    processor.mm_tokens = SimpleNamespace(
        image_token_id=42,
        video_token_id=None,
        audio_token_id=None,
    )
    processor.IM_TOKEN_ID = 42
    processor.VIDEO_TOKEN_ID = None
    processor.vision_start_token_id = 7
    processor.vision_end_token_id = 8
    processor.audio_start_token_id = None
    processor.video_config = {}
    return processor


@pytest.mark.asyncio
async def test_qwen_vl_token_in_uses_original_prompt_ids_for_final_multimodal_input(monkeypatch):
    processor = _build_test_processor()
    original_prompt_ids = [5, 7, 42, 8, 6]

    base_output = SimpleNamespace(videos=[], images=["raw-image"], audios=[])
    monkeypatch.setattr(processor, "load_mm_data", lambda **kwargs: base_output)

    wrong_input_ids = torch.tensor([999, 7, 42, 42, 8, 6], dtype=torch.long)
    mm_items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(1, 2)],
            precomputed_embeddings=torch.zeros((2, 3), dtype=torch.float32),
        )
    ]
    ret = SimpleNamespace(image_grid_thw=torch.tensor([[1, 2, 1]], dtype=torch.long))
    monkeypatch.setattr(
        processor,
        "process_and_combine_mm_data",
        lambda *args, **kwargs: (mm_items, wrong_input_ids, ret),
    )

    monkeypatch.setattr(
        qwen_vl_module.MRotaryEmbedding,
        "get_rope_index",
        staticmethod(
            lambda **kwargs: (
                torch.zeros((3, 1, kwargs["input_ids"].shape[1]), dtype=torch.long),
                0,
            )
        ),
    )

    output = await processor.process_mm_data_async(
        image_data=["raw-image"],
        input_text=original_prompt_ids,
        request_obj=SimpleNamespace(video_data=None, audio_data=None, rid="rid"),
    )

    assert output.input_ids == [5, 7, 42, 42, 8, 6]
    assert len(output.mm_items) == 1
    assert output.mm_items[0].offsets == [(2, 3)]


@pytest.mark.asyncio
async def test_qwen_vl_text_input_preserves_processor_retokenized_ids(monkeypatch):
    processor = _build_test_processor()

    base_output = SimpleNamespace(videos=[], images=["raw-image"], audios=[])
    monkeypatch.setattr(processor, "load_mm_data", lambda **kwargs: base_output)

    processor_input_ids = torch.tensor([11, 7, 42, 42, 8, 12], dtype=torch.long)
    mm_items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(2, 3)],
            precomputed_embeddings=torch.zeros((2, 3), dtype=torch.float32),
        )
    ]
    ret = SimpleNamespace(image_grid_thw=torch.tensor([[1, 2, 1]], dtype=torch.long))
    monkeypatch.setattr(
        processor,
        "process_and_combine_mm_data",
        lambda *args, **kwargs: (mm_items, processor_input_ids, ret),
    )

    monkeypatch.setattr(
        qwen_vl_module.MRotaryEmbedding,
        "get_rope_index",
        staticmethod(
            lambda **kwargs: (
                torch.zeros((3, 1, kwargs["input_ids"].shape[1]), dtype=torch.long),
                0,
            )
        ),
    )

    output = await processor.process_mm_data_async(
        image_data=["raw-image"],
        input_text="prompt text",
        request_obj=SimpleNamespace(video_data=None, audio_data=None, rid="rid"),
    )

    assert output.input_ids == processor_input_ids.tolist()
    assert output.mm_items[0].offsets == [(2, 3)]
