import pytest

from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    _extract_prompt_logprobs_sglang,
    _extract_skd_delta_prompt_logprobs_sglang,
)


def _top_row(start_token_id: int) -> list[tuple[float, int, str]]:
    return [
        (-float(start_token_id), start_token_id, f"tok-{start_token_id}"),
        (-float(start_token_id + 1), start_token_id + 1, f"tok-{start_token_id + 1}"),
    ]


def test_extract_prompt_logprobs_full_mode_keeps_sequence_length_contract():
    result: dict[str, list] = {}
    meta_info = {
        "input_token_logprobs": [
            (None, 10, "tok-10"),
            (-0.1, 11, "tok-11"),
            (-0.2, 12, "tok-12"),
            (-0.3, 13, "tok-13"),
        ],
        "input_top_logprobs": [
            None,
            _top_row(21),
            _top_row(31),
            _top_row(41),
        ],
    }

    _extract_prompt_logprobs_sglang(
        meta_info=meta_info,
        num_prompt_logprobs=2,
        sequence_length=4,
        result_dict=result,
    )

    assert result["prompt_ids"] == [
        [21, 22],
        [31, 32],
        [41, 42],
        [0, 0],
    ]
    assert result["prompt_logprobs"] == [
        [-21.0, -22.0],
        [-31.0, -32.0],
        [-41.0, -42.0],
        [0.0, 0.0],
    ]


def test_extract_prompt_logprobs_delta_mode_returns_suffix_without_dummy_row():
    result: dict[str, list] = {}
    meta_info = {
        "input_token_logprobs": [
            (-0.2, 12, "tok-12"),
            (-0.3, 13, "tok-13"),
        ],
        "input_top_logprobs": [
            None,
            _top_row(31),
            _top_row(41),
        ],
    }

    _extract_skd_delta_prompt_logprobs_sglang(
        meta_info=meta_info,
        num_prompt_logprobs=2,
        sequence_length=4,
        result_dict=result,
        prompt_logprobs_start_len=1,
    )

    assert result["prompt_ids"] == [
        [31, 32],
        [41, 42],
    ]
    assert result["prompt_logprobs"] == [
        [-31.0, -32.0],
        [-41.0, -42.0],
    ]


def test_extract_prompt_logprobs_delta_mode_validates_suffix_length():
    result: dict[str, list] = {}
    meta_info = {
        "input_token_logprobs": [
            (-0.2, 12, "tok-12"),
        ],
        "input_top_logprobs": [
            None,
            _top_row(31),
        ],
    }

    with pytest.raises(ValueError, match="SGLang SKD delta prompt_logprobs length"):
        _extract_skd_delta_prompt_logprobs_sglang(
            meta_info=meta_info,
            num_prompt_logprobs=2,
            sequence_length=4,
            result_dict=result,
            prompt_logprobs_start_len=1,
        )
