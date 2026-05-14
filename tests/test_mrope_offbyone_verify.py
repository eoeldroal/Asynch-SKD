"""
Verify whether the claimed mRoPE off-by-one bug in SGLang normal decode path exists.

Claim in the report:
  SGLang normal decode uses: (delta - 1) + seq_lens_cpu
  HF decode uses:            (past_key_values_length) + delta
  → alleged off-by-one

This test shows the claim is WRONG because prepare_for_decode adds +1 to seq_lens
BEFORE the mRoPE position is computed, making both formulas identical.

Run: conda run -n skd-cudnn python tests/test_mrope_offbyone_verify.py
"""

import torch


def hf_decode_position(past_kv_len: int, delta: int) -> int:
    """HF path: arange(past_kv_len, past_kv_len+1) + delta"""
    position_ids = torch.arange(past_kv_len, past_kv_len + 1)
    return (position_ids + delta).item()


def sglang_scheduler_seq_lens(prompt_len: int, output_ids_len: int) -> int:
    """scheduler.py:2184  — BEFORE prepare_for_decode"""
    return prompt_len + output_ids_len - 1


def sglang_after_prepare(seq_lens_before: int) -> int:
    """schedule_batch.py:2132 — prepare_for_decode adds +1"""
    return seq_lens_before + 1


def sglang_expand_mrope(delta: int, seq_lens_cpu_after_prepare: int) -> int:
    """forward_batch_info.py:747-749  _expand_mrope_from_input
      cache = (delta - 1)
      return cache + seq_len
    """
    cache = delta - 1
    return cache + seq_lens_cpu_after_prepare


def sglang_text_only_decode_position(seq_lens_cpu_after_prepare: int) -> int:
    """forward_batch_info.py:768  text-only path: seq_lens_cpu - 1"""
    return seq_lens_cpu_after_prepare - 1


def clamp_position(seq_lens_after_prepare: int) -> int:
    """forward_batch_info.py:1196  clamp_position = seq_lens - 1"""
    return max(seq_lens_after_prepare - 1, 0)


def run_cases():
    # Test cases: (prompt_len, num_generated_tokens_so_far, delta)
    # "num_generated_tokens_so_far" = len(output_ids) at the time make_decode_batch runs
    cases = [
        # First decode step after prefill (prompt=10 tokens, first output)
        (10, 1, 0),
        # Later decode step (prompt=10, 5 outputs so far), text-only (delta=0)
        (10, 5, 0),
        # Multimodal: prompt=417 tokens, delta=-1980 (from the user's example)
        (417, 1, -1980),
        (417, 9, -1980),
        # Generic multimodal
        (100, 3, -50),
        (2000, 50, 100),
    ]

    print("=" * 72)
    print(f"{'Case':<35} {'HF pos':>8} {'SGL pos':>8} {'clamp':>8} {'Match':>6}")
    print("=" * 72)

    all_pass = True
    for P, G, delta in cases:
        # In HF, past_kv_len = tokens in KV = P + (G-1)
        # (G tokens in output_ids, last one is current input → G-1 already in KV)
        past_kv_len = P + G - 1
        hf_pos = hf_decode_position(past_kv_len, delta)

        seq_lens_before = sglang_scheduler_seq_lens(P, G)         # P+G-1
        seq_lens_after  = sglang_after_prepare(seq_lens_before)   # P+G

        sgl_pos_text = sglang_text_only_decode_position(seq_lens_after)  # delta=0 path
        sgl_pos_mm   = sglang_expand_mrope(delta, seq_lens_after)
        sgl_clamp    = clamp_position(seq_lens_after)  # for attention (no delta)

        # Text-only: delta=0 everywhere, so hf_pos == P+G-1 == sgl_pos_text
        if delta == 0:
            match_text = (sgl_pos_text == hf_pos)
            case_str = f"P={P}, G={G}, delta={delta} [text]"
            ok = "OK" if match_text else "FAIL"
            if not match_text:
                all_pass = False
            print(f"{case_str:<35} {hf_pos:>8} {sgl_pos_text:>8} {sgl_clamp:>8} {ok:>6}")
        else:
            match_mm = (sgl_pos_mm == hf_pos)
            case_str = f"P={P}, G={G}, delta={delta} [mm]"
            ok = "OK" if match_mm else "FAIL"
            if not match_mm:
                all_pass = False
            print(f"{case_str:<35} {hf_pos:>8} {sgl_pos_mm:>8} {sgl_clamp:>8} {ok:>6}")

    print("=" * 72)
    if all_pass:
        print("\n✓ ALL PASS — no off-by-one. The reported bug does NOT exist.")
        print()
        print("Why the claim is wrong:")
        print("  schedule_batch.prepare_for_decode() adds +1 to seq_lens_cpu")
        print("  (schedule_batch.py:2132-2134) BEFORE the forward pass.")
        print()
        print("  So by the time _expand_mrope_from_input sees seq_lens_cpu:")
        print("    seq_lens_cpu = P + G  (not P + G - 1)")
        print()
        print("  And:  (delta - 1) + (P + G)  ==  (P + G - 1) + delta  ✓")
        print()
        print("  The '-1' in the cache is intentional: it pre-subtracts the +1")
        print("  that prepare_for_decode will add, so the formula stays correct.")
        print()
        print("  The speculative path does NOT subtract 1 from delta because")
        print("  spec_info.positions already holds the raw position (no +1 added),")
        print("  so it adds delta directly: seq_positions + delta.")
        print("  Both paths produce identical results — they just account for the")
        print("  +1 offset at different points.")
    else:
        print("\n✗ FAILURES detected — investigate further.")


if __name__ == "__main__":
    run_cases()
