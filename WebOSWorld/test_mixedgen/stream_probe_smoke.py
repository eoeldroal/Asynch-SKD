import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer


def encode_one(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Text did not encode to any tokens: {!r}".format(text))
    return int(token_ids[0])


def encode_many(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Text did not encode to any tokens: {!r}".format(text))
    return [int(token_id) for token_id in token_ids]


def eos_id(tokenizer):
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    if eos is None:
        special_ids = getattr(tokenizer, "all_special_ids", []) or []
        if not special_ids:
            raise ValueError("Tokenizer has no EOS token or special token ids")
        eos = special_ids[0]
    return int(eos)


def record(kind, request_id=None, **payload):
    return {
        "ts": time.time(),
        "kind": kind,
        "request_id": request_id,
        "payload": payload,
    }


def rows_for_top1(token_ids):
    return [[int(token_id)] for token_id in token_ids]


def final_result(sample_id, response_ids, tokenizer, mode="smoke"):
    return record(
        "final_result",
        sample_id=sample_id,
        mode=mode,
        response_ids=response_ids,
        decoded_skip_special=tokenizer.decode(response_ids, skip_special_tokens=True),
        decoded_with_special=tokenizer.decode(response_ids, skip_special_tokens=False),
        metrics={},
        extra_fields={},
    )


def generate_records(tokenizer):
    accepted = encode_many(tokenizer, " alpha beta")
    rejected_student = encode_one(tokenizer, " wrong")
    normal_replacement = encode_one(tokenizer, " right")
    special_replacement = eos_id(tokenizer)
    normal_tail = encode_many(tokenizer, " tail")
    eos_tail = encode_many(tokenizer, " ignored")
    mismatch_chunk = encode_many(tokenizer, " length mismatch")
    short_ids = encode_many(tokenizer, " ok")
    normal_final_ids = encode_many(tokenizer, " This is a normal final result for stream probe smoke testing.")

    records = []

    records.append(
        record(
            "student_generate_finish",
            request_id="accepted-all",
            token_ids=accepted,
            stop_reason=None,
        )
    )
    records.append(
        record(
            "teacher_verify_finish",
            request_id="accepted-all",
            teacher_ids=rows_for_top1(accepted),
            teacher_logprobs=[],
            rows=len(accepted),
            width=1,
        )
    )

    records.append(
        record(
            "student_generate_finish",
            request_id="reject-normal",
            token_ids=[rejected_student] + normal_tail,
            stop_reason=None,
        )
    )
    records.append(
        record(
            "teacher_verify_finish",
            request_id="reject-normal",
            teacher_ids=[[normal_replacement]] + rows_for_top1(normal_tail),
            teacher_logprobs=[],
            rows=1 + len(normal_tail),
            width=1,
        )
    )

    records.append(
        record(
            "student_generate_finish",
            request_id="reject-eos",
            token_ids=[rejected_student] + eos_tail,
            stop_reason=None,
        )
    )
    records.append(
        record(
            "teacher_verify_finish",
            request_id="reject-eos",
            teacher_ids=[[special_replacement]] + rows_for_top1(eos_tail),
            teacher_logprobs=[],
            rows=1 + len(eos_tail),
            width=1,
        )
    )

    records.append(
        record(
            "student_generate_finish",
            request_id="row-length-mismatch",
            token_ids=mismatch_chunk,
            stop_reason=None,
        )
    )
    records.append(
        record(
            "teacher_verify_finish",
            request_id="row-length-mismatch",
            teacher_ids=rows_for_top1(mismatch_chunk[:-1]),
            teacher_logprobs=[],
            rows=max(0, len(mismatch_chunk) - 1),
            width=1,
        )
    )

    records.append(
        record(
            "teacher_verify_error",
            request_id="teacher-error",
            error="RuntimeError('synthetic teacher verification failure')",
        )
    )

    records.append(final_result("final-empty", [special_replacement], tokenizer))
    records.append(final_result("final-short", short_ids, tokenizer))
    records.append(final_result("final-normal", normal_final_ids, tokenizer))

    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic stream_probe smoke JSONL.")
    parser.add_argument("--out", required=True, help="Path to write synthetic JSONL.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path or name.")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for item in generate_records(tokenizer):
            handle.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
        handle.write("{malformed json line\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
