import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


STUDENT_EVENT = "student_generate_finish"
TEACHER_EVENT = "teacher_verify_finish"
TEACHER_ERROR_EVENT = "teacher_verify_error"
FINAL_RESULT_EVENT = "final_result"


def load_jsonl(path):
    records = []
    parse_errors = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                parse_errors.append(
                    {
                        "line": line_no,
                        "error": str(exc),
                    }
                )
    return records, parse_errors


def is_special_token(token_id, tokenizer):
    if token_id is None:
        return False

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple, set)):
        special_ids.update(eos_token_id)
    elif eos_token_id is not None:
        special_ids.add(eos_token_id)
    return token_id in special_ids


def eos_token_ids(tokenizer):
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return set(eos_token_id)
    return {eos_token_id}


def is_eos_token(token_id, tokenizer):
    return token_id in eos_token_ids(tokenizer)


def _record_event(record):
    for key in ("kind", "event", "event_name", "type", "name"):
        value = record.get(key)
        if value:
            return value
    payload = _payload(record)
    for key in ("kind", "event", "event_name", "type", "name"):
        value = payload.get(key)
        if value:
            return value
    return None


def _payload(record):
    payload = record.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def _student_chunk(record):
    payload = _payload(record)
    token_ids = payload.get("token_ids", [])
    if isinstance(token_ids, list):
        return token_ids
    return []


def _teacher_rows(record):
    payload = _payload(record)
    teacher_ids = payload.get("teacher_ids", [])
    if isinstance(teacher_ids, list):
        return teacher_ids
    return []


def _teacher_top1(row):
    if isinstance(row, list) and row:
        return row[0]
    return None


def _in_top_k(token_id, row, verify_top_k):
    if not isinstance(row, list):
        return False
    return token_id in row[:verify_top_k]


def _decode(tokenizer, token_ids, skip_special_tokens):
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _token_diagnostics(tokenizer, token_id):
    if token_id is None:
        return {
            "token_id": None,
            "is_eos": False,
            "is_special": False,
            "decoded_skip_special": "",
            "decoded_with_special": "",
        }
    return {
        "token_id": token_id,
        "is_eos": is_eos_token(token_id, tokenizer),
        "is_special": is_special_token(token_id, tokenizer),
        "decoded_skip_special": _decode(tokenizer, [token_id], skip_special_tokens=True),
        "decoded_with_special": _decode(tokenizer, [token_id], skip_special_tokens=False),
    }


def _problematic(sample):
    return (
        sample["empty"]
        or sample["short"]
        or sample["first_token_teacher_special"]
        or sample.get("first_token_teacher_eos", False)
        or sample["delta_length_mismatch"]
    )


def _analyze_request(request_id, request_records, tokenizer, verify_top_k, short_chars):
    pending_student = []
    committed = []
    paired_chunks = []
    rejections = []
    delta_length_mismatch = False
    unpaired_teacher_finishes = 0
    teacher_errors = []
    terminated = False
    termination_reason = None

    for record in request_records:
        event = _record_event(record)
        if event == STUDENT_EVENT:
            pending_student.append(record)
        elif event == TEACHER_EVENT and pending_student:
            student_record = pending_student.pop(0)
            chunk = _student_chunk(student_record)
            teacher_ids = _teacher_rows(record)
            if len(teacher_ids) != len(chunk):
                delta_length_mismatch = True

            rejection_pos = None
            replacement = None
            replacement_is_special = False
            accepted_prefix_len = len(chunk)

            for idx, token_id in enumerate(chunk):
                row = teacher_ids[idx] if idx < len(teacher_ids) else None
                if not _in_top_k(token_id, row, verify_top_k):
                    rejection_pos = idx
                    replacement = _teacher_top1(row)
                    replacement_is_special = is_special_token(replacement, tokenizer)
                    accepted_prefix_len = idx
                    break

            if not terminated:
                committed.extend(chunk[:accepted_prefix_len])
                if rejection_pos is not None:
                    if replacement is not None:
                        committed.append(replacement)
                    student_diag = _token_diagnostics(tokenizer, chunk[rejection_pos])
                    replacement_diag = _token_diagnostics(tokenizer, replacement)
                    rejections.append(
                        {
                            "chunk_index": len(paired_chunks),
                            "rejection_pos": rejection_pos,
                            "student_token_id": chunk[rejection_pos],
                            "student_decoded": student_diag,
                            "replacement": replacement,
                            "replacement_is_special": replacement_is_special,
                            "replacement_is_eos": replacement_diag["is_eos"],
                            "replacement_decoded_skip_special": replacement_diag["decoded_skip_special"],
                            "replacement_decoded_with_special": replacement_diag["decoded_with_special"],
                        }
                    )
                if any(is_eos_token(token_id, tokenizer) for token_id in committed):
                    terminated = True
                    termination_reason = "eos"

            paired_chunks.append(
                {
                    "chunk_index": len(paired_chunks),
                    "chunk_len": len(chunk),
                    "teacher_len": len(teacher_ids),
                    "rejection_pos": rejection_pos,
                    "replacement": replacement,
                    "replacement_is_special": replacement_is_special,
                    "replacement_is_eos": is_eos_token(replacement, tokenizer),
                }
            )
        elif event == TEACHER_EVENT:
            unpaired_teacher_finishes += 1
        elif event == TEACHER_ERROR_EVENT:
            teacher_errors.append(_payload(record).get("error"))

    decoded_skip_special = _decode(tokenizer, committed, skip_special_tokens=True)
    decoded_with_special = _decode(tokenizer, committed, skip_special_tokens=False)
    first_token_reject = any(item["rejection_pos"] == 0 for item in rejections)
    first_token_teacher_special = any(
        item["rejection_pos"] == 0 and item["replacement_is_special"]
        for item in rejections
    )
    first_token_teacher_eos = any(
        item["rejection_pos"] == 0 and item["replacement_is_eos"]
        for item in rejections
    )

    return {
        "request_id": request_id,
        "num_records": len(request_records),
        "num_paired_chunks": len(paired_chunks),
        "unpaired_student_chunks": len(pending_student),
        "unpaired_teacher_finishes": unpaired_teacher_finishes,
        "teacher_errors": teacher_errors,
        "committed_token_ids": committed,
        "decoded_skip_special": decoded_skip_special,
        "decoded_with_special": decoded_with_special,
        "empty": decoded_skip_special == "",
        "short": len(decoded_skip_special.strip()) < short_chars,
        "first_token_reject": first_token_reject,
        "first_token_teacher_special": first_token_teacher_special,
        "first_token_teacher_eos": first_token_teacher_eos,
        "delta_length_mismatch": delta_length_mismatch,
        "termination_reason": termination_reason,
        "paired_chunks": paired_chunks,
        "rejections": rejections,
    }


def _analyze_final_results(records, short_chars):
    final_outputs = []
    for record in records:
        if _record_event(record) != FINAL_RESULT_EVENT:
            continue
        payload = _payload(record)
        decoded_skip_special = payload.get("decoded_skip_special", "")
        if not isinstance(decoded_skip_special, str):
            decoded_skip_special = str(decoded_skip_special)
        response_ids = payload.get("response_ids") or []
        final_outputs.append(
            {
                "sample_id": payload.get("sample_id"),
                "mode": payload.get("mode"),
                "response_len": len(response_ids),
                "decoded_skip_special": decoded_skip_special,
                "decoded_with_special": payload.get("decoded_with_special", ""),
                "empty": decoded_skip_special == "",
                "short": len(decoded_skip_special.strip()) < short_chars,
                "metrics": payload.get("metrics"),
                "extra_fields": payload.get("extra_fields"),
            }
        )
    return final_outputs


def analyze_records(
    records,
    tokenizer,
    verify_top_k=5,
    short_chars=16,
    parse_errors=None,
    include_samples=False,
):
    grouped = defaultdict(list)
    missing_request_id = 0
    for record in records:
        event = _record_event(record)
        request_id = record.get("request_id")
        if request_id is None:
            request_id = _payload(record).get("request_id")
        if request_id is None:
            if event in (STUDENT_EVENT, TEACHER_EVENT, TEACHER_ERROR_EVENT):
                missing_request_id += 1
            continue
        grouped[str(request_id)].append(record)

    samples = [
        _analyze_request(
            request_id,
            request_records,
            tokenizer,
            verify_top_k,
            short_chars,
        )
        for request_id, request_records in grouped.items()
    ]
    problematic_samples = [sample for sample in samples if _problematic(sample)]
    final_outputs = _analyze_final_results(records, short_chars)
    primary_outputs = final_outputs or samples

    summary = {
        "num_requests": len(samples),
        "num_final_outputs": len(final_outputs),
        "empty_outputs": sum(1 for sample in primary_outputs if sample["empty"]),
        "short_outputs": sum(1 for sample in primary_outputs if sample["short"]),
        "request_empty_outputs": sum(1 for sample in samples if sample["empty"]),
        "request_short_outputs": sum(1 for sample in samples if sample["short"]),
        "final_empty_outputs": sum(1 for sample in final_outputs if sample["empty"]),
        "final_short_outputs": sum(1 for sample in final_outputs if sample["short"]),
        "first_token_reject_outputs": sum(
            1 for sample in samples if sample["first_token_reject"]
        ),
        "first_token_teacher_special_outputs": sum(
            1 for sample in samples if sample["first_token_teacher_special"]
        ),
        "first_token_teacher_eos_outputs": sum(
            1 for sample in samples if sample["first_token_teacher_eos"]
        ),
        "delta_length_mismatch_requests": sum(
            1 for sample in samples if sample["delta_length_mismatch"]
        ),
        "unpaired_student_chunks": sum(sample["unpaired_student_chunks"] for sample in samples),
        "unpaired_teacher_finishes": sum(sample["unpaired_teacher_finishes"] for sample in samples),
        "teacher_verify_errors": sum(len(sample["teacher_errors"]) for sample in samples),
        "missing_request_id_records": missing_request_id,
        "parse_errors": parse_errors or [],
        "problematic_samples": problematic_samples,
        "problematic_final_outputs": [sample for sample in final_outputs if sample["empty"] or sample["short"]],
        "final_outputs": final_outputs,
    }
    if include_samples:
        summary["samples"] = samples
    return summary


def _summary_for_stdout(summary):
    excluded = {"final_outputs", "problematic_final_outputs", "problematic_samples", "samples"}
    return {key: value for key, value in summary.items() if key not in excluded}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze async SKD probe JSONL.")
    parser.add_argument("--probe-log", required=True, help="Path to probe JSONL log.")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer path or name.")
    parser.add_argument("--verify-top-k", type=int, default=5)
    parser.add_argument("--short-chars", type=int, default=16)
    parser.add_argument("--out", required=True, help="Path to output JSON summary.")
    parser.add_argument("--include-samples", action="store_true", default=False)
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    records, parse_errors = load_jsonl(args.probe_log)
    summary = analyze_records(
        records,
        tokenizer,
        verify_top_k=args.verify_top_k,
        short_chars=args.short_chars,
        parse_errors=parse_errors,
        include_samples=args.include_samples,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(_summary_for_stdout(summary), sort_keys=True))


if __name__ == "__main__":
    main()
