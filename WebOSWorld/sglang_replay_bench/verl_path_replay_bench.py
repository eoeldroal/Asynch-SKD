from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4

import pandas as pd
import yaml
from PIL import Image, ImageDraw

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids


def _now() -> float:
    return time.perf_counter()


def _ms_since(start: float) -> float:
    return round((_now() - start) * 1000, 1)


def _normalise_urls(raw: str) -> list[str]:
    urls = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not urls:
        raise ValueError("--urls must contain at least one URL")
    return urls


def _make_counter_screenshot(width: int = 1184, height: int = 816) -> Image.Image:
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 60), fill=(31, 41, 55))
    draw.text((24, 20), "WebGym Counter", fill=(255, 255, 255))
    draw.rectangle((80, 120, width - 80, height - 80), outline=(148, 163, 184), width=3)
    draw.text((130, 165), "Counter value: 0", fill=(15, 23, 42))
    draw.rounded_rectangle((520, 300, 760, 390), radius=8, fill=(255, 255, 255), outline=(71, 85, 105), width=2)
    draw.text((590, 335), "+1", fill=(15, 23, 42))
    draw.rounded_rectangle((520, 420, 760, 510), radius=8, fill=(255, 255, 255), outline=(71, 85, 105), width=2)
    draw.text((585, 455), "DONE", fill=(15, 23, 42))
    draw.ellipse((676, 455, 716, 495), outline=(220, 38, 38), width=4)
    draw.line((696, 440, 696, 510), fill=(220, 38, 38), width=2)
    draw.line((661, 475, 731, 475), fill=(220, 38, 38), width=2)
    return image


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_dataset_prompt(path: str) -> list[dict[str, Any]]:
    row = pd.read_parquet(path).iloc[0]
    return [dict(item) for item in row["prompt"].tolist()]


def _load_tool_schemas(path: str) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return [item["tool_schema"] for item in config.get("tools", [])]


def _tool_observation(turn: int) -> str:
    a11y = "\n".join(
        [
            "Browser observation:",
            "Task id: counter",
            "Cursor position in screenshot coordinates: x=696, y=475",
            "Accessibility tree:",
            "[1] role=button name='+1' bounds=(520,300,240,90)",
            "[2] role=button name='DONE' bounds=(520,420,240,90)",
            "[3] role=text name='Counter value: 0' bounds=(130,165,220,40)",
        ]
    )
    filler = (
        "Use the current screenshot coordinates. The target is to make the counter value 5. "
        "Click the increment button until the value is correct, then call DONE. "
    )
    return f"Observation turn {turn}.\n{a11y}\n{filler * 8}"


def _build_messages(base_prompt: list[dict[str, Any]], turns: int, *, with_image: bool) -> list[dict[str, Any]]:
    messages = [dict(item) for item in base_prompt]
    for turn in range(1, turns + 1):
        if turn > 1:
            messages.append(
                {
                    "role": "assistant",
                    "content": "<tool_call>\n{\"name\":\"CLICK\",\"arguments\":{\"x\":640,\"y\":345}}\n</tool_call>",
                }
            )
        if with_image:
            messages.append(
                {
                    "role": "tool",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _tool_observation(turn)},
                    ],
                }
            )
        else:
            messages.append({"role": "tool", "content": _tool_observation(turn)})
    return messages


def _build_case(args: argparse.Namespace, tokenizer: Any, processor: Any, with_image: bool) -> dict[str, Any]:
    base_prompt = _load_dataset_prompt(args.dataset_path)
    tools = _load_tool_schemas(args.tool_config_path)
    messages = _build_messages(base_prompt, args.turns, with_image=with_image)
    images = [_make_counter_screenshot() for _ in range(args.turns)] if with_image else None
    image_data = [_image_to_data_uri(image) for image in images] if images else None

    raw_prompt = apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    server_prompt_ids = normalize_token_ids(
        apply_chat_template(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
    )

    expanded_prompt_ids = server_prompt_ids
    if processor is not None and images:
        processor_raw_prompt = apply_chat_template(
            processor,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = processor(text=[processor_raw_prompt], images=images, return_tensors="pt")
        expanded_prompt_ids = normalize_token_ids(model_inputs["input_ids"])

    suffix_text = (
        "<tool_call>\n"
        "{\"name\":\"CLICK\",\"arguments\":{\"x\":640,\"y\":345,\"button\":\"left\",\"num_clicks\":1}}\n"
        "</tool_call>\n"
    )
    suffix_ids = normalize_token_ids(tokenizer(suffix_text, add_special_tokens=False)["input_ids"])
    if len(suffix_ids) < args.teacher_suffix_tokens:
        repeats = (args.teacher_suffix_tokens + len(suffix_ids) - 1) // max(len(suffix_ids), 1)
        suffix_ids = (suffix_ids * repeats)[: args.teacher_suffix_tokens]
    else:
        suffix_ids = suffix_ids[: args.teacher_suffix_tokens]

    return {
        "raw_prompt": raw_prompt,
        "server_prompt_ids": server_prompt_ids,
        "expanded_prompt_ids": expanded_prompt_ids,
        "suffix_ids": suffix_ids,
        "image_data": image_data,
        "image_count": len(image_data or []),
        "server_prompt_len": len(server_prompt_ids),
        "expanded_prompt_len": len(expanded_prompt_ids),
        "mm_prefix_surplus": max(len(expanded_prompt_ids) - len(server_prompt_ids), 0),
    }


def _unwrap_response(event: dict[str, Any]) -> dict[str, Any]:
    inner = event.get("response")
    if isinstance(inner, dict):
        merged = dict(inner)
        for key, value in event.items():
            merged.setdefault(key, value)
        return merged
    return event


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = _now()
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        f"{url}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url}/generate failed with HTTP {exc.code}: {body}") from exc

    parsed = json.loads(body)
    event = _unwrap_response(parsed if isinstance(parsed, dict) else {})
    meta_info = event.get("meta_info") if isinstance(event.get("meta_info"), dict) else {}
    output_ids = event.get("output_ids") or event.get("token_ids") or []
    return {
        "total_ms": _ms_since(started),
        "ttft_ms": None,
        "event_count": 1,
        "max_inter_chunk_gap_ms": None,
        "avg_inter_chunk_gap_ms": None,
        "output_ids_len": len(output_ids) if isinstance(output_ids, list) else None,
        "tokens_per_sec": round(len(output_ids) / (_ms_since(started) / 1000), 2)
        if isinstance(output_ids, list) and _ms_since(started) > 0
        else None,
        "finish_reason": meta_info.get("finish_reason"),
        "queue_time": meta_info.get("queue_time"),
        "prefill_launch_delay": meta_info.get("prefill_launch_delay"),
        "prefill_launch_latency": meta_info.get("prefill_launch_latency"),
        "inference_time": meta_info.get("inference_time"),
        "input_token_logprobs_len": len(meta_info.get("input_token_logprobs") or []),
        "input_top_logprobs_len": len(meta_info.get("input_top_logprobs") or []),
    }


def _extract_output_ids(event: dict[str, Any]) -> list[int] | None:
    event = _unwrap_response(event)
    for key in ("output_ids", "token_ids"):
        value = event.get(key)
        if isinstance(value, list):
            return [int(item) for item in value]
    return None


def _extract_meta(event: dict[str, Any]) -> dict[str, Any]:
    event = _unwrap_response(event)
    meta = event.get("meta_info")
    return meta if isinstance(meta, dict) else {}


def _post_stream(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = _now()
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        f"{url}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first_chunk_ms: float | None = None
    chunk_times: list[float] = []
    last_event: dict[str, Any] = {}
    last_output_ids: list[int] | None = None
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                item = line[len("data:") :].strip()
                if item == "[DONE]":
                    break
                event = json.loads(item)
                if not isinstance(event, dict):
                    continue
                if "error" in event:
                    raise RuntimeError(f"SGLang stream error: {event['error']}")
                if first_chunk_ms is None:
                    first_chunk_ms = _ms_since(started)
                chunk_times.append(_now())
                last_event = event
                ids = _extract_output_ids(event)
                if ids is not None:
                    last_output_ids = ids
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url}/generate failed with HTTP {exc.code}: {body}") from exc

    gaps_ms = [
        round((chunk_times[idx] - chunk_times[idx - 1]) * 1000, 1)
        for idx in range(1, len(chunk_times))
    ]
    total_ms = _ms_since(started)
    output_tokens = len(last_output_ids) if last_output_ids is not None else None
    meta_info = _extract_meta(last_event)
    return {
        "total_ms": total_ms,
        "ttft_ms": first_chunk_ms,
        "event_count": len(chunk_times),
        "max_inter_chunk_gap_ms": max(gaps_ms) if gaps_ms else None,
        "avg_inter_chunk_gap_ms": round(statistics.mean(gaps_ms), 1) if gaps_ms else None,
        "output_ids_len": output_tokens,
        "tokens_per_sec": round(output_tokens / (total_ms / 1000), 2) if output_tokens and total_ms > 0 else None,
        "finish_reason": meta_info.get("finish_reason"),
        "queue_time": meta_info.get("queue_time"),
        "prefill_launch_delay": meta_info.get("prefill_launch_delay"),
        "prefill_launch_latency": meta_info.get("prefill_launch_latency"),
        "inference_time": meta_info.get("inference_time"),
        "input_token_logprobs_len": len(meta_info.get("input_token_logprobs") or []),
        "input_top_logprobs_len": len(meta_info.get("input_top_logprobs") or []),
    }


def _student_payload(case: dict[str, Any], mode: str, request_id: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rid": request_id,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        "stream": bool(args.stream),
    }
    if mode.startswith("native"):
        payload["text"] = case["raw_prompt"]
    else:
        payload["input_ids"] = case["server_prompt_ids"]
    if case["image_data"]:
        payload["image_data"] = case["image_data"]
    return payload


def _teacher_payload(case: dict[str, Any], mode: str, request_id: str, args: argparse.Namespace) -> dict[str, Any]:
    sequence_ids = case["server_prompt_ids"] + case["suffix_ids"]
    payload: dict[str, Any] = {
        "rid": request_id,
        "input_ids": sequence_ids,
        "sampling_params": {"max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": len(case["server_prompt_ids"]) - 1 + case["mm_prefix_surplus"],
        "top_logprobs_num": args.top_logprobs_num,
        "stream": bool(args.stream),
    }
    if mode.startswith("native"):
        payload["text"] = case["raw_prompt"] + args.teacher_suffix_text
        payload.pop("input_ids", None)
    if case["image_data"]:
        payload["image_data"] = case["image_data"]
    return payload


async def run_one(
    *,
    url: str,
    mode: str,
    index: int,
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rid = f"verlpath-{mode}-c{args.concurrency}-r{index:04d}-{uuid4().hex[:8]}"
    if mode.endswith("_teacher"):
        payload = _teacher_payload(case, mode, rid, args)
    else:
        payload = _student_payload(case, mode, rid, args)

    try:
        post_fn = _post_stream if args.stream else _post_json
        result = await asyncio.to_thread(post_fn, url, payload, args.timeout)
        status = "ok"
        error_msg = None
    except Exception as exc:
        result = {}
        status = "error"
        error_msg = repr(exc)

    return {
        "stage": "request_result",
        "status": status,
        "error": error_msg,
        "mode": mode,
        "request_id": rid,
        "url": url,
        "index": index,
        "concurrency": args.concurrency,
        "stream": bool(args.stream),
        "turns": args.turns,
        "image_count": case["image_count"],
        "server_prompt_len": case["server_prompt_len"],
        "expanded_prompt_len": case["expanded_prompt_len"],
        "mm_prefix_surplus": case["mm_prefix_surplus"],
        **result,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return round(values[lo], 1)
    frac = rank - lo
    return round(values[lo] * (1 - frac) + values[hi] * frac, 1)


def summarize(mode: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    oks = [item for item in records if item["status"] == "ok"]
    total = [float(item["total_ms"]) for item in oks if item.get("total_ms") is not None]
    ttft = [float(item["ttft_ms"]) for item in oks if item.get("ttft_ms") is not None]
    gaps = [float(item["max_inter_chunk_gap_ms"]) for item in oks if item.get("max_inter_chunk_gap_ms") is not None]
    prefill = [float(item["prefill_launch_latency"]) for item in oks if item.get("prefill_launch_latency") is not None]
    return {
        "stage": "mode_summary",
        "mode": mode,
        "count": len(records),
        "ok": len(oks),
        "errors": len(records) - len(oks),
        "total_ms_avg": round(statistics.mean(total), 1) if total else None,
        "total_ms_p50": _percentile(total, 0.50),
        "total_ms_p95": _percentile(total, 0.95),
        "ttft_ms_avg": round(statistics.mean(ttft), 1) if ttft else None,
        "ttft_ms_p50": _percentile(ttft, 0.50),
        "ttft_ms_p95": _percentile(ttft, 0.95),
        "max_inter_chunk_gap_ms_p95": _percentile(gaps, 0.95),
        "prefill_launch_latency_avg_s": round(statistics.mean(prefill), 3) if prefill else None,
        "prefill_launch_latency_p95_s": _percentile(prefill, 0.95),
    }


async def run_mode(mode: str, args: argparse.Namespace, urls: list[str], cases: dict[str, dict[str, Any]], out: Any) -> None:
    case = cases["image"] if "image" in mode else cases["text"]
    print(
        f"[verl-path] mode={mode} concurrency={args.concurrency} requests={args.requests_per_mode} "
        f"turns={args.turns} image_count={case['image_count']} server_len={case['server_prompt_len']} "
        f"expanded_len={case['expanded_prompt_len']} surplus={case['mm_prefix_surplus']}",
        flush=True,
    )
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(index: int) -> dict[str, Any]:
        async with sem:
            return await run_one(
                url=urls[index % len(urls)],
                mode=mode,
                index=index,
                case=case,
                args=args,
            )

    tasks = [asyncio.create_task(guarded(index)) for index in range(args.requests_per_mode)]
    records: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(
            f"[verl-path] {record['mode']} {record['index']:04d} {record['status']} "
            f"ttft={record.get('ttft_ms')} total={record.get('total_ms')} "
            f"gap={record.get('max_inter_chunk_gap_ms')} prefill={record.get('prefill_launch_latency')} "
            f"url={record['url']}",
            flush=True,
        )

    summary = summarize(mode, records)
    out.write(json.dumps(summary, ensure_ascii=False) + "\n")
    out.flush()
    print(f"[verl-path] summary {json.dumps(summary, ensure_ascii=False)}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    urls = _normalise_urls(args.urls)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    allowed = {
        "native_image_student",
        "pretokenized_image_student",
        "native_text_student",
        "pretokenized_text_student",
        "native_image_teacher",
        "pretokenized_image_teacher",
        "native_text_teacher",
        "pretokenized_text_teacher",
    }
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Unsupported mode: {mode}")

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    processor = hf_processor(args.model_path, trust_remote_code=True)
    cases = {
        "image": _build_case(args, tokenizer, processor, with_image=True),
        "text": _build_case(args, tokenizer, processor, with_image=False),
    }

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "stage": "config",
                    "urls": urls,
                    "modes": modes,
                    "concurrency": args.concurrency,
                    "requests_per_mode": args.requests_per_mode,
                    "turns": args.turns,
                    "max_new_tokens": args.max_new_tokens,
                    "top_logprobs_num": args.top_logprobs_num,
                    "stream": bool(args.stream),
                    "image_case": {k: cases["image"][k] for k in ("image_count", "server_prompt_len", "expanded_prompt_len", "mm_prefix_surplus")},
                    "text_case": {k: cases["text"][k] for k in ("image_count", "server_prompt_len", "expanded_prompt_len", "mm_prefix_surplus")},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for mode in modes:
            await run_mode(mode, args, urls, cases, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the verl SGLang multimodal request contract.")
    parser.add_argument("--urls", required=True, help="Comma-separated SGLang base URLs.")
    parser.add_argument("--model-path", default="/home/sogang_nlpy/verl/models/Qwen3.5-9B")
    parser.add_argument("--dataset-path", default="/home/sogang_nlpy/verl/data/webgym_rl_counter/train.parquet")
    parser.add_argument("--tool-config-path", default="/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml")
    parser.add_argument(
        "--modes",
        default="native_image_student,pretokenized_image_student,native_text_student,pretokenized_text_student,pretokenized_image_teacher",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests-per-mode", type=int, default=16)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-suffix-tokens", type=int, default=64)
    parser.add_argument("--teacher-suffix-text", default=" <tool_call>{\"name\":\"CLICK\",\"arguments\":{\"x\":640,\"y\":345}}</tool_call>")
    parser.add_argument("--top-logprobs-num", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-jsonl", default="logs/sglang_replay_bench/verl_path_replay_results.jsonl")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
