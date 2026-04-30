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

from PIL import Image, ImageDraw


def _now() -> float:
    return time.perf_counter()


def _ms_since(start: float) -> float:
    return round((_now() - start) * 1000, 1)


def _normalise_urls(raw: str) -> list[str]:
    urls = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not urls:
        raise ValueError("--urls must contain at least one URL")
    return urls


def _make_screenshot_data_uri(width: int = 1280, height: int = 720) -> str:
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 56), fill=(36, 41, 54))
    draw.text((24, 18), "WebGym Counter Replay", fill=(255, 255, 255))
    draw.rectangle((80, 130, width - 80, height - 90), outline=(148, 163, 184), width=3)
    draw.text((120, 170), "Counter task", fill=(15, 23, 42))
    for idx in range(6):
        x0 = 120 + idx * 170
        y0 = 260
        draw.rounded_rectangle((x0, y0, x0 + 120, y0 + 76), radius=8, fill=(255, 255, 255), outline=(71, 85, 105))
        draw.text((x0 + 28, y0 + 28), f"Button {idx + 1}", fill=(30, 41, 59))
    draw.ellipse((676, 455, 716, 495), outline=(220, 38, 38), width=4)
    draw.line((696, 440, 696, 510), fill=(220, 38, 38), width=2)
    draw.line((661, 475, 731, 475), fill=(220, 38, 38), width=2)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _tool_observation(turn: int, *, image: bool) -> str:
    a11y = "\n".join(
        [
            f"[{idx}] role=button name='Counter option {idx}' bounds=({120 + idx * 40},{220 + idx * 7},120,40)"
            for idx in range(1, 18)
        ]
    )
    visual = "<image>\n" if image else "[screenshot omitted]\n"
    return (
        f"{visual}"
        f"Observation turn {turn}: the browser shows a counter task. "
        "The target is to click the control that increments the counter and then finish when the desired value is visible.\n"
        "Accessibility tree excerpt:\n"
        f"{a11y}\n"
        "Previous action status: success. Continue from the current screen."
    )


def _filler(turn: int, repeat: int) -> str:
    unit = (
        f"Turn {turn} context note: preserve the task goal, inspect the current screen, "
        "choose precise coordinates, and avoid irrelevant controls. "
    )
    return unit * repeat


def build_prompt(mode: str, *, turns: int, filler_repeat: int) -> tuple[str, list[str] | None]:
    use_images = mode == "image_multiturn"
    image_data = [_make_screenshot_data_uri() for _ in range(turns)] if use_images else None

    chunks = [
        "<|im_start|>system\n"
        "You are a web automation agent. Use the visual observation and the task text to decide the next action. "
        "Respond briefly with the next intended action.\n"
        "<|im_end|>\n",
        "<|im_start|>user\n"
        "Task: In the counter web page, reach the target counter value using the visible controls. "
        "Use the screenshot or observation history if available.\n"
        "<|im_end|>\n",
    ]
    for turn in range(1, turns + 1):
        chunks.append(
            "<|im_start|>assistant\n"
            f"I will inspect turn {turn} and choose the next counter action.\n"
            "<|im_end|>\n"
        )
        chunks.append(
            "<|im_start|>tool\n"
            f"{_tool_observation(turn, image=use_images)}\n"
            f"{_filler(turn, filler_repeat)}\n"
            "<|im_end|>\n"
        )
    chunks.append("<|im_start|>assistant\n")
    return "".join(chunks), image_data


def _unwrap_response(event: dict[str, Any]) -> dict[str, Any]:
    inner = event.get("response")
    if isinstance(inner, dict):
        merged = dict(inner)
        for key, value in event.items():
            merged.setdefault(key, value)
        return merged
    return event


def _extract_text(event: dict[str, Any]) -> str:
    event = _unwrap_response(event)
    value = event.get("text", event.get("output", ""))
    return "" if value is None else str(value)


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
    last_text = ""
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
                text = _extract_text(event)
                if text:
                    last_text = text
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
    return {
        "ttft_ms": first_chunk_ms,
        "total_ms": total_ms,
        "event_count": len(chunk_times),
        "max_inter_chunk_gap_ms": max(gaps_ms) if gaps_ms else None,
        "avg_inter_chunk_gap_ms": round(statistics.mean(gaps_ms), 1) if gaps_ms else None,
        "output_text_chars": len(last_text),
        "output_tokens": output_tokens,
        "tokens_per_sec": round(output_tokens / (total_ms / 1000), 2) if output_tokens and total_ms > 0 else None,
        "meta_info": _extract_meta(last_event),
    }


async def run_one(
    *,
    url: str,
    mode: str,
    index: int,
    args: argparse.Namespace,
    prompt: str,
    image_data: list[str] | None,
) -> dict[str, Any]:
    rid = f"replay-{mode}-c{args.concurrency}-r{index:04d}-{uuid4().hex[:8]}"
    payload: dict[str, Any] = {
        "rid": rid,
        "text": prompt,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        "stream": True,
    }
    if image_data is not None:
        payload["image_data"] = image_data
    try:
        result = await asyncio.to_thread(_post_stream, url, payload, args.timeout)
        status = "ok"
        error_msg = None
    except Exception as exc:
        result = {}
        status = "error"
        error_msg = repr(exc)
    record = {
        "stage": "request_result",
        "status": status,
        "error": error_msg,
        "mode": mode,
        "request_id": rid,
        "url": url,
        "index": index,
        "concurrency": args.concurrency,
        "stream": True,
        "turns": args.turns,
        "image_count": len(image_data or []),
        "prompt_chars": len(prompt),
        "max_new_tokens": args.max_new_tokens,
        **result,
    }
    return record


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
    ttft = [float(item["ttft_ms"]) for item in oks if item.get("ttft_ms") is not None]
    total = [float(item["total_ms"]) for item in oks if item.get("total_ms") is not None]
    gaps = [float(item["max_inter_chunk_gap_ms"]) for item in oks if item.get("max_inter_chunk_gap_ms") is not None]
    return {
        "stage": "mode_summary",
        "mode": mode,
        "count": len(records),
        "ok": len(oks),
        "errors": len(records) - len(oks),
        "ttft_ms_avg": round(statistics.mean(ttft), 1) if ttft else None,
        "ttft_ms_p50": _percentile(ttft, 0.50),
        "ttft_ms_p95": _percentile(ttft, 0.95),
        "total_ms_avg": round(statistics.mean(total), 1) if total else None,
        "total_ms_p50": _percentile(total, 0.50),
        "total_ms_p95": _percentile(total, 0.95),
        "max_inter_chunk_gap_ms_p95": _percentile(gaps, 0.95),
    }


async def run_mode(mode: str, args: argparse.Namespace, urls: list[str], out_handle: Any) -> None:
    prompt, image_data = build_prompt(mode, turns=args.turns, filler_repeat=args.filler_repeat)
    print(
        f"[replay] mode={mode} concurrency={args.concurrency} requests={args.requests_per_mode} "
        f"turns={args.turns} image_count={len(image_data or [])} prompt_chars={len(prompt)}",
        flush=True,
    )
    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(index: int) -> dict[str, Any]:
        async with sem:
            url = urls[index % len(urls)]
            return await run_one(url=url, mode=mode, index=index, args=args, prompt=prompt, image_data=image_data)

    tasks = [asyncio.create_task(guarded(index)) for index in range(args.requests_per_mode)]
    records: list[dict[str, Any]] = []
    for task in asyncio.as_completed(tasks):
        record = await task
        records.append(record)
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_handle.flush()
        print(
            f"[replay] {record['mode']} {record['index']:04d} {record['status']} "
            f"ttft={record.get('ttft_ms')} total={record.get('total_ms')} url={record['url']}",
            flush=True,
        )

    summary = summarize(mode, records)
    out_handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    out_handle.flush()
    print(f"[replay] summary {json.dumps(summary, ensure_ascii=False)}", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    urls = _normalise_urls(args.urls)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    for mode in modes:
        if mode not in {"text_multiturn", "image_multiturn"}:
            raise ValueError(f"Unsupported mode: {mode}")
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as out_handle:
        config_record = {
            "stage": "config",
            "urls": urls,
            "modes": modes,
            "concurrency": args.concurrency,
            "requests_per_mode": args.requests_per_mode,
            "turns": args.turns,
            "stream": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        }
        out_handle.write(json.dumps(config_record, ensure_ascii=False) + "\n")
        for mode in modes:
            await run_mode(mode, args, urls, out_handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay text/image multi-turn SGLang streaming requests.")
    parser.add_argument("--urls", required=True, help="Comma-separated SGLang base URLs.")
    parser.add_argument("--modes", default="text_multiturn,image_multiturn")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests-per-mode", type=int, default=16)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--stream", action="store_true", help="Kept explicit for command readability; this bench always streams.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--filler-repeat", type=int, default=28)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-jsonl", default="logs/sglang_replay_bench/stream_replay_results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stream:
        raise ValueError("This benchmark is intentionally stream=True only. Pass --stream.")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
