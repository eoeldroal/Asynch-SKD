from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request
from uuid import uuid4

import pandas as pd
import yaml

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.tools.web_osgym_tool import WebOsGymTool
from verl.utils import hf_tokenizer
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


def _image_to_data_uri(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _load_dataset_prompt(path: str) -> list[dict[str, Any]]:
    row = pd.read_parquet(path).iloc[0]
    return [dict(item) for item in row["prompt"].tolist()]


def _load_tool_config(
    path: str,
    *,
    base_url: str | None,
    include_a11y: bool,
) -> tuple[list[dict[str, Any]], list[OpenAIFunctionToolSchema], dict[str, WebOsGymTool]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    prompt_schemas: list[dict[str, Any]] = []
    parser_schemas: list[OpenAIFunctionToolSchema] = []
    tools: dict[str, WebOsGymTool] = {}
    for item in config.get("tools", []):
        tool_schema = OpenAIFunctionToolSchema(**item["tool_schema"])
        tool_config = dict(item["config"])
        if base_url:
            tool_config["base_url"] = base_url
        tool_config["include_a11y"] = include_a11y
        prompt_schemas.append(tool_schema.model_dump())
        parser_schemas.append(tool_schema)
        tools[tool_schema.function.name] = WebOsGymTool(config=tool_config, tool_schema=tool_schema)
    if not tools:
        raise ValueError(f"No tools loaded from {path}")
    return prompt_schemas, parser_schemas, tools


def _student_tool_message(tool_response: ToolResponse, *, include_text_with_image: bool) -> tuple[dict[str, Any] | None, list[Any]]:
    images = list(tool_response.image or [])
    text = tool_response.text or ""
    if images:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        if include_text_with_image and text:
            content.append({"type": "text", "text": text})
        return {"role": "tool", "content": content}, images
    if text:
        return {"role": "tool", "content": text}, []
    return None, []


def _tool_call_text(tool_call: FunctionCall) -> str:
    try:
        arguments = json.loads(tool_call.arguments)
    except Exception:
        arguments = {}
    return (
        "<tool_call>\n"
        + json.dumps({"name": tool_call.name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def _fallback_tool_call(args: argparse.Namespace) -> FunctionCall:
    arguments = {
        "x": args.fallback_click_x,
        "y": args.fallback_click_y,
        "button": "left",
        "num_clicks": 1,
    }
    return FunctionCall(name="CLICK", arguments=json.dumps(arguments, ensure_ascii=False))


def _extract_output_ids(event: dict[str, Any]) -> list[int] | None:
    inner = event.get("response")
    if isinstance(inner, dict):
        event = inner
    for key in ("output_ids", "token_ids"):
        value = event.get(key)
        if isinstance(value, list):
            return [int(item) for item in value]
    return None


def _extract_meta(event: dict[str, Any]) -> dict[str, Any]:
    inner = event.get("response")
    if isinstance(inner, dict):
        event = inner
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
                output_ids = _extract_output_ids(event)
                if output_ids is not None:
                    last_output_ids = output_ids
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url}/generate failed with HTTP {exc.code}: {body}") from exc

    total_ms = _ms_since(started)
    gaps = [
        round((chunk_times[index] - chunk_times[index - 1]) * 1000, 1)
        for index in range(1, len(chunk_times))
    ]
    meta = _extract_meta(last_event)
    output_ids = last_output_ids or []
    return {
        "total_ms": total_ms,
        "ttft_ms": first_chunk_ms,
        "event_count": len(chunk_times),
        "max_inter_chunk_gap_ms": max(gaps) if gaps else None,
        "avg_inter_chunk_gap_ms": round(statistics.mean(gaps), 1) if gaps else None,
        "output_ids": output_ids,
        "output_ids_len": len(output_ids),
        "tokens_per_sec": round(len(output_ids) / (total_ms / 1000), 2) if total_ms > 0 else None,
        "finish_reason": meta.get("finish_reason"),
        "queue_time": meta.get("queue_time"),
        "prefill_launch_delay": meta.get("prefill_launch_delay"),
        "prefill_launch_latency": meta.get("prefill_launch_latency"),
    }


def _build_payload(
    *,
    mode: str,
    request_id: str,
    messages: list[dict[str, Any]],
    images: list[Any],
    tools: list[dict[str, Any]],
    parser_tools: list[OpenAIFunctionToolSchema],
    tokenizer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_data = [_image_to_data_uri(image) for image in images]
    template_started = _now()
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
    template_ms = _ms_since(template_started)

    payload: dict[str, Any] = {
        "rid": request_id,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
        },
        "stream": True,
    }
    if mode == "native_image":
        payload["text"] = raw_prompt
    elif mode == "pretokenized_image":
        payload["input_ids"] = server_prompt_ids
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if image_data:
        payload["image_data"] = image_data

    return payload, {
        "template_ms": template_ms,
        "raw_prompt_len": len(raw_prompt),
        "server_prompt_len": len(server_prompt_ids),
        "image_count": len(image_data),
    }


@dataclass
class AgentStub:
    request_id: str
    extra_fields: dict[str, Any] = field(default_factory=dict)


def _restore_all_tools(tools: dict[str, WebOsGymTool], instance_id: str, extra_fields: dict[str, Any]) -> None:
    for tool in tools.values():
        tool.restore_instance(
            instance_id,
            task_id=extra_fields["web_osgym_task_id"],
            request_id=extra_fields["web_osgym_session_id"],
            include_a11y=extra_fields["web_osgym_include_a11y"],
            reward=extra_fields.get("web_osgym_reward_score"),
            cursor_x=extra_fields.get("web_osgym_cursor_x"),
            cursor_y=extra_fields.get("web_osgym_cursor_y"),
        )


async def _start_session(
    *,
    tools: dict[str, WebOsGymTool],
    args: argparse.Namespace,
    session_index: int,
) -> tuple[str, ToolResponse, dict[str, Any], float]:
    tool = next(iter(tools.values()))
    session_id = uuid4().int % (2**31 - 1) or 1
    started = _now()
    instance_id, response = await tool.create(
        task_id=args.task_id,
        request_id=session_id,
        include_a11y=args.include_a11y,
    )
    elapsed_ms = _ms_since(started)
    extra_fields = {
        "web_osgym_instance_id": instance_id,
        "web_osgym_task_id": args.task_id,
        "web_osgym_session_id": session_id,
        "web_osgym_include_a11y": args.include_a11y,
        "bench_session_index": session_index,
    }
    _restore_all_tools(tools, instance_id, extra_fields)
    return instance_id, response, extra_fields, elapsed_ms


async def _execute_tool_call(
    *,
    tool_call: FunctionCall,
    tools: dict[str, WebOsGymTool],
    instance_id: str,
    agent: AgentStub,
) -> tuple[ToolResponse, dict[str, Any], float, bool]:
    tool = tools.get(tool_call.name)
    if tool is None:
        return ToolResponse(text=f"Unknown function '{tool_call.name}'"), {
            "terminated": False,
            "termination_reason": None,
            "action_count": 0,
            "invalid_action": True,
        }, 0.0, False
    try:
        parameters = json.loads(tool_call.arguments)
    except Exception:
        parameters = {}
    started = _now()
    response, _, result = await tool.execute(instance_id, parameters, agent_data=agent)
    return response, result, _ms_since(started), True


async def run_session(
    *,
    mode: str,
    session_index: int,
    url: str,
    base_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_objects: dict[str, WebOsGymTool],
    tokenizer: Any,
    tool_parser: ToolParser,
    args: argparse.Namespace,
    out: Any,
) -> None:
    instance_id, start_response, extra_fields, start_ms = await _start_session(
        tools=tool_objects,
        args=args,
        session_index=session_index,
    )
    agent = AgentStub(request_id=f"webgym-loop-{mode}-{session_index:04d}", extra_fields=extra_fields)

    messages = [dict(item) for item in base_messages]
    images: list[Any] = []
    start_message, start_images = _student_tool_message(
        start_response,
        include_text_with_image=args.include_text_with_image,
    )
    if start_message is not None:
        messages.append(start_message)
        images.extend(start_images)

    out.write(
        json.dumps(
            {
                "stage": "session_start",
                "mode": mode,
                "session_index": session_index,
                "url": url,
                "start_ms": start_ms,
                "start_text_len": len(start_response.text or ""),
                "start_image_count": len(start_images),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    out.flush()

    terminated = False
    for turn in range(1, args.turns + 1):
        request_id = f"webgymloop-{mode}-s{session_index:04d}-t{turn}-{uuid4().hex[:8]}"
        payload, prompt_meta = _build_payload(
            mode=mode,
            request_id=request_id,
            messages=messages,
            images=images,
            tools=tools,
            tokenizer=tokenizer,
            args=args,
        )
        generate = await asyncio.to_thread(_post_stream, url, payload, args.timeout)
        output_ids = generate.pop("output_ids", [])
        parse_started = _now()
        decoded_text, tool_calls = await tool_parser.extract_tool_calls(output_ids, parser_tools)
        parse_ms = _ms_since(parse_started)

        selected_call: FunctionCall | None = tool_calls[0] if tool_calls else None
        used_fallback = False
        if args.always_use_fallback_action or selected_call is None or selected_call.name not in tool_objects:
            selected_call = _fallback_tool_call(args)
            used_fallback = True

        messages.append({"role": "assistant", "content": _tool_call_text(selected_call)})
        tool_response, tool_result, tool_ms, tool_known = await _execute_tool_call(
            tool_call=selected_call,
            tools=tool_objects,
            instance_id=instance_id,
            agent=agent,
        )
        tool_message, new_images = _student_tool_message(
            tool_response,
            include_text_with_image=args.include_text_with_image,
        )
        if tool_message is not None:
            messages.append(tool_message)
            images.extend(new_images)

        record = {
            "stage": "turn_result",
            "mode": mode,
            "session_index": session_index,
            "turn": turn,
            "url": url,
            "request_id": request_id,
            **prompt_meta,
            **generate,
            "parse_ms": parse_ms,
            "decoded_text_len": len(decoded_text or ""),
            "parsed_tool_calls": len(tool_calls),
            "selected_tool": selected_call.name,
            "used_fallback": used_fallback,
            "tool_known": tool_known,
            "tool_ms": tool_ms,
            "tool_action_count": tool_result.get("action_count", 0),
            "tool_invalid_action": bool(tool_result.get("invalid_action", False)),
            "tool_terminated": bool(tool_result.get("terminated", False)),
            "tool_termination_reason": tool_result.get("termination_reason"),
            "tool_text_len": len(tool_response.text or ""),
            "tool_image_count": len(new_images),
            "accumulated_image_count": len(images),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        print(
            f"[webgym-loop] {mode} s={session_index:04d} t={turn} "
            f"ttft={record['ttft_ms']} total={record['total_ms']} "
            f"tool={record['tool_ms']} images={record['accumulated_image_count']} "
            f"fallback={record['used_fallback']}",
            flush=True,
        )

        if tool_result.get("terminated"):
            terminated = True
            break

    reward = None
    try:
        reward = await tool_objects[next(iter(tool_objects))].calc_reward(instance_id)
    except Exception as exc:
        out.write(
            json.dumps(
                {
                    "stage": "reward_error",
                    "mode": mode,
                    "session_index": session_index,
                    "error": repr(exc),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out.flush()
    out.write(
        json.dumps(
            {
                "stage": "session_done",
                "mode": mode,
                "session_index": session_index,
                "terminated": terminated,
                "reward": reward,
                "final_image_count": len(images),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    out.flush()


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


async def run_mode(
    *,
    mode: str,
    urls: list[str],
    base_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    parser_tools: list[OpenAIFunctionToolSchema],
    tokenizer: Any,
    tool_parser: ToolParser,
    args: argparse.Namespace,
    out: Any,
) -> None:
    print(f"[webgym-loop] mode={mode} sessions={args.sessions} turns={args.turns}", flush=True)
    sem = asyncio.Semaphore(args.sessions)
    records_start_pos = out.tell()

    async def guarded(index: int) -> None:
        async with sem:
            _, _, tool_objects = _load_tool_config(
                args.tool_config_path,
                base_url=args.webgym_base_url,
                include_a11y=args.include_a11y,
            )
            await run_session(
                mode=mode,
                session_index=index,
                url=urls[index % len(urls)],
                base_messages=base_messages,
                tools=tools,
                parser_tools=parser_tools,
                tool_objects=tool_objects,
                tokenizer=tokenizer,
                tool_parser=tool_parser,
                args=args,
                out=out,
            )

    await asyncio.gather(*(guarded(index) for index in range(args.sessions)))

    out.flush()
    out.seek(records_start_pos)
    turn_records = []
    for line in out:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if item.get("stage") == "turn_result" and item.get("mode") == mode:
            turn_records.append(item)
    out.seek(0, 2)

    for turn in range(1, args.turns + 1):
        subset = [item for item in turn_records if item["turn"] == turn]
        if not subset:
            continue
        totals = [float(item["total_ms"]) for item in subset if item.get("total_ms") is not None]
        ttfts = [float(item["ttft_ms"]) for item in subset if item.get("ttft_ms") is not None]
        tools_ms = [float(item["tool_ms"]) for item in subset if item.get("tool_ms") is not None]
        summary = {
            "stage": "turn_summary",
            "mode": mode,
            "turn": turn,
            "count": len(subset),
            "total_ms_avg": round(statistics.mean(totals), 1) if totals else None,
            "total_ms_p95": _percentile(totals, 0.95),
            "ttft_ms_avg": round(statistics.mean(ttfts), 1) if ttfts else None,
            "ttft_ms_p95": _percentile(ttfts, 0.95),
            "tool_ms_avg": round(statistics.mean(tools_ms), 1) if tools_ms else None,
            "tool_ms_p95": _percentile(tools_ms, 0.95),
        }
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"[webgym-loop] summary {json.dumps(summary, ensure_ascii=False)}", flush=True)
    out.flush()


async def main_async(args: argparse.Namespace) -> None:
    urls = _normalise_urls(args.urls)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    allowed_modes = {"pretokenized_image", "native_image"}
    for mode in modes:
        if mode not in allowed_modes:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(allowed_modes)}")

    tokenizer = hf_tokenizer(args.model_path, trust_remote_code=True)
    tool_parser = ToolParser.get_tool_parser(args.tool_parser, tokenizer)
    base_messages = _load_dataset_prompt(args.dataset_path)
    tools, parser_tools, _ = _load_tool_config(
        args.tool_config_path,
        base_url=args.webgym_base_url,
        include_a11y=args.include_a11y,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w+", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {
                    "stage": "config",
                    "urls": urls,
                    "modes": modes,
                    "sessions": args.sessions,
                    "turns": args.turns,
                    "task_id": args.task_id,
                    "include_a11y": args.include_a11y,
                    "include_text_with_image": args.include_text_with_image,
                    "always_use_fallback_action": args.always_use_fallback_action,
                    "max_new_tokens": args.max_new_tokens,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        out.flush()
        for mode in modes:
            await run_mode(
                mode=mode,
                urls=urls,
                base_messages=base_messages,
                tools=tools,
                parser_tools=parser_tools,
                tokenizer=tokenizer,
                tool_parser=tool_parser,
                args=args,
                out=out,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-SGLang + real WebGym loop latency bench.")
    parser.add_argument("--urls", required=True, help="Comma-separated SGLang base URLs.")
    parser.add_argument("--webgym-base-url", default="http://127.0.0.1:18001")
    parser.add_argument("--model-path", default="/home/sogang_nlpy/verl/models/Qwen3.5-9B")
    parser.add_argument("--dataset-path", default="/home/sogang_nlpy/verl/data/webgym_rl_counter/train.parquet")
    parser.add_argument("--tool-config-path", default="/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config.yaml")
    parser.add_argument("--modes", default="pretokenized_image,native_image")
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--task-id", default="counter")
    parser.add_argument("--include-a11y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-text-with-image", action="store_true")
    parser.add_argument("--always-use-fallback-action", action="store_true")
    parser.add_argument("--fallback-click-x", type=int, default=696)
    parser.add_argument("--fallback-click-y", type=int, default=475)
    parser.add_argument("--tool-parser", default="qwen3_coder")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-jsonl", default="logs/sglang_replay_bench/webgym_fixed_server_loop.jsonl")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
