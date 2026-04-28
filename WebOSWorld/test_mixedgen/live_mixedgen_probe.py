from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error, request
from uuid import uuid4

from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer

from WebOSWorld.test_mixedgen.observer import JsonlProbeWriter, ObservedStudentManager, ObservedTeacherManager
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.skd_agent_loop import SkdAgentLoop
from verl.experimental.async_skd.state import SkdPartialState
from verl.workers.rollout.replica import TokenOutput


class ConfigWrap:
    def __init__(self, config: Any):
        self.config = config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)


class TextOnlyDataset:
    @staticmethod
    async def process_vision_info(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {}


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"uid", "raw_prompt", "data_source"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing required keys: {sorted(missing)}")
            rows.append(row)
    return rows


def _normalise_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc
    if not raw:
        return {}
    response = json.loads(raw)
    if isinstance(response, list):
        if not response:
            return {}
        response = response[0]
    if not isinstance(response, dict):
        raise TypeError(f"Unexpected SGLang response type from {url}: {type(response).__name__}")
    return response


async def _post_json_async(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    return await asyncio.to_thread(_post_json, url, payload, timeout)


def _post_json_stream(url: str, payload: dict[str, Any], timeout: float, on_event: Any) -> dict[str, Any]:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_event: dict[str, Any] = {}
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
                    raise RuntimeError(f"SGLang stream error from {url}: {event['error']}")
                last_event = event
                on_event(event)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc
    return last_event


async def _post_json_stream_async(url: str, payload: dict[str, Any], timeout: float, on_event: Any) -> dict[str, Any]:
    return await asyncio.to_thread(_post_json_stream, url, payload, timeout, on_event)


def _unwrap_response(response: dict[str, Any]) -> dict[str, Any]:
    inner = response.get("response")
    if isinstance(inner, dict):
        merged = dict(inner)
        for key, value in response.items():
            merged.setdefault(key, value)
        return merged
    return response


def _extract_meta_info(response: dict[str, Any]) -> dict[str, Any]:
    response = _unwrap_response(response)
    meta_info = response.get("meta_info")
    if isinstance(meta_info, dict):
        return meta_info
    inner = response.get("response")
    if isinstance(inner, dict) and isinstance(inner.get("meta_info"), dict):
        return inner["meta_info"]
    return {}


def _extract_output_ids(response: dict[str, Any]) -> list[int] | None:
    response = _unwrap_response(response)
    for key in ("output_ids", "token_ids"):
        token_ids = response.get(key)
        if token_ids is not None:
            return [int(token_id) for token_id in token_ids]
    return None


def _extract_finish_reason(meta_info: dict[str, Any]) -> str | None:
    finish_reason = meta_info.get("finish_reason")
    if isinstance(finish_reason, dict):
        value = finish_reason.get("type") or finish_reason.get("reason")
        return str(value) if value is not None else None
    if finish_reason is not None:
        return str(finish_reason)
    return None


def _sampling_params_for_sglang(sampling_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(sampling_params)
    # SGLang native /generate expects logprob controls at the request top level
    # (for example return_logprob), not inside sampling_params.
    params.pop("logprobs", None)
    if "max_tokens" in params and "max_new_tokens" not in params:
        params["max_new_tokens"] = params.pop("max_tokens")
    return params


class HttpSglangStudentManager:
    def __init__(
        self,
        base_url: str,
        tokenizer: Any,
        timeout: float = 600.0,
        stream: bool = False,
        stream_writer: Any = None,
    ):
        self.base_url = _normalise_url(base_url)
        self.tokenizer = tokenizer
        self.timeout = timeout
        self.stream = stream
        self.stream_writer = stream_writer

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del image_data, video_data, kwargs
        payload = {
            "rid": request_id,
            "input_ids": list(prompt_ids),
            "sampling_params": _sampling_params_for_sglang(sampling_params),
            "stream": self.stream,
        }
        streamed_ids: list[int] = []
        if self.stream:
            def on_event(event: dict[str, Any]) -> None:
                nonlocal streamed_ids
                event = _unwrap_response(event)
                event_ids = _extract_output_ids(event) or []
                if event_ids:
                    if len(event_ids) >= len(streamed_ids) and event_ids[: len(streamed_ids)] == streamed_ids:
                        delta_ids = event_ids[len(streamed_ids) :]
                        streamed_ids = list(event_ids)
                    else:
                        delta_ids = list(event_ids)
                        streamed_ids.extend(delta_ids)
                else:
                    delta_ids = []
                if delta_ids and self.stream_writer is not None:
                    self.stream_writer.write(
                        "student_generate_delta",
                        request_id=request_id,
                        token_ids=delta_ids,
                        cumulative_len=len(streamed_ids),
                        text=self.tokenizer.decode(delta_ids, skip_special_tokens=False),
                    )

            response = await _post_json_stream_async(f"{self.base_url}/generate", payload, self.timeout, on_event)
            if streamed_ids:
                response = dict(_unwrap_response(response))
                response["output_ids"] = streamed_ids
        else:
            response = await _post_json_async(f"{self.base_url}/generate", payload, self.timeout)
        response = _unwrap_response(response)
        meta_info = _extract_meta_info(response)
        token_ids = _extract_output_ids(response)
        stop_reason = _extract_finish_reason(meta_info)
        extra_fields = {"http_url": self.base_url}

        if token_ids is None:
            text = response.get("text")
            if text is None:
                text = response.get("output")
            if text is None:
                raise ValueError("SGLang student response has no output_ids, token_ids, text, or output field")
            token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            stop_reason = "unknown"
            extra_fields["tokenized_text_fallback"] = True

        output_token_logprobs = meta_info.get("output_token_logprobs") or []
        log_probs = None
        if output_token_logprobs and len(output_token_logprobs) == len(token_ids):
            log_probs = [float(entry[0]) for entry in output_token_logprobs]

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields=extra_fields,
        )


def _token_id_from_logprob_entry(entry: Any) -> int:
    if isinstance(entry, dict):
        for key in ("token_id", "id", "token"):
            if key in entry:
                return int(entry[key])
        raise ValueError(f"Cannot extract token id from logprob entry keys={sorted(entry)}")
    return int(entry[1])


def _logprob_from_logprob_entry(entry: Any) -> float:
    if isinstance(entry, dict):
        for key in ("logprob", "log_prob"):
            if key in entry:
                return float(entry[key])
        raise ValueError(f"Cannot extract logprob from logprob entry keys={sorted(entry)}")
    return float(entry[0])


def _top_entries_for_position(input_top_logprobs: Any, position: int) -> list[Any]:
    entries = input_top_logprobs[position]
    if entries is None:
        return []
    if isinstance(entries, dict):
        converted = []
        for token_id, logprob in entries.items():
            converted.append({"token_id": token_id, "logprob": logprob})
        return converted
    return list(entries)


class HttpSglangTeacherManager:
    def __init__(self, base_url: str, loss_top_k: int, timeout: float = 600.0):
        self.base_url = _normalise_url(base_url)
        self.loss_top_k = int(loss_top_k)
        self.timeout = timeout

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: dict[str, Any] | None = None,
        routing_key: str | None = None,
        request_id: str | None = None,
        logprob_start_len: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del multi_modal_data, routing_key
        request_id = request_id or uuid4().hex
        payload = {
            "rid": request_id,
            "input_ids": list(sequence_ids),
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
            },
            "return_logprob": True,
            "logprob_start_len": int(logprob_start_len),
            "stream": False,
        }
        if self.loss_top_k > 0:
            payload["top_logprobs_num"] = self.loss_top_k
        response = await _post_json_async(f"{self.base_url}/generate", payload, self.timeout)
        meta_info = _extract_meta_info(response)
        teacher_ids, teacher_logprobs = self._extract_delta_prompt_logprobs(
            meta_info=meta_info,
            sequence_length=len(sequence_ids),
            logprob_start_len=int(logprob_start_len),
        )
        return teacher_ids, teacher_logprobs

    def _extract_delta_prompt_logprobs(
        self,
        *,
        meta_info: dict[str, Any],
        sequence_length: int,
        logprob_start_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_token_logprobs = meta_info.get("input_token_logprobs") or []
        input_top_logprobs = meta_info.get("input_top_logprobs") or []
        expected_len = sequence_length - logprob_start_len - 1
        if expected_len < 0:
            raise ValueError(f"Invalid logprob_start_len={logprob_start_len} for sequence_length={sequence_length}")

        rows_ids: list[list[int]] = []
        rows_logprobs: list[list[float]] = []
        for position, raw_top_entries in enumerate(input_top_logprobs):
            if raw_top_entries is None:
                continue
            top_entries = _top_entries_for_position(input_top_logprobs, position)
            if len(top_entries) != self.loss_top_k:
                raise ValueError(
                    f"SGLang teacher returned {len(top_entries)} top logprobs at position {position}, "
                    f"expected {self.loss_top_k}."
                )
            rows_ids.append([_token_id_from_logprob_entry(entry) for entry in top_entries])
            rows_logprobs.append([_logprob_from_logprob_entry(entry) for entry in top_entries])

        if len(rows_ids) != expected_len or len(rows_logprobs) != expected_len:
            raise ValueError(
                f"Unexpected teacher delta rows: ids={len(rows_ids)}, logprobs={len(rows_logprobs)}, "
                f"expected={expected_len}, sequence_length={sequence_length}, start={logprob_start_len}, "
                f"input_token_logprobs={len(input_token_logprobs)}, input_top_logprobs={len(input_top_logprobs)}."
            )
        return torch.tensor(rows_ids, dtype=torch.int32), torch.tensor(rows_logprobs, dtype=torch.float32)

    async def release_sticky_session(self, request_id: str) -> None:
        del request_id
        return None


def make_loop_config(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {}},
            "actor_rollout_ref": {
                "model": {"path": args.student_model, "trust_remote_code": True},
                "rollout": {
                    "name": "sglang",
                    "prompt_length": args.max_prompt,
                    "response_length": args.max_response,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "min_p": args.min_p,
                    "presence_penalty": args.presence_penalty,
                    "repetition_penalty": args.repetition_penalty,
                    "calculate_log_probs": False,
                    "agent": {"default_agent_loop": "skd_agent"},
                    "multi_turn": {
                        "enable": False,
                        "max_assistant_turns": None,
                        "max_user_turns": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 256,
                        "tool_response_truncate_side": "middle",
                        "tool_config_path": None,
                        "format": "qwen3_coder",
                    },
                },
            },
            "distillation": {
                "enabled": True,
                "teacher_key": "data_source",
                "skd": {
                    "chunk_size": args.chunk_size,
                    "verify_top_k": args.verify_top_k,
                    "max_chunks_per_sample": args.max_chunks,
                    "teacher_system_prompt_path": args.teacher_system_prompt_path,
                },
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk",
                    "topk": args.loss_top_k,
                    "use_task_rewards": False,
                    "use_policy_gradient": False,
                },
            },
        }
    )


def make_sampling_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_response,
    }


def build_loop(
    args: argparse.Namespace,
    *,
    student_manager: Any,
    teacher_manager: Any,
    tokenizer: Any,
) -> SkdAgentLoop:
    config = make_loop_config(args)
    loop = SkdAgentLoop.__new__(SkdAgentLoop)
    try:
        SkdAgentLoop.__init__(
            loop,
            trainer_config=ConfigWrap(config),
            server_manager=student_manager,
            teacher_server_manager=teacher_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=TextOnlyDataset,
            data_config=ConfigWrap(config.data),
        )
    except Exception:
        SkdAgentLoop.__init__(
            loop,
            trainer_config=SimpleNamespace(config=config, get=config.get),
            server_manager=student_manager,
            teacher_server_manager=teacher_manager,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=TextOnlyDataset,
            data_config=ConfigWrap({"apply_chat_template_kwargs": {}}),
        )

    async def process_vision_info(messages: list[dict[str, Any]]) -> dict[str, Any]:
        del messages
        return {}

    loop.process_vision_info = process_vision_info
    return loop


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    return str(value)


def output_record(result: AgentLoopOutput, tokenizer: Any) -> dict[str, Any]:
    response_ids = list(result.response_ids)
    return {
        "response_ids": response_ids,
        "decoded_skip_special": tokenizer.decode(response_ids, skip_special_tokens=True),
        "decoded_with_special": tokenizer.decode(response_ids, skip_special_tokens=False),
        "metrics": _jsonable(result.metrics),
        "extra_fields": _jsonable(result.extra_fields),
    }


async def run_direct(
    *,
    loop: SkdAgentLoop,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    writer: JsonlProbeWriter,
    tokenizer: Any,
) -> None:
    sampling_params = make_sampling_params(args)
    for row in rows[: args.num_direct]:
        result = await loop.run(
            sampling_params,
            raw_prompt=row["raw_prompt"],
            data_source=row.get("data_source", "default"),
        )
        writer.write(
            "final_result",
            request_id=None,
            sample_id=row["uid"],
            mode="direct",
            teacher_model=args.teacher_model,
            **output_record(result, tokenizer),
        )


async def run_boundary(
    *,
    loop: SkdAgentLoop,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    writer: JsonlProbeWriter,
    tokenizer: Any,
) -> None:
    sampling_params = make_sampling_params(args)
    for row in rows[: args.num_direct]:
        result = await loop.run_until_exportable_boundary(
            sampling_params,
            sample_id=row["uid"],
            logical_step=1,
            source_type="lookahead",
            raw_prompt=row["raw_prompt"],
            data_source=row.get("data_source", "default"),
        )
        if isinstance(result, SkdPartialState):
            writer.write(
                "boundary_result",
                request_id=result.request_id,
                sample_id=row["uid"],
                result_type="SkdPartialState",
                partial_state=_jsonable(result),
            )
            result = await loop.run_from_partial_to_completion(
                sampling_params,
                partial_state=result,
            )
        else:
            writer.write(
                "boundary_result",
                request_id=None,
                sample_id=row["uid"],
                result_type=type(result).__name__,
                result=output_record(result, tokenizer),
            )

        writer.write(
            "final_result",
            request_id=None,
            sample_id=row["uid"],
            mode="boundary",
            teacher_model=args.teacher_model,
            **output_record(result, tokenizer),
        )


async def run_mixed(
    *,
    loop: SkdAgentLoop,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    writer: JsonlProbeWriter,
    tokenizer: Any,
) -> None:
    """Exercise the carryover-plus-fresh shape without importing trainer/Ray state.

    This is a test-only sequential probe: it first exports one unfinished SKD
    trajectory at an SKD boundary, then completes that carryover while also
    running fresh prompts in the same process.  The production async scheduler
    still owns queueing and batching; this mode keeps the live probe small and
    focused on the SKD loop contract.
    """
    if not rows:
        return

    sampling_params = make_sampling_params(args)
    carryover_row = rows[0]
    writer.write(
        "mixed_current_start",
        request_id=None,
        carryover_sample_id=carryover_row["uid"],
        fresh_sample_ids=[row["uid"] for row in rows[1 : args.num_direct]],
    )

    boundary_result = await loop.run_until_exportable_boundary(
        sampling_params,
        sample_id=carryover_row["uid"],
        logical_step=1,
        source_type="lookahead",
        raw_prompt=carryover_row["raw_prompt"],
        data_source=carryover_row.get("data_source", "default"),
    )
    if isinstance(boundary_result, SkdPartialState):
        writer.write(
            "mixed_current_partial",
            request_id=boundary_result.request_id,
            sample_id=carryover_row["uid"],
            partial_state=_jsonable(boundary_result),
        )
        carryover_final = await loop.run_from_partial_to_completion(
            sampling_params,
            partial_state=boundary_result,
        )
    else:
        writer.write(
            "mixed_current_completed_without_partial",
            request_id=None,
            sample_id=carryover_row["uid"],
            result=output_record(boundary_result, tokenizer),
        )
        carryover_final = boundary_result

    writer.write(
        "final_result",
        request_id=None,
        sample_id=carryover_row["uid"],
        mode="mixed_carryover",
        teacher_model=args.teacher_model,
        **output_record(carryover_final, tokenizer),
    )

    for row in rows[1 : args.num_direct]:
        fresh_result = await loop.run(
            sampling_params,
            raw_prompt=row["raw_prompt"],
            data_source=row.get("data_source", "default"),
        )
        writer.write(
            "final_result",
            request_id=None,
            sample_id=row["uid"],
            mode="mixed_fresh",
            teacher_model=args.teacher_model,
            **output_record(fresh_result, tokenizer),
        )

    writer.write("mixed_current_finish", request_id=None)


async def main_async(args: argparse.Namespace) -> None:
    writer = JsonlProbeWriter(args.probe_log)
    rows = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student = ObservedStudentManager(
        HttpSglangStudentManager(
            args.student_url,
            tokenizer,
            timeout=args.http_timeout,
            stream=args.student_stream,
            stream_writer=writer,
        ),
        writer,
    )
    teacher = ObservedTeacherManager(
        HttpSglangTeacherManager(args.teacher_url, args.loss_top_k, timeout=args.http_timeout),
        writer,
    )
    loop = build_loop(args, student_manager=student, teacher_manager=teacher, tokenizer=tokenizer)
    writer.write(
        "probe_start",
        request_id=None,
        mode=args.mode,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        student_url=args.student_url,
        teacher_url=args.teacher_url,
        teacher_system_prompt_path=args.teacher_system_prompt_path,
        prompt_count=len(rows),
        prefetch_limit=args.prefetch_limit,
        prefetch_worker_target=args.prefetch_worker_target,
        max_prompt=args.max_prompt,
        max_response=args.max_response,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        student_stream=args.student_stream,
    )
    if args.mode == "direct":
        await run_direct(loop=loop, rows=rows, args=args, writer=writer, tokenizer=tokenizer)
    elif args.mode == "boundary":
        await run_boundary(loop=loop, rows=rows, args=args, writer=writer, tokenizer=tokenizer)
    else:
        await run_mixed(loop=loop, rows=rows, args=args, writer=writer, tokenizer=tokenizer)
    writer.write("probe_finish", request_id=None, mode=args.mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual live Async SKD mixed-generation probe.")
    parser.add_argument("--student-model", required=True, help="Student model path or HF id, used for tokenizer.")
    parser.add_argument("--teacher-model", default=None, help="Teacher model path or HF id, recorded as metadata only.")
    parser.add_argument("--prompts", default="WebOSWorld/test_mixedgen/prompts.jsonl")
    parser.add_argument("--probe-log", required=True)
    parser.add_argument("--student-url", default="http://127.0.0.1:31000")
    parser.add_argument("--teacher-url", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--teacher-system-prompt-path",
        default=None,
        help="Optional teacher-only system prompt file used when building the SKD teacher stream.",
    )
    parser.add_argument("--max-prompt", type=int, default=2048)
    parser.add_argument("--max-response", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--verify-top-k", type=int, default=5)
    parser.add_argument("--loss-top-k", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--num-direct", type=int, default=2)
    parser.add_argument("--mode", choices=("direct", "boundary", "mixed"), default="boundary")
    parser.add_argument("--prefetch-limit", type=int, default=2, help="Recorded for parity with async SKD configs.")
    parser.add_argument(
        "--prefetch-worker-target",
        type=int,
        default=1,
        help="Recorded for parity with async SKD configs.",
    )
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--student-stream", action="store_true", help="Use SGLang SSE streaming for student chunks.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
