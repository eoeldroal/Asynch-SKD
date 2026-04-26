import dataclasses
import json
import os
import threading
import time
from collections.abc import Mapping


try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None


_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _jsonable(value):
    if isinstance(value, _PRIMITIVE_TYPES):
        return value

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))

    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())

    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())

    if isinstance(value, Mapping):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


def _safe_len(value):
    try:
        return len(value)
    except Exception:
        return None


def _tail(value, count):
    try:
        return _jsonable(value[-count:])
    except Exception:
        try:
            return _jsonable(list(value)[-count:])
        except Exception:
            return _jsonable(value)


def _field(output, name, default=None):
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _public_extra_fields(output, known_fields):
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        data = dataclasses.asdict(output)
    elif isinstance(output, Mapping):
        data = dict(output)
    elif hasattr(output, "__dict__"):
        data = {
            key: value
            for key, value in vars(output).items()
            if not key.startswith("_")
        }
    else:
        data = {}

    return {
        key: _jsonable(value)
        for key, value in data.items()
        if key not in known_fields
    }


def _extra_fields(output, known_fields):
    explicit = _field(output, "extra_fields", None)
    if explicit is not None:
        return _jsonable(explicit)
    return _public_extra_fields(output, known_fields)


def _matrix_shape(value):
    if value is None:
        return 0, 0

    if torch is not None and isinstance(value, torch.Tensor):
        shape = tuple(value.shape)
        if len(shape) == 0:
            return 0, 0
        if len(shape) == 1:
            return shape[0], 0
        return shape[0], shape[1]

    if np is not None and isinstance(value, np.ndarray):
        shape = value.shape
        if len(shape) == 0:
            return 0, 0
        if len(shape) == 1:
            return shape[0], 0
        return shape[0], shape[1]

    try:
        rows = len(value)
    except Exception:
        return 0, 0

    width = 0
    if rows:
        try:
            first = value[0]
        except Exception:
            try:
                first = next(iter(value))
            except Exception:
                first = None
        try:
            width = len(first) if first is not None else 0
        except Exception:
            width = 0

    return rows, width


class JsonlProbeWriter:
    def __init__(self, path, append=False):
        self.path = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        mode = "a" if append else "w"
        with open(self.path, mode, encoding="utf-8"):
            pass

    def write(self, kind, request_id=None, **payload):
        record = {
            "ts": time.time(),
            "kind": kind,
            "request_id": request_id,
            "payload": _jsonable(payload),
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")


class ObservedStudentManager:
    def __init__(self, inner, writer):
        self.inner = inner
        self.writer = writer

    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        image_data=None,
        video_data=None,
        **kwargs,
    ):
        self.writer.write(
            "student_generate_start",
            request_id=request_id,
            prompt_len=_safe_len(prompt_ids),
            prompt_tail=_tail(prompt_ids, 64),
            sampling_params=_jsonable(sampling_params),
            has_images=image_data is not None,
            has_videos=video_data is not None,
        )

        try:
            output = await self.inner.generate(
                request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
        except Exception as exc:
            self.writer.write(
                "student_generate_error",
                request_id=request_id,
                error=repr(exc),
            )
            raise

        known_fields = {
            "token_ids",
            "log_probs",
            "stop_reason",
            "num_preempted",
            "extra_fields",
        }
        self.writer.write(
            "student_generate_finish",
            request_id=request_id,
            token_ids=_jsonable(_field(output, "token_ids")),
            log_probs=_jsonable(_field(output, "log_probs")),
            stop_reason=_jsonable(_field(output, "stop_reason")),
            num_preempted=_jsonable(_field(output, "num_preempted")),
            extra_fields=_extra_fields(output, known_fields),
        )
        return output

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ObservedTeacherManager:
    def __init__(self, inner, writer):
        self.inner = inner
        self.writer = writer

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids,
        multi_modal_data=None,
        routing_key=None,
        request_id=None,
        logprob_start_len=0,
    ):
        sequence_len = _safe_len(sequence_ids)
        if sequence_len is None:
            expected_suffix_len = None
        elif logprob_start_len > 0:
            expected_suffix_len = sequence_len - logprob_start_len - 1
        else:
            expected_suffix_len = sequence_len - 1

        self.writer.write(
            "teacher_verify_start",
            request_id=request_id,
            sequence_len=sequence_len,
            sequence_tail=_tail(sequence_ids, 96),
            logprob_start_len=logprob_start_len,
            expected_suffix_len=expected_suffix_len,
            routing_key=_jsonable(routing_key),
            has_multimodal=multi_modal_data is not None,
        )

        try:
            result = await self.inner.compute_teacher_logprobs_single(
                sequence_ids,
                multi_modal_data=multi_modal_data,
                routing_key=routing_key,
                request_id=request_id,
                logprob_start_len=logprob_start_len,
            )
        except Exception as exc:
            self.writer.write(
                "teacher_verify_error",
                request_id=request_id,
                error=repr(exc),
            )
            raise

        teacher_ids, teacher_logprobs = result
        rows, width = _matrix_shape(teacher_ids)
        self.writer.write(
            "teacher_verify_finish",
            request_id=request_id,
            teacher_ids=_jsonable(teacher_ids),
            teacher_logprobs=_jsonable(teacher_logprobs),
            rows=rows,
            width=width,
        )
        return result

    def release_sticky_session(self, *args, **kwargs):
        release = getattr(self.inner, "release_sticky_session", None)
        if release is None:
            return None
        return release(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)
