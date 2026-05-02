# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

try:
    from tensordict import TensorDictBase
except ImportError:  # pragma: no cover - tensordict is always available in runtime env
    TensorDictBase = None

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - DTensor is available on supported torch builds
    DTensor = None

EXACT_SNAPSHOT_DIR_ENV = "VERL_VEOMNI_EXACT_SNAPSHOT_DIR"
EXACT_SNAPSHOT_ABORT_ENV = "VERL_VEOMNI_EXACT_SNAPSHOT_ABORT"

_EXACT_SNAPSHOT_DONE = False


def get_exact_snapshot_dir() -> Path | None:
    snapshot_dir = os.getenv(EXACT_SNAPSHOT_DIR_ENV)
    if not snapshot_dir:
        return None
    return Path(snapshot_dir).expanduser().resolve()


def should_abort_after_exact_snapshot() -> bool:
    raw_value = os.getenv(EXACT_SNAPSHOT_ABORT_ENV, "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def exact_snapshot_pending() -> bool:
    return get_exact_snapshot_dir() is not None and not _EXACT_SNAPSHOT_DONE


def mark_exact_snapshot_done() -> None:
    global _EXACT_SNAPSHOT_DONE
    _EXACT_SNAPSHOT_DONE = True


def _clone_tensor_for_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    if DTensor is not None and isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().cpu().clone()


def clone_snapshot_payload(payload: Any) -> Any:
    if isinstance(payload, torch.Tensor):
        return _clone_tensor_for_snapshot(payload)

    if DTensor is not None and isinstance(payload, DTensor):
        return _clone_tensor_for_snapshot(payload)

    if TensorDictBase is not None and isinstance(payload, TensorDictBase):
        cloned = copy.deepcopy(payload)
        return cloned.apply(_clone_tensor_for_snapshot)

    if isinstance(payload, dict):
        return {key: clone_snapshot_payload(value) for key, value in payload.items()}

    if isinstance(payload, list):
        return [clone_snapshot_payload(value) for value in payload]

    if isinstance(payload, tuple):
        return tuple(clone_snapshot_payload(value) for value in payload)

    return copy.deepcopy(payload)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    return repr(value)


def write_exact_snapshot(
    snapshot_dir: str | Path,
    rank: int,
    model_inputs: dict[str, Any],
    output_args: dict[str, Any],
    micro_batch: Any,
    meta: dict[str, Any],
) -> dict[str, Path]:
    snapshot_root = Path(snapshot_dir).expanduser().resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)

    prefix = f"rank{rank:02d}"
    paths = {
        "model_inputs": snapshot_root / f"{prefix}_model_inputs.pt",
        "output_args": snapshot_root / f"{prefix}_output_args.pt",
        "micro_batch": snapshot_root / f"{prefix}_micro_batch.pt",
        "meta": snapshot_root / f"{prefix}_meta.json",
    }

    torch.save(clone_snapshot_payload(model_inputs), paths["model_inputs"])
    torch.save(clone_snapshot_payload(output_args), paths["output_args"])
    torch.save(clone_snapshot_payload(micro_batch), paths["micro_batch"])
    paths["meta"].write_text(json.dumps(_json_ready(meta), indent=2, sort_keys=True))

    return paths
