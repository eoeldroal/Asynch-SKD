#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from verl.trainer.config import CheckpointConfig
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig
from verl.workers.engine.veomni.transformer_impl import VeOmniEngineWithLMHead


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _move_to_device(payload: Any, device: torch.device) -> Any:
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    if isinstance(payload, dict):
        return {key: _move_to_device(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(value, device) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_move_to_device(value, device) for value in payload)
    if hasattr(payload, "to"):
        try:
            return payload.to(device)
        except Exception:
            return payload
    return payload


def _resolve_module_attr(module: Any, attr_name: str) -> Any:
    current = module
    seen = set()
    while not hasattr(current, attr_name) and hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    if not hasattr(current, attr_name) and hasattr(current, "model") and hasattr(current.model, attr_name):
        return getattr(current.model, attr_name)
    if not hasattr(current, attr_name):
        raise AttributeError(f"Could not resolve attribute '{attr_name}' from wrapped module type {type(module).__name__}")
    return getattr(current, attr_name)


def _output_summary(output: Any) -> dict[str, Any]:
    if isinstance(output, torch.Tensor):
        return {"tensor_shape": list(output.shape), "dtype": str(output.dtype)}
    if hasattr(output, "logits"):
        logits = output.logits
        return {"logits_shape": list(logits.shape), "dtype": str(logits.dtype)}
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return {"tuple0_shape": list(output[0].shape), "dtype": str(output[0].dtype)}
    if hasattr(output, "last_hidden_state"):
        lhs = output.last_hidden_state
        return {"last_hidden_state_shape": list(lhs.shape), "dtype": str(lhs.dtype)}
    return {"output_type": type(output).__name__}

def _build_engine(
    meta: dict[str, Any],
    world_size: int,
    model_path: str,
    override_enable_gradient_checkpointing: bool | None = None,
) -> VeOmniEngineWithLMHead:
    hf_attn_implementation = str(meta["attn_implementation"])
    hf_attn_implementation = hf_attn_implementation.replace("veomni_", "")
    hf_attn_implementation = hf_attn_implementation.replace("_with_sp", "")
    engine_attn_implementation = str(meta["attn_implementation"])
    if os.getenv("MODELING_BACKEND") == "hf":
        engine_attn_implementation = hf_attn_implementation
    init_device = str(meta.get("init_device", "meta"))
    if world_size == 1 and init_device in {"meta", "cpu"}:
        init_device = "cuda"

    model_config = HFModelConfig(
        path=model_path,
        load_tokenizer=False,
        enable_gradient_checkpointing=(
            bool(meta.get("enable_gradient_checkpointing", True))
            if override_enable_gradient_checkpointing is None
            else override_enable_gradient_checkpointing
        ),
        use_remove_padding=bool(meta.get("use_remove_padding", True)),
        override_config={"attn_implementation": hf_attn_implementation},
    )
    engine_config = VeOmniEngineConfig(
        strategy="veomni",
        forward_only=True,
        param_offload=bool(meta.get("param_offload", False)),
        optimizer_offload=bool(meta.get("optimizer_offload", False)),
        use_torch_compile=bool(meta.get("use_torch_compile", True)),
        fsdp_size=-1 if world_size > 1 else 1,
        ulysses_parallel_size=int(meta.get("ulysses_parallel_size", 1)),
        expert_parallel_size=int(meta.get("expert_parallel_size", 1)),
        enable_full_shard=bool(meta.get("enable_full_shard", False)),
        mixed_precision=bool(meta.get("mixed_precision", False)),
        init_device=init_device,
        activation_gpu_limit=float(meta.get("activation_gpu_limit", 0.0)),
        enable_reentrant=bool(meta.get("enable_reentrant", False)),
        forward_prefetch=bool(meta.get("forward_prefetch", False)),
        basic_modules=list(meta.get("basic_modules", [])),
        attn_implementation=engine_attn_implementation,
        moe_implementation=str(meta.get("moe_implementation", "fused")),
        seed=42,
    )
    optimizer_config = VeOmniOptimizerConfig(total_training_steps=1, lr=1e-6)
    checkpoint_config = CheckpointConfig()
    engine = VeOmniEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    engine.initialize()
    return engine


def _run_mode(
    engine: VeOmniEngineWithLMHead,
    mode: str,
    model_inputs: dict[str, Any],
    freeze_vision_tower: bool = False,
) -> tuple[float, dict[str, Any]]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    if freeze_vision_tower:
        visual = _resolve_module_attr(engine.module, "visual")
        visual.requires_grad_(False)
        visual.eval()

    dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with engine.train_mode(disable_auto_offload=True), engine.model_fwd_context, torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        if mode == "full":
            output = engine.module(**model_inputs, use_cache=False)
        elif mode == "vision_only":
            get_image_features = _resolve_module_attr(engine.module, "get_image_features")
            context = torch.no_grad() if freeze_vision_tower else nullcontext()
            with context:
                output = get_image_features(
                    pixel_values=model_inputs["pixel_values"],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed_s = time.perf_counter() - start
    return elapsed_s, _output_summary(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exact VeOmni self.module inputs captured from training.")
    parser.add_argument("--snapshot-dir", required=True, help="Directory containing rankXX snapshot files.")
    parser.add_argument("--model-path", required=True, help="Path to the HF model used in training.")
    parser.add_argument(
        "--mode",
        choices=["full", "vision_only"],
        default="full",
        help="Replay the full self.module forward or only the vision/image feature extraction path.",
    )
    parser.add_argument(
        "--source-rank",
        type=int,
        default=None,
        help="Load snapshot from this rank instead of the current distributed rank. Useful for 1-rank controls.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path template for rank-local JSON results. Supports {rank}. Defaults inside snapshot dir.",
    )
    parser.add_argument(
        "--override-enable-gradient-checkpointing",
        choices=["true", "false"],
        default=None,
        help="Override the captured enable_gradient_checkpointing setting for replay.",
    )
    parser.add_argument(
        "--freeze-vision-tower",
        choices=["true", "false"],
        default="false",
        help="Replay with the vision tower frozen and executed under no_grad.",
    )
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    source_rank = rank if args.source_rank is None else args.source_rank

    meta = _load_json(snapshot_dir / f"rank{source_rank:02d}_meta.json")
    model_inputs = _load_torch(snapshot_dir / f"rank{source_rank:02d}_model_inputs.pt")
    model_inputs = _move_to_device(model_inputs, torch.device(f"cuda:{local_rank}"))

    override_enable_gradient_checkpointing = None
    if args.override_enable_gradient_checkpointing is not None:
        override_enable_gradient_checkpointing = args.override_enable_gradient_checkpointing == "true"

    engine = _build_engine(
        meta=meta,
        world_size=world_size,
        model_path=args.model_path,
        override_enable_gradient_checkpointing=override_enable_gradient_checkpointing,
    )

    freeze_vision_tower = args.freeze_vision_tower == "true"
    elapsed_s, summary = _run_mode(
        engine=engine,
        mode=args.mode,
        model_inputs=model_inputs,
        freeze_vision_tower=freeze_vision_tower,
    )

    effective_enable_gradient_checkpointing = (
        bool(meta.get("enable_gradient_checkpointing", True))
        if override_enable_gradient_checkpointing is None
        else override_enable_gradient_checkpointing
    )

    result = {
        "rank": rank,
        "world_size": world_size,
        "source_rank": source_rank,
        "mode": args.mode,
        "elapsed_s": round(elapsed_s, 4),
        "attn_implementation": meta.get("attn_implementation"),
        "use_remove_padding": meta.get("use_remove_padding"),
        "use_torch_compile": meta.get("use_torch_compile"),
        "enable_gradient_checkpointing": effective_enable_gradient_checkpointing,
        "freeze_vision_tower": freeze_vision_tower,
        "input_ids_shape": meta.get("input_ids_shape"),
        "pixel_values_shape": meta.get("pixel_values_shape"),
        "image_grid_shape": meta.get("image_grid_shape"),
        "summary": summary,
    }

    save_json_template = args.save_json or str(snapshot_dir / f"replay_{args.mode}_rank{{rank:02d}}.json")
    save_json_path = Path(save_json_template.format(rank=rank)).expanduser().resolve()
    save_json_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
