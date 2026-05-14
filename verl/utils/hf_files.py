import os
import shutil
from pathlib import Path


_WEIGHT_FILE_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".msgpack",
    ".onnx",
    ".tflite",
}

_WEIGHT_INDEX_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}


def is_hf_weight_file(path: str | os.PathLike[str]) -> bool:
    file_name = Path(path).name
    if file_name in _WEIGHT_INDEX_NAMES:
        return True
    return Path(file_name).suffix in _WEIGHT_FILE_SUFFIXES


def resolve_hf_non_weight_source_dir(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None

    candidate = Path(path)
    if not candidate.is_dir():
        return None

    huggingface_subdir = candidate / "huggingface"
    if huggingface_subdir.is_dir():
        return huggingface_subdir

    return candidate


def sync_hf_non_weight_files(
    source_dir: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    *,
    remove_extra: bool = True,
) -> None:
    """Copy non-parameter Hugging Face files from source to target byte-for-byte.

    This preserves tokenizer/config/processor metadata exactly as shipped by the
    original model while leaving merged parameter files in ``target_dir`` intact.
    """

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.is_dir():
        raise FileNotFoundError(f"Hugging Face source directory does not exist: {source_path}")

    target_path.mkdir(parents=True, exist_ok=True)

    source_files = {
        entry.name: entry
        for entry in source_path.iterdir()
        if entry.is_file() and not is_hf_weight_file(entry.name)
    }

    if remove_extra:
        for entry in target_path.iterdir():
            if not entry.is_file() or is_hf_weight_file(entry.name):
                continue
            if entry.name not in source_files:
                entry.unlink()

    for file_name, source_file in source_files.items():
        target_file = target_path / file_name
        if source_file.resolve() == target_file.resolve():
            continue
        shutil.copyfile(source_file, target_file)


def remove_hf_weight_files(target_dir: str | os.PathLike[str]) -> None:
    """Remove existing Hugging Face weight files from a target directory.

    This is used before writing a newly merged model into an existing target
    directory so that stale weight shards or index files do not survive across
    repeated merges.
    """

    target_path = Path(target_dir)
    if not target_path.is_dir():
        return

    for entry in target_path.iterdir():
        if entry.is_file() and is_hf_weight_file(entry.name):
            entry.unlink()
