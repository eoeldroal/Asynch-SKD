from pathlib import Path

from verl.utils.hf_files import resolve_hf_non_weight_source_dir, sync_hf_non_weight_files


def test_resolve_hf_non_weight_source_dir_prefers_huggingface_subdir(tmp_path: Path):
    source_root = tmp_path / "checkpoint"
    source_root.mkdir()
    huggingface_dir = source_root / "huggingface"
    huggingface_dir.mkdir()
    (huggingface_dir / "config.json").write_text('{"model_type":"qwen"}')

    assert resolve_hf_non_weight_source_dir(source_root) == huggingface_dir
    assert resolve_hf_non_weight_source_dir(huggingface_dir) == huggingface_dir
    assert resolve_hf_non_weight_source_dir(source_root / "missing") is None
    assert resolve_hf_non_weight_source_dir(None) is None


def test_sync_hf_non_weight_files_copies_metadata_and_preserves_weight_files(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()

    (source_dir / "config.json").write_text('{"hidden_size":4096}')
    (source_dir / "tokenizer.json").write_text('{"tokenizer":"qwen"}')
    (source_dir / "model.safetensors").write_text("source-weights")
    (source_dir / "pytorch_model.bin.index.json").write_text('{"weight_map":{}}')

    preserved_target_weight = target_dir / "model.safetensors"
    preserved_target_weight.write_text("merged-weights")
    (target_dir / "config.json").write_text('{"stale":true}')
    (target_dir / "obsolete.txt").write_text("remove me")

    sync_hf_non_weight_files(source_dir, target_dir)

    assert (target_dir / "config.json").read_text() == '{"hidden_size":4096}'
    assert (target_dir / "tokenizer.json").read_text() == '{"tokenizer":"qwen"}'
    assert preserved_target_weight.read_text() == "merged-weights"
    assert not (target_dir / "obsolete.txt").exists()
    assert not (target_dir / "pytorch_model.bin.index.json").exists()
