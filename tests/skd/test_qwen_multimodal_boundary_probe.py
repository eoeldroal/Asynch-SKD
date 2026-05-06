from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import subprocess


def _load_probe_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "debug"
        / "qwen_multimodal_boundary_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen_multimodal_boundary_probe_under_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_probe_module()


async def _return_row(row):
    return row


def test_probe_fails_when_no_requested_models_are_available(monkeypatch, tmp_path, capsys):
    models_root = tmp_path / "models"
    models_root.mkdir()

    monkeypatch.setattr(
        probe,
        "_parse_args",
        lambda: SimpleNamespace(
            models_root=models_root,
            image_dir=tmp_path / "images",
            image_path=tmp_path / "fake.png",
            models=["missing-model"],
        ),
    )
    monkeypatch.setattr(probe, "import_processors", lambda _: None)
    monkeypatch.setattr(probe, "_pick_image_path", lambda image_dir, image_path: image_path)

    exit_code = asyncio.run(probe._run())

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "missing-model" in captured.out


def test_probe_fails_when_compact_or_expanded_invariant_breaks(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    model_path = models_root / "Qwen3.5-9B"
    model_path.mkdir(parents=True)

    monkeypatch.setattr(
        probe,
        "_parse_args",
        lambda: SimpleNamespace(
            models_root=models_root,
            image_dir=tmp_path / "images",
            image_path=tmp_path / "fake.png",
            models=["Qwen3.5-9B"],
        ),
    )
    monkeypatch.setattr(probe, "import_processors", lambda _: None)
    monkeypatch.setattr(probe, "_pick_image_path", lambda image_dir, image_path: image_path)
    monkeypatch.setattr(
        probe,
        "_probe_model",
        lambda model_path, image_path: _return_row(
            probe.ProbeRow(
                model=model_path.name,
                status="ok",
                compact_len=10,
                expanded_len=20,
                expanded_errors=False,
                compact_matches_local_expanded=False,
            )
        ),
    )

    exit_code = asyncio.run(probe._run())

    assert exit_code != 0


def test_probe_succeeds_only_when_boundary_invariants_hold(monkeypatch, tmp_path):
    models_root = tmp_path / "models"
    model_path = models_root / "Qwen3.5-9B"
    model_path.mkdir(parents=True)

    monkeypatch.setattr(
        probe,
        "_parse_args",
        lambda: SimpleNamespace(
            models_root=models_root,
            image_dir=tmp_path / "images",
            image_path=tmp_path / "fake.png",
            models=["Qwen3.5-9B"],
        ),
    )
    monkeypatch.setattr(probe, "import_processors", lambda _: None)
    monkeypatch.setattr(probe, "_pick_image_path", lambda image_dir, image_path: image_path)
    monkeypatch.setattr(
        probe,
        "_probe_model",
        lambda model_path, image_path: _return_row(
            probe.ProbeRow(
                model=model_path.name,
                status="ok",
                compact_len=10,
                expanded_len=20,
                expanded_errors=True,
                compact_matches_local_expanded=True,
            )
        ),
    )

    exit_code = asyncio.run(probe._run())

    assert exit_code == 0


def test_default_image_dir_falls_back_to_available_git_worktree(monkeypatch, tmp_path):
    repo_root = tmp_path / "linked-worktree"
    repo_root.mkdir()
    main_root = tmp_path / "main-worktree"
    image_dir = (
        main_root
        / "logs"
        / "rollout_data"
        / "qwen35_webgym_fully_async_tool_veomni"
        / "webgym_tool_trace"
        / "images"
    )
    image_dir.mkdir(parents=True)

    monkeypatch.setattr(probe, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"worktree {repo_root}\n\nworktree {main_root}\n",
        ),
    )

    resolved = probe._default_image_dir()

    assert resolved == image_dir


def test_default_models_root_falls_back_to_available_git_worktree(monkeypatch, tmp_path):
    repo_root = tmp_path / "linked-worktree"
    repo_root.mkdir()
    main_root = tmp_path / "main-worktree"
    models_root = main_root / "models"
    models_root.mkdir(parents=True)

    monkeypatch.setattr(probe, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"worktree {repo_root}\n\nworktree {main_root}\n",
        ),
    )

    resolved = probe._default_models_root()

    assert resolved == models_root
