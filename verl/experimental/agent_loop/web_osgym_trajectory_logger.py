from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


def _safe_component(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text or fallback


@dataclass(frozen=True)
class WebOsGymTrajectoryLogger:
    root_dir: Path

    def session_dir(self, *, task_id: Any, sample_uid: Any, global_step: Any, session_id: Any) -> Path:
        task_part = _safe_component(task_id, fallback="unknown_task")
        sample_part = _safe_component(sample_uid, fallback="unknown_uid")
        global_step_part = _safe_component(f"global_step_{global_step}", fallback="global_step_unknown")
        session_part = _safe_component(session_id, fallback="unknown_session")
        path = self.root_dir / f"{task_part}___{sample_part}___{global_step_part}___{session_part}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _write_png(image_dir: Path, image_name: str, image: Any) -> dict[str, Any] | None:
        if image is None or not hasattr(image, "save"):
            return None
        image_dir.mkdir(parents=True, exist_ok=True)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_path = image_dir / image_name
        image_path.write_bytes(image_bytes)
        width, height = getattr(image, "size", (None, None))
        return {
            "path": str(Path("images") / image_name),
            "width": width,
            "height": height,
        }

    def write_images(
        self,
        session_dir: Path,
        *,
        assistant_turn: int,
        user_turn: int,
        images: list[Any] | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for image_index, image in enumerate(images or []):
            image_name = f"a{assistant_turn:03d}_u{user_turn:03d}_{image_index:02d}.png"
            record = self._write_png(session_dir / "images", image_name, image)
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def append_event(self, session_dir: Path, record: dict[str, Any]) -> None:
        self._append_jsonl(session_dir / "trajectory.jsonl", record)

    def write_summary(self, session_dir: Path, summary: dict[str, Any]) -> None:
        summary_path = session_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
