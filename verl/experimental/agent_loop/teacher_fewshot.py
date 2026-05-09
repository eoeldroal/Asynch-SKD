from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image


def load_teacher_fewshot_transcript(path: str | Path) -> tuple[list[dict[str, Any]], list[Image.Image] | None]:
    transcript_path = Path(path).expanduser().resolve()
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("teacher few-shot transcript must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("teacher few-shot transcript must contain a 'messages' list")

    normalized_messages = deepcopy(messages)
    images: list[Image.Image] = []
    _normalize_message_list(normalized_messages, transcript_path.parent, images)
    return normalized_messages, images or None


def _normalize_message_list(messages: list[dict[str, Any]], base_dir: Path, images: list[Image.Image]) -> None:
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each teacher few-shot message must be an object")

        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported teacher few-shot role: {role!r}")

        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("teacher few-shot message content must be a string or a content list")

        normalized_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("teacher few-shot content items must be objects")

            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ValueError("teacher few-shot text content must provide a string 'text'")
                normalized_content.append({"type": "text", "text": text})
                continue

            if item_type == "image":
                image_ref = item.get("image")
                if not isinstance(image_ref, str) or not image_ref:
                    raise ValueError("teacher few-shot image content must provide a non-empty string 'image'")
                image_path = (base_dir / image_ref).resolve()
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                normalized_content.append({"type": "image"})
                continue

            raise ValueError(f"unsupported teacher few-shot content type: {item_type!r}")

        message["content"] = normalized_content
