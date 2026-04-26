import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _normalise_messages(value: Any) -> list[dict[str, Any]]:
    value = _jsonable(value)
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        if "role" in value and "content" in value:
            return [value]
        if "content" in value:
            return [{"role": value.get("role", "user"), "content": value["content"]}]
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": str(item)})
        if messages:
            return messages
    raise ValueError(f"Cannot convert prompt value to chat messages: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a parquet prompt dataset to live mixed-gen probe JSONL.")
    parser.add_argument("--dataset", required=True, help="Input parquet dataset path.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--data-source-key", default="data_source")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--uid-prefix", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    out_path = Path(args.out).expanduser()
    if args.limit <= 0:
        raise ValueError(f"--limit must be positive, got {args.limit}")
    if args.offset < 0:
        raise ValueError(f"--offset must be non-negative, got {args.offset}")
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    dataframe = pd.read_parquet(dataset_path)
    if args.prompt_key not in dataframe.columns:
        raise KeyError(f"Dataset {dataset_path} has no prompt key {args.prompt_key!r}; columns={list(dataframe.columns)}")

    uid_prefix = args.uid_prefix or dataset_path.stem.replace(".", "_").replace("-", "_")
    rows = dataframe.iloc[args.offset : args.offset + args.limit]
    if rows.empty:
        raise ValueError(f"No rows selected from {dataset_path} at offset={args.offset}, limit={args.limit}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for absolute_index, (_, row) in enumerate(rows.iterrows(), start=args.offset):
            row_dict = row.to_dict()
            messages = _normalise_messages(row_dict[args.prompt_key])
            data_source = row_dict.get(args.data_source_key) if args.data_source_key in row_dict else dataset_path.stem
            record = {
                "uid": f"{uid_prefix}_{absolute_index:06d}",
                "data_source": str(data_source or dataset_path.stem),
                "raw_prompt": messages,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1

    print(f"wrote {count} prompts to {out_path} from {dataset_path}")


if __name__ == "__main__":
    main()
